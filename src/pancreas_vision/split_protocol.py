from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from pancreas_vision.training import EvaluationMetrics


@dataclass(frozen=True)
class BagSplitRow:
    bag_id: str
    lesion_id: str
    label_name: str
    source_buckets: str
    instance_count: int
    split_name: str
    split_role: str
    split_seed: int
    split_ratio: str


@dataclass(frozen=True)
class FoldAssignmentRow:
    bag_id: str
    lesion_id: str
    label_name: str
    source_buckets: str
    fold_index: int
    fold_role: str
    split_seed: int
    cv_scheme: str


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(
    output_path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_train_test_split(
    bag_manifest_rows: list[dict[str, str]],
    test_size: float = 0.3,
    random_seed: int = 42,
) -> list[BagSplitRow]:
    eligible_rows = [row for row in bag_manifest_rows if row["label_name"] != "CONFLICT"]
    bag_ids = [row["bag_id"] for row in eligible_rows]
    labels = [row["label_name"] for row in eligible_rows]

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_seed,
    )
    train_indices, test_indices = next(splitter.split(bag_ids, labels))
    train_index_set = set(train_indices.tolist())

    split_rows: list[BagSplitRow] = []
    for index, row in enumerate(eligible_rows):
        split_rows.append(
            BagSplitRow(
                bag_id=row["bag_id"],
                lesion_id=row["lesion_id"],
                label_name=row["label_name"],
                source_buckets=row["source_buckets"],
                instance_count=int(row["instance_count"]),
                split_name="main_train_test",
                split_role="train" if index in train_index_set else "test",
                split_seed=random_seed,
                split_ratio=f"{int(round((1 - test_size) * 10))}:{int(round(test_size * 10))}",
            )
        )
    return split_rows


def build_grouped_folds(
    bag_manifest_rows: list[dict[str, str]],
    n_splits: int = 5,
    random_seed: int = 42,
) -> list[FoldAssignmentRow]:
    eligible_rows = [row for row in bag_manifest_rows if row["label_name"] != "CONFLICT"]
    bag_ids = [row["bag_id"] for row in eligible_rows]
    labels = [row["label_name"] for row in eligible_rows]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    fold_rows: list[FoldAssignmentRow] = []
    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(bag_ids, labels)):
        train_index_set = set(train_indices.tolist())
        for index, row in enumerate(eligible_rows):
            fold_rows.append(
                FoldAssignmentRow(
                    bag_id=row["bag_id"],
                    lesion_id=row["lesion_id"],
                    label_name=row["label_name"],
                    source_buckets=row["source_buckets"],
                    fold_index=fold_index,
                    fold_role="train" if index in train_index_set else "test",
                    split_seed=random_seed,
                    cv_scheme=f"grouped_stratified_{n_splits}fold",
                )
            )
    return fold_rows


def build_split_summary(
    bag_manifest_rows: list[dict[str, str]],
    split_rows: list[BagSplitRow],
    fold_rows: list[FoldAssignmentRow],
    random_seed: int,
    test_size: float,
    n_splits: int,
) -> dict[str, object]:
    main_split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in split_rows:
        main_split_counts[row.split_role][row.label_name] += 1

    fold_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for row in fold_rows:
        fold_counts[str(row.fold_index)][row.fold_role][row.label_name] += 1

    return {
        "seed": random_seed,
        "test_size": test_size,
        "n_splits": n_splits,
        "formal_bag_count": len([row for row in bag_manifest_rows if row["label_name"] != "CONFLICT"]),
        "label_distribution": dict(
            sorted(Counter(row["label_name"] for row in bag_manifest_rows if row["label_name"] != "CONFLICT").items())
        ),
        "main_split_counts": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(main_split_counts.items())
        },
        "fold_counts": {
            fold_index: {
                role: dict(sorted(counts.items()))
                for role, counts in sorted(role_map.items())
            }
            for fold_index, role_map in sorted(fold_counts.items())
        },
    }


def render_split_summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Split Protocol V1 Summary",
        "",
        f"- Seed: {summary['seed']}",
        f"- Main split ratio: 7:3 (test_size={summary['test_size']})",
        f"- CV scheme: grouped stratified {summary['n_splits']}-fold",
        f"- Formal bag count: {summary['formal_bag_count']}",
        "",
        "## Overall Label Distribution",
        "",
    ]
    for label_name, count in summary["label_distribution"].items():  # type: ignore[index]
        lines.append(f"- {label_name}: {count} bags")

    lines.extend(["", "## Main Train/Test Split", ""])
    for split_role, counts in summary["main_split_counts"].items():  # type: ignore[index]
        line = ", ".join(f"{label_name}={count}" for label_name, count in counts.items())
        lines.append(f"- {split_role}: {line}")

    lines.extend(["", "## Fold Counts", ""])
    for fold_index, role_map in summary["fold_counts"].items():  # type: ignore[index]
        role_text = "; ".join(
            f"{role}: " + ", ".join(f"{label_name}={count}" for label_name, count in counts.items())
            for role, counts in role_map.items()
        )
        lines.append(f"- fold {fold_index}: {role_text}")

    lines.append("")
    return "\n".join(lines)


def build_evaluation_template() -> dict[str, object]:
    empty_metrics = EvaluationMetrics(
        accuracy=0.0,
        sensitivity=0.0,
        specificity=0.0,
        roc_auc=0.0,
        true_negative=0,
        false_positive=0,
        false_negative=0,
        true_positive=0,
    )
    return {
        "evaluation_protocol_version": "split_eval_v1",
        "required_outputs": [
            "bag_level_metrics",
            "bag_level_predictions",
            "error_by_source_bucket",
        ],
        "bag_level_metrics": empty_metrics.__dict__,
        "bag_level_predictions_columns": [
            "bag_id",
            "true_label_name",
            "predicted_label_name",
            "positive_score",
            "correct",
            "source_buckets",
            "instance_count",
            "dominant_channel",
            "dominant_magnification",
        ],
        "error_by_source_bucket_columns": [
            "source_bucket",
            "bag_count",
            "error_count",
            "false_positive_count",
            "false_negative_count",
            "accuracy",
        ],
        "aggregation_defaults": {
            "prediction_unit": "bag",
            "instance_to_bag_score_reducer": "mean_positive_score",
            "positive_class_name": "PanIN",
        },
    }


def write_split_outputs(
    output_dir: Path,
    split_rows: list[BagSplitRow],
    fold_rows: list[FoldAssignmentRow],
    summary: dict[str, object],
    evaluation_template: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_rows(
        output_dir / "main_split.csv",
        [
            "bag_id",
            "lesion_id",
            "label_name",
            "source_buckets",
            "instance_count",
            "split_name",
            "split_role",
            "split_seed",
            "split_ratio",
        ],
        [row.__dict__ for row in split_rows],
    )
    _write_csv_rows(
        output_dir / "grouped_5fold.csv",
        [
            "bag_id",
            "lesion_id",
            "label_name",
            "source_buckets",
            "fold_index",
            "fold_role",
            "split_seed",
            "cv_scheme",
        ],
        [row.__dict__ for row in fold_rows],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_split_summary_markdown(summary),
        encoding="utf-8",
    )
    (output_dir / "evaluation_template.json").write_text(
        json.dumps(evaluation_template, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
