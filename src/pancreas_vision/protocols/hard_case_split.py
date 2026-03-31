"""Hard-case split construction: force specific bags into test set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pancreas_vision.protocols.split_protocol import BagSplitRow


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load rows from CSV file."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_hard_case_split(
    bag_manifest_rows: list[dict[str, str]],
    force_test_bags: list[str],
    test_size: float = 0.3,
    random_seed: int = 42,
    split_name: str = "hard_case_split",
) -> list[BagSplitRow]:
    """Build train/test split forcing specific bags into test set.

    This is designed to evaluate model performance on known hard cases,
    particularly multichannel_kc_adm bags that have historically caused
    false positive errors.

    Args:
        bag_manifest_rows: List of bag manifest rows
        force_test_bags: List of bag_ids that MUST be in test set
        test_size: Approximate test set ratio (after accounting for forced bags)
        random_seed: Random seed for reproducibility
        split_name: Name for this split

    Returns:
        List of BagSplitRow objects
    """
    from pancreas_vision.protocols.split_protocol import BagSplitRow
    from sklearn.model_selection import StratifiedShuffleSplit

    # Filter eligible bags
    eligible_rows = [row for row in bag_manifest_rows if row["label_name"] != "CONFLICT"]

    # Separate forced test bags from remaining pool
    force_test_set = set(force_test_bags)
    forced_test_rows = [row for row in eligible_rows if row["bag_id"] in force_test_set]
    remaining_rows = [row for row in eligible_rows if row["bag_id"] not in force_test_set]

    # Check that forced bags exist
    missing_bags = force_test_set - {row["bag_id"] for row in eligible_rows}
    if missing_bags:
        raise ValueError(f"Force test bags not found in manifest: {missing_bags}")

    # Stratified split on remaining bags
    remaining_bag_ids = [row["bag_id"] for row in remaining_rows]
    remaining_labels = [row["label_name"] for row in remaining_rows]

    # Calculate adjusted test size to account for forced bags
    # We want approximately test_size overall, but forced bags are already in test
    forced_ratio = len(forced_test_rows) / len(eligible_rows)
    adjusted_test_size = max(0.1, test_size - forced_ratio)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=adjusted_test_size,
        random_state=random_seed,
    )

    # Handle edge case: if no remaining bags, skip split
    if remaining_rows:
        train_indices, test_indices = next(splitter.split(remaining_bag_ids, remaining_labels))
        train_index_set = set(train_indices.tolist())
    else:
        train_index_set = set()

    # Build split rows
    split_rows: list[BagSplitRow] = []

    # Add forced test bags first
    for row in forced_test_rows:
        split_rows.append(
            BagSplitRow(
                bag_id=row["bag_id"],
                lesion_id=row["lesion_id"],
                label_name=row["label_name"],
                source_buckets=row["source_buckets"],
                instance_count=int(row["instance_count"]),
                split_name=split_name,
                split_role="test",
                split_seed=random_seed,
                split_ratio=f"{int(round((1 - test_size) * 10))}:{int(round(test_size * 10))}",
            )
        )

    # Add remaining bags with stratified assignment
    for index, row in enumerate(remaining_rows):
        split_rows.append(
            BagSplitRow(
                bag_id=row["bag_id"],
                lesion_id=row["lesion_id"],
                label_name=row["label_name"],
                source_buckets=row["source_buckets"],
                instance_count=int(row["instance_count"]),
                split_name=split_name,
                split_role="train" if index in train_index_set else "test",
                split_seed=random_seed,
                split_ratio=f"{int(round((1 - test_size) * 10))}:{int(round(test_size * 10))}",
            )
        )

    return split_rows


def build_hard_case_split_summary(
    bag_manifest_rows: list[dict[str, str]],
    split_rows: list[BagSplitRow],
    force_test_bags: list[str],
    random_seed: int,
    test_size: float,
) -> dict[str, object]:
    """Build summary for hard-case split."""
    from collections import Counter, defaultdict

    # Count by role and label
    role_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in split_rows:
        role_counts[row.split_role][row.label_name] += 1

    # Check forced bags
    forced_in_test = [row for row in split_rows if row.bag_id in force_test_bags and row.split_role == "test"]
    forced_label_distribution = Counter(row.label_name for row in forced_in_test)

    return {
        "split_type": "hard_case_split",
        "seed": random_seed,
        "target_test_size": test_size,
        "forced_test_bags": force_test_bags,
        "forced_bag_count": len(force_test_bags),
        "forced_labels": dict(sorted(forced_label_distribution.items())),
        "actual_test_count": len([row for row in split_rows if row.split_role == "test"]),
        "actual_train_count": len([row for row in split_rows if row.split_role == "train"]),
        "role_distribution": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(role_counts.items())
        },
        "total_bag_count": len(split_rows),
    }


def write_hard_case_split(
    output_path: Path,
    split_rows: list[BagSplitRow],
    summary: dict[str, object],
) -> None:
    """Write hard-case split to CSV and summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fieldnames = [
        "bag_id",
        "lesion_id",
        "label_name",
        "source_buckets",
        "instance_count",
        "split_name",
        "split_role",
        "split_seed",
        "split_ratio",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in split_rows:
            writer.writerow(row.__dict__)

    # Write summary
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """CLI entry point for building hard-case split."""
    parser = argparse.ArgumentParser(
        description="Build train/test split forcing specific bags into test set"
    )
    parser.add_argument(
        "--bag-manifest",
        type=Path,
        required=True,
        help="Path to bag_manifest.csv",
    )
    parser.add_argument(
        "--force-test",
        nargs="+",
        required=True,
        help="Bag IDs to force into test set (e.g., LESION_MC_001 LESION_MC_002)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (summary JSON will be written alongside)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Target test set ratio (default: 0.3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="hard_case_split",
        help="Split name identifier",
    )
    args = parser.parse_args()

    # Load manifest
    bag_manifest_rows = load_csv_rows(args.bag_manifest)

    # Build split
    split_rows = build_hard_case_split(
        bag_manifest_rows=bag_manifest_rows,
        force_test_bags=args.force_test,
        test_size=args.test_size,
        random_seed=args.seed,
        split_name=args.split_name,
    )

    # Build summary
    summary = build_hard_case_split_summary(
        bag_manifest_rows=bag_manifest_rows,
        split_rows=split_rows,
        force_test_bags=args.force_test,
        random_seed=args.seed,
        test_size=args.test_size,
    )

    # Write outputs
    write_hard_case_split(
        output_path=args.output,
        split_rows=split_rows,
        summary=summary,
    )

    # Print summary
    print(f"Hard-case split written to: {args.output}")
    print(f"Summary written to: {args.output.with_suffix('.json')}")
    print(f"Forced test bags: {args.force_test}")
    print(f"Test set size: {summary['actual_test_count']} bags")
    print(f"Train set size: {summary['actual_train_count']} bags")


if __name__ == "__main__":
    main()