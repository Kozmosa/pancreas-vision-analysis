from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pancreas_vision.data import (
    ImageRecord,
    discover_records,
    load_metadata_index,
    normalize_metadata_label,
)


FORMAL_SOURCE_BUCKETS = (
    "caerulein_adm",
    "KC",
    "KPC",
    "multichannel_adm",
    "multichannel_kc_adm",
)


@dataclass(frozen=True)
class InstanceRow:
    instance_id: str
    bag_id: str
    image_path: str
    label_name: str
    source_bucket: str
    magnification: str
    channel_name: str
    sample_type: str
    is_roi: int
    split_key: str
    lesion_id: str
    record_key: str
    label_source: str
    crop_box: str


@dataclass(frozen=True)
class BagRow:
    bag_id: str
    lesion_id: str
    label_name: str
    source_buckets: str
    instance_count: int
    whole_image_count: int
    roi_count: int
    magnifications: str
    channel_names: str
    sample_types: str
    has_mixed_source_bucket: int
    has_unknown_magnification: int
    has_unknown_channel: int
    review_flag_count: int
    review_flags: str


def _sorted_join(values: Iterable[str]) -> str:
    cleaned = sorted({value for value in values if value})
    return "|".join(cleaned)


def _example_paths(records: Iterable[ImageRecord], limit: int = 3) -> str:
    unique_paths: list[str] = []
    seen: set[str] = set()
    for record in sorted(records, key=lambda item: item.record_key):
        path_text = record.image_path.as_posix()
        if path_text in seen:
            continue
        seen.add(path_text)
        unique_paths.append(path_text)
        if len(unique_paths) >= limit:
            break
    return " | ".join(unique_paths)


def _crop_box_text(record: ImageRecord) -> str:
    if record.crop_box is None:
        return ""
    return ",".join(str(value) for value in record.crop_box)


def _build_instance_rows(records: list[ImageRecord]) -> list[InstanceRow]:
    rows: list[InstanceRow] = []
    for index, record in enumerate(
        sorted(records, key=lambda item: (item.lesion_id, item.sample_type, item.record_key)),
        start=1,
    ):
        bag_id = record.lesion_id
        rows.append(
            InstanceRow(
                instance_id=f"INSTANCE_{index:04d}",
                bag_id=bag_id,
                image_path=record.image_path.as_posix(),
                label_name=record.label_name,
                source_bucket=record.source_bucket,
                magnification=record.magnification,
                channel_name=record.channel_name,
                sample_type=record.sample_type,
                is_roi=1 if record.sample_type == "roi_crop" else 0,
                split_key=bag_id,
                lesion_id=record.lesion_id,
                record_key=record.record_key,
                label_source=record.label_source,
                crop_box=_crop_box_text(record),
            )
        )
    return rows


def _bag_review_flags(records: list[ImageRecord]) -> list[str]:
    flags: list[str] = []
    sample_types = {record.sample_type for record in records}
    source_buckets = {record.source_bucket for record in records}
    labels = {record.label_name for record in records}
    magnifications = {record.magnification for record in records}
    channels = {record.channel_name for record in records}

    if len(records) == 1:
        flags.append("single_view_bag")
    if "unknown" in magnifications:
        flags.append("missing_magnification")
    if "unknown" in channels:
        flags.append("missing_channel")
    if len(source_buckets) > 1:
        flags.append("mixed_source_bucket")
    if len(labels) > 1:
        flags.append("label_conflict")
    if sample_types == {"roi_crop"}:
        flags.append("roi_only_bag")

    whole_view_slots: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        if record.sample_type != "whole_image":
            continue
        slot_key = (record.magnification, record.channel_name)
        whole_view_slots[slot_key].add(record.image_path.name)
        if record.source_bucket == "KPC" and record.image_path.name.lower().startswith("kc-"):
            flags.append("filename_bucket_conflict")
    if any(len(names) > 1 for names in whole_view_slots.values()):
        flags.append("duplicate_view_slot")

    return sorted(set(flags))


def _build_bag_rows(records: list[ImageRecord]) -> list[BagRow]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.lesion_id].append(record)

    bag_rows: list[BagRow] = []
    for bag_id, bag_records in sorted(grouped.items()):
        review_flags = _bag_review_flags(bag_records)
        label_names = sorted({record.label_name for record in bag_records})
        bag_rows.append(
            BagRow(
                bag_id=bag_id,
                lesion_id=bag_id,
                label_name=label_names[0] if len(label_names) == 1 else "CONFLICT",
                source_buckets=_sorted_join(record.source_bucket for record in bag_records),
                instance_count=len(bag_records),
                whole_image_count=sum(record.sample_type == "whole_image" for record in bag_records),
                roi_count=sum(record.sample_type == "roi_crop" for record in bag_records),
                magnifications=_sorted_join(record.magnification for record in bag_records),
                channel_names=_sorted_join(record.channel_name for record in bag_records),
                sample_types=_sorted_join(record.sample_type for record in bag_records),
                has_mixed_source_bucket=1 if len({record.source_bucket for record in bag_records}) > 1 else 0,
                has_unknown_magnification=1 if any(record.magnification == "unknown" for record in bag_records) else 0,
                has_unknown_channel=1 if any(record.channel_name == "unknown" for record in bag_records) else 0,
                review_flag_count=len(review_flags),
                review_flags="|".join(review_flags),
            )
        )
    return bag_rows


def _load_metadata_rows_by_path(
    data_root: Path,
    metadata_csv: Path | None,
) -> dict[str, dict[str, str]]:
    return load_metadata_index(data_root=data_root, metadata_csv=metadata_csv)


def _build_unresolved_review_rows(
    data_root: Path,
    metadata_csv: Path | None,
) -> list[dict[str, str]]:
    rows_by_path = _load_metadata_rows_by_path(data_root=data_root, metadata_csv=metadata_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for image_path_str, row in rows_by_path.items():
        image_path = Path(image_path_str)
        if image_path.parent.name != "multichannel_unresolved":
            continue
        if not image_path.exists():
            continue
        lesion_id = row.get("lesion_id", "").strip()
        if not lesion_id:
            lesion_id = f"UNRESOLVED:{image_path.stem}"
        grouped[lesion_id].append(row)

    review_rows: list[dict[str, str]] = []
    for lesion_id, grouped_rows in sorted(grouped.items()):
        flags = ["unresolved_bucket_excluded"]
        if any(row.get("needs_manual_check", "1") == "1" for row in grouped_rows):
            flags.append("needs_manual_check")
        if any(row.get("exclude_from_train", "1") != "0" for row in grouped_rows):
            flags.append("excluded_from_train")
        label_names = sorted(
            {
                label_name
                for label_name in (
                    normalize_metadata_label(row) for row in grouped_rows
                )
                if label_name is not None
            }
        )
        review_rows.append(
            {
                "bag_id": lesion_id,
                "label_name": label_names[0] if len(label_names) == 1 else "UNRESOLVED",
                "source_buckets": "multichannel_unresolved",
                "instance_count": str(len(grouped_rows)),
                "review_flags": "|".join(sorted(set(flags))),
                "example_paths": " | ".join(
                    sorted(
                        (
                            data_root / "multichannel_unresolved" / row["file_name"]
                        ).as_posix()
                        for row in grouped_rows[:3]
                    )
                ),
                "notes": "; ".join(sorted({row.get("notes", "").strip() for row in grouped_rows if row.get("notes", "").strip()})),
            }
        )
    return review_rows


def build_protocol_artifacts(
    data_root: Path,
    metadata_csv: Path | None,
    include_roi_crops: bool = True,
    roi_padding_fraction: float = 0.12,
) -> dict[str, object]:
    records = discover_records(
        data_root=data_root,
        include_kpc=True,
        include_multichannel=True,
        metadata_csv=metadata_csv,
        include_resolved_unresolved=False,
        include_roi_crops=include_roi_crops,
        roi_padding_fraction=roi_padding_fraction,
        allow_manual_check_rows=False,
    )
    records = [record for record in records if record.source_bucket in FORMAL_SOURCE_BUCKETS]
    instance_rows = _build_instance_rows(records)
    bag_rows = _build_bag_rows(records)

    review_rows: list[dict[str, str]] = []
    records_by_bag: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_bag[record.lesion_id].append(record)
    for bag_row in bag_rows:
        if bag_row.review_flag_count == 0:
            continue
        bag_records = records_by_bag[bag_row.bag_id]
        review_rows.append(
            {
                "bag_id": bag_row.bag_id,
                "label_name": bag_row.label_name,
                "source_buckets": bag_row.source_buckets,
                "instance_count": str(bag_row.instance_count),
                "review_flags": bag_row.review_flags,
                "example_paths": _example_paths(bag_records),
                "notes": "",
            }
        )
    review_rows.extend(_build_unresolved_review_rows(data_root=data_root, metadata_csv=metadata_csv))
    review_rows = sorted(review_rows, key=lambda row: (row["bag_id"], row["review_flags"]))

    return {
        "instance_rows": instance_rows,
        "bag_rows": bag_rows,
        "review_rows": review_rows,
        "summary": build_summary(records=records, bag_rows=bag_rows, review_rows=review_rows),
    }


def build_summary(
    records: list[ImageRecord],
    bag_rows: list[BagRow],
    review_rows: list[dict[str, str]],
) -> dict[str, object]:
    label_bag_counts = Counter(bag_row.label_name for bag_row in bag_rows)
    label_instance_counts = Counter(record.label_name for record in records)
    source_bag_counts: Counter[str] = Counter()
    for bag_row in bag_rows:
        for source_bucket in bag_row.source_buckets.split("|"):
            source_bag_counts[source_bucket] += 1
    source_instance_counts = Counter(record.source_bucket for record in records)
    view_count_distribution = Counter(bag_row.instance_count for bag_row in bag_rows)
    magnification_counts = Counter(record.magnification for record in records if record.sample_type == "whole_image")
    channel_counts = Counter(record.channel_name for record in records if record.sample_type == "whole_image")

    coverage_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        if record.sample_type != "whole_image":
            continue
        coverage_matrix[record.magnification][record.channel_name] += 1

    quality_flags = Counter()
    for bag_row in bag_rows:
        if bag_row.instance_count == 1:
            quality_flags["single_view_bag"] += 1
        if bag_row.has_unknown_magnification:
            quality_flags["missing_magnification"] += 1
        if bag_row.has_unknown_channel:
            quality_flags["missing_channel"] += 1
        if bag_row.has_mixed_source_bucket:
            quality_flags["mixed_source_bucket"] += 1
        if "roi_only_bag" in bag_row.review_flags.split("|"):
            quality_flags["roi_only_bag"] += 1

    return {
        "formal_scope_source_buckets": list(FORMAL_SOURCE_BUCKETS),
        "counts": {
            "bag_count": len(bag_rows),
            "instance_count": len(records),
            "roi_instance_count": sum(record.sample_type == "roi_crop" for record in records),
            "review_candidate_count": len(review_rows),
        },
        "label_distribution": {
            "bags": dict(sorted(label_bag_counts.items())),
            "instances": dict(sorted(label_instance_counts.items())),
        },
        "source_bucket_distribution": {
            "bags": dict(sorted(source_bag_counts.items())),
            "instances": dict(sorted(source_instance_counts.items())),
        },
        "view_count_distribution": {
            str(key): value for key, value in sorted(view_count_distribution.items())
        },
        "quality_flags": dict(sorted(quality_flags.items())),
        "magnification_counts": dict(sorted(magnification_counts.items())),
        "channel_counts": dict(sorted(channel_counts.items())),
        "coverage_matrix": {
            magnification: dict(sorted(channel_map.items()))
            for magnification, channel_map in sorted(coverage_matrix.items())
        },
    }


def write_csv_rows(output_path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_protocol_outputs(output_dir: Path, artifacts: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_rows = artifacts["instance_rows"]
    bag_rows = artifacts["bag_rows"]
    review_rows = artifacts["review_rows"]
    summary = artifacts["summary"]

    write_csv_rows(
        output_dir / "instance_manifest.csv",
        [
            "instance_id",
            "bag_id",
            "image_path",
            "label_name",
            "source_bucket",
            "magnification",
            "channel_name",
            "sample_type",
            "is_roi",
            "split_key",
            "lesion_id",
            "record_key",
            "label_source",
            "crop_box",
        ],
        [row.__dict__ for row in instance_rows],  # type: ignore[attr-defined]
    )
    write_csv_rows(
        output_dir / "bag_manifest.csv",
        [
            "bag_id",
            "lesion_id",
            "label_name",
            "source_buckets",
            "instance_count",
            "whole_image_count",
            "roi_count",
            "magnifications",
            "channel_names",
            "sample_types",
            "has_mixed_source_bucket",
            "has_unknown_magnification",
            "has_unknown_channel",
            "review_flag_count",
            "review_flags",
        ],
        [row.__dict__ for row in bag_rows],  # type: ignore[attr-defined]
    )
    write_csv_rows(
        output_dir / "review_candidates.csv",
        [
            "bag_id",
            "label_name",
            "source_buckets",
            "instance_count",
            "review_flags",
            "example_paths",
            "notes",
        ],
        review_rows,  # type: ignore[arg-type]
    )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")


def render_summary_markdown(summary: dict[str, object]) -> str:
    counts = summary["counts"]  # type: ignore[index]
    label_distribution = summary["label_distribution"]  # type: ignore[index]
    source_distribution = summary["source_bucket_distribution"]  # type: ignore[index]
    view_distribution = summary["view_count_distribution"]  # type: ignore[index]
    quality_flags = summary["quality_flags"]  # type: ignore[index]
    coverage_matrix = summary["coverage_matrix"]  # type: ignore[index]

    lines = [
        "# Bag Protocol V1 Summary",
        "",
        "## Scope",
        "",
        "- Formal source buckets: " + ", ".join(summary["formal_scope_source_buckets"]),  # type: ignore[index]
        "- `multichannel_unresolved` is excluded from formal manifests and only appears in `review_candidates.csv`.",
        "",
        "## Counts",
        "",
        f"- Bag count: {counts['bag_count']}",
        f"- Instance count: {counts['instance_count']}",
        f"- ROI instance count: {counts['roi_instance_count']}",
        f"- Review candidate count: {counts['review_candidate_count']}",
        "",
        "## Label Distribution",
        "",
    ]
    for label_name, bag_count in label_distribution["bags"].items():
        instance_count = label_distribution["instances"].get(label_name, 0)
        lines.append(f"- {label_name}: {bag_count} bags, {instance_count} instances")

    lines.extend(["", "## Source Bucket Distribution", ""])
    for source_bucket, bag_count in source_distribution["bags"].items():
        instance_count = source_distribution["instances"].get(source_bucket, 0)
        lines.append(f"- {source_bucket}: {bag_count} bags, {instance_count} instances")

    lines.extend(["", "## View Count Distribution", ""])
    for view_count, bag_count in view_distribution.items():
        lines.append(f"- {view_count} instances per bag: {bag_count} bags")

    lines.extend(["", "## Quality Flags", ""])
    if quality_flags:
        for flag_name, flag_count in quality_flags.items():
            lines.append(f"- {flag_name}: {flag_count}")
    else:
        lines.append("- No automated review flags were raised.")

    lines.extend(["", "## Coverage Matrix", ""])
    for magnification, channel_counts in coverage_matrix.items():
        channel_summary = ", ".join(
            f"{channel_name}={count}" for channel_name, count in channel_counts.items()
        )
        lines.append(f"- {magnification}: {channel_summary}")

    lines.append("")
    return "\n".join(lines)
