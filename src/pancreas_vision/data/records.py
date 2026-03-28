"""Data discovery, metadata parsing, ROI cropping, and record construction."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from pancreas_vision.types import ImageRecord


AUTHORITATIVE_LABELS = {
    "caerulein_adm": "ADM",
    "multichannel_adm": "ADM",
    "multichannel_kc_adm": "ADM",
    "KC": "PanIN",
    "KPC": "PanIN",
}

LABEL_TO_INDEX = {"ADM": 0, "PanIN": 1}
LEGACY_SOURCE_BUCKETS = {
    "caerulein": "caerulein_adm",
    "caerulein solved": "multichannel_unresolved",
    "kc": "KC",
    "kpc": "KPC",
}


def infer_magnification(image_path: Path) -> str:
    match = re.search(r"(10x|20x|40x)", image_path.stem.lower())
    return match.group(1) if match else "unknown"


def infer_channel_name(image_path: Path) -> str:
    stem = image_path.stem.lower()
    if "ck-19" in stem:
        return "ck19"
    if "amylase" in stem:
        return "amylase"
    if "merge" in stem:
        return "merge"
    return "single"


def infer_lesion_id(source_bucket: str, image_path: Path) -> str:
    stem = image_path.stem.lower()
    numeric_match = re.match(r"(\d+)", stem)
    if numeric_match:
        return f"{source_bucket}:{numeric_match.group(1)}"

    cleaned = stem.replace("ck-19", " ")
    cleaned = cleaned.replace("amylase", " ")
    cleaned = cleaned.replace("merge", " ")
    cleaned = re.sub(r"\b(?:10x|20x|40x|if|kc|adm)\b", " ", cleaned)
    cleaned = re.sub(r"[-_ ]+", " ", cleaned).strip()
    if cleaned:
        return f"{source_bucket}:{cleaned.replace(' ', '_')}"

    if source_bucket.startswith("multichannel_"):
        return f"{source_bucket}:paired_lesion"
    return f"{source_bucket}:{stem}"


def resolve_legacy_bucket(source_folder: str, file_name: str) -> str | None:
    lowered = source_folder.strip().lower()
    if lowered == "many colour":
        return "multichannel_kc_adm" if "kc" in file_name.lower() else "multichannel_adm"
    return LEGACY_SOURCE_BUCKETS.get(lowered)


def resolve_metadata_row_path(data_root: Path, row: dict[str, str]) -> Path | None:
    bucket = resolve_legacy_bucket(row["source_folder"], row["file_name"])
    if bucket is None:
        return None
    return data_root / bucket / row["file_name"]


def normalize_metadata_label(row: dict[str, str]) -> str | None:
    coarse_label = row.get("coarse_label", "").strip()
    if coarse_label == "ADM-like":
        return "ADM"
    if coarse_label == "PanIN-like":
        return "PanIN"
    return None


def metadata_value_or_fallback(
    row: dict[str, str] | None,
    field_name: str,
    fallback: str,
) -> str:
    if row is None:
        return fallback
    value = row.get(field_name, "").strip()
    if value and value.lower() != "unknown":
        return value
    return fallback


def load_metadata_index(
    data_root: Path,
    metadata_csv: Path | None,
) -> dict[str, dict[str, str]]:
    if metadata_csv is None or not metadata_csv.exists():
        return {}

    index: dict[str, dict[str, str]] = {}
    with metadata_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            resolved_path = resolve_metadata_row_path(data_root, row)
            if resolved_path is None:
                continue
            index[resolved_path.as_posix()] = row
    return index


def build_record(
    image_path: Path,
    source_bucket: str,
    label_name: str,
    metadata_row: dict[str, str] | None = None,
    sample_type: str = "whole_image",
    crop_box: tuple[int, int, int, int] | None = None,
    label_source: str = "folder_label",
) -> ImageRecord:
    inferred_lesion_id = infer_lesion_id(source_bucket, image_path)
    lesion_id = metadata_value_or_fallback(metadata_row, "lesion_id", inferred_lesion_id)
    magnification = metadata_value_or_fallback(
        metadata_row, "magnification", infer_magnification(image_path)
    )
    channel_name = metadata_value_or_fallback(
        metadata_row, "image_type", infer_channel_name(image_path)
    )
    group_id = lesion_id if lesion_id else f"{source_bucket}:{image_path.stem}"
    return ImageRecord(
        image_path=image_path,
        source_bucket=source_bucket,
        label_name=label_name,
        label_index=LABEL_TO_INDEX[label_name],
        lesion_id=lesion_id,
        group_id=group_id,
        magnification=magnification,
        channel_name=channel_name,
        sample_type=sample_type,
        crop_box=crop_box,
        label_source=label_source,
    )


def normalize_roi_label(label: str) -> str | None:
    lowered = label.strip().lower()
    if lowered in {"panin", "panin-like"}:
        return "PanIN"
    if lowered in {"adm", "adm-like"}:
        return "ADM"
    return None


def polygon_to_crop_box(
    points: list[list[float]],
    image_width: int,
    image_height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = min(xs)
    top = min(ys)
    right = max(xs)
    bottom = max(ys)
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    pad_x = width * padding_fraction
    pad_y = height * padding_fraction

    crop_left = max(0, int(round(left - pad_x)))
    crop_top = max(0, int(round(top - pad_y)))
    crop_right = min(image_width, int(round(right + pad_x)))
    crop_bottom = min(image_height, int(round(bottom + pad_y)))
    if crop_right <= crop_left or crop_bottom <= crop_top:
        return None
    return crop_left, crop_top, crop_right, crop_bottom


def bbox_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_left = max(ax1, bx1)
    inter_top = max(ay1, by1)
    inter_right = min(ax2, bx2)
    inter_bottom = min(ay2, by2)
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0
    intersection = (inter_right - inter_left) * (inter_bottom - inter_top)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def discover_roi_crop_records(
    base_record: ImageRecord,
    padding_fraction: float,
) -> list[ImageRecord]:
    annotation_path = base_record.image_path.with_suffix(".json")
    if not annotation_path.exists():
        return []

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_width = int(annotation.get("imageWidth", 0))
    image_height = int(annotation.get("imageHeight", 0))
    crop_records: list[ImageRecord] = []
    accepted_boxes: list[tuple[int, int, int, int]] = []
    for shape in annotation.get("shapes", []):
        shape_label = normalize_roi_label(shape.get("label", ""))
        if shape_label != base_record.label_name:
            continue
        if shape.get("shape_type") != "polygon":
            continue
        crop_box = polygon_to_crop_box(
            shape.get("points", []),
            image_width=image_width,
            image_height=image_height,
            padding_fraction=padding_fraction,
        )
        if crop_box is None:
            continue
        if any(bbox_iou(crop_box, existing_box) >= 0.95 for existing_box in accepted_boxes):
            continue
        accepted_boxes.append(crop_box)
        crop_records.append(
            ImageRecord(
                image_path=base_record.image_path,
                source_bucket=base_record.source_bucket,
                label_name=base_record.label_name,
                label_index=base_record.label_index,
                lesion_id=base_record.lesion_id,
                group_id=base_record.group_id,
                magnification=base_record.magnification,
                channel_name=base_record.channel_name,
                sample_type="roi_crop",
                crop_box=crop_box,
                label_source="roi_polygon",
            )
        )
    return crop_records


def discover_records(
    data_root: Path,
    include_kpc: bool = True,
    include_multichannel: bool = False,
    metadata_csv: Path | None = None,
    include_resolved_unresolved: bool = False,
    include_roi_crops: bool = False,
    roi_padding_fraction: float = 0.12,
    allow_manual_check_rows: bool = False,
) -> list[ImageRecord]:
    """Discover trainable records from folders, metadata CSV, and optional ROI JSON files."""
    metadata_index = load_metadata_index(data_root=data_root, metadata_csv=metadata_csv)
    enabled = {
        bucket: label
        for bucket, label in AUTHORITATIVE_LABELS.items()
        if include_multichannel or not bucket.startswith("multichannel_")
    }
    if not include_kpc:
        enabled.pop("KPC")

    records: list[ImageRecord] = []
    seen_keys: set[str] = set()

    for bucket, label_name in enabled.items():
        bucket_path = data_root / bucket
        if not bucket_path.exists():
            raise FileNotFoundError(f"Missing expected data bucket: {bucket_path}")
        for image_path in sorted(p for p in bucket_path.iterdir() if p.is_file() and p.suffix.lower() == ".tif"):
            metadata_row = metadata_index.get(image_path.as_posix())
            record = build_record(
                image_path=image_path,
                source_bucket=bucket,
                label_name=label_name,
                metadata_row=metadata_row,
                label_source="folder_label" if metadata_row is None else "folder_label+metadata",
            )
            if record.record_key not in seen_keys:
                records.append(record)
                seen_keys.add(record.record_key)
            if include_roi_crops:
                for crop_record in discover_roi_crop_records(
                    base_record=record,
                    padding_fraction=roi_padding_fraction,
                ):
                    if crop_record.record_key not in seen_keys:
                        records.append(crop_record)
                        seen_keys.add(crop_record.record_key)

    if include_resolved_unresolved:
        for image_path_str, metadata_row in metadata_index.items():
            image_path = Path(image_path_str)
            if image_path.parent.name != "multichannel_unresolved":
                continue
            if not image_path.exists():
                continue
            if metadata_row.get("exclude_from_train", "1") != "0":
                continue
            if (
                not allow_manual_check_rows
                and metadata_row.get("needs_manual_check", "1") == "1"
            ):
                continue
            label_name = normalize_metadata_label(metadata_row)
            if label_name is None:
                continue
            supplemental_record = build_record(
                image_path=image_path,
                source_bucket=image_path.parent.name,
                label_name=label_name,
                metadata_row=metadata_row,
                label_source="metadata_resolved",
            )
            if supplemental_record.record_key not in seen_keys:
                records.append(supplemental_record)
                seen_keys.add(supplemental_record.record_key)

    if not records:
        raise ValueError(f"No image records discovered under {data_root}")
    return records
