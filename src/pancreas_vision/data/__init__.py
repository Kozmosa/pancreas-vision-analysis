"""Data subpackage: discovery, datasets, and splitting.

Re-exports public API so that ``from pancreas_vision.data import discover_records``
continues to work after the subpackage split.
"""

from pancreas_vision.data.records import (
    AUTHORITATIVE_LABELS,
    LABEL_TO_INDEX,
    LEGACY_SOURCE_BUCKETS,
    ImageRecord,
    bbox_iou,
    build_record,
    discover_records,
    discover_roi_crop_records,
    infer_channel_name,
    infer_lesion_id,
    infer_magnification,
    load_metadata_index,
    metadata_value_or_fallback,
    normalize_metadata_label,
    normalize_roi_label,
    polygon_to_crop_box,
    resolve_legacy_bucket,
    resolve_metadata_row_path,
)
from pancreas_vision.data.dataset import MicroscopyDataset
from pancreas_vision.data.splitting import split_grouped_records, split_records

__all__ = [
    "AUTHORITATIVE_LABELS",
    "LABEL_TO_INDEX",
    "LEGACY_SOURCE_BUCKETS",
    "ImageRecord",
    "MicroscopyDataset",
    "bbox_iou",
    "build_record",
    "discover_records",
    "discover_roi_crop_records",
    "infer_channel_name",
    "infer_lesion_id",
    "infer_magnification",
    "load_metadata_index",
    "metadata_value_or_fallback",
    "normalize_metadata_label",
    "normalize_roi_label",
    "polygon_to_crop_box",
    "resolve_legacy_bucket",
    "resolve_metadata_row_path",
    "split_grouped_records",
    "split_records",
]
