"""Serialisation helpers for metrics, predictions, manifests, and experiment summaries."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from pancreas_vision.types import (
    BagPredictionRecord,
    EvaluationMetrics,
    ImageRecord,
    PredictionRecord,
    SourceBucketErrorRecord,
)


# ---------------------------------------------------------------------------
# Manifest I/O (moved from data.py)
# ---------------------------------------------------------------------------

def write_manifest(records: Iterable[ImageRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_key",
                "image_path",
                "source_bucket",
                "label_name",
                "label_index",
                "lesion_id",
                "group_id",
                "magnification",
                "channel_name",
                "sample_type",
                "crop_box",
                "label_source",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "record_key": record.record_key,
                    "image_path": record.image_path.as_posix(),
                    "source_bucket": record.source_bucket,
                    "label_name": record.label_name,
                    "label_index": record.label_index,
                    "lesion_id": record.lesion_id,
                    "group_id": record.group_id,
                    "magnification": record.magnification,
                    "channel_name": record.channel_name,
                    "sample_type": record.sample_type,
                    "crop_box": (
                        "" if record.crop_box is None else ",".join(str(v) for v in record.crop_box)
                    ),
                    "label_source": record.label_source,
                }
            )


# ---------------------------------------------------------------------------
# Metrics & predictions I/O (moved from training.py)
# ---------------------------------------------------------------------------

def save_metrics(metrics: EvaluationMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")


def save_experiment_summary(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_predictions(predictions: list[PredictionRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(item) for item in predictions], indent=2),
        encoding="utf-8",
    )


def save_bag_predictions(
    predictions: list[BagPredictionRecord],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(item) for item in predictions], indent=2),
        encoding="utf-8",
    )


def save_source_bucket_errors(
    rows: list[SourceBucketErrorRecord],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(item) for item in rows], indent=2),
        encoding="utf-8",
    )
