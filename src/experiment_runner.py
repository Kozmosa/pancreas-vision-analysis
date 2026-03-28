"""Shared experiment runner logic for train_baseline.py and train_improved.py.

This module extracts the common workflow:
    write_manifest -> print_stats -> create_dataloaders -> train -> evaluate -> save_results
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from pancreas_vision.data import ImageRecord
from pancreas_vision.engine import (
    aggregate_predictions_to_bags,
    create_dataloaders,
    evaluate_model,
    now_timestamp,
    set_random_seed,
    train_model,
)
from pancreas_vision.io import (
    save_bag_predictions,
    save_experiment_summary,
    save_metrics,
    save_predictions,
    save_source_bucket_errors,
    write_manifest,
)
from pancreas_vision.models import build_model


def run_experiment(
    train_records: list[ImageRecord],
    test_records: list[ImageRecord],
    output_dir: Path,
    *,
    model_name: str = "resnet18",
    model_kwargs: dict[str, Any] | None = None,
    epochs: int = 8,
    batch_size: int = 8,
    image_size: int = 224,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    num_workers: int = 2,
    use_weighted_sampler: bool = False,
    compute_bag_metrics: bool = False,
    seed: int = 42,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a complete training experiment.

    Args:
        train_records: List of training ImageRecord instances.
        test_records: List of test ImageRecord instances.
        output_dir: Directory to write outputs (manifests, metrics, etc.).
        model_name: Name of model to build from registry.
        model_kwargs: Additional kwargs passed to the model builder.
        epochs: Number of training epochs.
        batch_size: DataLoader batch size.
        image_size: Image resize dimension.
        learning_rate: Optimizer learning rate.
        weight_decay: AdamW weight decay.
        label_smoothing: CrossEntropyLoss label smoothing.
        num_workers: DataLoader worker count.
        use_weighted_sampler: Whether to use WeightedRandomSampler.
        compute_bag_metrics: Whether to compute bag-level aggregated metrics.
        seed: Random seed for reproducibility.
        extra_summary: Additional fields to include in experiment summary.

    Returns:
        Dictionary with metrics and experiment summary.
    """
    start_time = time.time()
    set_random_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(train_records, output_dir / "train_manifest.csv")
    write_manifest(test_records, output_dir / "test_manifest.csv")

    print("Train records:", len(train_records))
    print("Test records:", len(test_records))
    print("Train label counts:", Counter(record.label_name for record in train_records))
    print("Test label counts:", Counter(record.label_name for record in test_records))

    train_loader, test_loader = create_dataloaders(
        train_records=train_records,
        test_records=test_records,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampler=use_weighted_sampler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = model_kwargs or {}
    model = build_model(model_name, **model_kwargs)

    history = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        output_path=output_dir / "history.json",
    )

    metrics, predictions = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
    )
    save_metrics(metrics, output_dir / "metrics.json")
    save_predictions(predictions, output_dir / "predictions.json")

    result: dict[str, Any] = {
        "metrics": metrics,
        "predictions": predictions,
        "history": history,
    }

    if compute_bag_metrics:
        record_lookup = {record.image_path.as_posix(): record for record in test_records}
        bag_metrics, bag_predictions, source_bucket_errors = aggregate_predictions_to_bags(
            predictions=predictions,
            record_lookup=record_lookup,
        )
        save_metrics(bag_metrics, output_dir / "bag_metrics.json")
        save_bag_predictions(bag_predictions, output_dir / "bag_predictions.json")
        save_source_bucket_errors(
            source_bucket_errors, output_dir / "error_by_source_bucket.json"
        )
        result["bag_metrics"] = bag_metrics
        result["bag_predictions"] = bag_predictions
        result["source_bucket_errors"] = source_bucket_errors

    duration_seconds = time.time() - start_time
    summary: dict[str, Any] = {
        "started_at": now_timestamp(),
        "duration_seconds": duration_seconds,
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count(),
        "arguments": {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "epochs": epochs,
            "batch_size": batch_size,
            "image_size": image_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "num_workers": num_workers,
            "use_weighted_sampler": use_weighted_sampler,
            "compute_bag_metrics": compute_bag_metrics,
            "seed": seed,
        },
        "record_counts": {
            "train": len(train_records),
            "test": len(test_records),
        },
        "label_counts": {
            "train": dict(Counter(record.label_name for record in train_records)),
            "test": dict(Counter(record.label_name for record in test_records)),
        },
        "metrics": metrics.__dict__,
        "num_prediction_records": len(predictions),
        "history": [item.__dict__ for item in history],
    }
    if extra_summary:
        summary.update(extra_summary)
    if compute_bag_metrics:
        summary["bag_level_metrics"] = result["bag_metrics"].__dict__
        summary["num_bag_prediction_records"] = len(result["bag_predictions"])

    save_experiment_summary(summary, output_dir / "experiment_summary.json")

    print("Metrics saved to", (output_dir / "metrics.json").as_posix())
    print(metrics)

    return result


def load_split_csv(split_csv: Path) -> dict[str, str]:
    """Load a split CSV (e.g., main_split.csv) into a bag_id -> split_role lookup."""
    lookup: dict[str, str] = {}
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lookup[row["bag_id"]] = row["split_role"]
    return lookup