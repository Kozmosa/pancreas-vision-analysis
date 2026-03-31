"""Training engines for single-image and MIL models."""

from __future__ import annotations

# Single-image training (from _training.py)
from ._training import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    aggregate_predictions_to_bags,
    build_transforms,
    create_dataloaders,
    evaluate_model,
    now_timestamp,
    set_random_seed,
    train_model,
)

# MIL training (from mil.py)
from .mil import (
    evaluate_clam_model,
    train_clam_model,
)

__all__ = [
    # Constants
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    # Single-image training
    "aggregate_predictions_to_bags",
    "build_transforms",
    "create_dataloaders",
    "evaluate_model",
    "now_timestamp",
    "set_random_seed",
    "train_model",
    # MIL training
    "evaluate_clam_model",
    "train_clam_model",
]