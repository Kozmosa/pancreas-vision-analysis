"""Model registry and implementations."""

from __future__ import annotations

# Registry (from _registry.py)
from ._registry import (
    MODEL_REGISTRY,
    build_clam_single,
    build_model,
    build_resnet18,
    build_resnet34,
    list_models,
    register_model,
)

# CLAM model (from clam.py)
from .clam import CLAMSingleBranch

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "build_resnet18",
    "build_resnet34",
    "build_clam_single",
    "list_models",
    "register_model",
    "CLAMSingleBranch",
]