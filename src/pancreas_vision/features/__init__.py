"""Feature extraction for pancreatic microscopy images.

This module provides feature extractors for pathology images:
- UNIExtractor: Pathology-specific foundation model (1536-dim)
- DINOv2Extractor: General-purpose vision transformer
- BagFeatureDataset: Bag-level dataset for MIL training
"""

from __future__ import annotations

from pancreas_vision.features.extractors import (
    DINOv2Extractor,
    FeatureExtractor,
    UNIExtractor,
)
from pancreas_vision.features.cache import (
    build_feature_index,
    extract_and_cache_features,
)
from pancreas_vision.features.patches import sample_patches
from pancreas_vision.features.dataset import BagFeatureDataset

__all__ = [
    "BagFeatureDataset",
    "DINOv2Extractor",
    "FeatureExtractor",
    "UNIExtractor",
    "build_feature_index",
    "extract_and_cache_features",
    "sample_patches",
]