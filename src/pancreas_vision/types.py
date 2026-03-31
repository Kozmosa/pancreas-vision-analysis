"""Shared data classes used across the pancreas_vision package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    source_bucket: str
    label_name: str
    label_index: int
    lesion_id: str
    group_id: str
    magnification: str
    channel_name: str
    sample_type: str = "whole_image"
    crop_box: tuple[int, int, int, int] | None = None
    label_source: str = "folder_label"

    @property
    def record_key(self) -> str:
        if self.crop_box is None:
            return self.image_path.as_posix()
        left, top, right, bottom = self.crop_box
        return (
            f"{self.image_path.as_posix()}#crop={left},{top},{right},{bottom}"
            f"#type={self.sample_type}"
        )


@dataclass
class EvaluationMetrics:
    accuracy: float
    sensitivity: float
    specificity: float
    roc_auc: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


@dataclass
class PredictionRecord:
    image_path: str
    true_label: int
    predicted_label: int
    positive_score: float
    correct: bool
    bag_id: str | None = None
    source_bucket: str | None = None
    label_name: str | None = None
    sample_type: str | None = None


@dataclass
class TrainingHistory:
    epoch: int
    train_loss: float
    train_accuracy: float
    learning_rate: float


@dataclass
class BagPredictionRecord:
    bag_id: str
    true_label: int
    true_label_name: str
    predicted_label: int
    predicted_label_name: str
    positive_score: float
    correct: bool
    source_buckets: str
    instance_count: int
    dominant_channel: str
    dominant_magnification: str


@dataclass
class SourceBucketErrorRecord:
    source_bucket: str
    bag_count: int
    error_count: int
    false_positive_count: int
    false_negative_count: int
    accuracy: float


@dataclass
class AttentionSummaryRecord:
    """Attention weights summary for a bag in CLAM model."""
    bag_id: str
    attention_weights: list[float]
    top_instance_indices: list[int]
    top_magnifications: list[str]
    top_channels: list[str]
    top_feature_types: list[str]


@dataclass
class MILTrainingHistory:
    """Training history for MIL/CLAM models."""
    epoch: int
    train_loss: float
    learning_rate: float


@dataclass
class HardCaseBagSummary:
    """Hard-case bag analysis summary for GAN training consideration."""
    bag_id: str
    label_name: str
    source_buckets: str
    predicted_correctly: bool
    positive_score: float
    error_type: str | None  # "false_positive" or "false_negative"
    instance_count: int
    top_attention_instances: list[str]
    top_attention_channels: list[str]
    boundary_score: float  # Distance from 0.5
    recommended_for_gan: bool
    gan_reason: str


@dataclass
class GANPatchCandidate:
    """Patch candidate for GAN augmentation training data."""
    instance_id: str
    bag_id: str
    image_path: str
    magnification: str
    channel_name: str
    true_label_name: str
    attention_weight: float
    attention_rank: int
    selection_reason: str
    priority_score: float
