"""GAN subpackage for StyleGAN3-based data augmentation."""

from __future__ import annotations

from pancreas_vision.gan.augmentation import (
    create_augmented_instance_manifest,
    create_augmented_feature_index,
)
from pancreas_vision.gan.data_prep import (
    prepare_gan_training_dataset,
    prepare_stylegan3_dataset_format,
)
from pancreas_vision.gan.synthesis import (
    filter_synthetic_instances,
    generate_synthetic_instances,
    save_synthetic_instances_manifest,
    load_synthetic_instances_manifest,
)

__all__ = [
    "create_augmented_feature_index",
    "create_augmented_instance_manifest",
    "filter_synthetic_instances",
    "generate_synthetic_instances",
    "load_synthetic_instances_manifest",
    "prepare_gan_training_dataset",
    "prepare_stylegan3_dataset_format",
    "save_synthetic_instances_manifest",
]