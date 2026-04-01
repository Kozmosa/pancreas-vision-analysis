"""Tests for GAN data_prep module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def mock_instance_manifest(tmp_path: Path) -> Path:
    """Create a mock instance manifest."""
    manifest_path = tmp_path / "instance_manifest.csv"
    df = pd.DataFrame({
        "instance_id": ["INST_001", "INST_002", "INST_003", "INST_004"],
        "bag_id": ["BAG_ADM_001", "BAG_ADM_001", "BAG_ADM_002", "BAG_PANIN_001"],
        "image_path": [
            str(tmp_path / "img1.tif"),
            str(tmp_path / "img2.tif"),
            str(tmp_path / "img3.tif"),
            str(tmp_path / "img4.tif"),
        ],
        "label_name": ["ADM", "ADM", "ADM", "PanIN"],
        "source_bucket": ["multichannel_kc_adm", "multichannel_kc_adm", "caerulein_adm", "KC"],
        "magnification": ["40x", "20x", "unknown", "unknown"],
        "channel_name": ["ck19", "merge", "single", "single"],
        "sample_type": ["whole_image"] * 4,
        "is_roi": [0] * 4,
        "split_key": ["BAG_ADM_001", "BAG_ADM_001", "BAG_ADM_002", "BAG_PANIN_001"],
        "lesion_id": ["BAG_ADM_001", "BAG_ADM_001", "BAG_ADM_002", "BAG_PANIN_001"],
        "record_key": ["img1", "img2", "img3", "img4"],
        "label_source": ["folder_label"] * 4,
        "crop_box": [""] * 4,
    })
    df.to_csv(manifest_path, index=False)

    # Create dummy images
    from PIL import Image
    for i in range(1, 5):
        img = Image.new("RGB", (100, 100), color=(i * 50, 0, 0))
        img.save(tmp_path / f"img{i}.tif")

    return manifest_path


@pytest.fixture
def mock_split_csv(tmp_path: Path) -> Path:
    """Create a mock split CSV."""
    split_path = tmp_path / "split.csv"
    df = pd.DataFrame({
        "bag_id": ["BAG_ADM_001", "BAG_ADM_002", "BAG_PANIN_001"],
        "split_role": ["train", "train", "test"],
        "split_name": ["main_split"] * 3,
    })
    df.to_csv(split_path, index=False)
    return split_path


class TestPrepareGANTrainingDataset:
    """Tests for prepare_gan_training_dataset function."""

    def test_prepares_adm_images(
        self,
        mock_instance_manifest: Path,
        mock_split_csv: Path,
        tmp_path: Path,
    ):
        """Test that ADM images are prepared for GAN training."""
        from pancreas_vision.gan.data_prep import prepare_gan_training_dataset

        output_dir = tmp_path / "gan_output"

        summary = prepare_gan_training_dataset(
            instance_manifest_path=mock_instance_manifest,
            split_csv_path=mock_split_csv,
            output_dir=output_dir,
            target_label="ADM",
            target_buckets=["multichannel_kc_adm", "caerulein_adm"],
            split_role="train",
            image_size=256,
        )

        # Check summary
        assert summary["copied_images"] == 3  # 3 ADM in train set
        assert (output_dir / "images").exists()

    def test_respects_split_role(
        self,
        mock_instance_manifest: Path,
        mock_split_csv: Path,
        tmp_path: Path,
    ):
        """Test that only specified split role is included."""
        from pancreas_vision.gan.data_prep import prepare_gan_training_dataset

        output_dir = tmp_path / "gan_output"

        summary = prepare_gan_training_dataset(
            instance_manifest_path=mock_instance_manifest,
            split_csv_path=mock_split_csv,
            output_dir=output_dir,
            target_label="ADM",
            target_buckets=["multichannel_kc_adm", "caerulein_adm"],
            split_role="test",  # Only test set
            image_size=256,
        )

        # No ADM images in test set
        assert summary["copied_images"] == 0

    def test_filters_by_source_bucket(
        self,
        mock_instance_manifest: Path,
        mock_split_csv: Path,
        tmp_path: Path,
    ):
        """Test that source bucket filter works."""
        from pancreas_vision.gan.data_prep import prepare_gan_training_dataset

        output_dir = tmp_path / "gan_output"

        summary = prepare_gan_training_dataset(
            instance_manifest_path=mock_instance_manifest,
            split_csv_path=mock_split_csv,
            output_dir=output_dir,
            target_label="ADM",
            target_buckets=["multichannel_kc_adm"],  # Only MC
            split_role="train",
            image_size=256,
        )

        # Only 2 MC ADM images in train
        assert summary["copied_images"] == 2

    def test_resizes_images(
        self,
        mock_instance_manifest: Path,
        mock_split_csv: Path,
        tmp_path: Path,
    ):
        """Test that images are resized to target size."""
        from pancreas_vision.gan.data_prep import prepare_gan_training_dataset
        from PIL import Image

        output_dir = tmp_path / "gan_output"

        prepare_gan_training_dataset(
            instance_manifest_path=mock_instance_manifest,
            split_csv_path=mock_split_csv,
            output_dir=output_dir,
            target_label="ADM",
            target_buckets=["multichannel_kc_adm"],
            split_role="train",
            image_size=128,
        )

        # Check image size
        images = list((output_dir / "images").glob("*.png"))
        for img_path in images:
            with Image.open(img_path) as img:
                assert img.size == (128, 128)

    def test_creates_dataset_json(
        self,
        mock_instance_manifest: Path,
        mock_split_csv: Path,
        tmp_path: Path,
    ):
        """Test that dataset.json is created."""
        from pancreas_vision.gan.data_prep import prepare_gan_training_dataset

        output_dir = tmp_path / "gan_output"

        prepare_gan_training_dataset(
            instance_manifest_path=mock_instance_manifest,
            split_csv_path=mock_split_csv,
            output_dir=output_dir,
            target_label="ADM",
            target_buckets=["multichannel_kc_adm"],
            split_role="train",
            image_size=256,
        )

        dataset_json = output_dir / "dataset.json"
        assert dataset_json.exists()

        with dataset_json.open() as f:
            data = json.load(f)

        assert "name" in data
        assert "instances" in data
        assert data["image_size"] == 256