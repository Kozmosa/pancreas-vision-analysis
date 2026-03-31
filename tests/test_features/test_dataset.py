"""Tests for BagFeatureDataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from pancreas_vision.features.dataset import BagFeatureDataset

if TYPE_CHECKING:
    pass


@pytest.fixture
def mock_feature_cache(tmp_path: Path) -> Path:
    """Create a mock feature cache directory with test data."""
    cache_dir = tmp_path / "feature_cache"
    features_dir = cache_dir / "features"
    features_dir.mkdir(parents=True)

    # Create feature index
    feature_index_path = cache_dir / "feature_index.csv"
    with feature_index_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance_id", "bag_id", "feature_type", "feature_path",
            "feature_dim", "extractor", "magnification", "channel_name", "is_roi"
        ])
        # Bag 1: 3 instances (ADM)
        for i, (mag, chan) in enumerate([("10x", "single"), ("20x", "merge"), ("unknown", "single")]):
            feature_file = features_dir / f"INSTANCE_{i+1:04d}_global.pt"
            torch.save({
                "instance_id": f"INSTANCE_{i+1:04d}",
                "bag_id": "BAG_ADM_001",
                "feature_type": "global",
                "features": torch.randn(1536),
            }, feature_file)
            writer.writerow([
                f"INSTANCE_{i+1:04d}", "BAG_ADM_001", "global",
                f"features/INSTANCE_{i+1:04d}_global.pt", 1536, "uni2-h", mag, chan, "0"
            ])

        # Bag 2: 2 instances (PanIN)
        for i, (mag, chan) in enumerate([("40x", "ck19"), ("unknown", "amylase")]):
            idx = i + 4
            feature_file = features_dir / f"INSTANCE_{idx:04d}_global.pt"
            torch.save({
                "instance_id": f"INSTANCE_{idx:04d}",
                "bag_id": "BAG_PANIN_001",
                "feature_type": "global",
                "features": torch.randn(1536),
            }, feature_file)
            writer.writerow([
                f"INSTANCE_{idx:04d}", "BAG_PANIN_001", "global",
                f"features/INSTANCE_{idx:04d}_global.pt", 1536, "uni2-h", mag, chan, "0"
            ])

    return cache_dir


@pytest.fixture
def mock_bag_manifest(tmp_path: Path) -> Path:
    """Create a mock bag manifest."""
    manifest_path = tmp_path / "bag_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bag_id", "lesion_id", "label_name", "source_buckets",
            "instance_count", "whole_image_count", "roi_count"
        ])
        writer.writerow(["BAG_ADM_001", "BAG_ADM_001", "ADM", "test", 3, 3, 0])
        writer.writerow(["BAG_PANIN_001", "BAG_PANIN_001", "PanIN", "test", 2, 2, 0])
    return manifest_path


@pytest.fixture
def mock_split_csv(tmp_path: Path) -> Path:
    """Create a mock split CSV."""
    split_path = tmp_path / "main_split.csv"
    with split_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bag_id", "lesion_id", "label_name", "source_buckets",
            "instance_count", "split_name", "split_role", "split_seed", "split_ratio"
        ])
        writer.writerow(["BAG_ADM_001", "BAG_ADM_001", "ADM", "test", 3, "test_split", "train", "42", "5:5"])
        writer.writerow(["BAG_PANIN_001", "BAG_PANIN_001", "PanIN", "test", 2, "test_split", "test", "42", "5:5"])
    return split_path


class TestBagFeatureDataset:
    """Tests for BagFeatureDataset class."""

    def test_loads_train_bags(
        self,
        mock_feature_cache: Path,
        mock_bag_manifest: Path,
        mock_split_csv: Path,
    ):
        """Test loading training bags."""
        dataset = BagFeatureDataset(
            feature_index_path=mock_feature_cache / "feature_index.csv",
            bag_manifest_path=mock_bag_manifest,
            split_csv_path=mock_split_csv,
            split_role="train",
            cache_dir=mock_feature_cache,
        )

        assert len(dataset) == 1  # Only BAG_ADM_001 is train

    def test_loads_test_bags(
        self,
        mock_feature_cache: Path,
        mock_bag_manifest: Path,
        mock_split_csv: Path,
    ):
        """Test loading test bags."""
        dataset = BagFeatureDataset(
            feature_index_path=mock_feature_cache / "feature_index.csv",
            bag_manifest_path=mock_bag_manifest,
            split_csv_path=mock_split_csv,
            split_role="test",
            cache_dir=mock_feature_cache,
        )

        assert len(dataset) == 1  # Only BAG_PANIN_001 is test

    def test_getitem_returns_correct_structure(
        self,
        mock_feature_cache: Path,
        mock_bag_manifest: Path,
        mock_split_csv: Path,
    ):
        """Test __getitem__ returns correct data structure."""
        dataset = BagFeatureDataset(
            feature_index_path=mock_feature_cache / "feature_index.csv",
            bag_manifest_path=mock_bag_manifest,
            split_csv_path=mock_split_csv,
            split_role="train",
            cache_dir=mock_feature_cache,
        )

        item = dataset[0]

        assert "bag_id" in item
        assert "features" in item
        assert "magnification_ids" in item
        assert "channel_ids" in item
        assert "label" in item
        assert "instance_count" in item

        assert item["bag_id"] == "BAG_ADM_001"
        assert item["features"].shape[1] == 1536  # Feature dimension
        assert item["label"].item() == 0  # ADM = 0

    def test_metadata_encoding(
        self,
        mock_feature_cache: Path,
        mock_bag_manifest: Path,
        mock_split_csv: Path,
    ):
        """Test magnification and channel ID encoding."""
        dataset = BagFeatureDataset(
            feature_index_path=mock_feature_cache / "feature_index.csv",
            bag_manifest_path=mock_bag_manifest,
            split_csv_path=mock_split_csv,
            split_role="train",
            cache_dir=mock_feature_cache,
        )

        item = dataset[0]

        # Check magnification encoding
        mag_ids = item["magnification_ids"]
        assert mag_ids.dtype == torch.long
        assert all(0 <= mid < 5 for mid in mag_ids)

        # Check channel encoding
        channel_ids = item["channel_ids"]
        assert channel_ids.dtype == torch.long
        assert all(0 <= cid < 5 for cid in channel_ids)

    def test_label_distribution(
        self,
        mock_feature_cache: Path,
        mock_bag_manifest: Path,
        mock_split_csv: Path,
    ):
        """Test get_label_distribution method."""
        dataset = BagFeatureDataset(
            feature_index_path=mock_feature_cache / "feature_index.csv",
            bag_manifest_path=mock_bag_manifest,
            split_csv_path=mock_split_csv,
            split_role="train",
            cache_dir=mock_feature_cache,
        )

        distribution = dataset.get_label_distribution()

        assert "ADM" in distribution
        assert distribution["ADM"] == 1

    def test_get_bag_ids(
        self,
        mock_feature_cache: Path,
        mock_bag_manifest: Path,
        mock_split_csv: Path,
    ):
        """Test get_bag_ids method."""
        dataset = BagFeatureDataset(
            feature_index_path=mock_feature_cache / "feature_index.csv",
            bag_manifest_path=mock_bag_manifest,
            split_csv_path=mock_split_csv,
            split_role="train",
            cache_dir=mock_feature_cache,
        )

        bag_ids = dataset.get_bag_ids()

        assert len(bag_ids) == 1
        assert "BAG_ADM_001" in bag_ids

    def test_multiple_instances_per_bag(
        self,
        mock_feature_cache: Path,
        mock_bag_manifest: Path,
        mock_split_csv: Path,
    ):
        """Test that multiple instances in a bag are loaded correctly."""
        dataset = BagFeatureDataset(
            feature_index_path=mock_feature_cache / "feature_index.csv",
            bag_manifest_path=mock_bag_manifest,
            split_csv_path=mock_split_csv,
            split_role="train",
            cache_dir=mock_feature_cache,
        )

        item = dataset[0]

        # BAG_ADM_001 has 3 instances
        assert item["instance_count"] == 3
        assert item["features"].shape[0] == 3
        assert item["magnification_ids"].shape[0] == 3
        assert item["channel_ids"].shape[0] == 3


class TestMetadataEncoding:
    """Tests for metadata encoding constants."""

    def test_magnification_encoding(self):
        """Test magnification encoding values."""
        assert BagFeatureDataset.MAG_TO_ID["10x"] == 0
        assert BagFeatureDataset.MAG_TO_ID["20x"] == 1
        assert BagFeatureDataset.MAG_TO_ID["40x"] == 2
        assert BagFeatureDataset.MAG_TO_ID["unknown"] == 3
        assert BagFeatureDataset.MAG_TO_ID["none"] == 4

    def test_channel_encoding(self):
        """Test channel encoding values."""
        assert BagFeatureDataset.CHANNEL_TO_ID["single"] == 0
        assert BagFeatureDataset.CHANNEL_TO_ID["merge"] == 1
        assert BagFeatureDataset.CHANNEL_TO_ID["amylase"] == 2
        assert BagFeatureDataset.CHANNEL_TO_ID["ck19"] == 3
        assert BagFeatureDataset.CHANNEL_TO_ID["none"] == 4

    def test_label_encoding(self):
        """Test label encoding values."""
        assert BagFeatureDataset.LABEL_TO_ID["ADM"] == 0
        assert BagFeatureDataset.LABEL_TO_ID["PanIN"] == 1