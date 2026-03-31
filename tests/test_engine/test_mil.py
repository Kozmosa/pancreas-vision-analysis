"""Tests for MIL training functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest
import torch
from torch import nn

from pancreas_vision.engine.mil import (
    evaluate_clam_model,
    set_random_seed,
    train_clam_model,
)
from pancreas_vision.models.clam import CLAMSingleBranch

if TYPE_CHECKING:
    from pancreas_vision.features.dataset import BagFeatureDataset


class MockBagDataset(torch.utils.data.Dataset):
    """Mock dataset for testing MIL training."""

    MAG_TO_ID = {"10x": 0, "20x": 1, "40x": 2, "unknown": 3, "none": 4}
    CHANNEL_TO_ID = {"single": 0, "merge": 1, "amylase": 2, "ck19": 3, "none": 4}

    def __init__(self, num_bags: int = 5, instances_per_bag: int = 3, label: int = 0):
        self.num_bags = num_bags
        self.instances_per_bag = instances_per_bag
        self.label = label
        self.bags = [f"BAG_{i:03d}" for i in range(num_bags)]

        # Create mock bag_manifest and feature_index for evaluate_clam_model
        self.bag_manifest = pd.DataFrame({
            "bag_id": self.bags,
            "lesion_id": self.bags,
            "label_name": ["ADM" if label == 0 else "PanIN"] * num_bags,
            "source_buckets": ["test"] * num_bags,
        })
        self.feature_index = pd.DataFrame({
            "bag_id": [b for b in self.bags for _ in range(instances_per_bag)],
            "instance_id": [f"INST_{i}" for i in range(num_bags * instances_per_bag)],
            "feature_type": ["global"] * (num_bags * instances_per_bag),
            "magnification": ["unknown"] * (num_bags * instances_per_bag),
            "channel_name": ["single"] * (num_bags * instances_per_bag),
        })

    def __len__(self) -> int:
        return self.num_bags

    def __getitem__(self, idx: int) -> dict:
        n_instances = self.instances_per_bag
        return {
            "bag_id": self.bags[idx],
            "features": torch.randn(n_instances, 1536),
            "magnification_ids": torch.randint(0, 5, (n_instances,)),
            "channel_ids": torch.randint(0, 5, (n_instances,)),
            "label": torch.tensor(self.label),
            "instance_count": n_instances,
        }

    def get_bag_ids(self) -> list[str]:
        return self.bags.copy()

    def get_label_distribution(self) -> dict[str, int]:
        return {"ADM": self.num_bags if self.label == 0 else 0}


class TestSetRandomSeed:
    """Tests for set_random_seed function."""

    def test_sets_torch_seed(self):
        """Test that torch seed is set."""
        set_random_seed(42)
        t1 = torch.rand(5)
        set_random_seed(42)
        t2 = torch.rand(5)
        assert torch.allclose(t1, t2)

    def test_sets_cuda_seed_if_available(self):
        """Test that CUDA seed is set if CUDA is available."""
        # This test doesn't require CUDA, just verifies no error
        set_random_seed(42)
        set_random_seed(123)


class TestTrainClamModel:
    """Tests for train_clam_model function."""

    def test_training_returns_history(self):
        """Test that training returns history records."""
        model = CLAMSingleBranch(feature_dim=1536, hidden_dim=64, attention_dim=32)
        dataset = MockBagDataset(num_bags=3, instances_per_bag=2)
        device = torch.device("cpu")

        history = train_clam_model(
            model=model,
            train_dataset=dataset,
            device=device,
            epochs=3,
            learning_rate=1e-3,
        )

        assert len(history) == 3
        assert all(hasattr(h, "epoch") for h in history)
        assert all(hasattr(h, "train_loss") for h in history)
        assert all(hasattr(h, "learning_rate") for h in history)

    def test_training_reduces_loss_over_epochs(self):
        """Test that training generally reduces loss."""
        model = CLAMSingleBranch(feature_dim=1536, hidden_dim=64, attention_dim=32)
        dataset = MockBagDataset(num_bags=5, instances_per_bag=3)
        device = torch.device("cpu")

        history = train_clam_model(
            model=model,
            train_dataset=dataset,
            device=device,
            epochs=10,
            learning_rate=1e-3,
        )

        # Loss should generally decrease (not strictly, but trend should be down)
        losses = [h.train_loss for h in history]
        # At least check that final loss is lower than initial
        assert losses[-1] < losses[0]

    def test_training_with_custom_loss_weights(self):
        """Test training with custom bag and instance loss weights."""
        model = CLAMSingleBranch(feature_dim=1536, hidden_dim=64, attention_dim=32)
        dataset = MockBagDataset(num_bags=2, instances_per_bag=2)
        device = torch.device("cpu")

        # Should not raise
        history = train_clam_model(
            model=model,
            train_dataset=dataset,
            device=device,
            epochs=2,
            learning_rate=1e-3,
            bag_loss_weight=1.0,
            instance_loss_weight=0.5,
        )

        assert len(history) == 2

    def test_training_with_output_path(self, tmp_path: Path):
        """Test that training history is saved to file."""
        model = CLAMSingleBranch(feature_dim=1536, hidden_dim=64, attention_dim=32)
        dataset = MockBagDataset(num_bags=2, instances_per_bag=2)
        device = torch.device("cpu")
        output_path = tmp_path / "history.json"

        history = train_clam_model(
            model=model,
            train_dataset=dataset,
            device=device,
            epochs=2,
            learning_rate=1e-3,
            output_path=output_path,
        )

        assert output_path.exists()


class TestEvaluateClamModel:
    """Tests for evaluate_clam_model function."""

    def test_evaluation_returns_metrics_and_predictions(self):
        """Test that evaluation returns expected outputs."""
        model = CLAMSingleBranch(feature_dim=1536, hidden_dim=64, attention_dim=32)
        dataset = MockBagDataset(num_bags=3, instances_per_bag=2, label=0)
        device = torch.device("cpu")

        metrics, predictions, attentions = evaluate_clam_model(
            model=model,
            test_dataset=dataset,
            device=device,
        )

        assert metrics is not None
        assert len(predictions) == 3
        assert len(attentions) == 3

    def test_evaluation_metrics_structure(self):
        """Test that metrics have expected fields."""
        model = CLAMSingleBranch(feature_dim=1536, hidden_dim=64, attention_dim=32)
        # Mixed labels
        dataset = MockBagDataset(num_bags=4, instances_per_bag=2, label=0)
        device = torch.device("cpu")

        metrics, _, _ = evaluate_clam_model(
            model=model,
            test_dataset=dataset,
            device=device,
        )

        assert hasattr(metrics, "accuracy")
        assert hasattr(metrics, "sensitivity")
        assert hasattr(metrics, "specificity")
        assert hasattr(metrics, "roc_auc")

    def test_attention_extraction(self):
        """Test that attention weights are extracted correctly."""
        model = CLAMSingleBranch(feature_dim=1536, hidden_dim=64, attention_dim=32)
        dataset = MockBagDataset(num_bags=2, instances_per_bag=5, label=0)
        device = torch.device("cpu")

        _, _, attentions = evaluate_clam_model(
            model=model,
            test_dataset=dataset,
            device=device,
        )

        for att in attentions:
            assert hasattr(att, "bag_id")
            assert hasattr(att, "attention_weights")
            assert len(att.attention_weights) == 5  # 5 instances per bag

            # Attention should sum to approximately 1
            att_sum = sum(att.attention_weights)
            assert abs(att_sum - 1.0) < 1e-4

    def test_predictions_contain_bag_info(self):
        """Test that predictions contain bag-level information."""
        model = CLAMSingleBranch(feature_dim=1536, hidden_dim=64, attention_dim=32)
        dataset = MockBagDataset(num_bags=2, instances_per_bag=3, label=1)
        device = torch.device("cpu")

        _, predictions, _ = evaluate_clam_model(
            model=model,
            test_dataset=dataset,
            device=device,
        )

        for pred in predictions:
            assert pred.bag_id is not None
            assert pred.true_label == 1
            assert 0 <= pred.positive_score <= 1


class TestCLAMModelWithRealFeatures:
    """Tests using actual feature files if available."""

    @pytest.mark.skipif(
        not Path("artifacts/feature_cache_v1/feature_index.csv").exists(),
        reason="Feature cache not available"
    )
    def test_dataset_with_real_features(self):
        """Test dataset loading with real feature files."""
        from pancreas_vision.features.dataset import BagFeatureDataset

        dataset = BagFeatureDataset(
            feature_index_path=Path("artifacts/feature_cache_v1/feature_index.csv"),
            bag_manifest_path=Path("artifacts/bag_protocol_v1/bag_manifest.csv"),
            split_csv_path=Path("artifacts/split_protocol_v1/main_split.csv"),
            split_role="train",
            cache_dir=Path("artifacts/feature_cache_v1"),
        )

        assert len(dataset) > 0

        item = dataset[0]
        assert item["features"].shape[1] == 1536