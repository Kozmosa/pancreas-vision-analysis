"""Tests for CLAMSingleBranch model."""

from __future__ import annotations

import pytest
import torch

from pancreas_vision.models.clam import CLAMSingleBranch


class TestCLAMSingleBranch:
    """Tests for CLAMSingleBranch model."""

    def test_init_default_params(self):
        """Test model initialization with default parameters."""
        model = CLAMSingleBranch()

        assert model.feature_dim == 1536
        assert model.hidden_dim == 256
        assert model.num_classes == 2
        assert model._fused_dim == 256 + 16 + 8  # hidden + mag + channel

    def test_init_custom_params(self):
        """Test model initialization with custom parameters."""
        model = CLAMSingleBranch(
            feature_dim=512,
            hidden_dim=128,
            attention_dim=64,
            num_classes=3,
            dropout=0.2,
            magnification_dim=8,
            channel_dim=4,
        )

        assert model.feature_dim == 512
        assert model.hidden_dim == 128
        assert model.num_classes == 3

    def test_forward_output_shapes(self):
        """Test forward pass returns correct shapes."""
        model = CLAMSingleBranch(feature_dim=64, hidden_dim=32, attention_dim=16)

        # Create dummy input
        n_instances = 5
        features = torch.randn(n_instances, 64)
        mag_ids = torch.tensor([0, 1, 2, 3, 4])
        channel_ids = torch.tensor([0, 1, 2, 3, 0])

        logits, attention, instance_logits = model(features, mag_ids, channel_ids)

        assert logits.shape == (1, 2)  # (1 bag, 2 classes)
        assert attention.shape == (n_instances,)
        assert instance_logits.shape == (n_instances, 2)

    def test_attention_weights_sum_to_one(self):
        """Test attention weights are normalized (sum to 1)."""
        model = CLAMSingleBranch(feature_dim=64, hidden_dim=32, attention_dim=16)

        features = torch.randn(10, 64)
        mag_ids = torch.randint(0, 5, (10,))
        channel_ids = torch.randint(0, 5, (10,))

        _, attention, _ = model(features, mag_ids, channel_ids)

        # Attention should sum to approximately 1
        assert abs(attention.sum().item() - 1.0) < 1e-5

    def test_attention_weights_non_negative(self):
        """Test attention weights are non-negative."""
        model = CLAMSingleBranch(feature_dim=64, hidden_dim=32, attention_dim=16)

        features = torch.randn(10, 64)
        mag_ids = torch.randint(0, 5, (10,))
        channel_ids = torch.randint(0, 5, (10,))

        _, attention, _ = model(features, mag_ids, channel_ids)

        assert (attention >= 0).all()

    def test_metadata_embedding_dimensions(self):
        """Test magnification and channel embeddings have correct dimensions."""
        model = CLAMSingleBranch(
            feature_dim=64,
            hidden_dim=32,
            magnification_dim=16,
            channel_dim=8,
        )

        # Check embedding sizes
        assert model.magnification_embedding.num_embeddings == 5
        assert model.magnification_embedding.embedding_dim == 16
        assert model.channel_embedding.num_embeddings == 5
        assert model.channel_embedding.embedding_dim == 8

    def test_variable_instance_count(self):
        """Test model handles different numbers of instances."""
        model = CLAMSingleBranch(feature_dim=64, hidden_dim=32, attention_dim=16)

        for n_instances in [1, 5, 20]:
            features = torch.randn(n_instances, 64)
            mag_ids = torch.randint(0, 5, (n_instances,))
            channel_ids = torch.randint(0, 5, (n_instances,))

            logits, attention, instance_logits = model(features, mag_ids, channel_ids)

            assert logits.shape == (1, 2)
            assert attention.shape == (n_instances,)
            assert instance_logits.shape == (n_instances, 2)

    def test_single_instance_bag(self):
        """Test model handles bag with single instance."""
        model = CLAMSingleBranch(feature_dim=64, hidden_dim=32, attention_dim=16)

        features = torch.randn(1, 64)
        mag_ids = torch.tensor([3])  # unknown
        channel_ids = torch.tensor([0])  # single

        logits, attention, instance_logits = model(features, mag_ids, channel_ids)

        assert logits.shape == (1, 2)
        assert attention.shape == (1,)
        assert abs(attention.item() - 1.0) < 1e-5  # Single instance gets all attention

    def test_properties(self):
        """Test model properties return correct values."""
        model = CLAMSingleBranch(
            feature_dim=768,
            hidden_dim=128,
            num_classes=3,
        )

        assert model.feature_dim == 768
        assert model.hidden_dim == 128
        assert model.num_classes == 3

    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        model = CLAMSingleBranch(feature_dim=64, hidden_dim=32, attention_dim=16)

        features = torch.randn(5, 64, requires_grad=True)
        mag_ids = torch.tensor([0, 1, 2, 3, 4])
        channel_ids = torch.tensor([0, 1, 2, 3, 0])

        logits, _, _ = model(features, mag_ids, channel_ids)
        loss = logits.sum()
        loss.backward()

        # Check gradients exist
        assert features.grad is not None
        assert features.grad.shape == features.shape