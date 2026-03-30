"""CLAM (Clustering-constrained Attention Multiple Instance Learning) model.

Implements single-branch CLAM for binary ADM/PanIN classification at bag-level.
Architecture:
  1. Feature projection: UNI features (1536-dim) -> hidden_dim
  2. Metadata embedding: magnification (16-dim) + channel (8-dim)
  3. Attention pooling: A = softmax(W * tanh(V * H))
  4. Bag classifier: aggregated features -> logits

Reference: Lu et al., "Data Efficient and Weakly Supervised Learning
for Whole Slide Pathology Image Analysis", CVPR 2021
"""

from __future__ import annotations

import torch
from torch import nn
from torch import Tensor


class CLAMSingleBranch(nn.Module):
    """Single-branch CLAM model for binary bag-level classification.

    Each bag contains multiple instances (images/patches) with UNI features.
    The model uses attention-based pooling to aggregate instance features
    into a bag-level representation for classification.

    Args:
        feature_dim: Input feature dimension (1536 for UNI)
        hidden_dim: Projection dimension after feature projector
        attention_dim: Dimension of attention network
        num_classes: Number of output classes (2 for ADM/PanIN)
        dropout: Dropout rate in feature projector
        magnification_dim: Embedding dimension for magnification
        channel_dim: Embedding dimension for channel name
    """

    def __init__(
        self,
        feature_dim: int = 1536,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        magnification_dim: int = 16,
        channel_dim: int = 8,
    ):
        super().__init__()

        self._feature_dim = feature_dim
        self._hidden_dim = hidden_dim
        self._attention_dim = attention_dim
        self._num_classes = num_classes

        # Feature projector: 1536 -> hidden_dim
        self.feature_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Metadata embeddings
        # 5 categories: 10x, 20x, 40x, unknown, none
        self.magnification_embedding = nn.Embedding(5, magnification_dim)
        # 5 categories: single, merge, amylase, ck19, none
        self.channel_embedding = nn.Embedding(5, channel_dim)

        # Fused dimension: hidden_dim + mag_dim + channel_dim
        fused_dim = hidden_dim + magnification_dim + channel_dim
        self._fused_dim = fused_dim

        # Attention network: V and W for gated attention
        # A = softmax(W * tanh(V * H))
        self.attention_V = nn.Linear(fused_dim, attention_dim)
        self.attention_W = nn.Linear(attention_dim, 1)

        # Bag classifier
        self.classifier = nn.Linear(fused_dim, num_classes)

        # Instance classifier (for instance-level auxiliary loss)
        self.instance_classifier = nn.Linear(fused_dim, num_classes)

    def forward(
        self,
        features: Tensor,
        magnification_ids: Tensor,
        channel_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for a single bag.

        Args:
            features: Instance features (N_instances, feature_dim)
            magnification_ids: Magnification indices (N_instances,)
            channel_ids: Channel indices (N_instances,)

        Returns:
            logits: Bag-level logits (1, num_classes)
            attention_weights: Attention weights per instance (N_instances,)
            instance_logits: Instance-level logits (N_instances, num_classes)
        """
        # Project features
        H = self.feature_projector(features)  # (N, hidden_dim)

        # Add metadata embeddings
        mag_emb = self.magnification_embedding(magnification_ids)  # (N, mag_dim)
        chan_emb = self.channel_embedding(channel_ids)  # (N, channel_dim)
        H = torch.cat([H, mag_emb, chan_emb], dim=1)  # (N, fused_dim)

        # Attention pooling: A = softmax(W * tanh(V * H))
        A = torch.softmax(
            self.attention_W(torch.tanh(self.attention_V(H))),
            dim=0,
        )  # (N, 1)

        # Weighted aggregation
        bag_features = torch.sum(A * H, dim=0, keepdim=True)  # (1, fused_dim)

        # Bag-level prediction
        logits = self.classifier(bag_features)  # (1, num_classes)

        # Instance-level predictions (for auxiliary loss)
        instance_logits = self.instance_classifier(H)  # (N, num_classes)

        return logits, A.squeeze(-1), instance_logits

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_classes(self) -> int:
        return self._num_classes