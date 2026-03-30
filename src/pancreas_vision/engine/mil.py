"""MIL training and evaluation for bag-level models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from pancreas_vision.types import (
    AttentionSummaryRecord,
    BagPredictionRecord,
    EvaluationMetrics,
    MILTrainingHistory,
)

if TYPE_CHECKING:
    from pancreas_vision.features.dataset import BagFeatureDataset
    from pancreas_vision.models.clam import CLAMSingleBranch


def train_clam_model(
    model: "CLAMSingleBranch",
    train_dataset: "BagFeatureDataset",
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 1e-4,
    bag_loss_weight: float = 1.0,
    instance_loss_weight: float = 0.5,
    output_path: Path | None = None,
) -> list[MILTrainingHistory]:
    """Train CLAM model with bag-level and instance-level losses.

    Args:
        model: CLAM model instance
        train_dataset: BagFeatureDataset for training
        device: torch device
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        bag_loss_weight: Weight for bag-level cross-entropy loss
        instance_loss_weight: Weight for instance-level auxiliary loss
        output_path: Optional path to save training history

    Returns:
        List of MILTrainingHistory records
    """
    # DataLoader with batch_size=1 (bags have variable sizes)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Loss functions
    bag_criterion = nn.CrossEntropyLoss()
    instance_criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )

    model.to(device)
    history: list[MILTrainingHistory] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_bags = 0

        for batch in train_loader:
            # Extract batch data (batch_size=1, so we take index 0)
            features = batch["features"][0].to(device)
            magnification_ids = batch["magnification_ids"][0].to(device)
            channel_ids = batch["channel_ids"][0].to(device)
            label = batch["label"][0].to(device)

            # Forward pass
            logits, attention, instance_logits = model(
                features, magnification_ids, channel_ids
            )

            # Bag-level loss
            bag_loss = bag_criterion(logits, label.unsqueeze(0))

            # Instance-level loss (instances share bag label)
            instance_labels = label.expand(instance_logits.size(0))
            instance_loss = instance_criterion(instance_logits, instance_labels)

            # Combined loss
            total_batch_loss = (
                bag_loss_weight * bag_loss + instance_loss_weight * instance_loss
            )

            # Backward pass
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            num_bags += 1

        scheduler.step()

        avg_loss = total_loss / max(num_bags, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            MILTrainingHistory(
                epoch=epoch,
                train_loss=avg_loss,
                learning_rate=current_lr,
            )
        )

    # Save history if path provided
    if output_path is not None:
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([h.__dict__ for h in history], indent=2),
            encoding="utf-8",
        )

    return history


@torch.inference_mode()
def evaluate_clam_model(
    model: "CLAMSingleBranch",
    test_dataset: "BagFeatureDataset",
    device: torch.device,
) -> tuple[EvaluationMetrics, list[BagPredictionRecord], list[AttentionSummaryRecord]]:
    """Evaluate CLAM model and extract attention weights.

    Args:
        model: Trained CLAM model
        test_dataset: BagFeatureDataset for testing
        device: torch device

    Returns:
        Tuple of:
            - EvaluationMetrics: Bag-level metrics
            - List of BagPredictionRecord: Per-bag predictions
            - List of AttentionSummaryRecord: Attention analysis
    """
    model.eval()
    model.to(device)

    predictions: list[BagPredictionRecord] = []
    attentions: list[AttentionSummaryRecord] = []

    ID_TO_LABEL = {0: "ADM", 1: "PanIN"}
    ID_TO_MAG = {0: "10x", 1: "20x", 2: "40x", 3: "unknown", 4: "none"}
    ID_TO_CHANNEL = {0: "single", 1: "merge", 2: "amylase", 3: "ck19", 4: "none"}

    for i in range(len(test_dataset)):
        item = test_dataset[i]

        features = item["features"].to(device)
        magnification_ids = item["magnification_ids"].to(device)
        channel_ids = item["channel_ids"].to(device)
        true_label = item["label"].item()
        bag_id = item["bag_id"]

        # Forward pass
        logits, attention_weights, _ = model(features, magnification_ids, channel_ids)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
        positive_score = probabilities[0, 1].item()

        # Get source buckets from bag manifest
        bag_manifest = test_dataset.bag_manifest
        source_buckets = bag_manifest[
            bag_manifest["bag_id"] == bag_id
        ]["source_buckets"].iloc[0]

        # Create prediction record
        predictions.append(
            BagPredictionRecord(
                bag_id=bag_id,
                true_label=true_label,
                true_label_name=ID_TO_LABEL[true_label],
                predicted_label=predicted_label,
                predicted_label_name=ID_TO_LABEL[predicted_label],
                positive_score=positive_score,
                correct=predicted_label == true_label,
                source_buckets=source_buckets,
                instance_count=item["instance_count"],
                dominant_channel="unknown",  # Will be computed later
                dominant_magnification="unknown",
            )
        )

        # Create attention summary
        attention_list = attention_weights.cpu().numpy().tolist()
        top_k = min(5, len(attention_list))
        top_indices = sorted(
            range(len(attention_list)),
            key=lambda x: attention_list[x],
            reverse=True,
        )[:top_k]

        top_mags = [
            ID_TO_MAG[magnification_ids[j].item()] for j in top_indices
        ]
        top_channels = [
            ID_TO_CHANNEL[channel_ids[j].item()] for j in top_indices
        ]

        # Get feature types from feature_index
        feature_rows = test_dataset.feature_index[
            test_dataset.feature_index["bag_id"] == bag_id
        ]
        top_feature_types = [
            feature_rows.iloc[j]["feature_type"] if j < len(feature_rows) else "unknown"
            for j in top_indices
        ]

        attentions.append(
            AttentionSummaryRecord(
                bag_id=bag_id,
                attention_weights=attention_list,
                top_instance_indices=top_indices,
                top_magnifications=top_mags,
                top_channels=top_channels,
                top_feature_types=top_feature_types,
            )
        )

    # Compute metrics
    true_labels = [p.true_label for p in predictions]
    predicted_labels = [p.predicted_label for p in predictions]
    positive_scores = [p.positive_score for p in predictions]

    tn, fp, fn, tp = confusion_matrix(
        true_labels, predicted_labels, labels=[0, 1]
    ).ravel()

    metrics = EvaluationMetrics(
        accuracy=float(accuracy_score(true_labels, predicted_labels)),
        sensitivity=float(tp / (tp + fn) if (tp + fn) > 0 else 0.0),
        specificity=float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        roc_auc=float(
            roc_auc_score(true_labels, positive_scores)
            if len(set(true_labels)) > 1
            else float("nan")
        ),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )

    return metrics, predictions, attentions


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)