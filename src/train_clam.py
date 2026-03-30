#!/usr/bin/env python3
"""Train CLAM model for lesion-level ADM/PanIN classification.

This script trains a CLAM single-branch model using cached UNI features
for bag-level (lesion-level) weakly supervised classification.

Usage:
    PYTHONPATH=src python src/train_clam.py \
        --feature-cache artifacts/feature_cache_v1 \
        --bag-manifest artifacts/bag_protocol_v1/bag_manifest.csv \
        --split-csv artifacts/split_protocol_v1/main_split.csv \
        --output-dir artifacts/exp_clam_v1 \
        --epochs 50

Outputs:
    - training_history.json: Loss per epoch
    - bag_metrics.json: Bag-level metrics (accuracy, sensitivity, etc.)
    - bag_predictions.csv: Per-bag predictions
    - attention_summary.json: Attention weights per bag
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CLAM model for lesion-level ADM/PanIN classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        required=True,
        help="Path to feature cache directory (e.g., artifacts/feature_cache_v1)",
    )
    parser.add_argument(
        "--bag-manifest",
        type=Path,
        required=True,
        help="Path to bag_manifest.csv",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        required=True,
        help="Path to split CSV (main_split.csv or fold CSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for model and results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization (default: 1e-4)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for feature projector (default: 256)",
    )
    parser.add_argument(
        "--attention-dim",
        type=int,
        default=128,
        help="Attention network dimension (default: 128)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--bag-loss-weight",
        type=float,
        default=1.0,
        help="Weight for bag-level loss (default: 1.0)",
    )
    parser.add_argument(
        "--instance-loss-weight",
        type=float,
        default=0.5,
        help="Weight for instance-level loss (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for training (cuda or cpu)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.feature_cache.exists():
        print(f"Error: Feature cache not found: {args.feature_cache}")
        sys.exit(1)
    if not args.bag_manifest.exists():
        print(f"Error: Bag manifest not found: {args.bag_manifest}")
        sys.exit(1)
    if not args.split_csv.exists():
        print(f"Error: Split CSV not found: {args.split_csv}")
        sys.exit(1)

    # Check feature index exists
    feature_index = args.feature_cache / "feature_index.csv"
    if not feature_index.exists():
        print(f"Error: Feature index not found: {feature_index}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import modules
    print("Initializing CLAM training...")
    from pancreas_vision.models.clam import CLAMSingleBranch
    from pancreas_vision.features.dataset import BagFeatureDataset
    from pancreas_vision.engine.mil import (
        evaluate_clam_model,
        set_random_seed,
        train_clam_model,
    )
    import torch

    # Set random seed
    set_random_seed(args.seed)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    print(f"\nCreating CLAM model...")
    model = CLAMSingleBranch(
        feature_dim=1536,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        num_classes=2,
        dropout=args.dropout,
    )
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Attention dim: {args.attention_dim}")
    print(f"  Dropout: {args.dropout}")

    # Create datasets
    print(f"\nCreating datasets...")
    train_dataset = BagFeatureDataset(
        feature_index_path=feature_index,
        bag_manifest_path=args.bag_manifest,
        split_csv_path=args.split_csv,
        split_role="train",
        cache_dir=args.feature_cache,
    )
    test_dataset = BagFeatureDataset(
        feature_index_path=feature_index,
        bag_manifest_path=args.bag_manifest,
        split_csv_path=args.split_csv,
        split_role="test",
        cache_dir=args.feature_cache,
    )
    print(f"  Train bags: {len(train_dataset)}")
    print(f"  Test bags: {len(test_dataset)}")
    print(f"  Train label distribution: {train_dataset.get_label_distribution()}")
    print(f"  Test label distribution: {test_dataset.get_label_distribution()}")

    # Train model
    print(f"\nTraining for {args.epochs} epochs...")
    history = train_clam_model(
        model=model,
        train_dataset=train_dataset,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bag_loss_weight=args.bag_loss_weight,
        instance_loss_weight=args.instance_loss_weight,
    )

    # Evaluate model
    print(f"\nEvaluating on test set...")
    metrics, predictions, attentions = evaluate_clam_model(
        model=model,
        test_dataset=test_dataset,
        device=device,
    )

    # Print metrics
    print(f"\n{'='*50}")
    print("Test Results:")
    print(f"{'='*50}")
    print(f"  Accuracy: {metrics.accuracy:.4f}")
    print(f"  Sensitivity: {metrics.sensitivity:.4f}")
    print(f"  Specificity: {metrics.specificity:.4f}")
    print(f"  ROC AUC: {metrics.roc_auc:.4f}")
    print(f"  True Positive: {metrics.true_positive}")
    print(f"  True Negative: {metrics.true_negative}")
    print(f"  False Positive: {metrics.false_positive}")
    print(f"  False Negative: {metrics.false_negative}")

    # Save outputs
    print(f"\nSaving outputs to {args.output_dir}...")

    # Training history
    history_path = args.output_dir / "training_history.json"
    history_path.write_text(
        json.dumps([asdict(h) for h in history], indent=2),
        encoding="utf-8",
    )

    # Metrics
    metrics_path = args.output_dir / "bag_metrics.json"
    metrics_path.write_text(
        json.dumps(asdict(metrics), indent=2),
        encoding="utf-8",
    )

    # Predictions
    predictions_path = args.output_dir / "bag_predictions.csv"
    fieldnames = [
        "bag_id",
        "true_label",
        "true_label_name",
        "predicted_label",
        "predicted_label_name",
        "positive_score",
        "correct",
        "source_buckets",
        "instance_count",
        "dominant_channel",
        "dominant_magnification",
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in predictions:
            writer.writerow(asdict(p))

    # Attention summary
    attention_path = args.output_dir / "attention_summary.json"
    attention_path.write_text(
        json.dumps([asdict(a) for a in attentions], indent=2),
        encoding="utf-8",
    )

    # Model weights
    model_path = args.output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"{'='*50}")
    print(f"  Training history: {history_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Attention summary: {attention_path}")
    print(f"  Model weights: {model_path}")


if __name__ == "__main__":
    main()