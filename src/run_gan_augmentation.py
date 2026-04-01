#!/usr/bin/env python3
"""Run complete GAN augmentation pipeline.

This script runs the full pipeline for GAN-based data augmentation:
1. Prepare GAN training data
2. Generate synthetic instances (placeholder without StyleGAN3)
3. Extract features for synthetic instances
4. Create augmented training manifest

Usage:
    PYTHONPATH=src python src/run_gan_augmentation.py \
        --instance-manifest artifacts/bag_protocol_v1/instance_manifest.csv \
        --split-csv artifacts/split_protocol_v1/main_split.csv \
        --output-dir artifacts/gan_augmented_v1 \
        --target-bags LESION_MC_001 LESION_MC_002
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    """Run GAN augmentation pipeline."""
    parser = argparse.ArgumentParser(
        description="Run GAN augmentation pipeline"
    )
    parser.add_argument(
        "--instance-manifest",
        type=Path,
        required=True,
        help="Path to instance_manifest.csv",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        required=True,
        help="Path to split CSV",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Path to feature cache (for extracting synthetic features)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for augmented data",
    )
    parser.add_argument(
        "--target-bags",
        nargs="+",
        default=["LESION_MC_001", "LESION_MC_002"],
        help="Bags to inject synthetic instances into",
    )
    parser.add_argument(
        "--synthetics-per-bag",
        type=int,
        default=2,
        help="Number of synthetic instances per bag",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size for synthetic patches",
    )
    args = parser.parse_args()

    from pancreas_vision.gan import (
        create_augmented_instance_manifest,
        generate_synthetic_instances,
        prepare_gan_training_dataset,
        save_synthetic_instances_manifest,
    )

    print("=" * 60)
    print("GAN Augmentation Pipeline")
    print("=" * 60)

    # Step 1: Prepare GAN training data
    print("\n[Step 1] Preparing GAN training data...")
    gan_data_dir = args.output_dir / "gan_train"
    gan_summary = prepare_gan_training_dataset(
        instance_manifest_path=args.instance_manifest,
        split_csv_path=args.split_csv,
        output_dir=gan_data_dir,
        target_label="ADM",
        target_buckets=["multichannel_kc_adm", "multichannel_adm"],
        split_role="train",
        image_size=args.image_size,
    )

    # Step 2: Generate synthetic instances (placeholder)
    print("\n[Step 2] Generating synthetic instances...")
    synth_dir = args.output_dir / "synthetic_images"
    synthetic_instances = generate_synthetic_instances(
        output_dir=synth_dir,
        source_bag_ids=args.target_bags,
        num_instances_per_bag=args.synthetics_per_bag,
        seed_start=42,
        truncation_psi=0.7,
    )

    # Step 3: Filter synthetic instances
    print("\n[Step 3] Filtering synthetic instances...")
    # Get real image paths for comparison
    instance_df = pd.read_csv(args.instance_manifest)
    split_df = pd.read_csv(args.split_csv)
    train_bags = set(split_df[split_df["split_role"] == "train"]["bag_id"])

    real_paths = [
        Path(row["image_path"])
        for _, row in instance_df.iterrows()
        if row["bag_id"] in train_bags and Path(row["image_path"]).exists()
    ]

    from pancreas_vision.gan.synthesis import filter_synthetic_instances
    filtered_instances = filter_synthetic_instances(
        synthetic_instances=synthetic_instances,
        real_instance_paths=real_paths[:50],  # Limit for efficiency
        similarity_threshold=0.90,
    )

    # Save synthetic manifest
    synth_manifest_path = args.output_dir / "synthetic_instances.json"
    save_synthetic_instances_manifest(filtered_instances, synth_manifest_path)

    # Step 4: Create augmented instance manifest
    print("\n[Step 4] Creating augmented instance manifest...")
    aug_manifest_path = args.output_dir / "augmented_instance_manifest.csv"
    aug_stats = create_augmented_instance_manifest(
        original_instance_manifest_path=args.instance_manifest,
        synthetic_instances=filtered_instances,
        output_path=aug_manifest_path,
        max_synthetic_per_bag=args.synthetics_per_bag,
        target_bags=args.target_bags,
    )

    # Save pipeline summary
    summary = {
        "gan_training_data": gan_summary,
        "synthetic_instances": {
            "total_generated": len(synthetic_instances),
            "accepted": len([s for s in filtered_instances if s.is_filtered]),
            "rejected": len([s for s in filtered_instances if not s.is_filtered]),
        },
        "augmentation_stats": aug_stats,
    }

    summary_path = args.output_dir / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("GAN Augmentation Complete")
    print("=" * 60)
    print(f"GAN training images: {gan_summary['copied_images']}")
    print(f"Synthetic instances generated: {len(synthetic_instances)}")
    print(f"Synthetic instances accepted: {summary['synthetic_instances']['accepted']}")
    print(f"Total augmented instances: {aug_stats['total_instances']}")
    print(f"Output directory: {args.output_dir}")
    print("\nNext steps:")
    print(f"  1. Extract features for synthetic images (if needed)")
    print(f"  2. Train CLAM with augmented manifest:")
    print(f"     PYTHONPATH=src python src/train_clam.py \\")
    print(f"         --instance-manifest {aug_manifest_path} \\")
    print(f"         --split-csv {args.split_csv} \\")
    print(f"         --output-dir artifacts/exp_clam_gan_v1")


if __name__ == "__main__":
    main()