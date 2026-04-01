"""GAN data preparation utilities for StyleGAN3 training."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from PIL import Image

if TYPE_CHECKING:
    pass


def prepare_gan_training_dataset(
    instance_manifest_path: Path,
    split_csv_path: Path,
    output_dir: Path,
    target_label: str = "ADM",
    target_buckets: list[str] | None = None,
    split_role: str = "train",
    image_size: int = 256,
) -> dict:
    """Prepare images for StyleGAN3 training.

    Extracts training set ADM images, resizes to target resolution,
    and outputs in StyleGAN3-compatible format.

    Args:
        instance_manifest_path: Path to instance_manifest.csv
        split_csv_path: Path to split CSV
        output_dir: Output directory for GAN training data
        target_label: Target label for GAN training (default: ADM)
        target_buckets: Source buckets to include (default: all ADM)
        split_role: Split role to extract from (default: train)
        image_size: Target image size for StyleGAN3

    Returns:
        Dict with dataset statistics
    """
    if target_buckets is None:
        target_buckets = ["multichannel_kc_adm", "multichannel_adm", "caerulein_adm"]

    # Load manifests
    instance_manifest = pd.read_csv(instance_manifest_path)
    split_csv = pd.read_csv(split_csv_path)

    # Filter for training set
    train_bag_ids = set(
        split_csv[split_csv["split_role"] == split_role]["bag_id"].tolist()
    )

    # Filter instances
    filtered_instances = instance_manifest[
        (instance_manifest["bag_id"].isin(train_bag_ids))
        & (instance_manifest["label_name"] == target_label)
        & (instance_manifest["source_bucket"].isin(target_buckets))
    ]

    # Create output directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Copy and resize images
    copied_count = 0
    skipped_count = 0
    image_records = []

    for _, row in filtered_instances.iterrows():
        src_path = Path(row["image_path"])

        if not src_path.exists():
            skipped_count += 1
            continue

        # Generate destination filename
        dst_filename = f"{row['instance_id']}.png"
        dst_path = images_dir / dst_filename

        try:
            # Load, resize, and save as PNG
            with Image.open(src_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize to target size
                img_resized = img.resize((image_size, image_size), Image.LANCZOS)
                img_resized.save(dst_path, "PNG")

            copied_count += 1
            image_records.append({
                "instance_id": row["instance_id"],
                "bag_id": row["bag_id"],
                "source_path": str(src_path),
                "gan_path": str(dst_path),
                "source_bucket": row["source_bucket"],
                "magnification": row["magnification"],
                "channel_name": row["channel_name"],
            })

        except Exception as e:
            print(f"Warning: Failed to process {src_path}: {e}")
            skipped_count += 1

    # Write dataset metadata
    dataset_meta = {
        "name": f"pancreas_adm_{target_label.lower()}",
        "label": target_label,
        "source_buckets": target_buckets,
        "image_size": image_size,
        "num_images": copied_count,
        "split_role": split_role,
        "instances": image_records,
    }

    meta_path = output_dir / "dataset.json"
    meta_path.write_text(
        json.dumps(dataset_meta, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    # Write summary
    summary = {
        "total_instances": len(filtered_instances),
        "copied_images": copied_count,
        "skipped_images": skipped_count,
        "output_dir": str(output_dir),
        "images_dir": str(images_dir),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print(f"GAN training dataset prepared:")
    print(f"  Total instances found: {len(filtered_instances)}")
    print(f"  Images copied: {copied_count}")
    print(f"  Images skipped: {skipped_count}")
    print(f"  Output directory: {output_dir}")

    return summary


def prepare_stylegan3_dataset_format(
    images_dir: Path,
    output_dir: Path,
    dataset_name: str = "pancreas_adm",
) -> Path:
    """Convert images to StyleGAN3 dataset format.

    StyleGAN3 expects datasets in a specific format with a .zip file
    containing images, or a directory with specific structure.

    Args:
        images_dir: Directory containing PNG images
        output_dir: Output directory for StyleGAN3 dataset
        dataset_name: Name for the dataset

    Returns:
        Path to the formatted dataset directory
    """
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to dataset directory
    image_files = list(images_dir.glob("*.png"))

    for src_path in image_files:
        dst_path = dataset_dir / src_path.name
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

    print(f"StyleGAN3 dataset created: {dataset_dir}")
    print(f"  Total images: {len(image_files)}")

    return dataset_dir


def main() -> None:
    """CLI entry point for GAN data preparation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare ADM images for StyleGAN3 training"
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
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for GAN training data",
    )
    parser.add_argument(
        "--target-label",
        type=str,
        default="ADM",
        help="Target label for GAN training",
    )
    parser.add_argument(
        "--target-buckets",
        nargs="+",
        default=["multichannel_kc_adm", "multichannel_adm"],
        help="Source buckets to include",
    )
    parser.add_argument(
        "--split-role",
        type=str,
        default="train",
        help="Split role to extract from",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Target image size for StyleGAN3",
    )
    args = parser.parse_args()

    prepare_gan_training_dataset(
        instance_manifest_path=args.instance_manifest,
        split_csv_path=args.split_csv,
        output_dir=args.output_dir,
        target_label=args.target_label,
        target_buckets=args.target_buckets,
        split_role=args.split_role,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()