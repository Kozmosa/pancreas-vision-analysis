"""GAN augmentation utilities for creating augmented training manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from pancreas_vision.types import SyntheticInstance

if TYPE_CHECKING:
    pass


def create_augmented_instance_manifest(
    original_instance_manifest_path: Path,
    synthetic_instances: list[SyntheticInstance],
    output_path: Path,
    max_synthetic_per_bag: int = 2,
    target_bags: list[str] | None = None,
) -> dict:
    """Create augmented instance manifest with synthetic instances.

    Args:
        original_instance_manifest_path: Path to original instance_manifest.csv
        synthetic_instances: List of synthetic instances to add
        output_path: Output path for augmented manifest
        max_synthetic_per_bag: Maximum synthetic instances per bag
        target_bags: Bags to inject synthetic instances into (None = all ADM bags)

    Returns:
        Dict with augmentation statistics
    """
    # Load original manifest
    original_df = pd.read_csv(original_instance_manifest_path)

    # Filter synthetic instances
    accepted_synthetics = [s for s in synthetic_instances if s.is_filtered]

    # Limit per bag
    bag_counts: dict[str, int] = {}
    limited_synthetics: list[SyntheticInstance] = []

    for synth in accepted_synthetics:
        bag_id = synth.source_bag_id

        # Skip if not in target bags
        if target_bags is not None and bag_id not in target_bags:
            continue

        # Check per-bag limit
        current_count = bag_counts.get(bag_id, 0)
        if current_count < max_synthetic_per_bag:
            limited_synthetics.append(synth)
            bag_counts[bag_id] = current_count + 1

    # Create synthetic rows
    synthetic_rows = []
    for synth in limited_synthetics:
        synthetic_rows.append({
            "instance_id": synth.synthetic_id,
            "bag_id": synth.source_bag_id,
            "image_path": synth.image_path,
            "label_name": synth.label_name,
            "source_bucket": "synthetic",
            "magnification": "unknown",
            "channel_name": "synthetic",
            "sample_type": "synthetic",
            "is_roi": 0,
            "split_key": synth.source_bag_id,
            "lesion_id": synth.source_bag_id,
            "record_key": synth.image_path,
            "label_source": "synthetic",
            "crop_box": "",
            "is_synthetic": 1,
        })

    # Combine original and synthetic
    # Add is_synthetic column to original
    original_df["is_synthetic"] = 0

    synthetic_df = pd.DataFrame(synthetic_rows)
    augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)

    # Compute statistics
    stats = {
        "original_instances": len(original_df),
        "synthetic_instances_added": len(synthetic_rows),
        "total_instances": len(augmented_df),
        "bags_with_synthetic": list(bag_counts.keys()),
        "synthetic_per_bag": dict(bag_counts),
    }

    print(f"Augmented instance manifest created:")
    print(f"  Original instances: {stats['original_instances']}")
    print(f"  Synthetic instances added: {stats['synthetic_instances_added']}")
    print(f"  Total instances: {stats['total_instances']}")

    return stats


def create_augmented_feature_index(
    original_feature_index_path: Path,
    synthetic_instances: list[SyntheticInstance],
    synthetic_feature_dir: Path,
    output_path: Path,
) -> dict:
    """Create augmented feature index including synthetic instances.

    This assumes synthetic features have been extracted separately.

    Args:
        original_feature_index_path: Path to original feature_index.csv
        synthetic_instances: List of synthetic instances with features
        synthetic_feature_dir: Directory containing synthetic features
        output_path: Output path for augmented feature index

    Returns:
        Dict with statistics
    """
    # Load original feature index
    original_df = pd.read_csv(original_feature_index_path)

    # Create synthetic feature rows
    synthetic_rows = []
    for synth in synthetic_instances:
        if not synth.is_filtered:
            continue

        if synth.feature_path is None:
            continue

        synthetic_rows.append({
            "instance_id": synth.synthetic_id,
            "bag_id": synth.source_bag_id,
            "feature_type": "global",
            "feature_path": synth.feature_path,
            "feature_dim": 1536,
            "extractor": "uni2-h",
            "magnification": "unknown",
            "channel_name": "synthetic",
            "is_roi": 0,
        })

    if synthetic_rows:
        synthetic_df = pd.DataFrame(synthetic_rows)
        augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    else:
        augmented_df = original_df

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)

    stats = {
        "original_features": len(original_df),
        "synthetic_features": len(synthetic_rows),
        "total_features": len(augmented_df),
    }

    print(f"Augmented feature index created:")
    print(f"  Original features: {stats['original_features']}")
    print(f"  Synthetic features: {stats['synthetic_features']}")
    print(f"  Total features: {stats['total_features']}")

    return stats


def create_augmented_split_csv(
    original_split_path: Path,
    synthetic_instance_manifest_path: Path,
    output_path: Path,
) -> dict:
    """Create augmented split CSV including synthetic instances.

    Synthetic instances inherit the split role of their source bag.

    Args:
        original_split_path: Path to original split CSV
        synthetic_instance_manifest_path: Path to augmented instance manifest
        output_path: Output path for augmented split

    Returns:
        Dict with statistics
    """
    # Load original split
    original_split_df = pd.read_csv(original_split_path)

    # Load augmented instance manifest
    instance_df = pd.read_csv(synthetic_instance_manifest_path)

    # Get synthetic instances
    synthetic_df = instance_df[instance_df["is_synthetic"] == 1]

    if len(synthetic_df) == 0:
        # No synthetic instances, just copy original
        original_split_df.to_csv(output_path, index=False)
        return {"synthetic_split_rows": 0}

    # Build bag -> split_role mapping
    bag_to_role = dict(
        zip(original_split_df["bag_id"], original_split_df["split_role"])
    )

    # Create split rows for synthetic instances
    synthetic_split_rows = []
    for _, row in synthetic_df.iterrows():
        bag_id = row["bag_id"]
        split_role = bag_to_role.get(bag_id, "train")  # Default to train

        synthetic_split_rows.append({
            "bag_id": bag_id,
            "lesion_id": row["lesion_id"],
            "label_name": row["label_name"],
            "source_buckets": "synthetic",
            "instance_count": 1,
            "split_name": "gan_augmented",
            "split_role": split_role,
            "split_seed": 42,
            "split_ratio": "7:3",
        })

    # Note: split CSV is at bag level, not instance level
    # So we don't add synthetic rows to split CSV directly
    # Synthetic instances are added to bags that already exist

    # Just copy original split for now
    original_split_df.to_csv(output_path, index=False)

    stats = {
        "synthetic_bags": len(set(synthetic_df["bag_id"].tolist())),
        "total_bags": len(original_split_df),
    }

    return stats


def main() -> None:
    """CLI entry point for augmentation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create augmented training manifest with synthetic instances"
    )
    parser.add_argument(
        "--original-manifest",
        type=Path,
        required=True,
        help="Path to original instance_manifest.csv",
    )
    parser.add_argument(
        "--synthetic-manifest",
        type=Path,
        required=True,
        help="Path to synthetic_instances.json",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Output path for augmented instance manifest",
    )
    parser.add_argument(
        "--max-per-bag",
        type=int,
        default=2,
        help="Maximum synthetic instances per bag",
    )
    parser.add_argument(
        "--target-bags",
        nargs="+",
        default=None,
        help="Bags to inject synthetic instances into (default: all ADM)",
    )
    args = parser.parse_args()

    # Load synthetic instances
    synthetic_instances = load_synthetic_instances_json(args.synthetic_manifest)

    stats = create_augmented_instance_manifest(
        original_instance_manifest_path=args.original_manifest,
        synthetic_instances=synthetic_instances,
        output_path=args.output_manifest,
        max_synthetic_per_bag=args.max_per_bag,
        target_bags=args.target_bags,
    )

    # Save stats
    stats_path = args.output_manifest.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def load_synthetic_instances_json(path: Path) -> list[SyntheticInstance]:
    """Load synthetic instances from JSON manifest."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "instances" in data:
        return [SyntheticInstance(**s) for s in data["instances"]]
    return [SyntheticInstance(**s) for s in data]


if __name__ == "__main__":
    main()