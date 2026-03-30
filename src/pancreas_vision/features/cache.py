"""Feature caching and index management."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    from pancreas_vision.features.extractors import FeatureExtractor


def extract_and_cache_features(
    instance_manifest: list[dict],
    extractor: "FeatureExtractor",
    output_dir: Path,
    data_root: Path,
    num_patches: int = 4,
) -> dict:
    """Extract and cache features for all instances.

    For each instance:
    - Whole image: Extract global feature + up to num_patches local features
    - ROI crop: Extract global feature only (no additional patches)

    Args:
        instance_manifest: List of instance dicts from instance_manifest.csv
        extractor: FeatureExtractor instance (UNI or DINOv2)
        output_dir: Directory to save features
        data_root: Root directory for image paths
        num_patches: Number of local patches per whole image

    Returns:
        Dict with "index_rows" and "stats" keys
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)

    index_rows = []
    stats = {
        "total_instances": 0,
        "total_features": 0,
        "by_type": {"global": 0, "patch": 0},
        "errors": [],
    }

    for instance in tqdm(instance_manifest, desc="Extracting features"):
        image_path = data_root / instance["image_path"]
        is_roi = instance.get("is_roi", "0") == "1"

        try:
            # Load image
            with Image.open(image_path) as img:
                img = img.convert("RGB")

                # Apply crop box if ROI
                if is_roi and instance.get("crop_box"):
                    coords = [int(x) for x in instance["crop_box"].split(",")]
                    img = img.crop(tuple(coords))

                # Extract global feature
                global_feature = extractor.extract_features(img)
                global_path = features_dir / f"{instance['instance_id']}_global.pt"
                torch.save(
                    {
                        "instance_id": instance["instance_id"],
                        "bag_id": instance.get("bag_id", ""),
                        "feature_type": "global",
                        "features": torch.from_numpy(global_feature),
                    },
                    global_path,
                )

                index_rows.append(
                    {
                        "instance_id": instance["instance_id"],
                        "bag_id": instance.get("bag_id", ""),
                        "feature_type": "global",
                        "feature_path": str(global_path.relative_to(output_dir)),
                        "feature_dim": extractor.feature_dim,
                        "extractor": extractor.name,
                        "magnification": instance.get("magnification", "unknown"),
                        "channel_name": instance.get("channel_name", "unknown"),
                        "is_roi": instance.get("is_roi", "0"),
                    }
                )
                stats["total_features"] += 1
                stats["by_type"]["global"] += 1

                # Extract local patches (only for whole images, not ROI crops)
                if not is_roi:
                    from pancreas_vision.features.patches import sample_patches

                    patches = sample_patches(img, num_patches=num_patches)
                    for i, patch in enumerate(patches):
                        patch_feature = extractor.extract_features(patch)
                        patch_path = features_dir / f"{instance['instance_id']}_patch_{i}.pt"
                        torch.save(
                            {
                                "instance_id": instance["instance_id"],
                                "bag_id": instance.get("bag_id", ""),
                                "feature_type": "patch",
                                "patch_index": i,
                                "features": torch.from_numpy(patch_feature),
                            },
                            patch_path,
                        )

                        index_rows.append(
                            {
                                "instance_id": instance["instance_id"],
                                "bag_id": instance.get("bag_id", ""),
                                "feature_type": "patch",
                                "feature_path": str(patch_path.relative_to(output_dir)),
                                "feature_dim": extractor.feature_dim,
                                "extractor": extractor.name,
                                "magnification": instance.get("magnification", "unknown"),
                                "channel_name": instance.get("channel_name", "unknown"),
                                "is_roi": "0",
                            }
                        )
                        stats["total_features"] += 1
                        stats["by_type"]["patch"] += 1

            stats["total_instances"] += 1

        except Exception as e:
            stats["errors"].append(
                {
                    "instance_id": instance.get("instance_id", "unknown"),
                    "image_path": str(image_path),
                    "error": str(e),
                }
            )

    return {"index_rows": index_rows, "stats": stats}


def build_feature_index(output_dir: Path, index_rows: list[dict]) -> None:
    """Write feature_index.csv.

    Args:
        output_dir: Directory to save the index
        index_rows: List of feature metadata dicts
    """
    path = output_dir / "feature_index.csv"
    fieldnames = [
        "instance_id",
        "bag_id",
        "feature_type",
        "feature_path",
        "feature_dim",
        "extractor",
        "magnification",
        "channel_name",
        "is_roi",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)


def save_summary(output_dir: Path, stats: dict, extractor_name: str, feature_dim: int) -> None:
    """Save summary.json with extraction statistics.

    Args:
        output_dir: Directory to save the summary
        stats: Stats dict from extract_and_cache_features
        extractor_name: Name of the extractor used
        feature_dim: Dimension of extracted features
    """
    summary = {
        **stats,
        "extractor": extractor_name,
        "feature_dim": feature_dim,
    }

    # Convert any non-serializable types
    if "errors" in summary and not summary["errors"]:
        summary["errors"] = []

    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")