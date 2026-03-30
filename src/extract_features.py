#!/usr/bin/env python3
"""Extract and cache deep learning features for all instances.

This script extracts features from all instances in the instance_manifest.csv
using either UNI (pathology-specific) or DINOv2 (general-purpose) models.

For each instance:
- Whole images: Extract 1 global feature + N local patch features
- ROI crops: Extract 1 global feature only

Usage:
    # Extract features using UNI
    PYTHONPATH=src python src/extract_features.py --extractor uni

    # Extract features using DINOv2
    PYTHONPATH=src python src/extract_features.py --extractor dinov2 --output-dir artifacts/feature_cache_v1_dinov2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features using UNI or DINOv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instance-manifest",
        type=Path,
        default=Path("artifacts/bag_protocol_v1/instance_manifest.csv"),
        help="Path to instance_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/feature_cache_v1"),
        help="Output directory for cached features",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for image paths",
    )
    parser.add_argument(
        "--extractor",
        choices=["uni", "dinov2"],
        default="uni",
        help="Feature extractor to use",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=4,
        help="Number of local patches per whole image",
    )
    parser.add_argument(
        "--dinov2-model",
        default="dinov2_vits14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        help="DINOv2 model variant",
    )
    parser.add_argument(
        "--uni-model",
        default="uni2-h",
        choices=["uni", "uni2-h"],
        help="UNI model variant",
    )
    args = parser.parse_args()

    # Check input file exists
    if not args.instance_manifest.exists():
        print(f"Error: Instance manifest not found: {args.instance_manifest}")
        sys.exit(1)

    # Load manifest
    print(f"Loading instances from {args.instance_manifest}")
    with args.instance_manifest.open(encoding="utf-8") as f:
        instances = list(csv.DictReader(f))
    print(f"Found {len(instances)} instances")

    # Initialize extractor
    print(f"\nInitializing {args.extractor} extractor...")
    if args.extractor == "uni":
        from pancreas_vision.features.extractors import UNIExtractor

        extractor = UNIExtractor(model_name=args.uni_model, device=args.device)
    else:
        from pancreas_vision.features.extractors import DINOv2Extractor

        extractor = DINOv2Extractor(model_name=args.dinov2_model, device=args.device)

    print(f"  Model: {extractor.name}")
    print(f"  Feature dim: {extractor.feature_dim}")
    print(f"  Device: {args.device}")

    # Extract features
    print(f"\nExtracting features...")
    from pancreas_vision.features.cache import (
        build_feature_index,
        extract_and_cache_features,
        save_summary,
    )

    result = extract_and_cache_features(
        instance_manifest=instances,
        extractor=extractor,
        output_dir=args.output_dir,
        data_root=args.data_root,
        num_patches=args.num_patches,
    )

    # Write index
    build_feature_index(args.output_dir, result["index_rows"])

    # Save summary
    save_summary(
        args.output_dir,
        result["stats"],
        extractor.name,
        extractor.feature_dim,
    )

    # Print summary
    stats = result["stats"]
    print(f"\n{'='*50}")
    print("Extraction complete!")
    print(f"{'='*50}")
    print(f"  Instances processed: {stats['total_instances']}")
    print(f"  Total features: {stats['total_features']}")
    print(f"    Global features: {stats['by_type']['global']}")
    print(f"    Patch features: {stats['by_type']['patch']}")

    if stats["errors"]:
        print(f"\n  Errors: {len(stats['errors'])}")
        for err in stats["errors"][:5]:
            print(f"    - {err['instance_id']}: {err['error']}")

    print(f"\nOutput saved to: {args.output_dir}")
    print(f"  Feature index: {args.output_dir / 'feature_index.csv'}")
    print(f"  Summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()