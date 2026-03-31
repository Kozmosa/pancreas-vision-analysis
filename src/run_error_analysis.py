"""Run error analysis on CLAM experiment outputs.

This script aggregates predictions by source bucket, analyzes hard-case bags
(particularly multichannel_kc_adm), and generates GAN patch candidates for
training data augmentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pancreas_vision.analysis.error_analysis import (
    aggregate_errors_by_source_bucket,
    analyze_hard_case_bags,
    extract_gan_patch_candidates,
    generate_gan_review_shortlist,
    load_attention_summary,
    load_bag_predictions,
    write_error_analysis_outputs,
)


def main() -> None:
    """CLI entry point for error analysis."""
    parser = argparse.ArgumentParser(
        description="Run error analysis on CLAM experiment outputs"
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to CLAM experiment directory (e.g., artifacts/exp_clam_v1)",
    )
    parser.add_argument(
        "--bag-manifest",
        type=Path,
        required=True,
        help="Path to bag_manifest.csv",
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
        help="Path to split CSV (for filtering train-only GAN candidates)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--source-bucket-filter",
        nargs="+",
        default=["multichannel_kc_adm"],
        help="Source buckets to analyze as hard cases",
    )
    parser.add_argument(
        "--max-gan-shortlist",
        type=int,
        default=20,
        help="Maximum number of GAN patch candidates in shortlist",
    )
    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=0.2,
        help="Confidence threshold for boundary cases (distance from 0.5)",
    )
    args = parser.parse_args()

    # Load inputs
    print("Loading inputs...")

    predictions_path = args.experiment_dir / "bag_predictions.csv"
    attentions_path = args.experiment_dir / "attention_summary.json"

    predictions = load_bag_predictions(predictions_path)
    attentions = load_attention_summary(attentions_path)
    bag_manifest = pd.read_csv(args.bag_manifest)
    instance_manifest = pd.read_csv(args.instance_manifest)
    split_csv = pd.read_csv(args.split_csv)

    print(f"Loaded {len(predictions)} predictions, {len(attentions)} attention summaries")

    # Aggregate errors by source bucket
    print("Aggregating errors by source bucket...")
    error_by_bucket = aggregate_errors_by_source_bucket(predictions)

    # Analyze hard-case bags
    print(f"Analyzing hard-case bags: {args.source_bucket_filter}...")
    hard_case_summaries = analyze_hard_case_bags(
        predictions=predictions,
        attentions=attentions,
        bag_manifest=bag_manifest,
        instance_manifest=instance_manifest,
        source_bucket_filter=args.source_bucket_filter,
        boundary_threshold=args.boundary_threshold,
    )

    # Extract GAN patch candidates
    print("Extracting GAN patch candidates from training set...")
    gan_candidates = extract_gan_patch_candidates(
        hard_case_summaries=hard_case_summaries,
        attentions=attentions,
        instance_manifest=instance_manifest,
        split_csv=split_csv,
        split_role="train",
        boundary_threshold=args.boundary_threshold,
    )

    # Generate shortlist
    print(f"Generating shortlist (max {args.max_gan_shortlist} candidates)...")
    gan_shortlist, gan_metadata = generate_gan_review_shortlist(
        candidates=gan_candidates,
        max_shortlist_size=args.max_gan_shortlist,
    )

    # Write outputs
    print(f"Writing outputs to {args.output_dir}...")
    write_error_analysis_outputs(
        output_dir=args.output_dir,
        error_by_bucket=error_by_bucket,
        hard_case_summaries=hard_case_summaries,
        gan_candidates=gan_candidates,
        gan_shortlist=gan_shortlist,
        gan_metadata=gan_metadata,
    )

    # Print summary
    print("\n=== Error Analysis Summary ===")
    print(f"Total predictions: {len(predictions)}")
    print(f"Correct: {sum(1 for p in predictions if p.correct)}")
    print(f"Errors: {sum(1 for p in predictions if not p.correct)}")
    print(f"False positives: {sum(1 for p in predictions if p.true_label == 0 and p.predicted_label == 1)}")
    print(f"False negatives: {sum(1 for p in predictions if p.true_label == 1 and p.predicted_label == 0)}")

    print("\n=== Hard-case Analysis ===")
    print(f"Hard-case bags analyzed: {len(hard_case_summaries)}")
    print(f"Recommended for GAN: {sum(1 for s in hard_case_summaries if s.recommended_for_gan)}")

    print("\n=== GAN Candidates ===")
    print(f"Total candidates: {gan_metadata['total_candidates']}")
    print(f"Shortlisted: {gan_metadata['shortlisted_count']}")
    print(f"Selection criteria: {gan_metadata['selection_criteria']}")

    print("\nOutputs written:")
    print(f"  - error_by_source_bucket.csv")
    print(f"  - hard_case_analysis.json")
    print(f"  - gan_patch_candidates.csv")
    print(f"  - gan_review_shortlist.json")


if __name__ == "__main__":
    main()