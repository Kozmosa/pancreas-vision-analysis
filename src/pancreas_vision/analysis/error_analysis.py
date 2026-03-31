"""Error analysis module for CLAM model predictions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from pancreas_vision.types import (
    AttentionSummaryRecord,
    BagPredictionRecord,
    GANPatchCandidate,
    HardCaseBagSummary,
    SourceBucketErrorRecord,
)

if TYPE_CHECKING:
    pass


def load_bag_predictions(path: Path) -> list[BagPredictionRecord]:
    """Load bag predictions from CSV file."""
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        records.append(
            BagPredictionRecord(
                bag_id=row["bag_id"],
                true_label=int(row["true_label"]),
                true_label_name=row["true_label_name"],
                predicted_label=int(row["predicted_label"]),
                predicted_label_name=row["predicted_label_name"],
                positive_score=float(row["positive_score"]),
                correct=bool(row["correct"]),
                source_buckets=row["source_buckets"],
                instance_count=int(row["instance_count"]),
                dominant_channel=row["dominant_channel"],
                dominant_magnification=row["dominant_magnification"],
            )
        )
    return records


def load_attention_summary(path: Path) -> list[AttentionSummaryRecord]:
    """Load attention summary from JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for item in data:
        records.append(
            AttentionSummaryRecord(
                bag_id=item["bag_id"],
                attention_weights=item["attention_weights"],
                top_instance_indices=item["top_instance_indices"],
                top_magnifications=item["top_magnifications"],
                top_channels=item["top_channels"],
                top_feature_types=item["top_feature_types"],
            )
        )
    return records


def aggregate_errors_by_source_bucket(
    predictions: list[BagPredictionRecord],
) -> list[SourceBucketErrorRecord]:
    """Aggregate prediction errors by source bucket.

    Args:
        predictions: List of bag prediction records

    Returns:
        List of error records grouped by source bucket
    """
    # Some bags have mixed source buckets (e.g., "multichannel_adm|multichannel_kc_adm")
    # We split these and count them in both buckets

    bucket_data: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for pred in predictions:
        # Split source buckets on "|" separator
        buckets = pred.source_buckets.split("|")

        for bucket in buckets:
            bucket_data[bucket]["bag_count"] += 1
            if not pred.correct:
                bucket_data[bucket]["error_count"] += 1
                if pred.true_label == 0 and pred.predicted_label == 1:
                    bucket_data[bucket]["false_positive_count"] += 1
                elif pred.true_label == 1 and pred.predicted_label == 0:
                    bucket_data[bucket]["false_negative_count"] += 1

    records: list[SourceBucketErrorRecord] = []
    for bucket, counts in sorted(bucket_data.items()):
        bag_count = counts["bag_count"]
        error_count = counts["error_count"]
        accuracy = (bag_count - error_count) / bag_count if bag_count > 0 else 0.0
        records.append(
            SourceBucketErrorRecord(
                source_bucket=bucket,
                bag_count=bag_count,
                error_count=error_count,
                false_positive_count=counts["false_positive_count"],
                false_negative_count=counts["false_negative_count"],
                accuracy=accuracy,
            )
        )

    return records


def analyze_hard_case_bags(
    predictions: list[BagPredictionRecord],
    attentions: list[AttentionSummaryRecord],
    bag_manifest: pd.DataFrame,
    instance_manifest: pd.DataFrame,
    source_bucket_filter: list[str] | None = None,
    boundary_threshold: float = 0.2,
) -> list[HardCaseBagSummary]:
    """Analyze hard-case bags for GAN training consideration.

    Args:
        predictions: Bag prediction records
        attentions: Attention summary records
        bag_manifest: Bag manifest DataFrame
        instance_manifest: Instance manifest DataFrame
        source_bucket_filter: Optional filter for source buckets (default: multichannel_kc_adm)
        boundary_threshold: Distance from 0.5 to consider as boundary case

    Returns:
        List of hard-case bag summaries
    """
    if source_bucket_filter is None:
        source_bucket_filter = ["multichannel_kc_adm"]

    # Filter predictions for hard-case buckets
    hard_case_predictions = [
        pred
        for pred in predictions
        if any(filt in pred.source_buckets for filt in source_bucket_filter)
    ]

    # Build attention lookup
    attention_lookup = {att.bag_id: att for att in attentions}

    summaries: list[HardCaseBagSummary] = []

    for pred in hard_case_predictions:
        att = attention_lookup.get(pred.bag_id)

        # Get top attention instances
        top_attention_instances: list[str] = []
        top_attention_channels: list[str] = []

        if att:
            # Get instance IDs for this bag
            bag_instances = instance_manifest[instance_manifest["bag_id"] == pred.bag_id]
            instance_ids = bag_instances["instance_id"].tolist()

            # Map top indices to instance IDs
            for idx in att.top_instance_indices[:5]:
                if idx < len(instance_ids):
                    top_attention_instances.append(instance_ids[idx])

            # Get top channels from attention summary
            top_attention_channels = att.top_channels[:5]

        # Compute boundary score (distance from 0.5)
        boundary_score = abs(pred.positive_score - 0.5)

        # Determine error type
        error_type = None
        if not pred.correct:
            if pred.true_label == 0 and pred.predicted_label == 1:
                error_type = "false_positive"
            else:
                error_type = "false_negative"

        # Determine if recommended for GAN
        recommended_for_gan = False
        gan_reason = ""

        if not pred.correct:
            # Misclassified: highest priority
            recommended_for_gan = True
            gan_reason = f"misclassified_{error_type}"
        elif boundary_score < boundary_threshold:
            # Boundary case: medium priority
            recommended_for_gan = True
            gan_reason = "low_confidence_boundary"
        elif pred.true_label_name == "ADM" and top_attention_instances:
            # ADM bag with high attention instances
            recommended_for_gan = True
            gan_reason = "adm_bag_with_high_attention"

        summaries.append(
            HardCaseBagSummary(
                bag_id=pred.bag_id,
                label_name=pred.true_label_name,
                source_buckets=pred.source_buckets,
                predicted_correctly=pred.correct,
                positive_score=pred.positive_score,
                error_type=error_type,
                instance_count=pred.instance_count,
                top_attention_instances=top_attention_instances,
                top_attention_channels=top_attention_channels,
                boundary_score=boundary_score,
                recommended_for_gan=recommended_for_gan,
                gan_reason=gan_reason,
            )
        )

    return summaries


def extract_gan_patch_candidates(
    hard_case_summaries: list[HardCaseBagSummary],
    attentions: list[AttentionSummaryRecord],
    instance_manifest: pd.DataFrame,
    split_csv: pd.DataFrame,
    split_role: str = "train",
    top_k_attention: int = 5,
    boundary_threshold: float = 0.2,
    priority_channels: list[str] | None = None,
    priority_magnifications: list[str] | None = None,
) -> list[GANPatchCandidate]:
    """Extract ADM patches for GAN training from hard-case bags.

    Only extracts from TRAINING set bags for GAN augmentation.

    Args:
        hard_case_summaries: Hard-case bag summaries
        attentions: Attention summary records
        instance_manifest: Instance manifest DataFrame
        split_csv: Split assignment DataFrame
        split_role: Split role to extract from (default: "train")
        top_k_attention: Number of top attention instances to consider
        boundary_threshold: Confidence threshold for boundary cases
        priority_channels: Channels to prioritize (default: ck19)
        priority_magnifications: Magnifications to prioritize (default: 40x)

    Returns:
        List of GAN patch candidates with priority scores
    """
    if priority_channels is None:
        priority_channels = ["ck19"]
    if priority_magnifications is None:
        priority_magnifications = ["40x"]

    # Filter for training bags only
    train_bag_ids = set(
        split_csv[split_csv["split_role"] == split_role]["bag_id"].tolist()
    )

    # Build attention lookup
    attention_lookup = {att.bag_id: att for att in attentions}

    candidates: list[GANPatchCandidate] = []

    for summary in hard_case_summaries:
        # Skip bags not in training set
        if summary.bag_id not in train_bag_ids:
            continue

        # Only extract ADM bags
        if summary.label_name != "ADM":
            continue

        # Get instances for this bag
        bag_instances = instance_manifest[instance_manifest["bag_id"] == summary.bag_id]
        att = attention_lookup.get(summary.bag_id)

        if att is None:
            continue

        # Extract candidates based on attention ranking
        for rank, idx in enumerate(att.top_instance_indices[:top_k_attention]):
            if idx >= len(bag_instances):
                continue

            instance_row = bag_instances.iloc[idx]
            instance_id = instance_row["instance_id"]

            # Compute selection reason
            selection_reason = "high_attention"
            if not summary.predicted_correctly:
                selection_reason = "high_attention_fp"  # False positive
            elif summary.boundary_score < boundary_threshold:
                selection_reason = "boundary_case"

            # Compute priority score
            # Formula: (1 - boundary_score) * 0.4 + attention_weight * 0.3 + priority_boost * 0.3
            attention_weight = att.attention_weights[idx]
            boundary_factor = 1 - summary.boundary_score

            # Priority boost for channel/magnification
            priority_boost = 0.0
            if instance_row["channel_name"] in priority_channels:
                priority_boost = 1.0
            elif instance_row["magnification"] in priority_magnifications:
                priority_boost = 0.5

            # Misclassified bags get extra boost
            if not summary.predicted_correctly:
                priority_boost = max(priority_boost, 1.0)

            priority_score = (
                boundary_factor * 0.4
                + attention_weight * 0.3
                + priority_boost * 0.3
            )

            candidates.append(
                GANPatchCandidate(
                    instance_id=instance_id,
                    bag_id=summary.bag_id,
                    image_path=instance_row["image_path"],
                    magnification=instance_row["magnification"],
                    channel_name=instance_row["channel_name"],
                    true_label_name=summary.label_name,
                    attention_weight=attention_weight,
                    attention_rank=rank,
                    selection_reason=selection_reason,
                    priority_score=priority_score,
                )
            )

    # Sort by priority score descending
    candidates.sort(key=lambda c: c.priority_score, reverse=True)

    return candidates


def generate_gan_review_shortlist(
    candidates: list[GANPatchCandidate],
    max_shortlist_size: int = 20,
) -> tuple[list[GANPatchCandidate], dict]:
    """Generate final shortlist for manual review.

    Args:
        candidates: Full list of GAN patch candidates
        max_shortlist_size: Maximum number of candidates to shortlist

    Returns:
        Tuple of (shortlisted candidates, metadata dict)
    """
    shortlisted = candidates[:max_shortlist_size]

    # Compute metadata
    selection_reasons: dict[str, int] = Counter(c.selection_reason for c in candidates)
    source_bucket_summary: dict[str, int] = Counter(c.bag_id for c in candidates)
    channel_summary: dict[str, int] = Counter(c.channel_name for c in candidates)

    metadata = {
        "total_candidates": len(candidates),
        "shortlisted_count": len(shortlisted),
        "max_shortlist_size": max_shortlist_size,
        "selection_criteria": dict(selection_reasons),
        "source_bucket_summary": dict(source_bucket_summary),
        "channel_summary": dict(channel_summary),
        "notes": (
            "Shortlisted candidates are sorted by priority score. "
            "High attention FP instances from misclassified ADM bags are prioritized. "
            "ck19 channel and 40x magnification instances receive priority boost."
        ),
    }

    return shortlisted, metadata


def write_error_analysis_outputs(
    output_dir: Path,
    error_by_bucket: list[SourceBucketErrorRecord],
    hard_case_summaries: list[HardCaseBagSummary],
    gan_candidates: list[GANPatchCandidate],
    gan_shortlist: list[GANPatchCandidate],
    gan_metadata: dict,
) -> None:
    """Write all error analysis outputs to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write error by source bucket
    bucket_df = pd.DataFrame([r.__dict__ for r in error_by_bucket])
    bucket_df.to_csv(output_dir / "error_by_source_bucket.csv", index=False)

    # Write hard case analysis
    summary_dicts = []
    for s in hard_case_summaries:
        d = {
            "bag_id": s.bag_id,
            "label_name": s.label_name,
            "source_buckets": s.source_buckets,
            "predicted_correctly": s.predicted_correctly,
            "positive_score": s.positive_score,
            "error_type": s.error_type,
            "instance_count": s.instance_count,
            "top_attention_instances": s.top_attention_instances,
            "top_attention_channels": s.top_attention_channels,
            "boundary_score": s.boundary_score,
            "recommended_for_gan": s.recommended_for_gan,
            "gan_reason": s.gan_reason,
        }
        summary_dicts.append(d)

    with (output_dir / "hard_case_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(summary_dicts, f, ensure_ascii=True, indent=2)

    # Write all GAN candidates
    candidates_df = pd.DataFrame([c.__dict__ for c in gan_candidates])
    candidates_df.to_csv(output_dir / "gan_patch_candidates.csv", index=False)

    # Write shortlist
    shortlist_dicts = [c.__dict__ for c in gan_shortlist]
    with (output_dir / "gan_review_shortlist.json").open("w", encoding="utf-8") as f:
        json.dump({"metadata": gan_metadata, "shortlist": shortlist_dicts}, f, ensure_ascii=True, indent=2)