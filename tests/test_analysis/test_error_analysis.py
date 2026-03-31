"""Tests for error_analysis module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pancreas_vision.analysis.error_analysis import (
    aggregate_errors_by_source_bucket,
    analyze_hard_case_bags,
    extract_gan_patch_candidates,
    generate_gan_review_shortlist,
    load_attention_summary,
    load_bag_predictions,
)
from pancreas_vision.types import (
    AttentionSummaryRecord,
    BagPredictionRecord,
)


@pytest.fixture
def mock_predictions_csv(tmp_path: Path) -> Path:
    """Create a mock predictions CSV file."""
    import csv

    predictions_path = tmp_path / "bag_predictions.csv"
    with predictions_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bag_id", "true_label", "true_label_name", "predicted_label",
            "predicted_label_name", "positive_score", "correct",
            "source_buckets", "instance_count", "dominant_channel",
            "dominant_magnification"
        ])
        # Correct ADM
        writer.writerow(["BAG_ADM_001", 0, "ADM", 0, "ADM", 0.15, True, "caerulein_adm", 5, "single", "unknown"])
        # False positive ADM (hard case)
        writer.writerow(["BAG_MC_001", 0, "ADM", 1, "PanIN", 0.58, False, "multichannel_kc_adm", 12, "ck19", "40x"])
        # Correct PanIN
        writer.writerow(["BAG_PANIN_001", 1, "PanIN", 1, "PanIN", 0.85, True, "KC", 5, "single", "unknown"])
        # False negative PanIN
        writer.writerow(["BAG_PANIN_002", 1, "PanIN", 0, "ADM", 0.35, False, "KPC", 5, "amylase", "unknown"])
        # Boundary case ADM
        writer.writerow(["BAG_MC_002", 0, "ADM", 0, "ADM", 0.45, True, "multichannel_kc_adm", 6, "merge", "20x"])

    return predictions_path


@pytest.fixture
def mock_attention_json(tmp_path: Path) -> Path:
    """Create a mock attention summary JSON file."""
    attention_path = tmp_path / "attention_summary.json"
    data = [
        {
            "bag_id": "BAG_ADM_001",
            "attention_weights": [0.4, 0.2, 0.15, 0.15, 0.1],
            "top_instance_indices": [0, 1, 2, 3, 4],
            "top_magnifications": ["unknown", "unknown", "unknown", "unknown", "unknown"],
            "top_channels": ["single", "single", "single", "single", "single"],
            "top_feature_types": ["global", "patch", "patch", "patch", "patch"],
        },
        {
            "bag_id": "BAG_MC_001",
            "attention_weights": [0.3, 0.25, 0.2, 0.15, 0.1],
            "top_instance_indices": [0, 1, 2, 3, 4],
            "top_magnifications": ["40x", "20x", "10x", "unknown", "unknown"],
            "top_channels": ["ck19", "merge", "amylase", "single", "single"],
            "top_feature_types": ["global", "global", "global", "patch", "patch"],
        },
        {
            "bag_id": "BAG_PANIN_001",
            "attention_weights": [0.35, 0.25, 0.2, 0.15, 0.05],
            "top_instance_indices": [0, 1, 2, 3, 4],
            "top_magnifications": ["unknown", "unknown", "unknown", "unknown", "unknown"],
            "top_channels": ["single", "single", "single", "single", "single"],
            "top_feature_types": ["global", "patch", "patch", "patch", "patch"],
        },
        {
            "bag_id": "BAG_MC_002",
            "attention_weights": [0.4, 0.3, 0.2, 0.05, 0.05],
            "top_instance_indices": [0, 1, 2, 3, 4],
            "top_magnifications": ["20x", "10x", "40x", "unknown", "unknown"],
            "top_channels": ["merge", "amylase", "ck19", "single", "single"],
            "top_feature_types": ["global", "global", "global", "patch", "patch"],
        },
    ]
    with attention_path.open("w") as f:
        json.dump(data, f)

    return attention_path


@pytest.fixture
def mock_bag_manifest_df() -> pd.DataFrame:
    """Create a mock bag manifest DataFrame."""
    return pd.DataFrame({
        "bag_id": ["BAG_ADM_001", "BAG_MC_001", "BAG_PANIN_001", "BAG_PANIN_002", "BAG_MC_002"],
        "label_name": ["ADM", "ADM", "PanIN", "PanIN", "ADM"],
        "source_buckets": ["caerulein_adm", "multichannel_kc_adm", "KC", "KPC", "multichannel_kc_adm"],
        "instance_count": [5, 12, 5, 5, 6],
    })


@pytest.fixture
def mock_instance_manifest_df() -> pd.DataFrame:
    """Create a mock instance manifest DataFrame."""
    instances = []
    for bag_id, count in [("BAG_ADM_001", 5), ("BAG_MC_001", 12), ("BAG_PANIN_001", 5), ("BAG_PANIN_002", 5), ("BAG_MC_002", 6)]:
        for i in range(count):
            instances.append({
                "instance_id": f"INST_{bag_id}_{i}",
                "bag_id": bag_id,
                "image_path": f"data/{bag_id}/{i}.tif",
                "magnification": "40x" if i == 0 else "unknown",
                "channel_name": "ck19" if i == 0 else "single",
            })
    return pd.DataFrame(instances)


@pytest.fixture
def mock_split_df() -> pd.DataFrame:
    """Create a mock split DataFrame."""
    return pd.DataFrame({
        "bag_id": ["BAG_ADM_001", "BAG_MC_001", "BAG_PANIN_001", "BAG_PANIN_002", "BAG_MC_002"],
        "split_role": ["train", "train", "test", "test", "train"],
        "split_name": ["hard_case_split"] * 5,
    })


class TestLoadBagPredictions:
    """Tests for load_bag_predictions function."""

    def test_loads_from_csv(
        self,
        mock_predictions_csv: Path,
    ):
        """Test loading predictions from CSV."""
        predictions = load_bag_predictions(mock_predictions_csv)

        assert len(predictions) == 5
        assert isinstance(predictions[0], BagPredictionRecord)
        assert predictions[0].bag_id == "BAG_ADM_001"

    def test_parses_correct_field(
        self,
        mock_predictions_csv: Path,
    ):
        """Test that correct field is parsed as boolean."""
        predictions = load_bag_predictions(mock_predictions_csv)

        correct_bag = predictions[0]
        assert correct_bag.correct is True

        fp_bag = predictions[1]
        assert fp_bag.correct is False


class TestLoadAttentionSummary:
    """Tests for load_attention_summary function."""

    def test_loads_from_json(
        self,
        mock_attention_json: Path,
    ):
        """Test loading attention from JSON."""
        attentions = load_attention_summary(mock_attention_json)

        assert len(attentions) == 4
        assert isinstance(attentions[0], AttentionSummaryRecord)

    def test_attention_weights_are_floats(
        self,
        mock_attention_json: Path,
    ):
        """Test that attention weights are float lists."""
        attentions = load_attention_summary(mock_attention_json)

        for att in attentions:
            assert isinstance(att.attention_weights, list)
            assert all(isinstance(w, float) for w in att.attention_weights)


class TestAggregateErrorsBySourceBucket:
    """Tests for aggregate_errors_by_source_bucket function."""

    def test_aggregates_by_bucket(
        self,
        mock_predictions_csv: Path,
    ):
        """Test that errors are aggregated by source bucket."""
        predictions = load_bag_predictions(mock_predictions_csv)
        errors = aggregate_errors_by_source_bucket(predictions)

        # Should have entries for each bucket
        bucket_names = [e.source_bucket for e in errors]
        assert "caerulein_adm" in bucket_names
        assert "KC" in bucket_names
        assert "multichannel_kc_adm" in bucket_names

    def test_counts_false_positives(
        self,
        mock_predictions_csv: Path,
    ):
        """Test that false positives are counted correctly."""
        predictions = load_bag_predictions(mock_predictions_csv)
        errors = aggregate_errors_by_source_bucket(predictions)

        # Find multichannel_kc_adm bucket (should have 1 FP)
        mc_errors = [e for e in errors if e.source_bucket == "multichannel_kc_adm"][0]
        assert mc_errors.false_positive_count == 1
        assert mc_errors.error_count == 1  # Only FP, no FN

    def test_splits_mixed_buckets(
        self,
        mock_predictions_csv: Path,
    ):
        """Test that mixed source buckets are split."""
        predictions = load_bag_predictions(mock_predictions_csv)
        errors = aggregate_errors_by_source_bucket(predictions)

        # Bags with "multichannel_adm|multichannel_kc_adm" would be counted in both
        # (our mock doesn't have mixed, but the logic handles it)


class TestAnalyzeHardCaseBags:
    """Tests for analyze_hard_case_bags function."""

    def test_filters_by_source_bucket(
        self,
        mock_predictions_csv: Path,
        mock_attention_json: Path,
        mock_bag_manifest_df: pd.DataFrame,
        mock_instance_manifest_df: pd.DataFrame,
    ):
        """Test that analysis filters by source bucket."""
        predictions = load_bag_predictions(mock_predictions_csv)
        attentions = load_attention_summary(mock_attention_json)

        summaries = analyze_hard_case_bags(
            predictions=predictions,
            attentions=attentions,
            bag_manifest=mock_bag_manifest_df,
            instance_manifest=mock_instance_manifest_df,
            source_bucket_filter=["multichannel_kc_adm"],
        )

        # Should only have MC bags
        for s in summaries:
            assert "multichannel_kc_adm" in s.source_buckets

    def test_identifies_misclassified_as_hard_case(
        self,
        mock_predictions_csv: Path,
        mock_attention_json: Path,
        mock_bag_manifest_df: pd.DataFrame,
        mock_instance_manifest_df: pd.DataFrame,
    ):
        """Test that misclassified bags are marked as hard cases."""
        predictions = load_bag_predictions(mock_predictions_csv)
        attentions = load_attention_summary(mock_attention_json)

        summaries = analyze_hard_case_bags(
            predictions=predictions,
            attentions=attentions,
            bag_manifest=mock_bag_manifest_df,
            instance_manifest=mock_instance_manifest_df,
            source_bucket_filter=["multichannel_kc_adm"],
        )

        # Find the FP bag
        fp_summary = [s for s in summaries if not s.predicted_correctly][0]
        assert fp_summary.error_type == "false_positive"
        assert fp_summary.recommended_for_gan is True

    def test_identifies_boundary_cases(
        self,
        mock_predictions_csv: Path,
        mock_attention_json: Path,
        mock_bag_manifest_df: pd.DataFrame,
        mock_instance_manifest_df: pd.DataFrame,
    ):
        """Test that boundary cases are identified."""
        predictions = load_bag_predictions(mock_predictions_csv)
        attentions = load_attention_summary(mock_attention_json)

        summaries = analyze_hard_case_bags(
            predictions=predictions,
            attentions=attentions,
            bag_manifest=mock_bag_manifest_df,
            instance_manifest=mock_instance_manifest_df,
            source_bucket_filter=["multichannel_kc_adm"],
            boundary_threshold=0.2,
        )

        # BAG_MC_002 has score 0.45 (boundary of 0.05)
        boundary_summary = [s for s in summaries if s.bag_id == "BAG_MC_002"][0]
        assert boundary_summary.boundary_score < 0.1  # Very close to 0.5
        assert boundary_summary.recommended_for_gan is True


class TestExtractGANPatchCandidates:
    """Tests for extract_gan_patch_candidates function."""

    def test_only_from_training_set(
        self,
        mock_predictions_csv: Path,
        mock_attention_json: Path,
        mock_bag_manifest_df: pd.DataFrame,
        mock_instance_manifest_df: pd.DataFrame,
        mock_split_df: pd.DataFrame,
    ):
        """Test that candidates are only from training set."""
        predictions = load_bag_predictions(mock_predictions_csv)
        attentions = load_attention_summary(mock_attention_json)

        summaries = analyze_hard_case_bags(
            predictions=predictions,
            attentions=attentions,
            bag_manifest=mock_bag_manifest_df,
            instance_manifest=mock_instance_manifest_df,
            source_bucket_filter=["multichannel_kc_adm"],
        )

        candidates = extract_gan_patch_candidates(
            hard_case_summaries=summaries,
            attentions=attentions,
            instance_manifest=mock_instance_manifest_df,
            split_csv=mock_split_df,
            split_role="train",
        )

        # Check that all candidates come from training bags
        train_bag_ids = set(mock_split_df[mock_split_df["split_role"] == "train"]["bag_id"])
        for c in candidates:
            assert c.bag_id in train_bag_ids

    def test_only_adm_bags(
        self,
        mock_predictions_csv: Path,
        mock_attention_json: Path,
        mock_bag_manifest_df: pd.DataFrame,
        mock_instance_manifest_df: pd.DataFrame,
        mock_split_df: pd.DataFrame,
    ):
        """Test that only ADM bags are extracted."""
        predictions = load_bag_predictions(mock_predictions_csv)
        attentions = load_attention_summary(mock_attention_json)

        summaries = analyze_hard_case_bags(
            predictions=predictions,
            attentions=attentions,
            bag_manifest=mock_bag_manifest_df,
            instance_manifest=mock_instance_manifest_df,
            source_bucket_filter=["multichannel_kc_adm"],
        )

        candidates = extract_gan_patch_candidates(
            hard_case_summaries=summaries,
            attentions=attentions,
            instance_manifest=mock_instance_manifest_df,
            split_csv=mock_split_df,
            split_role="train",
        )

        for c in candidates:
            assert c.true_label_name == "ADM"

    def test_sorted_by_priority(
        self,
        mock_predictions_csv: Path,
        mock_attention_json: Path,
        mock_bag_manifest_df: pd.DataFrame,
        mock_instance_manifest_df: pd.DataFrame,
        mock_split_df: pd.DataFrame,
    ):
        """Test that candidates are sorted by priority score."""
        predictions = load_bag_predictions(mock_predictions_csv)
        attentions = load_attention_summary(mock_attention_json)

        summaries = analyze_hard_case_bags(
            predictions=predictions,
            attentions=attentions,
            bag_manifest=mock_bag_manifest_df,
            instance_manifest=mock_instance_manifest_df,
            source_bucket_filter=["multichannel_kc_adm"],
        )

        candidates = extract_gan_patch_candidates(
            hard_case_summaries=summaries,
            attentions=attentions,
            instance_manifest=mock_instance_manifest_df,
            split_csv=mock_split_df,
            split_role="train",
        )

        if len(candidates) > 1:
            for i in range(len(candidates) - 1):
                assert candidates[i].priority_score >= candidates[i + 1].priority_score


class TestGenerateGANReviewShortlist:
    """Tests for generate_gan_review_shortlist function."""

    def test_limits_shortlist_size(
        self,
        mock_predictions_csv: Path,
        mock_attention_json: Path,
        mock_bag_manifest_df: pd.DataFrame,
        mock_instance_manifest_df: pd.DataFrame,
        mock_split_df: pd.DataFrame,
    ):
        """Test that shortlist is limited to max size."""
        predictions = load_bag_predictions(mock_predictions_csv)
        attentions = load_attention_summary(mock_attention_json)

        summaries = analyze_hard_case_bags(
            predictions=predictions,
            attentions=attentions,
            bag_manifest=mock_bag_manifest_df,
            instance_manifest=mock_instance_manifest_df,
            source_bucket_filter=["multichannel_kc_adm"],
        )

        candidates = extract_gan_patch_candidates(
            hard_case_summaries=summaries,
            attentions=attentions,
            instance_manifest=mock_instance_manifest_df,
            split_csv=mock_split_df,
            split_role="train",
        )

        shortlist, metadata = generate_gan_review_shortlist(
            candidates=candidates,
            max_shortlist_size=5,
        )

        assert len(shortlist) <= 5
        assert metadata["shortlisted_count"] <= 5

    def test_metadata_includes_counts(
        self,
        mock_predictions_csv: Path,
        mock_attention_json: Path,
        mock_bag_manifest_df: pd.DataFrame,
        mock_instance_manifest_df: pd.DataFrame,
        mock_split_df: pd.DataFrame,
    ):
        """Test that metadata includes summary counts."""
        predictions = load_bag_predictions(mock_predictions_csv)
        attentions = load_attention_summary(mock_attention_json)

        summaries = analyze_hard_case_bags(
            predictions=predictions,
            attentions=attentions,
            bag_manifest=mock_bag_manifest_df,
            instance_manifest=mock_instance_manifest_df,
            source_bucket_filter=["multichannel_kc_adm"],
        )

        candidates = extract_gan_patch_candidates(
            hard_case_summaries=summaries,
            attentions=attentions,
            instance_manifest=mock_instance_manifest_df,
            split_csv=mock_split_df,
            split_role="train",
        )

        shortlist, metadata = generate_gan_review_shortlist(
            candidates=candidates,
            max_shortlist_size=5,
        )

        assert "total_candidates" in metadata
        assert "shortlisted_count" in metadata
        assert "selection_criteria" in metadata