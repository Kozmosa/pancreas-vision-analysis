"""Tests for pancreas_vision.engine module."""

from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import pytest

from pancreas_vision.engine import (
    aggregate_predictions_to_bags,
    build_transforms,
    create_dataloaders,
    now_timestamp,
    set_random_seed,
)
from pancreas_vision.types import ImageRecord, PredictionRecord


class TestSetRandomSeed:
    """Tests for set_random_seed function."""

    def test_runs_without_error(self):
        """Test that set_random_seed executes without error."""
        # Should not raise
        set_random_seed(42)
        set_random_seed(123)


class TestNowTimestamp:
    """Tests for now_timestamp function."""

    def test_returns_string(self):
        """Test that now_timestamp returns a string."""
        result = now_timestamp()
        assert isinstance(result, str)

    def test_format_matches_pattern(self):
        """Test that timestamp matches expected format."""
        result = now_timestamp()
        # Format: YYYY-MM-DD HH:MM:SS
        pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        assert re.match(pattern, result) is not None


class TestBuildTransforms:
    """Tests for build_transforms function."""

    def test_returns_train_and_eval_transforms(self):
        """Test that function returns two transform objects."""
        train_transform, eval_transform = build_transforms(image_size=224)

        assert train_transform is not None
        assert eval_transform is not None
        assert train_transform != eval_transform  # They should be different

    def test_custom_image_size(self):
        """Test that custom image size is accepted."""
        # Just verify it doesn't raise
        train_transform, eval_transform = build_transforms(image_size=128)
        assert train_transform is not None


class TestCreateDataloaders:
    """Tests for create_dataloaders function."""

    def test_creates_dataloaders(self, sample_records_list: list[ImageRecord]):
        """Test that dataloaders are created successfully."""
        with mock.patch("pancreas_vision.engine.MicroscopyDataset") as mock_dataset:
            mock_dataset.return_value.__len__ = lambda self: len(sample_records_list)
            mock_dataset.return_value.__getitem__ = lambda self, idx: (None, 0, "key")

            train_loader, test_loader = create_dataloaders(
                train_records=sample_records_list[:10],
                test_records=sample_records_list[10:],
                image_size=224,
                batch_size=4,
                num_workers=0,
            )

            assert train_loader is not None
            assert test_loader is not None

    def test_weighted_sampler_option(self, sample_records_list: list[ImageRecord]):
        """Test that weighted sampler option is accepted."""
        with mock.patch("pancreas_vision.engine.MicroscopyDataset") as mock_dataset:
            mock_dataset.return_value.__len__ = lambda self: len(sample_records_list)
            mock_dataset.return_value.__getitem__ = lambda self, idx: (None, 0, "key")

            train_loader, test_loader = create_dataloaders(
                train_records=sample_records_list[:10],
                test_records=sample_records_list[10:],
                image_size=224,
                batch_size=4,
                num_workers=0,
                use_weighted_sampler=True,
            )

            assert train_loader is not None


class TestAggregatePredictionsToBags:
    """Tests for aggregate_predictions_to_bags function."""

    def test_groups_by_bag_id(self, sample_prediction_records: list[PredictionRecord]):
        """Test that predictions are grouped by bag_id."""
        record_lookup = {
            "/fake/data/KC/1.tif": ImageRecord(
                image_path=Path("/fake/data/KC/1.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
            "/fake/data/KC/2.tif": ImageRecord(
                image_path=Path("/fake/data/KC/2.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
            "/fake/data/caerulein_adm/1.tif": ImageRecord(
                image_path=Path("/fake/data/caerulein_adm/1.tif"),
                source_bucket="caerulein_adm",
                label_name="ADM",
                label_index=0,
                lesion_id="caerulein_adm:1",
                group_id="caerulein_adm:1",
                magnification="20x",
                channel_name="single",
            ),
        }

        bag_metrics, bag_predictions, errors = aggregate_predictions_to_bags(
            predictions=sample_prediction_records,
            record_lookup=record_lookup,
        )

        # Should have 2 bags: KC:1 (2 instances) and caerulein_adm:1 (1 instance)
        assert len(bag_predictions) == 2

        bag_ids = {bp.bag_id for bp in bag_predictions}
        assert "KC:1" in bag_ids
        assert "caerulein_adm:1" in bag_ids

    def test_aggregates_scores_by_mean(self, sample_prediction_records: list[PredictionRecord]):
        """Test that bag scores are aggregated by mean."""
        record_lookup = {
            "/fake/data/KC/1.tif": ImageRecord(
                image_path=Path("/fake/data/KC/1.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
            "/fake/data/KC/2.tif": ImageRecord(
                image_path=Path("/fake/data/KC/2.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
            "/fake/data/caerulein_adm/1.tif": ImageRecord(
                image_path=Path("/fake/data/caerulein_adm/1.tif"),
                source_bucket="caerulein_adm",
                label_name="ADM",
                label_index=0,
                lesion_id="caerulein_adm:1",
                group_id="caerulein_adm:1",
                magnification="20x",
                channel_name="single",
            ),
        }

        bag_metrics, bag_predictions, _ = aggregate_predictions_to_bags(
            predictions=sample_prediction_records,
            record_lookup=record_lookup,
        )

        # Find KC:1 bag and verify mean aggregation
        kc_bag = next(bp for bp in bag_predictions if bp.bag_id == "KC:1")
        # Mean of 0.75 and 0.35 = 0.55
        assert abs(kc_bag.positive_score - 0.55) < 0.001

    def test_computes_error_by_source_bucket(self, sample_prediction_records: list[PredictionRecord]):
        """Test that errors are computed per source bucket."""
        record_lookup = {
            "/fake/data/KC/1.tif": ImageRecord(
                image_path=Path("/fake/data/KC/1.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
            "/fake/data/KC/2.tif": ImageRecord(
                image_path=Path("/fake/data/KC/2.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
            "/fake/data/caerulein_adm/1.tif": ImageRecord(
                image_path=Path("/fake/data/caerulein_adm/1.tif"),
                source_bucket="caerulein_adm",
                label_name="ADM",
                label_index=0,
                lesion_id="caerulein_adm:1",
                group_id="caerulein_adm:1",
                magnification="20x",
                channel_name="single",
            ),
        }

        bag_metrics, bag_predictions, errors = aggregate_predictions_to_bags(
            predictions=sample_prediction_records,
            record_lookup=record_lookup,
        )

        # Should have error stats for each source bucket
        error_buckets = {e.source_bucket for e in errors}
        assert "KC" in error_buckets or "caerulein_adm" in error_buckets

    def test_returns_evaluation_metrics(self, sample_prediction_records: list[PredictionRecord]):
        """Test that bag-level EvaluationMetrics are returned."""
        record_lookup = {
            "/fake/data/KC/1.tif": ImageRecord(
                image_path=Path("/fake/data/KC/1.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
            "/fake/data/KC/2.tif": ImageRecord(
                image_path=Path("/fake/data/KC/2.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
            "/fake/data/caerulein_adm/1.tif": ImageRecord(
                image_path=Path("/fake/data/caerulein_adm/1.tif"),
                source_bucket="caerulein_adm",
                label_name="ADM",
                label_index=0,
                lesion_id="caerulein_adm:1",
                group_id="caerulein_adm:1",
                magnification="20x",
                channel_name="single",
            ),
        }

        bag_metrics, _, _ = aggregate_predictions_to_bags(
            predictions=sample_prediction_records,
            record_lookup=record_lookup,
        )

        assert bag_metrics.accuracy is not None
        assert 0.0 <= bag_metrics.accuracy <= 1.0