"""Tests for pancreas_vision.types dataclasses."""

from __future__ import annotations

from pathlib import Path

import pytest

from pancreas_vision.types import (
    BagPredictionRecord,
    EvaluationMetrics,
    ImageRecord,
    PredictionRecord,
    SourceBucketErrorRecord,
    TrainingHistory,
)


class TestImageRecord:
    """Tests for ImageRecord dataclass."""

    def test_image_record_basic_fields(self, sample_image_record: ImageRecord):
        """Test that all basic fields are set correctly."""
        assert sample_image_record.source_bucket == "KC"
        assert sample_image_record.label_name == "PanIN"
        assert sample_image_record.label_index == 1
        assert sample_image_record.lesion_id == "KC:1"
        assert sample_image_record.magnification == "20x"
        assert sample_image_record.channel_name == "single"
        assert sample_image_record.sample_type == "whole_image"
        assert sample_image_record.crop_box is None
        assert sample_image_record.label_source == "folder_label"

    def test_record_key_without_crop(self, sample_image_record: ImageRecord):
        """Test record_key property returns image path for non-cropped records."""
        expected = "/fake/data/KC/test.tif"
        assert sample_image_record.record_key == expected

    def test_record_key_with_crop(self, sample_image_record_with_crop: ImageRecord):
        """Test record_key includes crop box for cropped records."""
        expected = "/fake/data/KC/test.tif#crop=100,200,300,400#type=roi_crop"
        assert sample_image_record_with_crop.record_key == expected

    def test_image_record_frozen(self, sample_image_record: ImageRecord):
        """Test that ImageRecord is frozen (immutable)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            sample_image_record.label_name = "ADM"

    def test_image_record_equality(self):
        """Test that two ImageRecords with same fields are equal."""
        record1 = ImageRecord(
            image_path=Path("/fake/test.tif"),
            source_bucket="KC",
            label_name="PanIN",
            label_index=1,
            lesion_id="KC:1",
            group_id="KC:1",
            magnification="20x",
            channel_name="single",
        )
        record2 = ImageRecord(
            image_path=Path("/fake/test.tif"),
            source_bucket="KC",
            label_name="PanIN",
            label_index=1,
            lesion_id="KC:1",
            group_id="KC:1",
            magnification="20x",
            channel_name="single",
        )
        assert record1 == record2

    def test_image_record_hash(self, sample_image_record: ImageRecord):
        """Test that ImageRecord is hashable (can be used in sets/dicts)."""
        records_set = {sample_image_record}
        assert sample_image_record in records_set


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_evaluation_metrics_fields(self, sample_evaluation_metrics: EvaluationMetrics):
        """Test that all fields are set correctly."""
        assert sample_evaluation_metrics.accuracy == 0.85
        assert sample_evaluation_metrics.sensitivity == 0.80
        assert sample_evaluation_metrics.specificity == 0.90
        assert sample_evaluation_metrics.roc_auc == 0.88
        assert sample_evaluation_metrics.true_negative == 45
        assert sample_evaluation_metrics.false_positive == 5
        assert sample_evaluation_metrics.false_negative == 8
        assert sample_evaluation_metrics.true_positive == 32

    def test_evaluation_metrics_default_values(self):
        """Test that metrics can be created with specific values."""
        metrics = EvaluationMetrics(
            accuracy=0.0,
            sensitivity=0.0,
            specificity=0.0,
            roc_auc=0.0,
            true_negative=0,
            false_positive=0,
            false_negative=0,
            true_positive=0,
        )
        assert metrics.accuracy == 0.0
        assert metrics.true_negative == 0


class TestPredictionRecord:
    """Tests for PredictionRecord dataclass."""

    def test_prediction_record_basic(self):
        """Test basic PredictionRecord creation."""
        record = PredictionRecord(
            image_path="/fake/test.tif",
            true_label=1,
            predicted_label=1,
            positive_score=0.75,
            correct=True,
        )
        assert record.image_path == "/fake/test.tif"
        assert record.true_label == 1
        assert record.predicted_label == 1
        assert record.positive_score == 0.75
        assert record.correct is True
        assert record.bag_id is None
        assert record.source_bucket is None

    def test_prediction_record_with_optional_fields(self):
        """Test PredictionRecord with optional fields."""
        record = PredictionRecord(
            image_path="/fake/test.tif",
            true_label=1,
            predicted_label=0,
            positive_score=0.35,
            correct=False,
            bag_id="KC:1",
            source_bucket="KC",
            label_name="PanIN",
            sample_type="whole_image",
        )
        assert record.bag_id == "KC:1"
        assert record.source_bucket == "KC"
        assert record.label_name == "PanIN"


class TestTrainingHistory:
    """Tests for TrainingHistory dataclass."""

    def test_training_history_fields(self, sample_training_history: TrainingHistory):
        """Test TrainingHistory fields."""
        assert sample_training_history.epoch == 1
        assert sample_training_history.train_loss == 0.65
        assert sample_training_history.train_accuracy == 0.72
        assert sample_training_history.learning_rate == 1e-4


class TestBagPredictionRecord:
    """Tests for BagPredictionRecord dataclass."""

    def test_bag_prediction_record_fields(self):
        """Test BagPredictionRecord fields."""
        record = BagPredictionRecord(
            bag_id="KC:1",
            true_label=1,
            true_label_name="PanIN",
            predicted_label=1,
            predicted_label_name="PanIN",
            positive_score=0.72,
            correct=True,
            source_buckets="KC",
            instance_count=3,
            dominant_channel="single",
            dominant_magnification="20x",
        )
        assert record.bag_id == "KC:1"
        assert record.instance_count == 3
        assert record.correct is True


class TestSourceBucketErrorRecord:
    """Tests for SourceBucketErrorRecord dataclass."""

    def test_source_bucket_error_record_fields(self):
        """Test SourceBucketErrorRecord fields."""
        record = SourceBucketErrorRecord(
            source_bucket="KC",
            bag_count=10,
            error_count=2,
            false_positive_count=1,
            false_negative_count=1,
            accuracy=0.80,
        )
        assert record.source_bucket == "KC"
        assert record.bag_count == 10
        assert record.error_count == 2
        assert record.accuracy == 0.80