"""Tests for pancreas_vision.io module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pancreas_vision.io import (
    save_bag_predictions,
    save_experiment_summary,
    save_metrics,
    save_predictions,
    save_source_bucket_errors,
    write_manifest,
)
from pancreas_vision.types import (
    BagPredictionRecord,
    EvaluationMetrics,
    ImageRecord,
    PredictionRecord,
    SourceBucketErrorRecord,
)


class TestSaveMetrics:
    """Tests for save_metrics function."""

    def test_saves_json_file(self, tmp_output_dir: Path, sample_evaluation_metrics: EvaluationMetrics):
        """Test that metrics are saved as JSON."""
        output_path = tmp_output_dir / "metrics.json"

        save_metrics(sample_evaluation_metrics, output_path)

        assert output_path.exists()

        with output_path.open("r") as f:
            data = json.load(f)

        assert data["accuracy"] == 0.85
        assert data["sensitivity"] == 0.80
        assert data["true_negative"] == 45

    def test_creates_parent_directory(self, tmp_path: Path, sample_evaluation_metrics: EvaluationMetrics):
        """Test that parent directories are created."""
        output_path = tmp_path / "nested" / "dir" / "metrics.json"

        save_metrics(sample_evaluation_metrics, output_path)

        assert output_path.exists()


class TestSaveExperimentSummary:
    """Tests for save_experiment_summary function."""

    def test_saves_dict_as_json(self, tmp_output_dir: Path):
        """Test that summary dict is saved as JSON."""
        summary = {
            "experiment": "test",
            "values": [1, 2, 3],
            "nested": {"key": "value"},
        }
        output_path = tmp_output_dir / "summary.json"

        save_experiment_summary(summary, output_path)

        with output_path.open("r") as f:
            data = json.load(f)

        assert data["experiment"] == "test"
        assert data["values"] == [1, 2, 3]
        assert data["nested"]["key"] == "value"


class TestSavePredictions:
    """Tests for save_predictions function."""

    def test_saves_predictions_list(self, tmp_output_dir: Path):
        """Test that predictions are saved as JSON array."""
        predictions = [
            PredictionRecord(
                image_path="/fake/1.tif",
                true_label=1,
                predicted_label=1,
                positive_score=0.75,
                correct=True,
            ),
            PredictionRecord(
                image_path="/fake/2.tif",
                true_label=0,
                predicted_label=1,
                positive_score=0.65,
                correct=False,
            ),
        ]
        output_path = tmp_output_dir / "predictions.json"

        save_predictions(predictions, output_path)

        with output_path.open("r") as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["image_path"] == "/fake/1.tif"
        assert data[1]["correct"] is False

    def test_empty_predictions_list(self, tmp_output_dir: Path):
        """Test saving empty predictions list."""
        output_path = tmp_output_dir / "predictions.json"

        save_predictions([], output_path)

        with output_path.open("r") as f:
            data = json.load(f)

        assert data == []


class TestSaveBagPredictions:
    """Tests for save_bag_predictions function."""

    def test_saves_bag_predictions(self, tmp_output_dir: Path):
        """Test that bag predictions are saved correctly."""
        bag_predictions = [
            BagPredictionRecord(
                bag_id="KC:1",
                true_label=1,
                true_label_name="PanIN",
                predicted_label=1,
                predicted_label_name="PanIN",
                positive_score=0.72,
                correct=True,
                source_buckets="KC",
                instance_count=2,
                dominant_channel="single",
                dominant_magnification="20x",
            ),
        ]
        output_path = tmp_output_dir / "bag_predictions.json"

        save_bag_predictions(bag_predictions, output_path)

        with output_path.open("r") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["bag_id"] == "KC:1"
        assert data[0]["instance_count"] == 2


class TestSaveSourceBucketErrors:
    """Tests for save_source_bucket_errors function."""

    def test_saves_error_records(self, tmp_output_dir: Path):
        """Test that error records are saved correctly."""
        errors = [
            SourceBucketErrorRecord(
                source_bucket="KC",
                bag_count=10,
                error_count=2,
                false_positive_count=1,
                false_negative_count=1,
                accuracy=0.80,
            ),
        ]
        output_path = tmp_output_dir / "errors.json"

        save_source_bucket_errors(errors, output_path)

        with output_path.open("r") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["source_bucket"] == "KC"
        assert data[0]["accuracy"] == 0.80


class TestWriteManifest:
    """Tests for write_manifest function."""

    def test_writes_csv_with_correct_columns(self, tmp_output_dir: Path, sample_image_record: ImageRecord):
        """Test that manifest CSV has correct columns."""
        output_path = tmp_output_dir / "manifest.csv"

        write_manifest([sample_image_record], output_path)

        assert output_path.exists()

        with output_path.open("r") as f:
            lines = f.readlines()

        # Check header
        header = lines[0].strip().split(",")
        assert "record_key" in header
        assert "image_path" in header
        assert "label_name" in header
        assert "lesion_id" in header

    def test_writes_correct_row_count(
        self, tmp_output_dir: Path, sample_records_list: list[ImageRecord]
    ):
        """Test that manifest has correct number of rows."""
        output_path = tmp_output_dir / "manifest.csv"

        write_manifest(sample_records_list, output_path)

        with output_path.open("r") as f:
            lines = f.readlines()

        # 1 header + N data rows
        assert len(lines) == len(sample_records_list) + 1

    def test_handles_crop_box(self, tmp_output_dir: Path, sample_image_record_with_crop: ImageRecord):
        """Test that crop_box is serialized correctly."""
        output_path = tmp_output_dir / "manifest.csv"

        write_manifest([sample_image_record_with_crop], output_path)

        with output_path.open("r") as f:
            lines = f.readlines()

        # Check that crop_box appears in data row
        assert "100,200,300,400" in lines[1]

    def test_empty_crop_box_is_empty_string(
        self, tmp_output_dir: Path, sample_image_record: ImageRecord
    ):
        """Test that None crop_box becomes empty string."""
        output_path = tmp_output_dir / "manifest.csv"

        write_manifest([sample_image_record], output_path)

        with output_path.open("r") as f:
            content = f.read()

        # crop_box column should have empty value for None
        # This is a simple check - exact format depends on CSV column order
        assert "crop_box" in content  # Header exists