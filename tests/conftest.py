"""Shared pytest fixtures for pancreas_vision tests."""

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


# ---------------------------------------------------------------------------
# Basic record fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image_record() -> ImageRecord:
    """A minimal ImageRecord for testing."""
    return ImageRecord(
        image_path=Path("/fake/data/KC/test.tif"),
        source_bucket="KC",
        label_name="PanIN",
        label_index=1,
        lesion_id="KC:1",
        group_id="KC:1",
        magnification="20x",
        channel_name="single",
        sample_type="whole_image",
        crop_box=None,
        label_source="folder_label",
    )


@pytest.fixture
def sample_image_record_with_crop() -> ImageRecord:
    """An ImageRecord with a crop box."""
    return ImageRecord(
        image_path=Path("/fake/data/KC/test.tif"),
        source_bucket="KC",
        label_name="PanIN",
        label_index=1,
        lesion_id="KC:1",
        group_id="KC:1",
        magnification="20x",
        channel_name="single",
        sample_type="roi_crop",
        crop_box=(100, 200, 300, 400),
        label_source="roi_polygon",
    )


@pytest.fixture
def sample_records_list() -> list[ImageRecord]:
    """A list of 20 ImageRecords (10 ADM + 10 PanIN) for splitting tests."""
    records = []
    # 10 ADM samples
    for i in range(10):
        records.append(
            ImageRecord(
                image_path=Path(f"/fake/data/caerulein_adm/{i}.tif"),
                source_bucket="caerulein_adm",
                label_name="ADM",
                label_index=0,
                lesion_id=f"caerulein_adm:{i}",
                group_id=f"caerulein_adm:{i}",
                magnification="20x",
                channel_name="single",
            )
        )
    # 10 PanIN samples
    for i in range(10):
        records.append(
            ImageRecord(
                image_path=Path(f"/fake/data/KC/{i}.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id=f"KC:{i}",
                group_id=f"KC:{i}",
                magnification="20x",
                channel_name="single",
            )
        )
    return records


@pytest.fixture
def grouped_records_list() -> list[ImageRecord]:
    """Records with group_id shared across multiple images (for group-aware split tests)."""
    records = []
    # Group 1: 3 images of same lesion (ADM)
    for i, mag in enumerate(["10x", "20x", "40x"]):
        records.append(
            ImageRecord(
                image_path=Path(f"/fake/data/caerulein_adm/1_{mag}.tif"),
                source_bucket="caerulein_adm",
                label_name="ADM",
                label_index=0,
                lesion_id="caerulein_adm:1",
                group_id="caerulein_adm:1",
                magnification=mag,
                channel_name="single",
            )
        )
    # Group 2: 2 images of same lesion (PanIN)
    for i, mag in enumerate(["10x", "20x"]):
        records.append(
            ImageRecord(
                image_path=Path(f"/fake/data/KC/1_{mag}.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification=mag,
                channel_name="single",
            )
        )
    # Group 3-7: Single-image groups (ADM)
    for i in range(2, 7):
        records.append(
            ImageRecord(
                image_path=Path(f"/fake/data/caerulein_adm/{i}.tif"),
                source_bucket="caerulein_adm",
                label_name="ADM",
                label_index=0,
                lesion_id=f"caerulein_adm:{i}",
                group_id=f"caerulein_adm:{i}",
                magnification="20x",
                channel_name="single",
            )
        )
    # Group 8-12: Single-image groups (PanIN)
    for i in range(2, 7):
        records.append(
            ImageRecord(
                image_path=Path(f"/fake/data/KC/{i}.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id=f"KC:{i}",
                group_id=f"KC:{i}",
                magnification="20x",
                channel_name="single",
            )
        )
    return records


# ---------------------------------------------------------------------------
# Metrics and prediction fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_evaluation_metrics() -> EvaluationMetrics:
    """Sample evaluation metrics."""
    return EvaluationMetrics(
        accuracy=0.85,
        sensitivity=0.80,
        specificity=0.90,
        roc_auc=0.88,
        true_negative=45,
        false_positive=5,
        false_negative=8,
        true_positive=32,
    )


@pytest.fixture
def sample_prediction_records() -> list[PredictionRecord]:
    """Sample prediction records for aggregation tests."""
    return [
        PredictionRecord(
            image_path="/fake/data/KC/1.tif",
            true_label=1,
            predicted_label=1,
            positive_score=0.75,
            correct=True,
            bag_id="KC:1",
            source_bucket="KC",
            label_name="PanIN",
            sample_type="whole_image",
        ),
        PredictionRecord(
            image_path="/fake/data/KC/2.tif",
            true_label=1,
            predicted_label=0,
            positive_score=0.35,
            correct=False,
            bag_id="KC:1",
            source_bucket="KC",
            label_name="PanIN",
            sample_type="whole_image",
        ),
        PredictionRecord(
            image_path="/fake/data/caerulein_adm/1.tif",
            true_label=0,
            predicted_label=0,
            positive_score=0.25,
            correct=True,
            bag_id="caerulein_adm:1",
            source_bucket="caerulein_adm",
            label_name="ADM",
            sample_type="whole_image",
        ),
    ]


@pytest.fixture
def sample_training_history() -> TrainingHistory:
    """Sample training history entry."""
    return TrainingHistory(
        epoch=1,
        train_loss=0.65,
        train_accuracy=0.72,
        learning_rate=1e-4,
    )


# ---------------------------------------------------------------------------
# Directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory structure for file I/O tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "KC").mkdir()
    (data_dir / "caerulein_adm").mkdir()
    (data_dir / "KPC").mkdir()
    (data_dir / "multichannel_adm").mkdir()
    (data_dir / "multichannel_kc_adm").mkdir()
    (data_dir / "multichannel_unresolved").mkdir()
    return data_dir


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir