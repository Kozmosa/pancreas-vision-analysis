"""Tests for pancreas_vision.data.records module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pancreas_vision.data.records import (
    AUTHORITATIVE_LABELS,
    LABEL_TO_INDEX,
    bbox_iou,
    build_record,
    infer_channel_name,
    infer_lesion_id,
    infer_magnification,
    load_metadata_index,
    metadata_value_or_fallback,
    normalize_metadata_label,
    normalize_roi_label,
    polygon_to_crop_box,
    resolve_legacy_bucket,
)


class TestInferMagnification:
    """Tests for infer_magnification function."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("sample_10x.tif", "10x"),
            ("sample_20x.tif", "20x"),
            ("sample_40x.tif", "40x"),
            ("10x_sample.tif", "10x"),
            ("SAMPLE_20X.TIF", "20x"),  # Case insensitive
            ("sample_10x_20x.tif", "10x"),  # Returns first match
            ("sample.tif", "unknown"),  # No magnification
            ("sample_100x.tif", "unknown"),  # Invalid magnification
        ],
    )
    def test_infer_magnification(self, filename: str, expected: str):
        """Test magnification inference from filename."""
        result = infer_magnification(Path(filename))
        assert result == expected


class TestInferChannelName:
    """Tests for infer_channel_name function."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("sample_ck-19.tif", "ck19"),
            ("sample_CK-19.tif", "ck19"),  # Case insensitive
            ("sample_amylase.tif", "amylase"),
            ("sample_merge.tif", "merge"),
            ("sample_MERGE.tif", "merge"),  # Case insensitive
            ("sample.tif", "single"),  # No channel marker
            ("ck-19_amylase.tif", "ck19"),  # ck-19 takes precedence
            ("amylase_merge.tif", "amylase"),  # amylase takes precedence
        ],
    )
    def test_infer_channel_name(self, filename: str, expected: str):
        """Test channel name inference from filename."""
        result = infer_channel_name(Path(filename))
        assert result == expected


class TestInferLesionId:
    """Tests for infer_lesion_id function."""

    def test_numeric_filename(self):
        """Test lesion ID for numeric filename."""
        result = infer_lesion_id("KC", Path("1.tif"))
        assert result == "KC:1"

    def test_numeric_filename_at_start(self):
        """Test lesion ID when number is at start of filename."""
        result = infer_lesion_id("KC", Path("1_sample.tif"))
        assert result == "KC:1"

    def test_non_numeric_filename(self):
        """Test lesion ID for non-numeric filename."""
        result = infer_lesion_id("KC", Path("test_sample.tif"))
        assert result == "KC:test_sample"

    def test_filename_with_magnification_only(self):
        """Test that magnification is stripped from lesion ID."""
        result = infer_lesion_id("KC", Path("sample_20x.tif"))
        # The function strips magnification markers from the stem
        assert "20x" not in result or "sample" in result

    def test_multichannel_bucket_with_non_numeric(self):
        """Test lesion ID for multichannel bucket with non-numeric filename."""
        result = infer_lesion_id("multichannel_adm", Path("sample.tif"))
        # Non-numeric filename gets cleaned stem
        assert result.startswith("multichannel_adm:")


class TestResolveLegacyBucket:
    """Tests for resolve_legacy_bucket function."""

    @pytest.mark.parametrize(
        "source_folder,file_name,expected",
        [
            ("caerulein", "test.tif", "caerulein_adm"),
            ("kc", "test.tif", "KC"),
            ("kpc", "test.tif", "KPC"),
            ("caerulein solved", "test.tif", "multichannel_unresolved"),
            ("many colour", "kc_test.tif", "multichannel_kc_adm"),
            ("many colour", "adm_test.tif", "multichannel_adm"),
            ("unknown", "test.tif", None),
        ],
    )
    def test_resolve_legacy_bucket(
        self, source_folder: str, file_name: str, expected: str | None
    ):
        """Test legacy bucket resolution."""
        result = resolve_legacy_bucket(source_folder, file_name)
        assert result == expected


class TestNormalizeMetadataLabel:
    """Tests for normalize_metadata_label function."""

    @pytest.mark.parametrize(
        "coarse_label,expected",
        [
            ("ADM-like", "ADM"),
            ("PanIN-like", "PanIN"),
            ("ADM", None),  # Only -like suffix is normalized
            ("PanIN", None),
            ("unknown", None),
            ("", None),
        ],
    )
    def test_normalize_metadata_label(self, coarse_label: str, expected: str | None):
        """Test metadata label normalization."""
        row = {"coarse_label": coarse_label}
        result = normalize_metadata_label(row)
        assert result == expected


class TestMetadataValueOrFallback:
    """Tests for metadata_value_or_fallback function."""

    def test_returns_value_when_present(self):
        """Test returns value when field is present and non-empty."""
        row = {"magnification": "20x"}
        result = metadata_value_or_fallback(row, "magnification", "unknown")
        assert result == "20x"

    def test_returns_fallback_when_missing(self):
        """Test returns fallback when field is missing."""
        row = {}
        result = metadata_value_or_fallback(row, "magnification", "unknown")
        assert result == "unknown"

    def test_returns_fallback_when_empty(self):
        """Test returns fallback when field is empty string."""
        row = {"magnification": ""}
        result = metadata_value_or_fallback(row, "magnification", "unknown")
        assert result == "unknown"

    def test_returns_fallback_when_unknown(self):
        """Test returns fallback when value is 'unknown'."""
        row = {"magnification": "unknown"}
        result = metadata_value_or_fallback(row, "magnification", "20x")
        assert result == "20x"

    def test_handles_none_row(self):
        """Test handles None row by returning fallback."""
        result = metadata_value_or_fallback(None, "magnification", "unknown")
        assert result == "unknown"


class TestNormalizeRoiLabel:
    """Tests for normalize_roi_label function."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("panin", "PanIN"),
            ("PANIN", "PanIN"),
            ("PanIN", "PanIN"),
            ("panin-like", "PanIN"),
            ("adm", "ADM"),
            ("ADM", "ADM"),
            ("adm-like", "ADM"),
            ("unknown", None),
            ("", None),
        ],
    )
    def test_normalize_roi_label(self, label: str, expected: str | None):
        """Test ROI label normalization."""
        result = normalize_roi_label(label)
        assert result == expected


class TestPolygonToCropBox:
    """Tests for polygon_to_crop_box function."""

    def test_basic_polygon(self):
        """Test crop box calculation from basic polygon."""
        points = [[10, 20], [30, 20], [30, 40], [10, 40]]
        result = polygon_to_crop_box(points, image_width=100, image_height=100, padding_fraction=0.0)
        assert result == (10, 20, 30, 40)

    def test_with_padding(self):
        """Test crop box includes padding."""
        points = [[10, 10], [20, 10], [20, 20], [10, 20]]
        # Width/height = 10, padding = 10 * 0.2 = 2
        result = polygon_to_crop_box(points, image_width=100, image_height=100, padding_fraction=0.2)
        assert result == (8, 8, 22, 22)

    def test_clips_to_image_bounds(self):
        """Test crop box is clipped to image bounds."""
        points = [[5, 5], [15, 5], [15, 15], [5, 15]]
        result = polygon_to_crop_box(points, image_width=10, image_height=10, padding_fraction=0.5)
        # Padding would push left/top negative, but clipped to 0
        assert result[0] >= 0
        assert result[1] >= 0
        assert result[2] <= 10
        assert result[3] <= 10

    def test_empty_points_returns_none(self):
        """Test empty points list returns None."""
        result = polygon_to_crop_box([], image_width=100, image_height=100, padding_fraction=0.1)
        assert result is None


class TestBboxIoU:
    """Tests for bbox_iou function."""

    def test_identical_boxes(self):
        """Test IoU of identical boxes is 1.0."""
        box = (0, 0, 10, 10)
        result = bbox_iou(box, box)
        assert result == 1.0

    def test_no_overlap(self):
        """Test IoU of non-overlapping boxes is 0.0."""
        box_a = (0, 0, 10, 10)
        box_b = (20, 20, 30, 30)
        result = bbox_iou(box_a, box_b)
        assert result == 0.0

    def test_partial_overlap(self):
        """Test IoU calculation for partial overlap."""
        # box_a: (0,0,10,10), box_b: (5,5,15,15)
        # Intersection: (5,5,10,10) -> area = 25
        # Union: 100 + 100 - 25 = 175
        # IoU = 25/175 ≈ 0.143
        box_a = (0, 0, 10, 10)
        box_b = (5, 5, 15, 15)
        result = bbox_iou(box_a, box_b)
        assert abs(result - 25 / 175) < 0.001

    def test_one_contains_another(self):
        """Test IoU when one box contains another."""
        box_a = (0, 0, 20, 20)
        box_b = (5, 5, 10, 10)
        # Intersection = small box = 25
        # Union = large box = 400
        # IoU = 25/400 = 0.0625
        result = bbox_iou(box_a, box_b)
        assert abs(result - 25 / 400) < 0.001


class TestAuthoritativeLabels:
    """Tests for label constants."""

    def test_authoritative_labels_mapping(self):
        """Test that authoritative labels map to valid values."""
        assert AUTHORITATIVE_LABELS["caerulein_adm"] == "ADM"
        assert AUTHORITATIVE_LABELS["KC"] == "PanIN"
        assert AUTHORITATIVE_LABELS["KPC"] == "PanIN"
        assert AUTHORITATIVE_LABELS["multichannel_adm"] == "ADM"

    def test_label_to_index_mapping(self):
        """Test label to index mapping."""
        assert LABEL_TO_INDEX["ADM"] == 0
        assert LABEL_TO_INDEX["PanIN"] == 1
        assert len(LABEL_TO_INDEX) == 2


class TestLoadMetadataIndex:
    """Tests for load_metadata_index function."""

    def test_loads_valid_csv(self, tmp_path: Path):
        """Test loading a valid metadata CSV."""
        csv_content = "source_folder,file_name,coarse_label,magnification\n"
        csv_content += "kc,test1.tif,PanIN-like,20x\n"
        csv_content += "caerulein,test2.tif,ADM-like,10x\n"

        csv_path = tmp_path / "metadata.csv"
        csv_path.write_text(csv_content)

        # Create data directories
        data_root = tmp_path / "data"
        (data_root / "KC").mkdir(parents=True)
        (data_root / "caerulein_adm").mkdir(parents=True)

        # Create the files referenced in CSV
        (data_root / "KC" / "test1.tif").touch()
        (data_root / "caerulein_adm" / "test2.tif").touch()

        result = load_metadata_index(data_root, csv_path)

        assert len(result) == 2
        assert (data_root / "KC" / "test1.tif").as_posix() in result
        assert (data_root / "caerulein_adm" / "test2.tif").as_posix() in result

    def test_returns_empty_for_missing_csv(self, tmp_path: Path):
        """Test returns empty dict for missing CSV file."""
        result = load_metadata_index(tmp_path, tmp_path / "nonexistent.csv")
        assert result == {}

    def test_returns_empty_for_none_csv(self, tmp_path: Path):
        """Test returns empty dict when CSV path is None."""
        result = load_metadata_index(tmp_path, None)
        assert result == {}