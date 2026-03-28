"""Tests for pancreas_vision.protocols.bag_protocol module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pancreas_vision.protocols.bag_protocol import (
    BagRow,
    InstanceRow,
    _bag_review_flags,
    _build_bag_rows,
    _build_instance_rows,
    _crop_box_text,
    _example_paths,
    _sorted_join,
    build_summary,
    render_summary_markdown,
)
from pancreas_vision.types import ImageRecord


class TestSortedJoin:
    """Tests for _sorted_join helper function."""

    def test_joins_sorted_unique(self):
        """Test that values are sorted and unique before joining."""
        result = _sorted_join(["c", "a", "b", "a"])
        assert result == "a|b|c"

    def test_handles_empty_values(self):
        """Test that empty strings are filtered out."""
        result = _sorted_join(["a", "", "b", None])  # type: ignore
        assert result == "a|b"

    def test_empty_input(self):
        """Test empty input returns empty string."""
        result = _sorted_join([])
        assert result == ""


class TestExamplePaths:
    """Tests for _example_paths helper function."""

    def test_returns_limited_paths(self, sample_records_list: list[ImageRecord]):
        """Test that limited number of paths are returned."""
        result = _example_paths(sample_records_list, limit=2)

        paths = result.split(" | ")
        assert len(paths) == 2

    def test_deduplicates_paths(self):
        """Test that duplicate paths are removed."""
        records = [
            ImageRecord(
                image_path=Path("/fake/test.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
            ImageRecord(
                image_path=Path("/fake/test.tif"),  # Same path
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="40x",  # Different mag
                channel_name="single",
            ),
        ]

        result = _example_paths(records, limit=3)

        # Should only have one path (deduplicated)
        assert "/fake/test.tif" in result


class TestCropBoxText:
    """Tests for _crop_box_text helper function."""

    def test_formats_crop_box(self, sample_image_record_with_crop: ImageRecord):
        """Test that crop box is formatted correctly."""
        result = _crop_box_text(sample_image_record_with_crop)
        assert result == "100,200,300,400"

    def test_returns_empty_for_no_crop(self, sample_image_record: ImageRecord):
        """Test that None crop_box returns empty string."""
        result = _crop_box_text(sample_image_record)
        assert result == ""


class TestBagReviewFlags:
    """Tests for _bag_review_flags function."""

    def test_single_view_bag_flag(self, sample_image_record: ImageRecord):
        """Test that single-view bag gets flagged."""
        flags = _bag_review_flags([sample_image_record])
        assert "single_view_bag" in flags

    def test_missing_magnification_flag(self):
        """Test that unknown magnification gets flagged."""
        record = ImageRecord(
            image_path=Path("/fake/test.tif"),
            source_bucket="KC",
            label_name="PanIN",
            label_index=1,
            lesion_id="KC:1",
            group_id="KC:1",
            magnification="unknown",
            channel_name="single",
        )
        flags = _bag_review_flags([record])
        assert "missing_magnification" in flags

    def test_missing_channel_flag(self):
        """Test that unknown channel gets flagged."""
        record = ImageRecord(
            image_path=Path("/fake/test.tif"),
            source_bucket="KC",
            label_name="PanIN",
            label_index=1,
            lesion_id="KC:1",
            group_id="KC:1",
            magnification="20x",
            channel_name="unknown",
        )
        flags = _bag_review_flags([record])
        assert "missing_channel" in flags

    def test_mixed_source_bucket_flag(self):
        """Test that mixed source buckets get flagged."""
        records = [
            ImageRecord(
                image_path=Path("/fake/1.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
            ImageRecord(
                image_path=Path("/fake/2.tif"),
                source_bucket="KPC",  # Different source
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
        ]
        flags = _bag_review_flags(records)
        assert "mixed_source_bucket" in flags

    def test_label_conflict_flag(self):
        """Test that conflicting labels get flagged."""
        records = [
            ImageRecord(
                image_path=Path("/fake/1.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
            ImageRecord(
                image_path=Path("/fake/2.tif"),
                source_bucket="KC",
                label_name="ADM",  # Different label
                label_index=0,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
        ]
        flags = _bag_review_flags(records)
        assert "label_conflict" in flags

    def test_roi_only_bag_flag(self):
        """Test that ROI-only bag gets flagged."""
        records = [
            ImageRecord(
                image_path=Path("/fake/1.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
                sample_type="roi_crop",
            ),
        ]
        flags = _bag_review_flags(records)
        assert "roi_only_bag" in flags

    def test_clean_bag_has_no_flags(self):
        """Test that clean bag has no flags."""
        records = [
            ImageRecord(
                image_path=Path("/fake/1.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
                sample_type="whole_image",
            ),
            ImageRecord(
                image_path=Path("/fake/2.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="40x",
                channel_name="single",
                sample_type="whole_image",
            ),
        ]
        flags = _bag_review_flags(records)
        assert len(flags) == 0


class TestBuildInstanceRows:
    """Tests for _build_instance_rows function."""

    def test_creates_rows_from_records(self, sample_records_list: list[ImageRecord]):
        """Test that instance rows are created from records."""
        rows = _build_instance_rows(sample_records_list)

        assert len(rows) == len(sample_records_list)
        assert all(isinstance(row, InstanceRow) for row in rows)

    def test_assigns_sequential_ids(self, sample_records_list: list[ImageRecord]):
        """Test that instance IDs are sequential."""
        rows = _build_instance_rows(sample_records_list[:5])

        ids = [row.instance_id for row in rows]
        assert ids == ["INSTANCE_0001", "INSTANCE_0002", "INSTANCE_0003", "INSTANCE_0004", "INSTANCE_0005"]

    def test_uses_lesion_id_as_bag_id(self, sample_records_list: list[ImageRecord]):
        """Test that bag_id is set from lesion_id."""
        rows = _build_instance_rows(sample_records_list[:3])

        for row in rows:
            assert row.bag_id == row.lesion_id


class TestBuildBagRows:
    """Tests for _build_bag_rows function."""

    def test_groups_by_lesion_id(self, grouped_records_list: list[ImageRecord]):
        """Test that records are grouped by lesion_id."""
        bag_rows = _build_bag_rows(grouped_records_list)

        # Count unique lesion_ids
        lesion_ids = {r.lesion_id for r in grouped_records_list}
        assert len(bag_rows) == len(lesion_ids)

    def test_counts_instances_correctly(self, grouped_records_list: list[ImageRecord]):
        """Test that instance counts are correct."""
        bag_rows = _build_bag_rows(grouped_records_list)

        for bag_row in bag_rows:
            expected_count = sum(
                1 for r in grouped_records_list if r.lesion_id == bag_row.bag_id
            )
            assert bag_row.instance_count == expected_count

    def test_counts_roi_correctly(self):
        """Test that ROI counts are correct."""
        records = [
            ImageRecord(
                image_path=Path("/fake/1.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
                sample_type="whole_image",
            ),
            ImageRecord(
                image_path=Path("/fake/2.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
                sample_type="roi_crop",
            ),
            ImageRecord(
                image_path=Path("/fake/3.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
                sample_type="roi_crop",
            ),
        ]

        bag_rows = _build_bag_rows(records)
        assert len(bag_rows) == 1
        assert bag_rows[0].whole_image_count == 1
        assert bag_rows[0].roi_count == 2


class TestBuildSummary:
    """Tests for build_summary function."""

    def test_counts_bags_and_instances(self, sample_records_list: list[ImageRecord]):
        """Test that summary counts are correct."""
        bag_rows = _build_bag_rows(sample_records_list)
        summary = build_summary(sample_records_list, bag_rows, [])

        assert summary["counts"]["bag_count"] == len(bag_rows)
        assert summary["counts"]["instance_count"] == len(sample_records_list)

    def test_label_distribution(self, sample_records_list: list[ImageRecord]):
        """Test that label distribution is computed."""
        bag_rows = _build_bag_rows(sample_records_list)
        summary = build_summary(sample_records_list, bag_rows, [])

        label_dist = summary["label_distribution"]
        assert "ADM" in label_dist["bags"]
        assert "PanIN" in label_dist["bags"]

    def test_includes_formal_buckets(self, sample_records_list: list[ImageRecord]):
        """Test that formal source buckets are listed."""
        bag_rows = _build_bag_rows(sample_records_list)
        summary = build_summary(sample_records_list, bag_rows, [])

        assert "formal_scope_source_buckets" in summary
        assert len(summary["formal_scope_source_buckets"]) > 0


class TestRenderSummaryMarkdown:
    """Tests for render_summary_markdown function."""

    def test_produces_valid_markdown(self, sample_records_list: list[ImageRecord]):
        """Test that markdown output has expected sections."""
        bag_rows = _build_bag_rows(sample_records_list)
        summary = build_summary(sample_records_list, bag_rows, [])

        markdown = render_summary_markdown(summary)

        assert "# Bag Protocol" in markdown
        assert "## Counts" in markdown
        assert "## Label Distribution" in markdown
        assert "## Source Bucket Distribution" in markdown

    def test_includes_key_numbers(self, sample_records_list: list[ImageRecord]):
        """Test that key numbers are included in output."""
        bag_rows = _build_bag_rows(sample_records_list)
        summary = build_summary(sample_records_list, bag_rows, [])

        markdown = render_summary_markdown(summary)

        assert str(len(bag_rows)) in markdown  # Bag count
        assert str(len(sample_records_list)) in markdown  # Instance count