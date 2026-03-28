"""Tests for pancreas_vision.data.splitting module."""

from __future__ import annotations

import pytest

from pancreas_vision.data.splitting import split_grouped_records, split_records
from pancreas_vision.types import ImageRecord


class TestSplitRecords:
    """Tests for split_records function."""

    def test_basic_split(self, sample_records_list: list[ImageRecord]):
        """Test basic train/test split."""
        train, test = split_records(
            sample_records_list,
            test_size=0.3,
            random_seed=42,
            group_aware=False,
        )

        # Check that all records are accounted for
        assert len(train) + len(test) == len(sample_records_list)

        # Check approximate ratio (within 1 record)
        expected_test = int(len(sample_records_list) * 0.3)
        assert abs(len(test) - expected_test) <= 1

    def test_stratified_split(self, sample_records_list: list[ImageRecord]):
        """Test that split is stratified (both classes in train and test)."""
        train, test = split_records(
            sample_records_list,
            test_size=0.3,
            random_seed=42,
            group_aware=False,
        )

        train_labels = {r.label_name for r in train}
        test_labels = {r.label_name for r in test}

        assert train_labels == {"ADM", "PanIN"}
        assert test_labels == {"ADM", "PanIN"}

    def test_reproducibility(self, sample_records_list: list[ImageRecord]):
        """Test that same seed produces same split."""
        train1, test1 = split_records(
            sample_records_list,
            test_size=0.3,
            random_seed=42,
            group_aware=False,
        )
        train2, test2 = split_records(
            sample_records_list,
            test_size=0.3,
            random_seed=42,
            group_aware=False,
        )

        train1_paths = {r.image_path for r in train1}
        train2_paths = {r.image_path for r in train2}

        assert train1_paths == train2_paths

    def test_different_seeds_produce_different_splits(
        self, sample_records_list: list[ImageRecord]
    ):
        """Test that different seeds produce different splits."""
        train1, _ = split_records(
            sample_records_list,
            test_size=0.3,
            random_seed=42,
            group_aware=False,
        )
        train2, _ = split_records(
            sample_records_list,
            test_size=0.3,
            random_seed=123,
            group_aware=False,
        )

        train1_paths = {r.image_path for r in train1}
        train2_paths = {r.image_path for r in train2}

        # Different seeds should usually produce different splits
        # (though theoretically could be the same by chance)
        assert train1_paths != train2_paths

    def test_raises_for_too_few_records(self):
        """Test that split raises for fewer than 4 records."""
        records = [
            ImageRecord(
                image_path=__import__("pathlib").Path(f"/fake/{i}.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id=f"KC:{i}",
                group_id=f"KC:{i}",
                magnification="20x",
                channel_name="single",
            )
            for i in range(3)
        ]

        with pytest.raises(ValueError, match="at least 4"):
            split_records(records, test_size=0.3, random_seed=42, group_aware=False)


class TestSplitGroupedRecords:
    """Tests for split_grouped_records function."""

    def test_groups_stay_together(self, grouped_records_list: list[ImageRecord]):
        """Test that records with same group_id stay together."""
        train, test = split_grouped_records(
            grouped_records_list,
            test_size=0.3,
            random_seed=42,
        )

        # Collect all group_ids in train and test
        train_groups = {r.group_id for r in train}
        test_groups = {r.group_id for r in test}

        # No group should appear in both sets
        assert train_groups.isdisjoint(test_groups)

    def test_all_records_accounted_for(self, grouped_records_list: list[ImageRecord]):
        """Test that all records are in either train or test."""
        train, test = split_grouped_records(
            grouped_records_list,
            test_size=0.3,
            random_seed=42,
        )

        all_paths = {r.image_path for r in grouped_records_list}
        split_paths = {r.image_path for r in train} | {r.image_path for r in test}

        assert all_paths == split_paths

    def test_stratified_at_group_level(self, grouped_records_list: list[ImageRecord]):
        """Test that both classes appear in train and test at group level."""
        train, test = split_grouped_records(
            grouped_records_list,
            test_size=0.3,
            random_seed=42,
        )

        train_labels = {r.label_name for r in train}
        test_labels = {r.label_name for r in test}

        assert train_labels == {"ADM", "PanIN"}
        assert test_labels == {"ADM", "PanIN"}

    def test_raises_for_too_few_groups(self):
        """Test that grouped split raises for fewer than 4 groups."""
        records = [
            ImageRecord(
                image_path=__import__("pathlib").Path("/fake/1.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:1",
                group_id="KC:1",
                magnification="20x",
                channel_name="single",
            ),
            ImageRecord(
                image_path=__import__("pathlib").Path("/fake/2.tif"),
                source_bucket="KC",
                label_name="PanIN",
                label_index=1,
                lesion_id="KC:2",
                group_id="KC:2",
                magnification="20x",
                channel_name="single",
            ),
            ImageRecord(
                image_path=__import__("pathlib").Path("/fake/3.tif"),
                source_bucket="caerulein_adm",
                label_name="ADM",
                label_index=0,
                lesion_id="caerulein_adm:1",
                group_id="caerulein_adm:1",
                magnification="20x",
                channel_name="single",
            ),
        ]

        with pytest.raises(ValueError, match="at least 4"):
            split_grouped_records(records, test_size=0.3, random_seed=42)

    def test_reproducibility(self, grouped_records_list: list[ImageRecord]):
        """Test that same seed produces same grouped split."""
        train1, test1 = split_grouped_records(
            grouped_records_list,
            test_size=0.3,
            random_seed=42,
        )
        train2, test2 = split_grouped_records(
            grouped_records_list,
            test_size=0.3,
            random_seed=42,
        )

        train1_groups = {r.group_id for r in train1}
        train2_groups = {r.group_id for r in train2}

        assert train1_groups == train2_groups


class TestSplitRecordsGroupAware:
    """Tests for split_records with group_aware=True."""

    def test_group_aware_uses_grouped_split(
        self, grouped_records_list: list[ImageRecord]
    ):
        """Test that group_aware=True delegates to split_grouped_records."""
        train, test = split_records(
            grouped_records_list,
            test_size=0.3,
            random_seed=42,
            group_aware=True,
        )

        # Verify groups stay together
        train_groups = {r.group_id for r in train}
        test_groups = {r.group_id for r in test}
        assert train_groups.isdisjoint(test_groups)