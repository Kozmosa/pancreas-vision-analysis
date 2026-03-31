"""Tests for hard_case_split module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pancreas_vision.protocols.hard_case_split import (
    build_hard_case_split,
    build_hard_case_split_summary,
)


@pytest.fixture
def mock_bag_manifest(tmp_path: Path) -> Path:
    """Create a mock bag manifest for testing."""
    import csv

    manifest_path = tmp_path / "bag_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bag_id", "lesion_id", "label_name", "source_buckets",
            "instance_count", "whole_image_count", "roi_count"
        ])
        # ADM bags
        writer.writerow(["BAG_ADM_001", "BAG_ADM_001", "ADM", "caerulein_adm", 5, 5, 0])
        writer.writerow(["BAG_ADM_002", "BAG_ADM_002", "ADM", "caerulein_adm", 5, 5, 0])
        writer.writerow(["BAG_ADM_003", "BAG_ADM_003", "ADM", "caerulein_adm", 5, 5, 0])
        writer.writerow(["BAG_ADM_004", "BAG_ADM_004", "ADM", "caerulein_adm", 5, 5, 0])
        writer.writerow(["BAG_MC_001", "BAG_MC_001", "ADM", "multichannel_kc_adm", 12, 12, 0])
        writer.writerow(["BAG_MC_002", "BAG_MC_002", "ADM", "multichannel_kc_adm", 6, 6, 0])
        # PanIN bags
        writer.writerow(["BAG_PANIN_001", "BAG_PANIN_001", "PanIN", "KC", 5, 5, 0])
        writer.writerow(["BAG_PANIN_002", "BAG_PANIN_002", "PanIN", "KC", 5, 5, 0])
        writer.writerow(["BAG_PANIN_003", "BAG_PANIN_003", "PanIN", "KC", 5, 5, 0])
        writer.writerow(["BAG_PANIN_004", "BAG_PANIN_004", "PanIN", "KC", 5, 5, 0])
        writer.writerow(["BAG_PANIN_005", "BAG_PANIN_005", "PanIN", "KC", 5, 5, 0])
    return manifest_path


class TestBuildHardCaseSplit:
    """Tests for build_hard_case_split function."""

    def test_forces_bags_to_test(
        self,
        mock_bag_manifest: Path,
    ):
        """Test that specified bags are forced into test set."""
        from pancreas_vision.protocols.split_protocol import load_csv_rows

        bag_manifest_rows = load_csv_rows(mock_bag_manifest)

        split_rows = build_hard_case_split(
            bag_manifest_rows=bag_manifest_rows,
            force_test_bags=["BAG_MC_001", "BAG_MC_002"],
            test_size=0.3,
            random_seed=42,
        )

        # Check that forced bags are in test set
        test_bag_ids = [row.bag_id for row in split_rows if row.split_role == "test"]
        assert "BAG_MC_001" in test_bag_ids
        assert "BAG_MC_002" in test_bag_ids

    def test_maintains_stratified_balance(
        self,
        mock_bag_manifest: Path,
    ):
        """Test that remaining bags are stratified."""
        from pancreas_vision.protocols.split_protocol import load_csv_rows

        bag_manifest_rows = load_csv_rows(mock_bag_manifest)

        split_rows = build_hard_case_split(
            bag_manifest_rows=bag_manifest_rows,
            force_test_bags=["BAG_MC_001"],
            test_size=0.3,
            random_seed=42,
        )

        # Count labels in test set
        test_rows = [row for row in split_rows if row.split_role == "test"]
        test_labels = [row.label_name for row in test_rows]

        # Should have both ADM and PanIN in test
        assert "ADM" in test_labels
        assert "PanIN" in test_labels

    def test_raises_on_missing_bag(
        self,
        mock_bag_manifest: Path,
    ):
        """Test that missing forced bags raise error."""
        from pancreas_vision.protocols.split_protocol import load_csv_rows

        bag_manifest_rows = load_csv_rows(mock_bag_manifest)

        with pytest.raises(ValueError, match="not found"):
            build_hard_case_split(
                bag_manifest_rows=bag_manifest_rows,
                force_test_bags=["BAG_NONEXISTENT"],
                test_size=0.3,
                random_seed=42,
            )

    def test_split_name_customizable(
        self,
        mock_bag_manifest: Path,
    ):
        """Test that split name can be customized."""
        from pancreas_vision.protocols.split_protocol import load_csv_rows

        bag_manifest_rows = load_csv_rows(mock_bag_manifest)

        split_rows = build_hard_case_split(
            bag_manifest_rows=bag_manifest_rows,
            force_test_bags=["BAG_MC_001"],
            test_size=0.3,
            random_seed=42,
            split_name="custom_split",
        )

        # All rows should have custom split name
        for row in split_rows:
            assert row.split_name == "custom_split"


class TestBuildHardCaseSplitSummary:
    """Tests for build_hard_case_split_summary function."""

    def test_summary_includes_forced_bags(
        self,
        mock_bag_manifest: Path,
    ):
        """Test that summary includes forced bag information."""
        from pancreas_vision.protocols.split_protocol import load_csv_rows

        bag_manifest_rows = load_csv_rows(mock_bag_manifest)

        split_rows = build_hard_case_split(
            bag_manifest_rows=bag_manifest_rows,
            force_test_bags=["BAG_MC_001", "BAG_MC_002"],
            test_size=0.3,
            random_seed=42,
        )

        summary = build_hard_case_split_summary(
            bag_manifest_rows=bag_manifest_rows,
            split_rows=split_rows,
            force_test_bags=["BAG_MC_001", "BAG_MC_002"],
            random_seed=42,
            test_size=0.3,
        )

        assert summary["forced_test_bags"] == ["BAG_MC_001", "BAG_MC_002"]
        assert summary["forced_bag_count"] == 2
        assert "actual_test_count" in summary
        assert "actual_train_count" in summary

    def test_summary_counts_by_role(
        self,
        mock_bag_manifest: Path,
    ):
        """Test that summary counts bags by role."""
        from pancreas_vision.protocols.split_protocol import load_csv_rows

        bag_manifest_rows = load_csv_rows(mock_bag_manifest)

        split_rows = build_hard_case_split(
            bag_manifest_rows=bag_manifest_rows,
            force_test_bags=["BAG_MC_001"],
            test_size=0.3,
            random_seed=42,
        )

        summary = build_hard_case_split_summary(
            bag_manifest_rows=bag_manifest_rows,
            split_rows=split_rows,
            force_test_bags=["BAG_MC_001"],
            random_seed=42,
            test_size=0.3,
        )

        assert summary["total_bag_count"] == len(split_rows)
        assert summary["actual_test_count"] == len([r for r in split_rows if r.split_role == "test"])
        assert summary["actual_train_count"] == len([r for r in split_rows if r.split_role == "train"])