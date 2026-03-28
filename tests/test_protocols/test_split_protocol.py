"""Tests for pancreas_vision.protocols.split_protocol module."""

from __future__ import annotations

import pytest

from pancreas_vision.protocols.split_protocol import (
    BagSplitRow,
    FoldAssignmentRow,
    build_evaluation_template,
    build_grouped_folds,
    build_split_summary,
    build_train_test_split,
    render_split_summary_markdown,
)


class TestBuildEvaluationTemplate:
    """Tests for build_evaluation_template function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        result = build_evaluation_template()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        """Test that template has required keys."""
        result = build_evaluation_template()

        assert "evaluation_protocol_version" in result
        assert "required_outputs" in result
        assert "bag_level_metrics" in result
        assert "aggregation_defaults" in result

    def test_required_outputs_list(self):
        """Test that required_outputs is a list."""
        result = build_evaluation_template()

        assert isinstance(result["required_outputs"], list)
        assert "bag_level_metrics" in result["required_outputs"]
        assert "bag_level_predictions" in result["required_outputs"]

    def test_aggregation_defaults(self):
        """Test aggregation defaults are set."""
        result = build_evaluation_template()

        defaults = result["aggregation_defaults"]
        assert defaults["prediction_unit"] == "bag"
        assert defaults["positive_class_name"] == "PanIN"


class TestBuildTrainTestSplit:
    """Tests for build_train_test_split function."""

    @pytest.fixture
    def bag_manifest_rows(self) -> list[dict[str, str]]:
        """Sample bag manifest rows for testing."""
        rows = []
        # 10 ADM bags
        for i in range(10):
            rows.append({
                "bag_id": f"caerulein_adm:{i}",
                "lesion_id": f"caerulein_adm:{i}",
                "label_name": "ADM",
                "source_buckets": "caerulein_adm",
                "instance_count": str(i % 3 + 1),
            })
        # 10 PanIN bags
        for i in range(10):
            rows.append({
                "bag_id": f"KC:{i}",
                "lesion_id": f"KC:{i}",
                "label_name": "PanIN",
                "source_buckets": "KC",
                "instance_count": str(i % 3 + 1),
            })
        return rows

    def test_returns_bag_split_rows(self, bag_manifest_rows: list[dict[str, str]]):
        """Test that function returns list of BagSplitRow."""
        result = build_train_test_split(bag_manifest_rows)

        assert isinstance(result, list)
        assert all(isinstance(row, BagSplitRow) for row in result)

    def test_all_bags_assigned(self, bag_manifest_rows: list[dict[str, str]]):
        """Test that all eligible bags are assigned."""
        result = build_train_test_split(bag_manifest_rows)

        assert len(result) == len(bag_manifest_rows)

    def test_has_train_and_test(self, bag_manifest_rows: list[dict[str, str]]):
        """Test that both train and test roles are present."""
        result = build_train_test_split(bag_manifest_rows)

        roles = {row.split_role for row in result}
        assert "train" in roles
        assert "test" in roles

    def test_stratified_split(self, bag_manifest_rows: list[dict[str, str]]):
        """Test that split is roughly stratified."""
        result = build_train_test_split(bag_manifest_rows, test_size=0.3)

        train_rows = [r for r in result if r.split_role == "train"]
        test_rows = [r for r in result if r.split_role == "test"]

        # Both should have both classes
        train_labels = {r.label_name for r in train_rows}
        test_labels = {r.label_name for r in test_rows}

        assert "ADM" in train_labels
        assert "PanIN" in train_labels
        assert "ADM" in test_labels
        assert "PanIN" in test_labels

    def test_excludes_conflict_labels(self):
        """Test that CONFLICT labels are excluded."""
        manifest = [
            {"bag_id": "good:1", "lesion_id": "good:1", "label_name": "ADM", "source_buckets": "ADM", "instance_count": "1"},
            {"bag_id": "good:2", "lesion_id": "good:2", "label_name": "ADM", "source_buckets": "ADM", "instance_count": "1"},
            {"bag_id": "good:3", "lesion_id": "good:3", "label_name": "PanIN", "source_buckets": "KC", "instance_count": "1"},
            {"bag_id": "good:4", "lesion_id": "good:4", "label_name": "PanIN", "source_buckets": "KC", "instance_count": "1"},
            {"bag_id": "bad:1", "lesion_id": "bad:1", "label_name": "CONFLICT", "source_buckets": "KC", "instance_count": "1"},
        ]

        result = build_train_test_split(manifest)

        bag_ids = {r.bag_id for r in result}
        assert "bad:1" not in bag_ids
        # All non-CONFLICT bags should be included
        assert "good:1" in bag_ids
        assert "good:2" in bag_ids
        assert "good:3" in bag_ids
        assert "good:4" in bag_ids
        # Total should be 4 (excluding CONFLICT)
        assert len(bag_ids) == 4

    def test_reproducibility(self, bag_manifest_rows: list[dict[str, str]]):
        """Test that same seed gives same split."""
        result1 = build_train_test_split(bag_manifest_rows, random_seed=42)
        result2 = build_train_test_split(bag_manifest_rows, random_seed=42)

        roles1 = {r.bag_id: r.split_role for r in result1}
        roles2 = {r.bag_id: r.split_role for r in result2}

        assert roles1 == roles2


class TestBuildGroupedFolds:
    """Tests for build_grouped_folds function."""

    @pytest.fixture
    def bag_manifest_rows(self) -> list[dict[str, str]]:
        """Sample bag manifest rows for fold testing."""
        rows = []
        for i in range(15):
            rows.append({
                "bag_id": f"ADM:{i}",
                "lesion_id": f"ADM:{i}",
                "label_name": "ADM",
                "source_buckets": "caerulein_adm",
                "instance_count": "1",
            })
        for i in range(15):
            rows.append({
                "bag_id": f"PanIN:{i}",
                "lesion_id": f"PanIN:{i}",
                "label_name": "PanIN",
                "source_buckets": "KC",
                "instance_count": "1",
            })
        return rows

    def test_returns_fold_rows(self, bag_manifest_rows: list[dict[str, str]]):
        """Test that function returns FoldAssignmentRow list."""
        result = build_grouped_folds(bag_manifest_rows)

        assert isinstance(result, list)
        assert all(isinstance(row, FoldAssignmentRow) for row in result)

    def test_correct_number_of_folds(self, bag_manifest_rows: list[dict[str, str]]):
        """Test that n_splits folds are created."""
        n_splits = 5
        result = build_grouped_folds(bag_manifest_rows, n_splits=n_splits)

        fold_indices = {row.fold_index for row in result}
        assert fold_indices == set(range(n_splits))

    def test_each_bag_appears_in_all_folds(self, bag_manifest_rows: list[dict[str, str]]):
        """Test that each bag appears exactly once per fold."""
        result = build_grouped_folds(bag_manifest_rows, n_splits=5)

        bag_ids = {r["bag_id"] for r in bag_manifest_rows if r["label_name"] != "CONFLICT"}

        for bag_id in bag_ids:
            fold_appearances = [r for r in result if r.bag_id == bag_id]
            assert len(fold_appearances) == 5  # Once per fold

    def test_each_fold_has_train_and_test(self, bag_manifest_rows: list[dict[str, str]]):
        """Test that each fold has both train and test roles."""
        result = build_grouped_folds(bag_manifest_rows, n_splits=5)

        for fold_idx in range(5):
            fold_rows = [r for r in result if r.fold_index == fold_idx]
            roles = {r.fold_role for r in fold_rows}
            assert "train" in roles
            assert "test" in roles


class TestBuildSplitSummary:
    """Tests for build_split_summary function."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for summary tests."""
        bag_manifest = [
            {"bag_id": "ADM:1", "lesion_id": "ADM:1", "label_name": "ADM", "source_buckets": "ADM", "instance_count": "1"},
            {"bag_id": "ADM:2", "lesion_id": "ADM:2", "label_name": "ADM", "source_buckets": "ADM", "instance_count": "1"},
            {"bag_id": "PanIN:1", "lesion_id": "PanIN:1", "label_name": "PanIN", "source_buckets": "KC", "instance_count": "1"},
            {"bag_id": "PanIN:2", "lesion_id": "PanIN:2", "label_name": "PanIN", "source_buckets": "KC", "instance_count": "1"},
        ]

        split_rows = [
            BagSplitRow(bag_id="ADM:1", lesion_id="ADM:1", label_name="ADM", source_buckets="ADM", instance_count=1, split_name="main", split_role="train", split_seed=42, split_ratio="7:3"),
            BagSplitRow(bag_id="ADM:2", lesion_id="ADM:2", label_name="ADM", source_buckets="ADM", instance_count=1, split_name="main", split_role="test", split_seed=42, split_ratio="7:3"),
            BagSplitRow(bag_id="PanIN:1", lesion_id="PanIN:1", label_name="PanIN", source_buckets="KC", instance_count=1, split_name="main", split_role="train", split_seed=42, split_ratio="7:3"),
            BagSplitRow(bag_id="PanIN:2", lesion_id="PanIN:2", label_name="PanIN", source_buckets="KC", instance_count=1, split_name="main", split_role="test", split_seed=42, split_ratio="7:3"),
        ]

        fold_rows = [
            FoldAssignmentRow(bag_id="ADM:1", lesion_id="ADM:1", label_name="ADM", source_buckets="ADM", fold_index=0, fold_role="train", split_seed=42, cv_scheme="5fold"),
            FoldAssignmentRow(bag_id="ADM:2", lesion_id="ADM:2", label_name="ADM", source_buckets="ADM", fold_index=0, fold_role="test", split_seed=42, cv_scheme="5fold"),
            FoldAssignmentRow(bag_id="PanIN:1", lesion_id="PanIN:1", label_name="PanIN", source_buckets="KC", fold_index=0, fold_role="train", split_seed=42, cv_scheme="5fold"),
            FoldAssignmentRow(bag_id="PanIN:2", lesion_id="PanIN:2", label_name="PanIN", source_buckets="KC", fold_index=0, fold_role="test", split_seed=42, cv_scheme="5fold"),
        ]

        return bag_manifest, split_rows, fold_rows

    def test_returns_dict(self, sample_data):
        """Test that summary is a dictionary."""
        bag_manifest, split_rows, fold_rows = sample_data
        result = build_split_summary(bag_manifest, split_rows, fold_rows, 42, 0.3, 5)

        assert isinstance(result, dict)

    def test_includes_seed_and_params(self, sample_data):
        """Test that seed and parameters are included."""
        bag_manifest, split_rows, fold_rows = sample_data
        result = build_split_summary(bag_manifest, split_rows, fold_rows, 42, 0.3, 5)

        assert result["seed"] == 42
        assert result["test_size"] == 0.3
        assert result["n_splits"] == 5

    def test_counts_by_role(self, sample_data):
        """Test that main_split_counts has train/test breakdown."""
        bag_manifest, split_rows, fold_rows = sample_data
        result = build_split_summary(bag_manifest, split_rows, fold_rows, 42, 0.3, 5)

        assert "main_split_counts" in result
        assert "train" in result["main_split_counts"]
        assert "test" in result["main_split_counts"]


class TestRenderSplitSummaryMarkdown:
    """Tests for render_split_summary_markdown function."""

    def test_produces_valid_markdown(self):
        """Test that markdown has expected sections."""
        summary = {
            "seed": 42,
            "test_size": 0.3,
            "n_splits": 5,
            "formal_bag_count": 20,
            "label_distribution": {"ADM": 10, "PanIN": 10},
            "main_split_counts": {"train": {"ADM": 7, "PanIN": 7}, "test": {"ADM": 3, "PanIN": 3}},
            "fold_counts": {"0": {"train": {"ADM": 8, "PanIN": 8}, "test": {"ADM": 2, "PanIN": 2}}},
        }

        result = render_split_summary_markdown(summary)

        assert "# Split Protocol" in result
        assert "Seed: 42" in result
        assert "## Main Train/Test Split" in result
        assert "## Fold Counts" in result

    def test_includes_label_distribution(self):
        """Test that label distribution is shown."""
        summary = {
            "seed": 42,
            "test_size": 0.3,
            "n_splits": 5,
            "formal_bag_count": 20,
            "label_distribution": {"ADM": 10, "PanIN": 10},
            "main_split_counts": {},
            "fold_counts": {},
        }

        result = render_split_summary_markdown(summary)

        assert "ADM: 10" in result
        assert "PanIN: 10" in result