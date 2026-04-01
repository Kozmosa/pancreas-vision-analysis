"""Tests for GAN synthesis module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pancreas_vision.types import SyntheticInstance


class TestGenerateSyntheticInstances:
    """Tests for generate_synthetic_instances function."""

    def test_generates_correct_count(self, tmp_path: Path):
        """Test that correct number of instances are generated."""
        from pancreas_vision.gan.synthesis import generate_synthetic_instances

        output_dir = tmp_path / "synth"
        instances = generate_synthetic_instances(
            output_dir=output_dir,
            source_bag_ids=["BAG_001", "BAG_002"],
            num_instances_per_bag=2,
            seed_start=0,
        )

        assert len(instances) == 4  # 2 bags * 2 per bag

    def test_creates_images(self, tmp_path: Path):
        """Test that images are created."""
        from pancreas_vision.gan.synthesis import generate_synthetic_instances

        output_dir = tmp_path / "synth"
        instances = generate_synthetic_instances(
            output_dir=output_dir,
            source_bag_ids=["BAG_001"],
            num_instances_per_bag=2,
            seed_start=0,
        )

        for inst in instances:
            assert Path(inst.image_path).exists()

    def test_synthetic_instance_fields(self, tmp_path: Path):
        """Test that SyntheticInstance has correct fields."""
        from pancreas_vision.gan.synthesis import generate_synthetic_instances

        output_dir = tmp_path / "synth"
        instances = generate_synthetic_instances(
            output_dir=output_dir,
            source_bag_ids=["BAG_001"],
            num_instances_per_bag=1,
            seed_start=42,
            truncation_psi=0.7,
        )

        inst = instances[0]
        assert inst.source_bag_id == "BAG_001"
        assert inst.label_name == "ADM"
        assert inst.seed == 42
        assert inst.truncation_psi == 0.7
        assert inst.is_filtered is False  # Not filtered yet

    def test_deterministic_with_seed(self, tmp_path: Path):
        """Test that generation is deterministic with same seed."""
        from pancreas_vision.gan.synthesis import generate_synthetic_instances

        output_dir1 = tmp_path / "synth1"
        output_dir2 = tmp_path / "synth2"

        instances1 = generate_synthetic_instances(
            output_dir=output_dir1,
            source_bag_ids=["BAG_001"],
            num_instances_per_bag=1,
            seed_start=42,
        )

        instances2 = generate_synthetic_instances(
            output_dir=output_dir2,
            source_bag_ids=["BAG_001"],
            num_instances_per_bag=1,
            seed_start=42,
        )

        # Same seed should produce same images
        from PIL import Image
        import numpy as np

        img1 = np.array(Image.open(instances1[0].image_path))
        img2 = np.array(Image.open(instances2[0].image_path))

        assert np.array_equal(img1, img2)


class TestFilterSyntheticInstances:
    """Tests for filter_synthetic_instances function."""

    def test_filters_similar_instances(self, tmp_path: Path):
        """Test that similar instances are filtered out."""
        from pancreas_vision.gan.synthesis import (
            generate_synthetic_instances,
            filter_synthetic_instances,
        )
        from PIL import Image
        import numpy as np

        # Create a real image
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        real_path = real_dir / "real.png"
        img = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
        img.save(real_path)

        # Generate synthetic
        synth_dir = tmp_path / "synth"
        instances = generate_synthetic_instances(
            output_dir=synth_dir,
            source_bag_ids=["BAG_001"],
            num_instances_per_bag=2,
            seed_start=0,
        )

        # Filter with high similarity threshold
        filtered = filter_synthetic_instances(
            synthetic_instances=instances,
            real_instance_paths=[real_path],
            similarity_threshold=0.99,  # Very high threshold
        )

        # All should pass filter (different from real)
        assert all(inst.is_filtered for inst in filtered)

    def test_similarity_score_computed(self, tmp_path: Path):
        """Test that similarity scores are computed."""
        from pancreas_vision.gan.synthesis import (
            generate_synthetic_instances,
            filter_synthetic_instances,
        )
        from PIL import Image
        import numpy as np

        # Create a real image
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        real_path = real_dir / "real.png"
        img = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
        img.save(real_path)

        # Generate synthetic
        synth_dir = tmp_path / "synth"
        instances = generate_synthetic_instances(
            output_dir=synth_dir,
            source_bag_ids=["BAG_001"],
            num_instances_per_bag=1,
            seed_start=0,
        )

        filtered = filter_synthetic_instances(
            synthetic_instances=instances,
            real_instance_paths=[real_path],
            similarity_threshold=0.90,
        )

        assert filtered[0].similarity_score is not None


class TestSyntheticInstancesManifest:
    """Tests for saving and loading synthetic instances."""

    def test_save_and_load(self, tmp_path: Path):
        """Test that manifest can be saved and loaded."""
        from pancreas_vision.gan.synthesis import (
            generate_synthetic_instances,
            save_synthetic_instances_manifest,
            load_synthetic_instances_manifest,
        )

        output_dir = tmp_path / "synth"
        instances = generate_synthetic_instances(
            output_dir=output_dir,
            source_bag_ids=["BAG_001"],
            num_instances_per_bag=2,
            seed_start=0,
        )

        manifest_path = tmp_path / "manifest.json"
        save_synthetic_instances_manifest(instances, manifest_path)

        loaded = load_synthetic_instances_manifest(manifest_path)

        assert len(loaded) == len(instances)
        assert loaded[0].synthetic_id == instances[0].synthetic_id