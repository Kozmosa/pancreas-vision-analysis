"""Tests for GAN augmentation module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pancreas_vision.types import SyntheticInstance


@pytest.fixture
def mock_instance_manifest(tmp_path: Path) -> Path:
    """Create a mock instance manifest."""
    manifest_path = tmp_path / "instance_manifest.csv"
    df = pd.DataFrame({
        "instance_id": ["INST_001", "INST_002", "INST_003"],
        "bag_id": ["BAG_001", "BAG_001", "BAG_002"],
        "image_path": ["img1.tif", "img2.tif", "img3.tif"],
        "label_name": ["ADM", "ADM", "ADM"],
        "source_bucket": ["multichannel_kc_adm", "multichannel_kc_adm", "caerulein_adm"],
        "magnification": ["40x", "20x", "unknown"],
        "channel_name": ["ck19", "merge", "single"],
        "sample_type": ["whole_image"] * 3,
        "is_roi": [0] * 3,
        "split_key": ["BAG_001", "BAG_001", "BAG_002"],
        "lesion_id": ["BAG_001", "BAG_001", "BAG_002"],
        "record_key": ["img1", "img2", "img3"],
        "label_source": ["folder_label"] * 3,
        "crop_box": [""] * 3,
    })
    df.to_csv(manifest_path, index=False)
    return manifest_path


@pytest.fixture
def mock_synthetic_instances(tmp_path: Path) -> list[SyntheticInstance]:
    """Create mock synthetic instances."""
    synth_dir = tmp_path / "synth"
    synth_dir.mkdir()

    # Create dummy images
    from PIL import Image
    import numpy as np

    instances = []
    for i, bag_id in enumerate(["BAG_001", "BAG_001", "BAG_002"]):
        img_path = synth_dir / f"synth_{i}.png"
        img = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
        img.save(img_path)

        instances.append(SyntheticInstance(
            synthetic_id=f"SYNTH_{i:03d}",
            source_bag_id=bag_id,
            image_path=str(img_path),
            label_name="ADM",
            seed=i,
            truncation_psi=0.7,
            is_filtered=True,
            similarity_score=0.5,
        ))

    return instances


class TestCreateAugmentedInstanceManifest:
    """Tests for create_augmented_instance_manifest function."""

    def test_adds_synthetic_instances(
        self,
        mock_instance_manifest: Path,
        mock_synthetic_instances: list[SyntheticInstance],
        tmp_path: Path,
    ):
        """Test that synthetic instances are added."""
        from pancreas_vision.gan.augmentation import create_augmented_instance_manifest

        output_path = tmp_path / "augmented.csv"
        stats = create_augmented_instance_manifest(
            original_instance_manifest_path=mock_instance_manifest,
            synthetic_instances=mock_synthetic_instances,
            output_path=output_path,
            max_synthetic_per_bag=2,
        )

        assert stats["synthetic_instances_added"] == 3
        assert output_path.exists()

    def test_respects_max_per_bag(
        self,
        mock_instance_manifest: Path,
        mock_synthetic_instances: list[SyntheticInstance],
        tmp_path: Path,
    ):
        """Test that max synthetic per bag is respected."""
        from pancreas_vision.gan.augmentation import create_augmented_instance_manifest

        output_path = tmp_path / "augmented.csv"
        stats = create_augmented_instance_manifest(
            original_instance_manifest_path=mock_instance_manifest,
            synthetic_instances=mock_synthetic_instances,
            output_path=output_path,
            max_synthetic_per_bag=1,  # Only 1 per bag
        )

        # BAG_001 and BAG_002 each get 1
        assert stats["synthetic_instances_added"] == 2

    def test_filters_by_target_bags(
        self,
        mock_instance_manifest: Path,
        mock_synthetic_instances: list[SyntheticInstance],
        tmp_path: Path,
    ):
        """Test that only target bags receive synthetic instances."""
        from pancreas_vision.gan.augmentation import create_augmented_instance_manifest

        output_path = tmp_path / "augmented.csv"
        stats = create_augmented_instance_manifest(
            original_instance_manifest_path=mock_instance_manifest,
            synthetic_instances=mock_synthetic_instances,
            output_path=output_path,
            max_synthetic_per_bag=2,
            target_bags=["BAG_001"],  # Only BAG_001
        )

        # Only BAG_001 gets synthetic instances
        assert "BAG_001" in stats["bags_with_synthetic"]
        assert "BAG_002" not in stats["bags_with_synthetic"]

    def test_only_filtered_instances(
        self,
        mock_instance_manifest: Path,
        tmp_path: Path,
    ):
        """Test that only filtered instances are added."""
        from pancreas_vision.gan.augmentation import create_augmented_instance_manifest

        # Create instances with is_filtered=False
        synth_dir = tmp_path / "synth"
        synth_dir.mkdir()

        from PIL import Image
        import numpy as np

        img_path = synth_dir / "synth.png"
        img = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
        img.save(img_path)

        instances = [
            SyntheticInstance(
                synthetic_id="SYNTH_001",
                source_bag_id="BAG_001",
                image_path=str(img_path),
                label_name="ADM",
                seed=0,
                truncation_psi=0.7,
                is_filtered=False,  # Rejected
                similarity_score=0.99,
            )
        ]

        output_path = tmp_path / "augmented.csv"
        stats = create_augmented_instance_manifest(
            original_instance_manifest_path=mock_instance_manifest,
            synthetic_instances=instances,
            output_path=output_path,
            max_synthetic_per_bag=2,
        )

        assert stats["synthetic_instances_added"] == 0

    def test_augmented_manifest_structure(
        self,
        mock_instance_manifest: Path,
        mock_synthetic_instances: list[SyntheticInstance],
        tmp_path: Path,
    ):
        """Test that augmented manifest has correct structure."""
        from pancreas_vision.gan.augmentation import create_augmented_instance_manifest

        output_path = tmp_path / "augmented.csv"
        create_augmented_instance_manifest(
            original_instance_manifest_path=mock_instance_manifest,
            synthetic_instances=mock_synthetic_instances,
            output_path=output_path,
            max_synthetic_per_bag=2,
        )

        df = pd.read_csv(output_path)

        # Check is_synthetic column exists
        assert "is_synthetic" in df.columns

        # Check synthetic rows have correct values
        synthetic_rows = df[df["is_synthetic"] == 1]
        assert len(synthetic_rows) == 3
        assert all(synthetic_rows["source_bucket"] == "synthetic")