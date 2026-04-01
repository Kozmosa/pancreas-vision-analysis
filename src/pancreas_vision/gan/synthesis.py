"""GAN synthesis utilities for generating synthetic ADM patches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from pancreas_vision.types import SyntheticInstance

if TYPE_CHECKING:
    pass


def generate_synthetic_instances(
    output_dir: Path,
    source_bag_ids: list[str],
    num_instances_per_bag: int = 2,
    label_name: str = "ADM",
    seed_start: int = 0,
    truncation_psi: float = 0.7,
) -> list[SyntheticInstance]:
    """Generate synthetic instances for training augmentation.

    This is a placeholder that generates random noise images.
    When StyleGAN3 is installed, replace with actual GAN generation.

    Args:
        output_dir: Output directory for synthetic images
        source_bag_ids: List of bag IDs to generate instances for
        num_instances_per_bag: Number of synthetic instances per bag
        label_name: Label for synthetic instances
        seed_start: Starting seed for generation
        truncation_psi: Truncation parameter for StyleGAN3

    Returns:
        List of SyntheticInstance records
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    instances: list[SyntheticInstance] = []

    current_seed = seed_start

    for bag_id in source_bag_ids:
        for i in range(num_instances_per_bag):
            synthetic_id = f"SYNTH_{bag_id}_{i:03d}"
            image_path = output_dir / f"{synthetic_id}.png"

            # Generate placeholder image (random noise)
            # When StyleGAN3 is installed, replace with actual generation
            np.random.seed(current_seed)
            noise = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(noise, mode="RGB")
            img.save(image_path)

            instances.append(
                SyntheticInstance(
                    synthetic_id=synthetic_id,
                    source_bag_id=bag_id,
                    image_path=str(image_path),
                    label_name=label_name,
                    seed=current_seed,
                    truncation_psi=truncation_psi,
                    is_filtered=False,
                    similarity_score=None,
                )
            )

            current_seed += 1

    print(f"Generated {len(instances)} synthetic instances")
    return instances


def compute_perceptual_hash(image_path: Path, hash_size: int = 16) -> str:
    """Compute perceptual hash of an image.

    Args:
        image_path: Path to the image
        hash_size: Size of the hash

    Returns:
        Hex string representing the perceptual hash
    """
    with Image.open(image_path) as img:
        # Resize to hash size
        img_resized = img.resize((hash_size, hash_size), Image.LANCZOS)
        # Convert to grayscale
        img_gray = img_resized.convert("L")
        # Get pixel values
        pixels = list(img_gray.getdata())
        # Compute average
        avg = sum(pixels) / len(pixels)
        # Create binary hash
        bits = "".join("1" if p > avg else "0" for p in pixels)
        # Convert to hex
        return hex(int(bits, 2))[2:]


def compute_image_similarity(img1_path: Path, img2_path: Path) -> float:
    """Compute similarity between two images using perceptual hashing.

    Args:
        img1_path: Path to first image
        img2_path: Path to second image

    Returns:
        Similarity score between 0 and 1
    """
    hash1 = compute_perceptual_hash(img1_path)
    hash2 = compute_perceptual_hash(img2_path)

    # Hamming distance normalized to similarity
    if len(hash1) != len(hash2):
        return 0.0

    distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    max_distance = len(hash1)
    similarity = 1.0 - (distance / max_distance)

    return similarity


def filter_synthetic_instances(
    synthetic_instances: list[SyntheticInstance],
    real_instance_paths: list[Path],
    similarity_threshold: float = 0.90,
) -> list[SyntheticInstance]:
    """Filter out synthetic instances too similar to real training data.

    Args:
        synthetic_instances: List of synthetic instances to filter
        real_instance_paths: Paths to real training images
        similarity_threshold: Maximum allowed similarity to real images

    Returns:
        Filtered list of synthetic instances
    """
    filtered_instances: list[SyntheticInstance] = []

    for synth in synthetic_instances:
        synth_path = Path(synth.image_path)
        max_similarity = 0.0

        # Compute similarity to all real instances
        # In practice, use approximate nearest neighbor for efficiency
        for real_path in real_instance_paths[:50]:  # Limit for efficiency
            if not real_path.exists():
                continue
            try:
                similarity = compute_image_similarity(synth_path, real_path)
                max_similarity = max(max_similarity, similarity)
            except Exception:
                continue

        # Create updated instance with similarity score
        is_filtered = max_similarity < similarity_threshold

        filtered_instances.append(
            SyntheticInstance(
                synthetic_id=synth.synthetic_id,
                source_bag_id=synth.source_bag_id,
                image_path=synth.image_path,
                label_name=synth.label_name,
                seed=synth.seed,
                truncation_psi=synth.truncation_psi,
                is_filtered=is_filtered,
                similarity_score=max_similarity,
            )
        )

    # Filter out rejected instances
    accepted = [s for s in filtered_instances if s.is_filtered]
    rejected = [s for s in filtered_instances if not s.is_filtered]

    print(f"Filtered synthetic instances:")
    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected (too similar): {len(rejected)}")

    return filtered_instances


def save_synthetic_instances_manifest(
    synthetic_instances: list[SyntheticInstance],
    output_path: Path,
) -> None:
    """Save synthetic instances to JSON manifest.

    Args:
        synthetic_instances: List of synthetic instances
        output_path: Output path for manifest
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total_instances": len(synthetic_instances),
        "accepted_instances": len([s for s in synthetic_instances if s.is_filtered]),
        "rejected_instances": len([s for s in synthetic_instances if not s.is_filtered]),
        "instances": [s.__dict__ for s in synthetic_instances],
    }

    output_path.write_text(
        json.dumps(data, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def load_synthetic_instances_manifest(
    manifest_path: Path,
) -> list[SyntheticInstance]:
    """Load synthetic instances from JSON manifest.

    Args:
        manifest_path: Path to manifest file

    Returns:
        List of SyntheticInstance records
    """
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        SyntheticInstance(**instance_data)
        for instance_data in data["instances"]
    ]


def main() -> None:
    """CLI entry point for synthetic instance generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic ADM patches for training augmentation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for synthetic images",
    )
    parser.add_argument(
        "--source-bags",
        nargs="+",
        required=True,
        help="Bag IDs to generate synthetic instances for",
    )
    parser.add_argument(
        "--num-per-bag",
        type=int,
        default=2,
        help="Number of synthetic instances per bag",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed for generation",
    )
    parser.add_argument(
        "--truncation-psi",
        type=float,
        default=0.7,
        help="Truncation parameter for StyleGAN3",
    )
    args = parser.parse_args()

    instances = generate_synthetic_instances(
        output_dir=args.output_dir,
        source_bag_ids=args.source_bags,
        num_instances_per_bag=args.num_per_bag,
        seed_start=args.seed_start,
        truncation_psi=args.truncation_psi,
    )

    # Save manifest
    manifest_path = args.output_dir / "synthetic_instances.json"
    save_synthetic_instances_manifest(instances, manifest_path)

    print(f"Synthetic instances saved to: {manifest_path}")


if __name__ == "__main__":
    main()