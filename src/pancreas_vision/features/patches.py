"""Local patch sampling for multi-scale feature extraction."""

from __future__ import annotations

import random
from typing import Optional

from PIL import Image


def sample_patches(
    image: Image.Image,
    num_patches: int = 4,
    min_scale: float = 0.25,
    max_scale: float = 0.5,
    seed: Optional[int] = 42,
) -> list[Image.Image]:
    """Sample random patches from an image.

    Patches are cropped regions of the original image, useful for
    capturing local morphology details at higher resolution.

    Args:
        image: PIL Image
        num_patches: Maximum number of patches to sample
        min_scale: Minimum patch scale relative to image size
                   (0.25 = patch is at least 1/4 of image width/height)
        max_scale: Maximum patch scale relative to image size
                   (0.5 = patch is at most 1/2 of image width/height)
        seed: Random seed for reproducibility. None for random behavior.

    Returns:
        List of PIL Image patches (may be fewer than num_patches if image is small)
    """
    if seed is not None:
        random.seed(seed)

    w, h = image.size
    patches = []

    for _ in range(num_patches):
        # Random scale for this patch
        scale = random.uniform(min_scale, max_scale)
        patch_w = int(w * scale)
        patch_h = int(h * scale)

        # Skip if patch would be too small
        if patch_w < 32 or patch_h < 32:
            continue

        # Random position (ensure patch fits within image)
        max_x = max(0, w - patch_w)
        max_y = max(0, h - patch_h)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Crop patch
        patch = image.crop((x, y, x + patch_w, y + patch_h))
        patches.append(patch)

    return patches


def sample_grid_patches(
    image: Image.Image,
    grid_size: int = 2,
    scale: float = 0.5,
) -> list[Image.Image]:
    """Sample patches on a regular grid.

    Instead of random sampling, divide the image into a grid and extract
    patches centered on each grid cell.

    Args:
        image: PIL Image
        grid_size: Number of patches along each dimension (2 = 2x2 grid = 4 patches)
        scale: Patch scale relative to image size

    Returns:
        List of PIL Image patches
    """
    w, h = image.size
    patch_w = int(w * scale)
    patch_h = int(h * scale)

    patches = []
    cell_w = w / grid_size
    cell_h = h / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            # Center of this grid cell
            center_x = int((i + 0.5) * cell_w)
            center_y = int((j + 0.5) * cell_h)

            # Top-left corner of patch
            x = max(0, min(w - patch_w, center_x - patch_w // 2))
            y = max(0, min(h - patch_h, center_y - patch_h // 2))

            patch = image.crop((x, y, x + patch_w, y + patch_h))
            patches.append(patch)

    return patches