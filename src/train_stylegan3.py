#!/usr/bin/env python3
"""Train StyleGAN3 model for ADM patch generation.

This script trains a StyleGAN3 model for generating synthetic ADM patches
to augment the training data for CLAM model.

Usage:
    # With StyleGAN3 installed
    python src/train_stylegan3.py \
        --data-dir artifacts/gan_train_v1/pancreas_adm \
        --output-dir artifacts/stylegan3_adm_v1 \
        --gpus 4 \
        --kimg 5000 \
        --batch 32

    # Quick test (fewer iterations)
    python src/train_stylegan3.py \
        --data-dir artifacts/gan_train_v1/pancreas_adm \
        --output-dir artifacts/stylegan3_adm_v1 \
        --gpus 4 \
        --kimg 500 \
        --batch 32

Note:
    This script requires StyleGAN3 to be installed:
    pip install stylegan3 ninja

    Or from source:
    git clone https://github.com/NVlabs/stylegan3
    cd stylegan3 && pip install -e .
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def check_stylegan3_installed() -> bool:
    """Check if StyleGAN3 is installed."""
    try:
        import torch  # noqa: F401
        # Check for CUDA
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, StyleGAN3 requires GPU")
            return False
        return True
    except ImportError:
        return False


def train_stylegan3(
    data_dir: Path,
    output_dir: Path,
    gpus: int = 4,
    kimg: int = 5000,
    batch: int = 32,
    resolution: int = 256,
    pretrained: Path | None = None,
    cfg: str = "stylegan3-t",
    gamma: float = 8.0,
    mirror: bool = True,
    dry_run: bool = False,
) -> dict:
    """Train StyleGAN3 model.

    Args:
        data_dir: Path to training data directory
        output_dir: Output directory for model and logs
        gpus: Number of GPUs to use
        kimg: Training duration in thousands of images
        batch: Total batch size across all GPUs
        resolution: Image resolution
        pretrained: Path to pretrained model for transfer learning
        cfg: StyleGAN3 configuration (stylegan3-t or stylegan3-r)
        gamma: R1 regularization weight
        mirror: Enable horizontal augmentation
        dry_run: If True, print command without running

    Returns:
        Dict with training configuration
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python", "-m", "stylegan3.train",
        "--outdir", str(output_dir),
        "--data", str(data_dir),
        "--gpus", str(gpus),
        "--batch", str(batch),
        "--kimg", str(kimg),
        "--resolution", str(resolution),
        "--cfg", cfg,
        "--gamma", str(gamma),
    ]

    if pretrained is not None and pretrained.exists():
        cmd.extend(["--resume", str(pretrained)])

    if mirror:
        cmd.append("--mirror")

    # Save configuration
    config = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "gpus": gpus,
        "kimg": kimg,
        "batch": batch,
        "resolution": resolution,
        "pretrained": str(pretrained) if pretrained else None,
        "cfg": cfg,
        "gamma": gamma,
        "mirror": mirror,
        "command": " ".join(cmd),
    }

    config_path = output_dir / "training_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("StyleGAN3 Training Configuration")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"GPUs: {gpus}")
    print(f"Training iterations: {kimg}k images")
    print(f"Batch size: {batch}")
    print(f"Resolution: {resolution}")
    print(f"Config: {cfg}")
    print(f"Pretrained: {pretrained}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    if dry_run:
        print("Dry run - command not executed")
        return config

    # Check if StyleGAN3 is available
    if not check_stylegan3_installed():
        print("\nERROR: StyleGAN3 is not installed.")
        print("Install with: pip install stylegan3 ninja")
        print("\nAlternatively, use the placeholder implementation:")
        print("  python src/pancreas_vision/gan/synthesis.py --output-dir ...")
        return config

    # Run training
    print("\nStarting training...")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nTraining failed with return code: {result.returncode}")
    else:
        print("\nTraining completed successfully!")

    return config


def generate_samples(
    generator_path: Path,
    output_dir: Path,
    num_samples: int = 100,
    truncation_psi: float = 0.7,
    seeds: str = "0-99",
    dry_run: bool = False,
) -> dict:
    """Generate samples from trained StyleGAN3 model.

    Args:
        generator_path: Path to trained generator network
        output_dir: Output directory for generated images
        num_samples: Number of samples to generate
        truncation_psi: Truncation parameter
        seeds: Seed range (e.g., "0-99")
        dry_run: If True, print command without running

    Returns:
        Dict with generation configuration
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "stylegan3.gen_images",
        "--network", str(generator_path),
        "--outdir", str(output_dir),
        "--trunc", str(truncation_psi),
        "--seeds", seeds,
    ]

    config = {
        "generator_path": str(generator_path),
        "output_dir": str(output_dir),
        "num_samples": num_samples,
        "truncation_psi": truncation_psi,
        "seeds": seeds,
        "command": " ".join(cmd),
    }

    print("=" * 60)
    print("StyleGAN3 Sample Generation")
    print("=" * 60)
    print(f"Generator: {generator_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {num_samples}")
    print(f"Truncation psi: {truncation_psi}")
    print("=" * 60)

    if dry_run:
        print("Dry run - command not executed")
        return config

    if not check_stylegan3_installed():
        print("\nERROR: StyleGAN3 is not installed.")
        return config

    subprocess.run(cmd, check=False)
    return config


def main() -> None:
    """CLI entry point for StyleGAN3 training."""
    parser = argparse.ArgumentParser(
        description="Train StyleGAN3 for ADM patch generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to training data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for model and logs",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="Number of GPUs to use (default: 4)",
    )
    parser.add_argument(
        "--kimg",
        type=int,
        default=5000,
        help="Training duration in thousands of images (default: 5000)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Total batch size (default: 32)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Image resolution (default: 256)",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Path to pretrained model for transfer learning",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="stylegan3-t",
        choices=["stylegan3-t", "stylegan3-r"],
        help="StyleGAN3 configuration",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=8.0,
        help="R1 regularization weight",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal augmentation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )
    args = parser.parse_args()

    train_stylegan3(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        gpus=args.gpus,
        kimg=args.kimg,
        batch=args.batch,
        resolution=args.resolution,
        pretrained=args.pretrained,
        cfg=args.cfg,
        gamma=args.gamma,
        mirror=not args.no_mirror,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()