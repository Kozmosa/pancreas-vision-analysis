"""Build the fixed lesion-level train/test split, grouped 5-fold CV, and shared evaluation template."""

from __future__ import annotations

import argparse
from pathlib import Path

from pancreas_vision.protocols.split_protocol import (
    build_evaluation_template,
    build_grouped_folds,
    build_split_summary,
    build_train_test_split,
    load_csv_rows,
    write_split_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the fixed lesion-level train/test split, grouped 5-fold CV, "
            "and shared evaluation template from the current bag protocol."
        )
    )
    parser.add_argument(
        "--bag-manifest",
        type=Path,
        default=Path("artifacts/bag_protocol_v1/bag_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/split_protocol_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bag_manifest_rows = load_csv_rows(args.bag_manifest)
    split_rows = build_train_test_split(
        bag_manifest_rows=bag_manifest_rows,
        test_size=args.test_size,
        random_seed=args.seed,
    )
    fold_rows = build_grouped_folds(
        bag_manifest_rows=bag_manifest_rows,
        n_splits=args.n_splits,
        random_seed=args.seed,
    )
    summary = build_split_summary(
        bag_manifest_rows=bag_manifest_rows,
        split_rows=split_rows,
        fold_rows=fold_rows,
        random_seed=args.seed,
        test_size=args.test_size,
        n_splits=args.n_splits,
    )
    evaluation_template = build_evaluation_template()
    write_split_outputs(
        output_dir=args.output_dir,
        split_rows=split_rows,
        fold_rows=fold_rows,
        summary=summary,
        evaluation_template=evaluation_template,
    )
    print("Split protocol artifacts written to", args.output_dir.as_posix())


if __name__ == "__main__":
    main()