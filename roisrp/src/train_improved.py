from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

from pancreas_vision.data import discover_records, split_records, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an improved ADM-vs-PanIN classifier using current folders, "
            "data/2.csv metadata, and KC ROI json crops when available."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/2.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/improved_hybrid"),
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--roi-padding-fraction", type=float, default=0.12)
    parser.add_argument(
        "--allow-manual-check-rows",
        action="store_true",
        help="Include metadata-resolved rows that still carry needs_manual_check=1.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the pretrained backbone and only train the classifier head.",
    )
    parser.add_argument(
        "--no-kpc",
        dest="include_kpc",
        action="store_false",
        help="Exclude KPC from the improved run.",
    )
    parser.add_argument(
        "--no-multichannel",
        dest="include_multichannel",
        action="store_false",
        help="Exclude multichannel_adm and multichannel_kc_adm.",
    )
    parser.add_argument(
        "--no-resolved-unresolved",
        dest="include_resolved_unresolved",
        action="store_false",
        help="Do not pull additional labeled rows from multichannel_unresolved via data/2.csv.",
    )
    parser.add_argument(
        "--no-roi-crops",
        dest="include_roi_crops",
        action="store_false",
        help="Disable ROI crop augmentation from KC/*.json.",
    )
    parser.add_argument(
        "--no-group-aware-split",
        dest="group_aware_split",
        action="store_false",
        help="Fallback to the older image-level random split.",
    )
    parser.add_argument(
        "--no-weighted-sampler",
        dest="use_weighted_sampler",
        action="store_false",
        help="Disable class-balanced weighted sampling in the training loader.",
    )
    parser.set_defaults(
        include_kpc=True,
        include_multichannel=True,
        include_resolved_unresolved=True,
        include_roi_crops=True,
        group_aware_split=True,
        use_weighted_sampler=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    from pancreas_vision.training import (
        build_model_with_backbone,
        create_dataloaders,
        evaluate_model,
        now_timestamp,
        save_experiment_summary,
        save_metrics,
        save_predictions,
        set_random_seed,
        train_model,
    )

    start_time = time.time()
    set_random_seed(args.seed)

    records = discover_records(
        data_root=args.data_root,
        include_kpc=args.include_kpc,
        include_multichannel=args.include_multichannel,
        metadata_csv=args.metadata_csv,
        include_resolved_unresolved=args.include_resolved_unresolved,
        include_roi_crops=args.include_roi_crops,
        roi_padding_fraction=args.roi_padding_fraction,
        allow_manual_check_rows=args.allow_manual_check_rows,
    )
    train_records, test_records = split_records(
        records,
        test_size=args.test_size,
        random_seed=args.seed,
        group_aware=args.group_aware_split,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(train_records, output_dir / "train_manifest.csv")
    write_manifest(test_records, output_dir / "test_manifest.csv")

    print("Discovered records:", len(records))
    print("Train label counts:", Counter(record.label_name for record in train_records))
    print("Test label counts:", Counter(record.label_name for record in test_records))
    print("Train sample types:", Counter(record.sample_type for record in train_records))
    print("Test sample types:", Counter(record.sample_type for record in test_records))

    train_loader, test_loader = create_dataloaders(
        train_records=train_records,
        test_records=test_records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.use_weighted_sampler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_with_backbone(
        backbone_name=args.backbone,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    )
    history = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        output_path=output_dir / "history.json",
    )

    metrics, predictions = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
    )
    save_metrics(metrics, output_dir / "metrics.json")
    save_predictions(predictions, output_dir / "predictions.json")

    duration_seconds = time.time() - start_time
    summary = {
        "started_at": now_timestamp(),
        "duration_seconds": duration_seconds,
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count(),
        "arguments": {
            "data_root": args.data_root.as_posix(),
            "metadata_csv": args.metadata_csv.as_posix(),
            "output_dir": args.output_dir.as_posix(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "test_size": args.test_size,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "backbone": args.backbone,
            "dropout": args.dropout,
            "freeze_backbone": args.freeze_backbone,
            "include_kpc": args.include_kpc,
            "include_multichannel": args.include_multichannel,
            "include_resolved_unresolved": args.include_resolved_unresolved,
            "include_roi_crops": args.include_roi_crops,
            "group_aware_split": args.group_aware_split,
            "use_weighted_sampler": args.use_weighted_sampler,
            "roi_padding_fraction": args.roi_padding_fraction,
            "allow_manual_check_rows": args.allow_manual_check_rows,
        },
        "record_counts": {
            "total": len(records),
            "train": len(train_records),
            "test": len(test_records),
        },
        "label_counts": {
            "train": dict(Counter(record.label_name for record in train_records)),
            "test": dict(Counter(record.label_name for record in test_records)),
        },
        "sample_type_counts": {
            "train": dict(Counter(record.sample_type for record in train_records)),
            "test": dict(Counter(record.sample_type for record in test_records)),
        },
        "source_bucket_counts": {
            "train": dict(Counter(record.source_bucket for record in train_records)),
            "test": dict(Counter(record.source_bucket for record in test_records)),
        },
        "metrics": metrics.__dict__,
        "num_prediction_records": len(predictions),
        "history": [item.__dict__ for item in history],
    }
    save_experiment_summary(summary, output_dir / "experiment_summary.json")

    print("Metrics saved to", (output_dir / "metrics.json").as_posix())
    print(metrics)


if __name__ == "__main__":
    main()
