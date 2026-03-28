"""Build lesion-level bag manifests, QC summary, and manual review candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from pancreas_vision.protocols.bag_protocol import build_protocol_artifacts, write_protocol_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build lesion-level bag manifests, QC summary, and manual review candidates "
            "for the ADM-vs-PanIN repository."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/2.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/bag_protocol_v1"),
    )
    parser.add_argument(
        "--roi-padding-fraction",
        type=float,
        default=0.12,
        help="Padding fraction applied when converting KC ROI polygons into crop boxes.",
    )
    parser.add_argument(
        "--no-roi-crops",
        dest="include_roi_crops",
        action="store_false",
        help="Disable ROI crop instances from KC JSON annotations.",
    )
    parser.set_defaults(include_roi_crops=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_protocol_artifacts(
        data_root=args.data_root,
        metadata_csv=args.metadata_csv,
        include_roi_crops=args.include_roi_crops,
        roi_padding_fraction=args.roi_padding_fraction,
    )
    write_protocol_outputs(args.output_dir, artifacts)
    print("Bag protocol artifacts written to", args.output_dir.as_posix())


if __name__ == "__main__":
    main()