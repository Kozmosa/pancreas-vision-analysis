from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


AUTHORITATIVE_LABELS = {
    "caerulein_adm": "ADM",
    "multichannel_adm": "ADM",
    "multichannel_kc_adm": "ADM",
    "KC": "PanIN",
    "KPC": "PanIN",
}


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    source_bucket: str
    label_name: str
    label_index: int
    lesion_id: str
    magnification: str
    channel_name: str


def infer_magnification(image_path: Path) -> str:
    match = re.search(r"(10x|20x|40x)", image_path.stem.lower())
    return match.group(1) if match else "unknown"


def infer_channel_name(image_path: Path) -> str:
    stem = image_path.stem.lower()
    if "ck-19" in stem:
        return "ck19"
    if "amylase" in stem:
        return "amylase"
    if "merge" in stem:
        return "merge"
    return "single"


def infer_lesion_id(source_bucket: str, image_path: Path) -> str:
    stem = image_path.stem.lower()
    numeric_match = re.match(r"(\d+)", stem)
    if numeric_match:
        return f"{source_bucket}:{numeric_match.group(1)}"

    cleaned = stem.replace("ck-19", " ")
    cleaned = cleaned.replace("amylase", " ")
    cleaned = cleaned.replace("merge", " ")
    cleaned = re.sub(r"\b(?:10x|20x|40x|if|kc|adm)\b", " ", cleaned)
    cleaned = re.sub(r"[-_ ]+", " ", cleaned).strip()
    if cleaned:
        return f"{source_bucket}:{cleaned.replace(' ', '_')}"

    if source_bucket.startswith("multichannel_"):
        return f"{source_bucket}:paired_lesion"
    return f"{source_bucket}:{stem}"


def discover_records(
    data_root: Path,
    include_kpc: bool = True,
    include_multichannel: bool = False,
) -> list[ImageRecord]:
    """Discover image-level records from the current authoritative-label folders."""
    enabled = {
        bucket: label
        for bucket, label in AUTHORITATIVE_LABELS.items()
        if include_multichannel or not bucket.startswith("multichannel_")
    }
    if not include_kpc:
        enabled.pop("KPC")

    label_to_index = {"ADM": 0, "PanIN": 1}
    records: list[ImageRecord] = []
    for bucket, label_name in enabled.items():
        bucket_path = data_root / bucket
        if not bucket_path.exists():
            raise FileNotFoundError(f"Missing expected data bucket: {bucket_path}")
        for image_path in sorted(p for p in bucket_path.iterdir() if p.is_file()):
            records.append(
                ImageRecord(
                    image_path=image_path,
                    source_bucket=bucket,
                    label_name=label_name,
                    label_index=label_to_index[label_name],
                    lesion_id=infer_lesion_id(bucket, image_path),
                    magnification=infer_magnification(image_path),
                    channel_name=infer_channel_name(image_path),
                )
            )

    if not records:
        raise ValueError(f"No image records discovered under {data_root}")
    return records


def split_records(
    records: list[ImageRecord],
    test_size: float = 0.3,
    random_seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Create a reproducible image-level split.

    This follows the current requirement document's 7:3 split, while explicitly noting
    that the repo does not yet contain mouse- or slide-level provenance to prevent leakage.
    """
    if len(records) < 4:
        raise ValueError("Need at least 4 images to produce a train/test split")

    indices = list(range(len(records)))
    labels = [record.label_index for record in records]
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )
    train_records = [records[index] for index in train_indices]
    test_records = [records[index] for index in test_indices]
    return train_records, test_records


def write_manifest(records: Iterable[ImageRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "source_bucket",
                "label_name",
                "label_index",
                "lesion_id",
                "magnification",
                "channel_name",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": record.image_path.as_posix(),
                    "source_bucket": record.source_bucket,
                    "label_name": record.label_name,
                    "label_index": record.label_index,
                    "lesion_id": record.lesion_id,
                    "magnification": record.magnification,
                    "channel_name": record.channel_name,
                }
            )


class MicroscopyDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label_index, record.image_path.as_posix()
