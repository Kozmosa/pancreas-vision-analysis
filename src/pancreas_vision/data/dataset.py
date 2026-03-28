"""PyTorch Dataset for pancreatic microscopy images."""

from __future__ import annotations

from pancreas_vision.types import ImageRecord

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - keeps metadata tools usable in light envs
    class Dataset:  # type: ignore[no-redef]
        pass


class MicroscopyDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        from PIL import Image

        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
            if record.crop_box is not None:
                image = image.crop(record.crop_box)
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label_index, record.record_key
