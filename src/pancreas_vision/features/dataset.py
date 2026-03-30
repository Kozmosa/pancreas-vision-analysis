"""Bag-level feature dataset for MIL training.

Loads cached UNI features from feature_index.csv and organizes them
by bag for Multiple Instance Learning (CLAM) training.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    pass


class BagFeatureDataset(Dataset):
    """Dataset that loads cached features for bag-level MIL training.

    Each item is a bag with all its instance features loaded from cache.

    Metadata encoding:
        - Magnification: {"10x": 0, "20x": 1, "40x": 2, "unknown": 3, "none": 4}
        - Channel: {"single": 0, "merge": 1, "amylase": 2, "ck19": 3, "none": 4}
        - Label: {"ADM": 0, "PanIN": 1}

    Args:
        feature_index_path: Path to feature_index.csv
        bag_manifest_path: Path to bag_manifest.csv
        split_csv_path: Path to main_split.csv or fold CSV
        split_role: "train" or "test" (or "train"/"val" for folds)
        cache_dir: Root directory for feature file paths
    """

    MAG_TO_ID = {"10x": 0, "20x": 1, "40x": 2, "unknown": 3, "none": 4}
    CHANNEL_TO_ID = {"single": 0, "merge": 1, "amylase": 2, "ck19": 3, "none": 4}
    LABEL_TO_ID = {"ADM": 0, "PanIN": 1}
    ID_TO_LABEL = {0: "ADM", 1: "PanIN"}

    def __init__(
        self,
        feature_index_path: Path,
        bag_manifest_path: Path,
        split_csv_path: Path,
        split_role: str,
        cache_dir: Path,
    ):
        self.feature_index = pd.read_csv(feature_index_path)
        self.bag_manifest = pd.read_csv(bag_manifest_path)
        self.cache_dir = cache_dir
        self.split_role = split_role

        # Filter bags by split role
        split_df = pd.read_csv(split_csv_path)
        if "split_role" in split_df.columns:
            self.bags = split_df[split_df["split_role"] == split_role]["bag_id"].tolist()
        elif "fold_role" in split_df.columns:
            # For CV folds
            self.bags = split_df[split_df["fold_role"] == split_role]["bag_id"].tolist()
        else:
            raise ValueError(f"Cannot determine split role in {split_csv_path}")

        # Build bag -> label mapping
        self.bag_to_label = dict(
            zip(self.bag_manifest["bag_id"], self.bag_manifest["label_name"])
        )

        # Filter feature_index to only include bags in this split
        self.feature_index = self.feature_index[
            self.feature_index["bag_id"].isin(self.bags)
        ]

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> dict:
        """Get a bag with all its instance features.

        Returns:
            dict with:
                - bag_id: str
                - features: Tensor (N_instances, feature_dim)
                - magnification_ids: Tensor (N_instances,)
                - channel_ids: Tensor (N_instances,)
                - label: Tensor (scalar)
                - instance_count: int
        """
        bag_id = self.bags[idx]

        # Get all feature rows for this bag
        bag_features = self.feature_index[self.feature_index["bag_id"] == bag_id]

        if len(bag_features) == 0:
            raise ValueError(f"No features found for bag {bag_id}")

        features_list = []
        mag_ids_list = []
        channel_ids_list = []

        for _, row in bag_features.iterrows():
            # Load feature from cache
            feature_path = self.cache_dir / row["feature_path"]
            data = torch.load(feature_path, weights_only=True)
            features_list.append(data["features"])

            # Encode metadata
            mag_id = self.MAG_TO_ID.get(row.get("magnification", "unknown"), 3)
            channel_id = self.CHANNEL_TO_ID.get(row.get("channel_name", "single"), 0)
            mag_ids_list.append(mag_id)
            channel_ids_list.append(channel_id)

        # Stack into tensors
        features = torch.stack(features_list)
        magnification_ids = torch.tensor(mag_ids_list, dtype=torch.long)
        channel_ids = torch.tensor(channel_ids_list, dtype=torch.long)

        # Get bag label
        label_name = self.bag_to_label.get(bag_id)
        if label_name is None:
            raise ValueError(f"No label found for bag {bag_id}")
        label = torch.tensor(self.LABEL_TO_ID[label_name], dtype=torch.long)

        return {
            "bag_id": bag_id,
            "features": features,
            "magnification_ids": magnification_ids,
            "channel_ids": channel_ids,
            "label": label,
            "instance_count": len(features_list),
        }

    def get_bag_ids(self) -> list[str]:
        """Return list of all bag IDs in this dataset."""
        return self.bags.copy()

    def get_label_distribution(self) -> dict[str, int]:
        """Return label distribution in this dataset."""
        labels = [self.bag_to_label[bag_id] for bag_id in self.bags]
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return counts