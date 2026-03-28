"""Train/test splitting logic with optional group-aware (lesion-level) splits."""

from __future__ import annotations

import random
from collections import defaultdict

from pancreas_vision.types import ImageRecord


def split_grouped_records(
    records: list[ImageRecord],
    test_size: float,
    random_seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    grouped_records: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped_records[record.group_id].append(record)

    if len(grouped_records) < 4:
        raise ValueError("Need at least 4 record groups to produce a grouped split")

    rng = random.Random(random_seed)
    label_to_group_items: dict[int, list[tuple[str, list[ImageRecord]]]] = defaultdict(list)
    for group_id, group_records in grouped_records.items():
        label = group_records[0].label_index
        label_to_group_items[label].append((group_id, group_records))

    selected_test_groups: set[str] = set()
    for label_index, group_items in label_to_group_items.items():
        total_count = sum(len(group_records) for _, group_records in group_items)
        target_count = max(1, int(round(total_count * test_size)))
        target_group_count = max(1, int(round(len(group_items) * test_size)))
        ordered_items = sorted(
            group_items,
            key=lambda item: (rng.random(), len(item[1])),
        )
        selected_for_label: list[str] = []
        selected_count = 0

        while ordered_items and (
            selected_count < target_count or len(selected_for_label) < target_group_count
        ):
            best_idx = min(
                range(len(ordered_items)),
                key=lambda idx: (
                    abs(target_count - (selected_count + len(ordered_items[idx][1]))),
                    abs(target_group_count - (len(selected_for_label) + 1)),
                    len(ordered_items[idx][1]),
                ),
            )
            group_id, group_records = ordered_items.pop(best_idx)
            selected_for_label.append(group_id)
            selected_count += len(group_records)

        if len(selected_for_label) == len(group_items) and len(group_items) > 1:
            smallest_group_id = min(
                selected_for_label,
                key=lambda gid: len(grouped_records[gid]),
            )
            selected_for_label.remove(smallest_group_id)
        selected_test_groups.update(selected_for_label)

    train_records = [
        record for record in records if record.group_id not in selected_test_groups
    ]
    test_records = [
        record for record in records if record.group_id in selected_test_groups
    ]
    if not train_records or not test_records:
        raise ValueError("Grouped split produced an empty train or test partition")
    return train_records, test_records


def split_records(
    records: list[ImageRecord],
    test_size: float = 0.3,
    random_seed: int = 42,
    group_aware: bool = False,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Create a reproducible split, optionally keeping lesion/image groups together."""
    if len(records) < 4:
        raise ValueError("Need at least 4 images to produce a train/test split")

    if group_aware:
        return split_grouped_records(
            records=records,
            test_size=test_size,
            random_seed=random_seed,
        )

    rng = random.Random(random_seed)
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        label_to_indices[record.label_index].append(index)

    test_indices: set[int] = set()
    for indices in label_to_indices.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        target_count = max(1, int(round(len(shuffled) * test_size)))
        test_indices.update(shuffled[:target_count])

    train_records = [
        record for index, record in enumerate(records) if index not in test_indices
    ]
    test_records = [
        record for index, record in enumerate(records) if index in test_indices
    ]
    return train_records, test_records
