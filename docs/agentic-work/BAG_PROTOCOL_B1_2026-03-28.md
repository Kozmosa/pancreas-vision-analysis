# B-1 Bag Protocol

## Purpose

This document defines the first lesion-level bag data contract for the current
`ADM vs PanIN` main line. It upgrades the repository from image-level manifests
to bag-oriented manifests that can be consumed directly by later grouped split,
feature caching, and CLAM training work.

## Scope

Formal bag protocol scope in this round:

- `data/caerulein_adm/`
- `data/KC/`
- `data/KPC/`
- `data/multichannel_adm/`
- `data/multichannel_kc_adm/`

Excluded from the formal manifests in this round:

- `data/multichannel_unresolved/`

`multichannel_unresolved` is still inspected through metadata and appears only
in `review_candidates.csv` so unresolved rows can be triaged without polluting
the formal training contract.

## Command

Bag protocol artifacts are generated with:

```bash
PYTHONPATH=src python3 src/build_bag_protocol.py
```

Default output directory:

- `artifacts/bag_protocol_v1/`

## Grouping Rules

- `bag_id` is equal to normalized `lesion_id`.
- Metadata in `data/2.csv` is preferred for `lesion_id`, `magnification`, and
  `image_type`.
- If metadata is absent, the code falls back to filename-derived inference.
- Multiple magnifications, multiple channels, and ROI crops are all attached to
  the same bag when they share the same `lesion_id`.
- KC ROI polygons are emitted as additional instance rows with `sample_type =
  roi_crop` and `is_roi = 1`.
- ROI crop instances keep the same `bag_id` and `split_key` as the source whole
  image.

## Output Files

### `instance_manifest.csv`

One row per trainable instance. Locked fields:

- `instance_id`
- `bag_id`
- `image_path`
- `label_name`
- `source_bucket`
- `magnification`
- `channel_name`
- `sample_type`
- `is_roi`
- `split_key`
- `lesion_id`
- `record_key`
- `label_source`
- `crop_box`

Contract notes:

- `split_key` is intentionally equal to `bag_id` so `B-2` can reuse it for
  grouped splitting without remapping.
- `is_roi = 1` only for ROI crop instances.
- `record_key` remains the stable unique instance key when multiple ROI crops
  come from the same source TIFF.

### `bag_manifest.csv`

One row per lesion bag. Locked fields:

- `bag_id`
- `lesion_id`
- `label_name`
- `source_buckets`
- `instance_count`
- `whole_image_count`
- `roi_count`
- `magnifications`
- `channel_names`
- `sample_types`
- `has_mixed_source_bucket`
- `has_unknown_magnification`
- `has_unknown_channel`
- `review_flag_count`
- `review_flags`

Contract notes:

- `label_name` is the bag-level label if all instances agree; otherwise it is
  set to `CONFLICT`.
- `source_buckets`, `magnifications`, `channel_names`, and `sample_types` are
  pipe-delimited summaries intended for human inspection and lightweight QC.

### `review_candidates.csv`

One row per bag or unresolved metadata group that should be inspected by A. The
current fields are:

- `bag_id`
- `label_name`
- `source_buckets`
- `instance_count`
- `review_flags`
- `example_paths`
- `notes`

Automated review triggers in this round:

- `single_view_bag`
- `missing_magnification`
- `missing_channel`
- `mixed_source_bucket`
- `label_conflict`
- `roi_only_bag`
- `duplicate_view_slot`
- `filename_bucket_conflict`
- `unresolved_bucket_excluded`
- `needs_manual_check`
- `excluded_from_train`

## QC Summary

The builder also emits:

- `summary.json`
- `summary.md`

These summarize:

- bag count and instance count
- ROI instance count
- label distribution
- source bucket distribution
- per-bag view-count distribution
- quality-flag totals
- whole-image magnification counts
- whole-image channel counts
- `magnification x channel` coverage matrix

## Current Observations From V1 Protocol Output

The first generated protocol under `artifacts/bag_protocol_v1/` shows:

- `35` formal bags
- `80` formal instances
- `27` ROI instances
- `45` review candidates including unresolved metadata groups

Two findings are especially important for downstream work:

- `LESION_KPC_001` correctly groups `1.tif`, `1-amylase.tif`, and
  `1-ck-19.tif` into one KPC bag.
- `LESION_MC_001` merges one ADM multichannel group across
  `multichannel_adm` and `multichannel_kc_adm`, which is now explicitly visible
  as a `mixed_source_bucket` review item instead of being silently flattened.

## Limitations

- Many current single-image buckets still have `unknown` magnification in
  metadata, so the review list is intentionally broad.
- This round does not resolve whether every `lesion_id` is already safe to use
  as the final `bag_id`; it only makes exceptions visible for A to confirm.
- `multichannel_unresolved` is still excluded from the formal training contract
  until later work explicitly upgrades its status.
