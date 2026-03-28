# B-2 Split And Evaluation Protocol

## Purpose

This document freezes the lesion-level split and evaluation contract that will
be reused by later `UNI/DINOv2`, `CLAM`, and comparison experiments.

It depends on the existing bag protocol under `artifacts/bag_protocol_v1/` and
promotes `lesion_id` / `bag_id` to the split unit. No future training run
should split instances from the same bag across train and test.

## Command

Generate the split protocol with:

```bash
PYTHONPATH=src python3 src/build_split_protocol.py
```

Default input:

- `artifacts/bag_protocol_v1/bag_manifest.csv`

Default output:

- `artifacts/split_protocol_v1/`

## Output Files

### `main_split.csv`

One row per formal bag. Locked fields:

- `bag_id`
- `lesion_id`
- `label_name`
- `source_buckets`
- `instance_count`
- `split_name`
- `split_role`
- `split_seed`
- `split_ratio`

Current defaults:

- `split_name = main_train_test`
- `split_seed = 42`
- `split_ratio = 7:3`

The current generated main split contains:

- train: `10` ADM bags, `14` PanIN bags
- test: `4` ADM bags, `7` PanIN bags

### `grouped_5fold.csv`

One row per bag per fold. Locked fields:

- `bag_id`
- `lesion_id`
- `label_name`
- `source_buckets`
- `fold_index`
- `fold_role`
- `split_seed`
- `cv_scheme`

Current defaults:

- `fold_index in [0, 1, 2, 3, 4]`
- `cv_scheme = grouped_stratified_5fold`
- `split_seed = 42`

Contract:

- the same `bag_id` may appear in multiple folds, but only once per fold
- within a fold, a `bag_id` must appear in exactly one role: `train` or `test`
- no fold is allowed to share the same `bag_id` between train and test

### `evaluation_template.json`

This is the repository-level evaluation contract for all later comparison
experiments. Every future experiment should emit:

- `bag_level_metrics`
- `bag_level_predictions`
- `error_by_source_bucket`

The current template freezes:

- required output sections
- required bag-level prediction columns
- required source-bucket error summary columns
- default aggregation assumption: instance scores are reduced to bag scores by
  `mean_positive_score`

## Split Policy

- Split unit: `bag_id`
- Label stratification unit: bag-level `label_name`
- Main split method: stratified shuffle split
- Cross-validation method: stratified 5-fold on bag labels
- Seed: `42`

This round intentionally splits by bag counts rather than instance counts,
because `B-2` is about lesion-level leakage control, not instance balancing.

## Why This Protocol Exists

The earlier v1 experiments split at image level, which can overestimate
generalization when multiple views or ROI crops come from the same lesion. This
protocol closes that gap before feature caching and MIL work continue.

## Current Output Summary

The first frozen split protocol under `artifacts/split_protocol_v1/` contains:

- `35` formal bags in scope
- one fixed main split file
- one grouped `5-fold` file with `175` rows
- one evaluation template file

Per-fold counts in the current output are:

- fold `0`: test `3` ADM, `4` PanIN
- fold `1`: test `3` ADM, `4` PanIN
- fold `2`: test `3` ADM, `4` PanIN
- fold `3`: test `3` ADM, `4` PanIN
- fold `4`: test `2` ADM, `5` PanIN

## Implementation Notes

- Split generation is implemented in `src/pancreas_vision/split_protocol.py`.
- The CLI entrypoint is `src/build_split_protocol.py`.
- The protocol excludes bags with `label_name = CONFLICT`, though the current
  formal bag manifest contains none.

## Limitations

- This protocol does not yet connect directly to training entrypoints; later
  experiment runners still need to consume `main_split.csv` or `grouped_5fold.csv`.
- Current folds are stratified by bag label only; they do not yet stratify by
  source bucket, magnification completeness, or review-flag severity.
