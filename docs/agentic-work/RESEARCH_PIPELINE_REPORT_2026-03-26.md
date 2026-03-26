# Research Pipeline Report

**Direction**: 本地就有4xa800可以启动实验。认为现有的目录语义是病理标注的结果。并给出下一步提高模型精度需要的数据（写入文档）
**Chosen Idea**: 以单图像 CNN 为基线、以病灶级多视角多通道融合为下一阶段主线
**Date**: 2026-03-26
**Pipeline**: idea-discovery -> implement -> local multi-GPU experiment -> documentation update

## Journey Summary

- Ideas generated: 1 main direction + 3 alternatives; auto-selected the top-ranked path under `AUTO_PROCEED=true`
- Implementation: extended the minimal `src/` training scaffold into a reproducible experiment runner with richer manifests, per-epoch history, and experiment summaries
- Experiments: 4 local GPU experiments on `4x A800`; total wall-clock about 88 seconds for the slowest run and about `0.08 GPU-hours` in aggregate
- Review status: no external auto-review loop run yet; this round focused on implementation and first benchmark expansion

## Experiment Matrix

| Experiment | Key setting | Accuracy | Sensitivity | Specificity | ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| `exp_single_image_all` | `caerulein_adm + KC + KPC` | 1.000 | 1.000 | 1.000 | 1.000 |
| `exp_single_image_kc_only` | exclude `KPC` | 1.000 | 1.000 | 1.000 | 1.000 |
| `exp_with_multichannel` | include `multichannel_*` as ordinary images | 0.875 | 0.857 | 0.889 | 0.984 |
| `exp_frozen_backbone` | freeze pretrained backbone | 0.727 | 1.000 | 0.250 | 1.000 |

## Main Findings

- Current folder-level labels are sufficient to launch reproducible local experiments immediately.
- End-to-end fine-tuning is materially better than a frozen-feature setup on the present sample size.
- Directly flattening multichannel, multi-magnification images into independent training samples hurts performance, which supports moving to lesion-level grouped modeling.
- The next likely accuracy gain will come more from better sample grouping and targeted data addition than from swapping in a slightly larger single-image backbone.

## Documentation Outputs Updated

- `docs/agentic-work/IDEA_REPORT.md`
- `docs/agentic-work/PROJECT_BASIS.md`
- `docs/Idea_Discovery_2026-03-26.md`

## Key Files Changed

- `src/pancreas_vision/data.py`
- `src/pancreas_vision/training.py`
- `src/train_baseline.py`
- `artifacts/exp_single_image_all/`
- `artifacts/exp_single_image_kc_only/`
- `artifacts/exp_with_multichannel/`
- `artifacts/exp_frozen_backbone/`

## Final Status

- [x] Local multi-GPU experiments started and completed
- [x] Accuracy-improvement data needs written into documentation
- [ ] Lesion-level grouped dataset and fusion model implemented
- [ ] External auto-review loop completed

## Remaining TODOs

- Build a lesion-group manifest so `10x/20x/40x` and `merge/amylase/ck-19` are treated as one lesion sample
- Add lesion-level or mouse-level split logic
- Run the grouped multi-view model against the current single-image baselines
- Add error analysis and visual inspection of false positives/false negatives

## Error Analysis Update

- Added per-sample prediction export to the training pipeline; each run can now write `predictions.json`
- Re-ran the multichannel experiment as `artifacts/exp_with_multichannel_error_analysis/`
- The test set had 2 mistakes out of 16 samples:
  - false positive: `data/multichannel_kc_adm/ck-19-KC-adm-IF-40x.tif`
  - false negative: `data/KPC/1-amylase.tif`
- Both errors come from channel-specific views rather than ordinary single images, which strengthens the case for lesion-level grouped modeling instead of flattening channels into independent samples
