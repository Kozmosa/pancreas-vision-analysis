# Idea Discovery Note

Date: 2026-03-26

Scope assumption used in this note:
- The current directory semantics are treated as authoritative pathology labels for experiment design.
- Local compute is available at the level of 4x A800 GPUs, so multi-stage training and moderate ablations are in scope.

## Top-Ranked Idea

### Lesion-consistent multi-view, multi-channel classification for ADM vs PanIN

Core thesis:
Train a pathology model that treats 10x/20x/40x images and matched stain views as coordinated observations of the same lesion, instead of independent samples, and learns ADM vs PanIN from their shared morphology plus channel-specific evidence.

Why this ranks first:
- It matches the strongest confirmed structure in the repository: same-lesion progressive magnification and matched `merge` / `amylase` / `ck-19` views.
- It uses available supervision efficiently without requiring pixel masks before the first experiment.
- It directly addresses the expected failure mode in this task: morphology overlap between ADM and early PanIN at single-view resolution.
- It is practical on 4x A800: pretraining, multi-branch fusion, and cross-validation can all be launched immediately.

Recommended formulation:
- Sample unit: lesion group, not single TIFF.
- Input branches: magnification branch (`10x`, `20x`, `40x`) and stain/channel branch (`merge`, `amylase`, `ck-19`, brightfield-like standalone TIFF when only one image exists).
- Backbone: image encoder per view with shared weights within the same modality family.
- Fusion: attention or gated pooling over available views so missing channels do not break training.
- Objective: supervised ADM vs PanIN classification plus lesion-level contrastive consistency loss across magnifications/channels.
- Evaluation: mouse-level or lesion-level split first, then report AUROC, sensitivity, specificity, F1, and calibration.

## Key Alternatives

### Alternative 1: Self-supervised morphology pretraining, then lightweight classifier

Train a stain-robust encoder first with contrastive or masked-image pretraining on all TIFFs, then fine-tune a smaller ADM/PanIN head.

Why it is strong:
- Best backup if labeled sample count is too small for an end-to-end multi-view model.
- Likely improves representation quality for subtle glandular morphology.

Why it is not ranked first:
- It improves feature quality, but does not fully exploit the explicit same-lesion multi-view structure already present in the data.

### Alternative 2: Two-stage pipeline with lesion detection/cropping before classification

First detect candidate ductal/acinar lesion regions, then classify crops into ADM vs PanIN.

Why it is strong:
- Can reduce background noise from stroma, vessels, and irrelevant tissue.
- May become the best long-term pipeline if ROI annotations are added later.

Why it is not ranked first:
- The repository currently has no checked-in region annotations, so the first version would depend on weak heuristics or manual ROI collection.

### Alternative 3: Magnification curriculum model

Pretrain or fine-tune sequentially from 10x to 20x to 40x so the model learns coarse tissue context before fine cellular detail.

Why it is strong:
- Very easy to implement and ablate.
- Useful if the multi-view fusion model is too complex for the initial benchmark.

Why it is not ranked first:
- It captures part of the cross-scale signal, but still treats views more sequentially than jointly.

## Concrete Implementation Implications

### Dataset construction

- Build a lesion manifest CSV/JSON that groups files into one training sample using directory label + lesion id + magnification + stain/channel.
- Keep authoritative labels from folder semantics for the first pass, but split at lesion or mouse level rather than raw image level.
- Represent missing views explicitly so the model can train on incomplete groups.
- Exclude or separately flag ambiguous storage buckets only if their label semantics are not explicitly declared authoritative.

### Training plan on 4x A800

- Stage 1: encoder warm-up or self-supervised pretraining on all TIFFs.
- Stage 2: supervised multi-view fusion training.
- Stage 3: ablations for single-view, single-magnification, and no-channel-fusion baselines.
- Stage 4: hard-example mining on mistakes such as vessel-vs-duct confusion and subtle ADM/PanIN boundary cases.

### First benchmark matrix

- Baseline A: single-image CNN on folder labels.
- Baseline B: single-magnification model at 20x only.
- Baseline C: early-fusion multi-channel model.
- Main model: lesion-consistent multi-view, multi-channel attention model.
- Ablations: remove contrastive consistency loss; remove 40x; remove stain-specific branches; replace attention with mean pooling.

### Success criteria

- Primary: improved lesion-level AUROC and sensitivity at clinically relevant specificity.
- Secondary: better cross-folder generalization between different pathological contexts.
- Tertiary: interpretable attention over magnification/channel views consistent with pathology expectations.

## Next-Step Data Needed to Improve Model Accuracy

The following data will most directly improve model accuracy beyond the first experiment round.

### Highest priority

- Mouse-level or specimen-level IDs so splits can prevent leakage across related images.
- Lesion-level linkage table confirming which `10x` / `20x` / `40x` and `merge` / `amylase` / `ck-19` files belong to the same lesion.
- More ADM and early PanIN examples with balanced counts across folders and staining conditions.
- Expert confirmation for hard negatives that mimic ducts, especially vessels, inflamed tissue, and benign duct-like structures.

### High value for the second phase

- ROI boxes or coarse masks marking the lesion area within each TIFF.
- Borderline or mixed lesions labeled by confidence, so uncertain cases can be down-weighted or modeled explicitly.
- Time-course or progression labels, especially if KPC samples span early progression states.
- Replicate images from additional mice and batches to improve robustness to staining and acquisition variation.

### High value for interpretability and deployment

- Marker-grounded labels such as Sox9, Claudin18, amylase, and CK19 confirmation at the lesion level.
- Acquisition metadata: scanner/camera settings, batch, date, operator, magnification calibration.
- Pathologist review labels on a held-out test set for clinically meaningful error analysis.

## Immediate recommendation

Start with the top-ranked multi-view, multi-channel lesion model as the main line, and use a single-image CNN plus self-supervised pretraining as low-risk baselines. The next data collection priority should be lesion linkage, mouse IDs, hard negatives, and lesion ROI annotations, because those four additions are the most likely to unlock a real accuracy jump rather than a marginal architecture-only gain.
