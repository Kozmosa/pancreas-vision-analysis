# Model V1

## Positioning

Model v1 is the first runnable ADM vs PanIN microscopy classifier built directly inside this repository. Its purpose is to establish a reproducible local training baseline, expose data protocol issues early, and provide a reference point for later lesion-level multi-view modeling.

This version is intentionally simple: it treats each TIFF as one independent sample, uses a pretrained `ResNet18` backbone, and optimizes a binary classifier under the current folder-based pathology labels.

## Task Definition

- Task type: binary image classification
- Class `0`: `ADM`
- Class `1`: `PanIN`
- Current sample unit: single TIFF image
- Current split unit: image level with stratified 7:3 train/test split

The v1 task definition follows `docs/ProjectDemand.md`, which asks for AI-assisted ADM vs PanIN discrimination and evaluation with accuracy, sensitivity, specificity, and ROC-related metrics.

## Data Used In V1

### Primary baseline data

- `data/caerulein_adm/`: 12 TIFF files, used as `ADM`
- `data/KC/`: 15 TIFF files, used as `PanIN`
- `data/KPC/`: 8 TIFF files, used as `PanIN`

This yields 35 total image-level samples for the main v1 baseline.

### Optional extension data used in benchmark ablations

- `data/multichannel_adm/`: 9 TIFF files, used as `ADM`
- `data/multichannel_kc_adm/`: 9 TIFF files, used as `ADM`

These files were included only in the multichannel ablation run to test whether channel-specific images help when treated as ordinary independent samples.

### Data explicitly excluded from v1

- `data/multichannel_unresolved/`: 35 TIFF files, excluded because they do not yet have a sufficiently explicit label contract for the v1 benchmark line

## Label Protocol Used In V1

The v1 label map is implemented in `src/pancreas_vision/data.py`.

- `caerulein_adm -> ADM`
- `multichannel_adm -> ADM`
- `multichannel_kc_adm -> ADM`
- `KC -> PanIN`
- `KPC -> PanIN`

For v1, these directory semantics are treated as the current experiment-stage pathology labels, following the clarified user instruction from 2026-03-26.

## Data Organization And Manifest Fields

Each discovered image is represented by an `ImageRecord` with the following fields:

- `image_path`
- `source_bucket`
- `label_name`
- `label_index`
- `lesion_id`
- `magnification`
- `channel_name`

These fields are exported to `train_manifest.csv` and `test_manifest.csv` for every run.

### Meaning of the derived fields

- `lesion_id`: a filename-derived grouping hint for future lesion-level modeling
- `magnification`: inferred from the filename when `10x`, `20x`, or `40x` is present
- `channel_name`: inferred as `single`, `merge`, `amylase`, or `ck19`

In v1, these fields are recorded but not yet used as model inputs. They are present to support the transition to the next version, where grouped multi-view samples will replace single-image flattening.

## Model Architecture

V1 uses a standard pretrained image classifier from `torchvision`.

- Backbone: `ResNet18`
- Initialization: `ResNet18_Weights.DEFAULT`
- Classifier head: final `fc` replaced with a 2-class linear layer

Two architectural modes were tested:

- Full fine-tuning: all parameters trainable
- Frozen-backbone ablation: only the classification head trainable

This model is defined in `src/pancreas_vision/training.py`.

## Image Preprocessing And Augmentation

All images are loaded with `PIL`, converted to `RGB`, and resized to a fixed spatial resolution.

### Training transform

- `Resize((224, 224))`
- `RandomHorizontalFlip()`
- `RandomVerticalFlip()`
- `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

### Evaluation transform

- `Resize((224, 224))`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

The normalization follows ImageNet conventions because the backbone is initialized with ImageNet pretrained weights.

## Training Method

### Framework and runtime

- Framework: PyTorch
- Local hardware used: `4x NVIDIA A800 80GB PCIe`
- Python environment verified with CUDA-enabled `torch 2.9.0+cu128`

### Optimization setup

- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning rate: `1e-4`
- Batch size: `8`
- Image size: `224`
- Random seed: `42`
- Data loader workers: `2`

### Baseline epoch settings

- First baseline verification run: `12` epochs
- Expanded benchmark runs: `20` epochs

### Output artifacts saved per run

- `train_manifest.csv`
- `test_manifest.csv`
- `history.json`
- `metrics.json`
- `experiment_summary.json`
- `predictions.json` for runs using the newer evaluation export path

## Evaluation Metrics

V1 reports the following metrics:

- Accuracy
- Sensitivity
- Specificity
- ROC AUC
- Confusion counts: `TN`, `FP`, `FN`, `TP`

These metrics are computed in `src/pancreas_vision/training.py` and align with the requirement language in `docs/ProjectDemand.md`.

## V1 Experiment Matrix

### 1. Initial baseline verification

- Command: `PYTHONPATH=src python src/train_baseline.py --epochs 12 --batch-size 8 --image-size 224 --output-dir artifacts/baseline`
- Dataset: `caerulein_adm + KC + KPC`
- Result:
  - accuracy `1.000`
  - sensitivity `1.000`
  - specificity `1.000`
  - roc_auc `1.000`
  - confusion counts `TN=4, FP=0, FN=0, TP=7`

Interpretation: this run proves that the v1 engineering path works end to end, but it is not sufficient evidence of robust biological generalization.

### 2. Expanded benchmark on local 4x A800

#### `exp_single_image_all`

- Data: `caerulein_adm + KC + KPC`
- Setting: full fine-tuning
- Records: total `35`, train `24`, test `11`
- Metrics:
  - accuracy `1.000`
  - sensitivity `1.000`
  - specificity `1.000`
  - roc_auc `1.000`

#### `exp_single_image_kc_only`

- Data: `caerulein_adm + KC`, excluding `KPC`
- Setting: full fine-tuning
- Records: total `27`, train `18`, test `9`
- Metrics:
  - accuracy `1.000`
  - sensitivity `1.000`
  - specificity `1.000`
  - roc_auc `1.000`

#### `exp_with_multichannel`

- Data: `caerulein_adm + KC + KPC + multichannel_adm + multichannel_kc_adm`
- Setting: full fine-tuning, but flatten all channel-specific images into independent samples
- Records: total `53`, train `37`, test `16`
- Metrics:
  - accuracy `0.875`
  - sensitivity `0.857`
  - specificity `0.889`
  - roc_auc `0.984`

#### `exp_frozen_backbone`

- Data: `caerulein_adm + KC + KPC`
- Setting: frozen `ResNet18` backbone, train classifier head only
- Records: total `35`, train `24`, test `11`
- Metrics:
  - accuracy `0.727`
  - sensitivity `1.000`
  - specificity `0.250`
  - roc_auc `1.000`

## Training Dynamics

The epoch history stored in `artifacts/exp_single_image_all/experiment_summary.json` shows rapid fitting on the small dataset.

- Epoch 1: train loss `0.554`, train accuracy `0.667`
- Epoch 3: train loss `0.248`, train accuracy `0.917`
- Epoch 4: train loss `0.080`, train accuracy `1.000`
- Most later epochs remain near perfect training accuracy

This confirms that v1 is capacity-sufficient for the current sample size and that the main risk has shifted from underfitting to over-optimistic evaluation under a small image-level split.

## Detailed Performance Interpretation

### What v1 does well

- It establishes a complete local training and evaluation loop in this repository.
- It trains reliably on the current data without requiring additional infrastructure.
- It demonstrates that a pretrained CNN can strongly separate the present single-image ADM and PanIN buckets.

### What v1 does not yet solve

- It does not use lesion-level grouping, even though the dataset contains clear multi-view structure.
- It does not prevent mouse-level or lesion-level leakage because provenance metadata is not yet available.
- It does not explicitly model channel semantics, despite the biological importance of `merge`, `amylase`, and `ck-19`.
- It does not yet provide interpretability outputs such as Grad-CAM.

## Error Analysis

The dedicated multichannel error-analysis rerun identified two mistakes out of 16 test samples.

### False positive

- File: `data/multichannel_kc_adm/ck-19-KC-adm-IF-40x.tif`
- Ground truth: `ADM`
- Predicted: `PanIN`
- Positive score: `0.5767`

Interpretation: a CK19-dominant view under a `KC-adm` setting can push the classifier toward PanIN, likely because duct-like marker evidence dominates when that channel is isolated from its companion views.

### False negative

- File: `data/KPC/1-amylase.tif`
- Ground truth: `PanIN`
- Predicted: `ADM`
- Positive score: `0.4162`

Interpretation: an amylase-centric image inside the PanIN bucket can shift the model toward ADM when the lesion is seen only through one channel.

### Main takeaway from the errors

The two mistakes are both channel-specific images rather than ordinary single-view TIFFs. This strongly suggests that channel-specific evidence should be fused at the lesion level instead of treated as fully independent image samples.

## Main Lessons From V1

- Full fine-tuning is clearly better than freezing the pretrained backbone.
- The current single-image baseline is already strong enough to expose data protocol issues.
- Adding multichannel files as ordinary flat images decreases performance, which is itself a useful scientific and engineering finding.
- The next bottleneck is no longer basic model training, but correct sample structuring and better supervision around ambiguous cases.

## Improvement Recommendations

### Model-side improvements

- Replace image-level flattening with lesion-level grouped samples.
- Build a multi-view fusion model that combines `10x/20x/40x` and `merge/amylase/ck-19` evidence.
- Compare attention-based fusion against simpler mean pooling or gated pooling.
- Add Grad-CAM or related visual explanation tools for boundary cases.
- Add stricter evaluation protocols such as lesion-level split, mouse-level split, and cross-validation once metadata supports them.

### Data-side improvements

- Add a lesion linkage table that explicitly maps which files belong to the same lesion.
- Add mouse-level or specimen-level IDs to prevent leakage across related samples.
- Collect more balanced ADM and early PanIN images across staining conditions.
- Add hard negatives such as vessels, inflamed tissue, and benign duct-like structures.
- Add ROI boxes or coarse masks to focus the model on lesion tissue.
- Add marker-grounded lesion confirmation such as Sox9, Claudin18, CK19, and amylase at the lesion level.
- Add acquisition metadata such as batch, device, date, operator, and calibration to assess domain shift.

## Recommended Next Version

V2 should keep the v1 single-image baseline as a control, but move the main modeling effort to lesion-consistent grouped training. The most important upgrade is to treat paired magnification and channel views as one sample and let the network learn how to fuse them, rather than forcing each view to act as a separate independent label-bearing image.
