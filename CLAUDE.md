# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository is still primarily a data-and-docs research workspace for pancreatic microscopy image analysis.
It now also contains a structured Python package (`pancreas_vision`) for baseline and improved ADM-vs-PanIN CNN experiments.

The current project scope should be taken from `docs/ProjectDemand.md`. The proposal PDF `docs/基于基因组学的胰腺癌早期ADM向PanIN生物标记物的发现和验证-申报书.pdf` is a predecessor project, useful for background only, and should not be treated as the authoritative requirements document for this repo.

The repo uses `pixi.toml` for system-level package management. Install with `pip install -e .` after activating a suitable Python environment with torch and torchvision. There is no CI pipeline or automated test harness.

## Current commands

Setup (after activating a Python 3.10+ environment with torch):

```
pip install -e .
```

Useful commands in the current repo:

- `git status --short`
- `git log --oneline -5`
- `ls data docs src artifacts`
- `python3 src/train_baseline.py --help`
- `python3 src/train_improved.py --help`
- `PYTHONPATH=src python3 src/build_bag_protocol.py --help`
- `PYTHONPATH=src python3 src/build_split_protocol.py --help`
- `PYTHONPATH=src python3 src/extract_features.py --help`
- `PYTHONPATH=src python3 src/train_clam.py --help`
- `python -c "from pancreas_vision.models import list_models; print(list_models())"`

## Current sources of truth

- `docs/ProjectDemand.md`
- `docs/Answer_Batch_2.md`
- `docs/Question_Batch_2.txt`
- `docs/Question_Batch_1.txt`
- `docs/refs.md` and `docs/references/*.pdf`
- `data/2.csv` for current metadata-backed training resolution
- `data/KC/*.json` for current ROI polygon annotations
- `artifacts/bag_protocol_v1/` for the current lesion-level bag protocol outputs
- `artifacts/split_protocol_v1/` for the current lesion-level split and evaluation protocol outputs
- `artifacts/feature_cache_v1/` for cached UNI features (1536-dim)

## Current project objective

Based on `docs/ProjectDemand.md`, the current project is aiming to:

- use AI-assisted morphology analysis to distinguish two early pancreatic lesion types: ADM and PanIN
- extract morphological features that may help predict malignant transformation risk
- train a CNN-based classifier as the initial modeling approach
- evaluate with a 7:3 train/test split using metrics such as accuracy, sensitivity, specificity, and ROC, and compare against traditional immunohistochemistry-based assessment

Treat those as current project requirements, but verify which channels, stains, and labels are actually present in the checked-in data before proposing preprocessing or experiments.

## Big-picture repository structure

- `data/` holds TIFF buckets, metadata CSVs, and KC ROI JSON annotations
- `src/` holds the `pancreas_vision` package and top-level training scripts
- `artifacts/` holds experiment outputs produced in this repo
- `docs/` holds the requirement document, dataset clarification notes, and literature
- `roisrp/` is an archived experimental snapshot retained for reference; **do not modify or depend on it**

## Package structure (`src/pancreas_vision/`)

After the 2026-03 refactoring, the package is organized by responsibility:

```
src/pancreas_vision/
├── __init__.py
├── types.py              # Dataclasses: ImageRecord, EvaluationMetrics, etc.
├── data/                 # Data discovery, metadata, datasets, splitting
│   ├── __init__.py
│   ├── records.py        # ImageRecord, discover_records, metadata parsing, ROI crops
│   ├── dataset.py        # MicroscopyDataset (PyTorch Dataset)
│   └── splitting.py      # split_records, group-aware splits
├── models/               # Model implementations
│   ├── __init__.py
│   └── clam.py           # CLAMSingleBranch for MIL classification
├── models.py             # Model registry (@register_model) + ResNet builders
├── features/             # Feature extraction and caching
│   ├── __init__.py
│   ├── extractors.py     # UNIExtractor, DINOv2Extractor
│   ├── cache.py          # Feature caching utilities
│   ├── patches.py        # Local patch sampling
│   └── dataset.py        # BagFeatureDataset for MIL training
├── engine/               # Training engines
│   ├── __init__.py
│   └── mil.py            # MIL training loop for CLAM
├── engine.py             # Training loop, evaluation, bag aggregation, dataloaders
├── io.py                 # JSON/CSV serialization, manifest writing
└── protocols/            # Bag and split protocol construction
    ├── __init__.py
    ├── bag_protocol.py   # Build bag manifests and QC reports
    └── split_protocol.py # Build train/test splits and CV folds
```

Top-level scripts in `src/`:

- `train_baseline.py` — Minimal ResNet18 baseline
- `train_improved.py` — ResNet34 with ROI crops, weighted sampling, etc.
- `build_bag_protocol.py` — Generate bag manifests
- `build_split_protocol.py` — Generate train/test splits and CV folds
- `experiment_runner.py` — Shared experiment workflow (used by both training scripts)
- `extract_features.py` — Extract and cache UNI/DINOv2 features
- `train_clam.py` — Train CLAM model for lesion-level MIL classification

## Current dataset semantics

Based on `docs/Answer_Batch_2.md` plus current filenames and metadata:

- `caerulein_adm` is ADM-focused material
- `KC` images are more aligned with PanIN
- `KPC` images currently represent early PanIN and do not include confirmed PDAC examples
- `merge`, `amylase`, and `ck-19` images are intended to be the same sample or field in different channels or renderings
- `10x`, `20x`, and `40x` indicate progressive magnification of the same lesion region
- filenames like `kc-adm` should be interpreted as ADM regions under a KC model background

These are working research assumptions, not definitive pathology labels.

## Important unknowns and constraints

- There is still no authoritative metadata table linking every image to mouse, slice, field, and pathology-validated label
- ROI annotations exist only for part of the KC set
- `multichannel_unresolved` should be treated as ambiguous unless metadata explicitly resolves a row for training use
- The current code is script-oriented and should stay small and explicit
- `roisrp/` should not be treated as the canonical implementation path anymore; new changes should land in root `src/`

## How to work in this repo

- Read `docs/ProjectDemand.md` first for current scope
- Use `docs/Answer_Batch_2.md` and `data/2.csv` to interpret filenames and training eligibility
- The `roisrp/` directory is archived and independent; do not modify it or add dependencies to it
- Be explicit about what is known from repo contents versus what remains an assumption
- If future metadata, annotations, notebooks, or scripts appear, prefer them over older filename-based assumptions and update this file accordingly
- To add a new model, use `@register_model("name")` in `models.py` or a new file that imports the registry
