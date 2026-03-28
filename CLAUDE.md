# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository is still primarily a data-and-docs research workspace for pancreatic microscopy image analysis.
It now also contains lightweight Python scripts for baseline and improved ADM-vs-PanIN CNN experiments.

The current project scope should be taken from `docs/ProjectDemand.md`. The proposal PDF `docs/基于基因组学的胰腺癌早期ADM向PanIN生物标记物的发现和验证-申报书.pdf` is a predecessor project, useful for background only, and should not be treated as the authoritative requirements document for this repo.

The repo still does not have a package manifest, environment definition, notebook suite, CI pipeline, or automated test harness. Do not invent a build system, dependency manager, or evaluation workflow that is not present.

## Current commands

There are still no project-specific build, lint, or test commands.

Useful commands in the current repo:

- `git status --short`
- `git log --oneline -5`
- `ls data docs src artifacts`
- `python3 src/train_baseline.py --help`
- `python3 src/train_improved.py --help`
- `PYTHONPATH=src python3 src/build_bag_protocol.py --help`
- `PYTHONPATH=src python3 src/build_split_protocol.py --help`

If setup or test tooling is added later, replace this section with the real environment setup, build, lint, and single-test commands.

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

## Current project objective

Based on `docs/ProjectDemand.md`, the current project is aiming to:

- use AI-assisted morphology analysis to distinguish two early pancreatic lesion types: ADM and PanIN
- extract morphological features that may help predict malignant transformation risk
- train a CNN-based classifier as the initial modeling approach
- evaluate with a 7:3 train/test split using metrics such as accuracy, sensitivity, specificity, and ROC, and compare against traditional immunohistochemistry-based assessment

Treat those as current project requirements, but verify which channels, stains, and labels are actually present in the checked-in data before proposing preprocessing or experiments.

## Big-picture repository structure

- `data/` holds TIFF buckets, metadata CSVs, and KC ROI JSON annotations
- `src/` holds lightweight training and data-loading utilities
- `artifacts/` holds experiment outputs produced in this repo
- `docs/` holds the requirement document, dataset clarification notes, and literature
- `roisrp/` is an archived experimental snapshot retained for reference during the merge into mainline code

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
- Keep the distinction clear between current canonical code under `src/` and archived work under `roisrp/`
- Be explicit about what is known from repo contents versus what remains an assumption
- If future metadata, annotations, notebooks, or scripts appear, prefer them over older filename-based assumptions and update this file accordingly
