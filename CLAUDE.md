# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository is still primarily a data-and-docs research workspace for pancreatic microscopy image analysis. It does not yet contain an implemented software pipeline.

The current project scope should be taken from `docs/ProjectDemand.md`. The proposal PDF `docs/基于基因组学的胰腺癌早期ADM向PanIN生物标记物的发现和验证-申报书.pdf` is a predecessor project, useful for background only, and should not be treated as the authoritative requirements document for this repo.

As checked in now, there are still no package manifests, Python environment files, notebooks, source directories, training scripts, or test suites. Do not invent a build system, dependency manager, or evaluation workflow that is not present.

## Current commands

There are no project-specific build, lint, or test commands yet.

Useful commands in the current repo:

- `git status --short` — inspect local changes
- `git log --oneline -5` — inspect recent history
- `ls data docs` — inspect the main content areas
- `ls "data/图片"` — inspect the current dataset buckets
- `ls docs` — inspect the current research documents

If code is added later, replace this section with the real environment setup, build, lint, and single-test commands.

## Current sources of truth

- `docs/ProjectDemand.md` — current project objective and research content
- `docs/Question_Batch_2.txt` and `docs/Answer_Batch_2.md` — current clarifications about dataset semantics and image naming
- `docs/Question_Batch_1.txt` — earlier unresolved questions about folder meaning, provenance, labels, and compute constraints
- `docs/refs.md` and `docs/references/*.pdf` — literature background
- `docs/基于基因组学的胰腺癌早期ADM向PanIN生物标记物的发现和验证-申报书.pdf` — predecessor-project background, not current requirements

## Current project objective

Based on `docs/ProjectDemand.md`, the current project is aiming to:

- use AI-assisted morphology analysis to distinguish two early pancreatic lesion types: ADM and PanIN
- extract morphological features that may help predict malignant transformation risk
- train a CNN-based classifier as the initial modeling approach
- evaluate with a 7:3 train/test split using metrics such as accuracy, sensitivity, specificity, and ROC, and compare against traditional immunohistochemistry-based assessment

`docs/ProjectDemand.md` also describes intended biological context and markers, including ADM material induced by `雨蛙素`, PanIN material from `p48Cre/+;LSL-KrasG12D/+` mice, and staining/marker expectations such as H&E, amylase, Sox9, and Claudin18.

Treat those as current project requirements, but verify which channels, stains, and labels are actually present in the checked-in data before proposing preprocessing or experiments.

## Big-picture repository structure

The repo's important structure is currently informational rather than software-based:

- `data/图片/` holds raw microscopy TIFF files grouped by broad folders such as `kc`, `KPC`, `雨蛙素`, and `新建文件夹`
- `docs/` holds the requirement document, dataset clarification notes, and literature
- `.claude/settings.json` contains local Claude Code settings for this repo

Some image groups include `merge`, `amylase`, and `ck-19` variants plus `10x`, `20x`, and `40x` magnification markers. That filename structure currently matters more than any code architecture.

## Current dataset semantics

Based on `docs/Answer_Batch_2.md`, use these working assumptions unless newer annotations override them:

- `雨蛙素` is ADM-focused material
- `kc` images are more aligned with PanIN
- `KPC` images currently represent early PanIN and do not include confirmed PDAC examples
- `merge`, `amylase`, and `ck-19` images are intended to be the same sample / field in different channels or renderings
- `10x`, `20x`, and `40x` indicate progressive magnification of the same lesion region
- filenames like `kc-adm` should be interpreted as ADM regions under a KC model background

These are working research assumptions, not definitive pathology labels.

## Important unknowns and constraints

- There is no authoritative metadata table linking an image to mouse, slice, field, label, or split
- There is no confirmed train / validation / test split implementation in the repo yet, even though `ProjectDemand.md` proposes a 7:3 train/test split
- There are no region-level or pixel-level annotations in the repo today
- `新建文件夹` should be treated as an ambiguous storage bucket, not a trustworthy biological class label, unless external metadata clarifies it
- `ProjectDemand.md` describes the intended research workflow, but the current checked-in files do not yet prove that all planned stains, markers, or labels are available

## How to work in this repo

- Read `docs/ProjectDemand.md` first for current project scope
- Use `docs/Answer_Batch_2.md` to interpret the existing image folders and filenames
- Keep the distinction clear between current requirements and predecessor-project background material
- Be explicit about what is known from repo contents versus what is only stated as intended methodology
- If future metadata, annotations, notebooks, or scripts appear, prefer them over current filename-based assumptions and update this file accordingly

## Verify first before suggesting implementation details

Before giving workflow advice, first check whether the repo has gained any of the following:

- a Python environment definition (`pyproject.toml`, `requirements.txt`, `environment.yml`, etc.)
- notebooks or scripts for preprocessing, training, or evaluation
- metadata tables mapping images to mouse, slice, field, and label
- pathology annotations distinguishing ADM, PanIN, or PDAC
- actual H&E, Sox9, Claudin18, or other marker/channel assets referenced by `ProjectDemand.md`
- experiment-tracking or split conventions
