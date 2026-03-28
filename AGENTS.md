# AGENTS.md

## Purpose

This file guides agentic coding assistants working in this repository.
It reflects the repository as it exists today, not an imagined future pipeline.

## Repository Reality Check

- This repository is still a research workspace centered on pancreatic microscopy data, notes, and lightweight training scripts.
- The checked-in code supports baseline and improved CNN training experiments, but the repo still does not have a package manifest, notebook suite, CI, or automated test harness.
- Do not invent missing infrastructure such as `pyproject.toml`, `package.json`, Docker flows, CI jobs, or deployment pipelines.
- Base project scope on `docs/ProjectDemand.md`.
- Treat `docs/基于基因组学的胰腺癌早期ADM向PanIN生物标记物的发现和验证-申报书.pdf` as background only, not the primary requirements source.

## Current Source of Truth

- `docs/ProjectDemand.md`: active project objective and planned research workflow.
- `docs/Answer_Batch_2.md`: current working assumptions for folder and filename semantics.
- `docs/Question_Batch_2.txt`: companion clarification prompts.
- `docs/Question_Batch_1.txt`: older unresolved questions about metadata and provenance.
- `docs/refs.md` and `docs/references/*.pdf`: literature context.
- `CLAUDE.md`: repository-specific operating guidance already present in the repo.

## Repo Layout

- `data/`: canonical training inputs, including TIFF buckets, metadata CSVs, and KC ROI JSON annotations.
- `src/`: lightweight training and data-loading utilities for baseline and improved experiments.
- `artifacts/`: experiment outputs already produced in this repo.
- `docs/`: requirement notes, clarifications, and reference material.
- `roisrp/`: archived experimental snapshot kept for reference while its useful changes are merged into the main project.
- `.claude/settings.json`: local tool settings; not a project build config.

## Cursor and Copilot Rules

- No `.cursorrules` file was found.
- No `.cursor/rules/` directory was found.
- No `.github/copilot-instructions.md` file was found.
- If any of these files appear later, update this document and follow them as higher-priority repository guidance.

## Working Domain Assumptions

- `caerulein_adm` data is currently treated as ADM-focused material.
- `KC` data is more aligned with PanIN.
- `KPC` data currently represents early PanIN, not confirmed PDAC.
- `merge`, `amylase`, and `ck-19` filenames likely refer to the same field/sample in different channels or renderings.
- `10x`, `20x`, and `40x` denote progressive magnification of the same lesion region.
- Names like `kc-adm` should be read as ADM morphology under a KC-model background.
- These are working assumptions from notes and metadata, not authoritative pathology labels.

## Known Unknowns

- There is still no authoritative metadata table linking every image to mouse, slice, field, or pathology-validated label.
- Region-level annotations currently exist only for a subset of KC images via ROI JSON files.
- No automated test suite exists yet.
- `multichannel_unresolved` remains partially ambiguous and should only be pulled into training through explicit metadata-gated logic.
- Planned markers in `docs/ProjectDemand.md` are not yet proven to all exist in the checked-in data.

## Build, Lint, and Test Commands

There is still no project-specific build, lint, or automated test command.
Use that fact explicitly rather than guessing.

### Supported Commands Today

- `git status --short`: inspect working tree changes.
- `git log --oneline -5`: inspect recent history.
- `python3 src/train_baseline.py --help`: inspect baseline training CLI.
- `python3 src/train_improved.py --help`: inspect improved training CLI.
- `ls data docs src artifacts`: inspect main project content.

### Build

- No build command exists yet.
- Do not claim `make`, `python -m build`, `npm run build`, or similar commands work unless the repo gains the required files.

### Lint

- No lint command exists yet.
- Do not invent `ruff`, `flake8`, `eslint`, `prettier`, or other tooling without a checked-in config and dependency manifest.

### Test

- No automated test suite exists yet.
- Use targeted smoke runs of the training entrypoints when verification is needed.

### Single Test

- No single-test command exists because there is no test framework or test target in the repository.

## How Agents Should Explore the Repo

- Read `docs/ProjectDemand.md` before proposing workflow or implementation ideas.
- Read `docs/Answer_Batch_2.md` before interpreting dataset folder names or image filenames.
- Verify repository contents before making assumptions about code, tooling, or metadata.
- Prefer statements like "not present in repo" over speculative claims.
- Distinguish clearly between current assets, archived experiments, and intended future methodology.

## Code Style Guidance for This Repo Today

Because the repo is still light on infrastructure, keep additions minimal and explicit.

- Prefer small, script-friendly modules over framework scaffolding.
- Keep filesystem discovery, metadata parsing, preprocessing, and modeling concerns separate.
- Use `pathlib` for paths and preserve raw inputs under `data/`.
- Add type hints for public helpers and structured records.
- Make random seeds explicit and log split behavior when experiments create manifests.
- Fail loudly on impossible filename patterns or missing required paths.
- Record biological or metadata assumptions close to the code that depends on them.

## Agent Behavior Expectations

- Before proposing implementation details, inspect whether new scripts, metadata, or annotations have appeared.
- Prefer editing documentation or small utilities over speculative infrastructure.
- When adding code later, also update this file and `CLAUDE.md` with real commands and current repository facts.
- Do not remove or contradict `CLAUDE.md`; keep this file aligned with it.
- When in doubt, be explicit about uncertainty and cite the file that supports a claim.
