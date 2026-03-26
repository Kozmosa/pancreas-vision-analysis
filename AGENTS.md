# AGENTS.md

## Purpose

This file guides agentic coding assistants working in this repository.
It reflects the repository as it exists today, not an imagined future pipeline.

## Repository Reality Check

- This repository is currently a research workspace centered on data and documentation.
- There is no implemented training pipeline, application package, notebook suite, or test harness yet.
- Do not invent missing infrastructure such as `pyproject.toml`, `package.json`, Docker flows, CI jobs, or evaluation scripts.
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

- `data/图片/`: microscopy TIFF assets grouped into buckets such as `kc`, `KPC`, `雨蛙素`, and `新建文件夹`.
- `docs/`: requirement notes, clarifications, and reference material.
- `.claude/settings.json`: local tool settings; not a project build config.
- There are currently no `src/`, `tests/`, `scripts/`, `notebooks/`, or environment manifest directories/files.

## Cursor and Copilot Rules

- No `.cursorrules` file was found.
- No `.cursor/rules/` directory was found.
- No `.github/copilot-instructions.md` file was found.
- If any of these files appear later, update this document and follow them as higher-priority repository guidance.

## Working Domain Assumptions

- `雨蛙素` data is currently treated as ADM-focused material.
- `kc` data is more aligned with PanIN.
- `KPC` data currently represents early PanIN, not confirmed PDAC.
- `merge`, `amylase`, and `ck-19` filenames likely refer to the same field/sample in different channels or renderings.
- `10x`, `20x`, and `40x` denote progressive magnification of the same lesion region.
- Names like `kc-adm` should be read as ADM morphology under a KC-model background.
- These are working assumptions from notes, not authoritative pathology labels.

## Known Unknowns

- No authoritative metadata table links image to mouse, slice, field, label, or split.
- No region-level or pixel-level annotations are checked in.
- No confirmed train/validation/test split implementation exists.
- `新建文件夹` is an ambiguous storage bucket, not a reliable biological class.
- Planned markers in `docs/ProjectDemand.md` are not yet proven to all exist in the checked-in data.

## Build, Lint, and Test Commands

At the time of writing, there are no project-specific build, lint, or test commands.
Use that fact explicitly rather than guessing.

### Supported Commands Today

- `git status --short`: inspect working tree changes.
- `git log --oneline -5`: inspect recent history.
- `ls data docs`: inspect top-level project content.
- `ls "data/图片"`: inspect dataset buckets.
- `ls docs`: inspect documentation assets.

### Build

- No build command exists yet.
- Do not claim `make`, `python -m build`, `npm run build`, or similar commands work unless the repo gains the required files.

### Lint

- No lint command exists yet.
- Do not invent `ruff`, `flake8`, `eslint`, `prettier`, or other tooling without a checked-in config and dependency manifest.

### Test

- No automated test suite exists yet.
- Do not claim `pytest`, `unittest`, `npm test`, or notebook test workflows are available.

### Single Test

- No single-test command exists because there is no test framework or test target in the repository.
- If tests are added later, update this section with exact commands such as `pytest path/to/test_file.py::test_name` or equivalent.

## How Agents Should Explore the Repo

- Read `docs/ProjectDemand.md` before proposing workflow or implementation ideas.
- Read `docs/Answer_Batch_2.md` before interpreting dataset folder names or image filenames.
- Verify repository contents before making assumptions about code, tooling, or metadata.
- Prefer statements like "not present in repo" over speculative claims.
- Distinguish clearly between current assets and intended future methodology.

## Code Style Guidance for This Repo Today

Because there is almost no code yet, style guidance here is mainly for future additions.
Follow these conventions if you add scripts, utilities, or small research code.

### General Principles

- Keep additions minimal and justified by current repository needs.
- Prefer small, explicit scripts over premature framework scaffolding.
- Preserve the repository's current role as a research workspace.
- Document assumptions near the code that depends on filename-derived labels.
- Avoid hidden magic, implicit data discovery rules, and fragile path conventions.

### Imports

- Group imports into standard library, third-party, and local imports.
- Keep imports explicit; avoid wildcard imports.
- Import only what is used.
- Prefer absolute imports if a real package structure is later introduced.
- If the repo remains script-based, keep intra-repo imports simple and predictable.

### Formatting

- Use UTF-8 file encoding.
- Prefer 88-100 character lines if a formatter is later adopted; until then, stay readable and consistent.
- Use 4 spaces for Python indentation.
- Keep one logical operation per line unless a compact expression is clearly clearer.
- Avoid noisy vertical spacing, but separate conceptual blocks cleanly.

### Types

- Add type hints for public functions, data-loading utilities, and reusable helpers.
- Prefer concrete types over `Any` when practical.
- Use `TypedDict`, `dataclass`, or small domain structs when representing metadata records.
- Validate filename-derived metadata rather than trusting string parsing blindly.
- If labels are uncertain, model uncertainty explicitly instead of forcing a false precise type.

### Naming

- Use descriptive names tied to microscopy and pathology domain meaning.
- Prefer `snake_case` for Python functions, variables, and file names.
- Prefer `PascalCase` for classes.
- Use singular names for one record/item and plural names for collections.
- Avoid ambiguous names like `data1`, `tmp`, `misc`, or `final_v2`.
- Reflect biological uncertainty in names, for example `working_label` or `assumed_channel`.

### Function Design

- Keep functions focused on one responsibility.
- Separate filesystem discovery, metadata parsing, preprocessing, and modeling concerns.
- Return structured data instead of printing from reusable functions.
- Keep side effects at the edges: CLI wrappers, file writes, or plotting code.
- Parameterize paths instead of hardcoding machine-specific locations.

### Error Handling

- Fail loudly on corrupted files, missing required paths, and impossible filename patterns.
- Raise clear exceptions with actionable context, including the file path involved.
- Do not silently drop samples unless that behavior is explicitly requested and logged.
- When using filename heuristics, warn when parsing is ambiguous.
- Handle missing optional metadata separately from invalid required metadata.

### Data and Path Handling

- Use `pathlib` for filesystem paths in Python.
- Avoid embedding assumptions that all folders are trustworthy class labels.
- Treat `新建文件夹` as ambiguous until external metadata clarifies it.
- Preserve raw data; write derived outputs to a separate, clearly named location.
- Never overwrite source images in `data/图片/`.

### Documentation in Code

- Add brief docstrings to non-trivial public functions.
- Record biological or dataset assumptions close to the parsing logic that uses them.
- Cite the relevant repo document when a behavior comes from documentation rather than data inspection.
- Keep comments factual and sparse; prefer better names over explanatory noise.

### Logging and Reporting

- Prefer structured summaries over ad hoc print spam.
- Report counts of discovered files, skipped files, ambiguous labels, and parsing failures.
- Make it easy to trace every derived artifact back to source inputs.
- If generating splits later, log the exact split logic and random seed.

### Reproducibility

- Make random seeds explicit.
- Keep preprocessing deterministic unless stochastic augmentation is intentionally part of an experiment.
- Version output manifests and intermediate metadata when they become important to experiments.
- Do not rely on unstated local environment state.

## Agent Behavior Expectations

- Before proposing implementation details, check whether environment files, scripts, notebooks, or tests have appeared.
- Prefer editing documentation over creating speculative code when the repo still lacks execution infrastructure.
- When adding code later, also update this file with real commands and style/tooling details.
- Do not remove or contradict `CLAUDE.md`; keep this file aligned with it.
- When in doubt, be explicit about uncertainty and cite the file that supports a claim.

## Updating This File Later

Revise this document when any of the following appear:

- Python environment or dependency manifests.
- Training, preprocessing, or evaluation scripts.
- Notebook workflows.
- Test suites and single-test entry points.
- Metadata tables or annotation files.
- Cursor or Copilot instruction files.
- Formal formatter, linter, typing, or CI configuration.
