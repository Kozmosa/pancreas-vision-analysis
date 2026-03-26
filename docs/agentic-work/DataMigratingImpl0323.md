# Data Migrating Impl 0323

## Objective

Reorganize `data/` into semantically clearer English directories without doing a large,
destructive restructure. The immediate goal is to stop relying on the ambiguous top-level
 folders `图片` and `多色`, while preserving all original data in place.

## Scope And Constraints

- Keep this migration lightweight: copy files into new English directories instead of moving or deleting source data.
- Reuse existing subdirectory semantics when they are already acceptable, such as `kc` and `KPC`.
- For ambiguous source folders, inspect contained filenames and assign the narrowest safe English name.
- Do not overclaim pathology labels when the repo lacks authoritative metadata tables.
- Preserve old directories for traceability and rollback.

## Source Review

Current top-level data layout before migration:

- `data/图片/kc/`
- `data/图片/KPC/`
- `data/图片/雨蛙素/`
- `data/图片/新建文件夹/`
- `data/多色/`

Relevant repository notes used during interpretation:

- `docs/ProjectDemand.md` states that caerulein-induced material corresponds to ADM and KC-model material is used for PanIN analysis.
- `docs/Answer_Batch_2.md` states that `雨蛙素` is ADM-focused, `kc` is PanIN-oriented, `KPC` currently represents early PanIN rather than confirmed PDAC, and `merge` / `amylase` / `ck-19` are paired views of the same lesion region.

## Naming Decisions

### Direct Copies For Semantically Acceptable Folders

- `data/图片/kc/` -> `data/KC/`
- `data/图片/KPC/` -> `data/KPC/`
- `data/图片/雨蛙素/` -> `data/caerulein_adm/`

Reasoning:

- `KC` and `KPC` are already compact biological/model identifiers and only needed normalization to English top-level placement.
- `雨蛙素` is better expressed as `caerulein_adm`, which preserves both the inducing agent and the current working ADM interpretation.

### Split Of `data/多色/`

`data/多色/` was not copied as a single folder because the directory name describes image appearance rather than sample meaning.
Its filenames separate into two clearer semantic groups:

- `data/multichannel_adm/`
- `data/multichannel_kc_adm/`

Routing rule used:

- Filenames containing `kc` or `KC-` were copied into `data/multichannel_kc_adm/`.
- Remaining files were copied into `data/multichannel_adm/`.

Examples:

- `amylase-adm-IF-10x.tif` -> `data/multichannel_adm/`
- `KC-20x-merge.tif` -> `data/multichannel_kc_adm/`
- `ck-19-KC-adm-IF-40x.tif` -> `data/multichannel_kc_adm/`

### Renaming Of `data/图片/新建文件夹/`

`data/图片/新建文件夹/` was copied to:

- `data/multichannel_unresolved/`

Reasoning:

- The folder contents clearly represent a multichannel acquisition set because they include paired `merge`, `amylase`, and `ck-19` files across `10x`, `20x`, and `40x` magnifications.
- However, the repo does not contain authoritative metadata that safely identifies the pathology class of this folder as ADM, PanIN, or something else.
- `multichannel_unresolved` is therefore the narrowest accurate English name: it captures modality while preserving label uncertainty.

## Executed Directory Plan

Created these target directories under `data/`:

- `data/KC/`
- `data/KPC/`
- `data/caerulein_adm/`
- `data/multichannel_adm/`
- `data/multichannel_kc_adm/`
- `data/multichannel_unresolved/`

## Executed Commands And Actions

### 1. Create Targets And Copy Clear Sources

Used shell copy operations to preserve source data while materializing the new layout:

```bash
mkdir -p data/KC data/KPC data/caerulein_adm \
  data/multichannel_adm data/multichannel_kc_adm data/multichannel_unresolved

cp -a data/图片/kc/. data/KC/
cp -a data/图片/KPC/. data/KPC/
cp -a data/图片/雨蛙素/. data/caerulein_adm/
cp -a data/图片/新建文件夹/. data/multichannel_unresolved/
```

### 2. Split `data/多色/` By Filename Semantics

Used a small Python script to classify files by filename:

```python
from pathlib import Path
import shutil

src = Path("data/多色")
adm = Path("data/multichannel_adm")
kc_adm = Path("data/multichannel_kc_adm")

for path in src.iterdir():
    if not path.is_file():
        continue
    name = path.name.lower()
    target = kc_adm if "kc" in name else adm
    shutil.copy2(path, target / path.name)
```

This kept the rule explicit and reproducible.

## Verification Summary

Expected post-copy file counts based on source inspection:

- `data/KC/`: 15 files copied from `data/图片/kc/`
- `data/KPC/`: 8 files copied from `data/图片/KPC/`
- `data/caerulein_adm/`: 12 files copied from `data/图片/雨蛙素/`
- `data/multichannel_unresolved/`: 35 files copied from `data/图片/新建文件夹/`
- `data/multichannel_adm/`: 9 files copied from `data/多色/`
- `data/multichannel_kc_adm/`: 9 files copied from `data/多色/`

Total copied files in new English directories: 88.

## Important Notes For Later Cleanup

- Old source directories remain in place and should be treated as legacy storage until downstream code, manifests, and documentation are updated.
- Future cleanup can remove `data/图片/` and `data/多色/` only after all references are audited.
- If a metadata table is introduced later, `data/multichannel_unresolved/` should be the first directory reviewed for relabeling.
- If future code is added, it should target the new English directories rather than the legacy Chinese top-level names.

## Recommended Next Follow-Up

- Add a small manifest file that records source path, target path, and migration rationale per directory.
- Update any future preprocessing scripts to treat the new English directories as canonical inputs.
- Revisit unresolved labels after laboratory metadata or pathology annotations become available.
