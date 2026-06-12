# Method-Oriented Repository Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the repository by historical method family while retaining source code, manual annotation assets, essential candidate metadata, concise documentation, and representative results, with large data and weights reserved for public Google Drive distribution.

**Architecture:** The current root tools and the `v3_v41_source_manual_scripts` experiment snapshot will be normalized into six method directories. Method-specific code remains near its README, cross-method batch entry points move to `scripts/`, annotation assets move to `annotation/`, and representative visuals move to `examples/`. Existing local data is left intact and excluded through `.gitignore`; Git tracks only the curated repository structure.

**Tech Stack:** Python 3.10+, NumPy, Pillow, PyTorch, Transformers, static HTML/CSS/JavaScript, Markdown, Git.

---

## Target File Map

### Root

- `README.md`: concise project overview, method index, setup, resource status, and representative results.
- `pyproject.toml`: direct project dependencies only; no Sa2VA-wide dependency set.
- `.gitignore`: explicit rules for weights, datasets, generated masks, batch results, caches, and local snapshots.

### Shared and Method Code

- `methods/01_sam_generation/run_sam_automatic_mask.py`
- `methods/01_sam_generation/README.md`
- `methods/02_area_shape_filter/filter_sam_masks_by_area.py`
- `methods/02_area_shape_filter/README.md`
- `methods/03_seed_heuristic/filter_sam_masks_by_ball_seeds.py`
- `methods/03_seed_heuristic/README.md`
- `methods/04_learned_seed/filter_sam_masks_by_learned_standard_seeds.py`
- `methods/04_learned_seed/filter_sam_masks_by_learned_seed_arc_shape.py`
- `methods/04_learned_seed/README.md`
- `methods/05_cnn_v3/train_cnn_prototype_classifier.py`
- `methods/05_cnn_v3/apply_cnn_prototype_classifier.py`
- `methods/05_cnn_v3/export_combined_manual_cnn_dataset_upscaled_fast.py`
- `methods/05_cnn_v3/export_v3_scale_normalized_dataset.py`
- `methods/05_cnn_v3/README.md`
- `methods/06_cnn_v4_v41/train_cnn_scale_aware_classifier_v41.py`
- `methods/06_cnn_v4_v41/apply_cnn_scale_aware_classifier_v41.py`
- `methods/06_cnn_v4_v41/export_2k_manual_cnn_dataset_v4_from_candidates.py`
- `methods/06_cnn_v4_v41/export_split_upscale_cnn_inference_dataset.py`
- `methods/06_cnn_v4_v41/README.md`

No new shared Python module will be introduced during the first reorganization pass. Existing scripts use sibling imports and were developed as CLI programs; moving code without refactoring behavior is safer. Import path cleanup is limited to inserting the required method directories into `sys.path` where cross-method imports are unavoidable.

### Annotation, Orchestration, and Documentation

- `annotation/viewers/*.html`: annotation and result viewers.
- `annotation/candidates/*.{json,csv}`: candidate metadata tied to manual labels.
- `annotation/labels/*.csv`: manual ground-truth labels.
- `annotation/build_2k_six_class_annotation_candidates.py`
- `annotation/normalize_candidate_paths.py`
- `annotation/README.md`
- `scripts/batch_process_new_membrane_cnn_v3_upscaled.py`
- `scripts/batch_process_cnn_v3_upscaled_dynamic.py`
- `scripts/batch_apply_v3_scale_normalized.py`
- `scripts/README.md`
- `docs/method_history.md`
- `docs/resources.md`
- `docs/reproduction.md`

### Examples

- `examples/basic/`: existing tracked SAM, area filter, and seed heuristic examples.
- `examples/cnn_v41/`: a reduced two-sample viewer containing samples 1 and 8 to show dense and sparse cases.

The other local full-resolution viewer assets remain untouched under `public_ball2k_viewer/` but ignored by Git after the representative subset is copied.

## Task 1: Establish Ignore and Dependency Boundaries

**Files:**
- Modify: `.gitignore`
- Create: `pyproject.toml`

- [ ] **Step 1: Extend `.gitignore` for local scientific assets**

Add rules covering the current snapshot and future outputs:

```gitignore
.claude/
v3_v41_source_manual_scripts/
public_ball2k_viewer/

*.pt
*.pth
*.ckpt
*.safetensors
pretrained/
weights/
checkpoints/

data/
work/
outputs/
demo/小球2K放大倍数/
demo/小球2K放大倍数_split*/
demo/*_dataset*/
demo/*_model*/
demo/*_apply*/
```

Keep explicit exceptions unnecessary because curated files will live under `examples/` and `annotation/`.

- [ ] **Step 2: Create the direct dependency manifest**

Create `pyproject.toml`:

```toml
[project]
name = "ball-segmentation"
version = "0.1.0"
description = "SAM-based ball segmentation methods from heuristic filters to scale-aware CNN classifiers"
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
dependencies = [
  "numpy>=1.24",
  "Pillow>=9.5",
  "torch>=2.0",
  "transformers>=4.40",
]
```

- [ ] **Step 3: Commit the repository boundary files**

```powershell
git add .gitignore pyproject.toml
git commit -m "Define minimal repository dependencies and ignored assets"
```

## Task 2: Move the Three Foundational Methods

**Files:**
- Move: `tools/run_sam_automatic_mask.py` to `methods/01_sam_generation/run_sam_automatic_mask.py`
- Move: `tools/filter_sam_masks_by_area.py` to `methods/02_area_shape_filter/filter_sam_masks_by_area.py`
- Move: `tools/filter_sam_masks_by_ball_seeds.py` to `methods/03_seed_heuristic/filter_sam_masks_by_ball_seeds.py`
- Create: one README in each method directory

- [ ] **Step 1: Create method directories and move tracked scripts with `git mv`**

```powershell
New-Item -ItemType Directory -Force methods/01_sam_generation,methods/02_area_shape_filter,methods/03_seed_heuristic
git mv tools/run_sam_automatic_mask.py methods/01_sam_generation/run_sam_automatic_mask.py
git mv tools/filter_sam_masks_by_area.py methods/02_area_shape_filter/filter_sam_masks_by_area.py
git mv tools/filter_sam_masks_by_ball_seeds.py methods/03_seed_heuristic/filter_sam_masks_by_ball_seeds.py
```

- [ ] **Step 2: Write focused method READMEs**

Each README must document:

- the problem addressed;
- algorithm steps;
- required inputs and generated outputs;
- exact CLI command using the new path;
- strengths and known limitations;
- where the method sits in the project history.

The seed README must preserve radial contrast, percentile thresholding, NMS, radial balance, tile-boundary support, and high-score fallback behavior.

- [ ] **Step 3: Commit foundational method organization**

```powershell
git add methods
git commit -m "Organize SAM and heuristic filtering methods"
```

## Task 3: Curate Learned-Seed Methods

**Files:**
- Copy: snapshot learned-seed scripts into `methods/04_learned_seed/`
- Create: `methods/04_learned_seed/README.md`
- Modify: copied scripts only where imports reference old sibling locations

- [ ] **Step 1: Copy the two learned-seed scripts**

Copy from:

```text
v3_v41_source_manual_scripts/sam_ball_segmentation_project/tools/filter_sam_masks_by_learned_standard_seeds.py
v3_v41_source_manual_scripts/sam_ball_segmentation_project/tools/filter_sam_masks_by_learned_seed_arc_shape.py
```

to `methods/04_learned_seed/`.

- [ ] **Step 2: Make the heuristic dependency portable**

At the top of each copied script, resolve the repository root and prepend `methods/03_seed_heuristic` to `sys.path` before importing `filter_sam_masks_by_ball_seeds`:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "methods" / "03_seed_heuristic"))
```

The arc-shape script may continue importing the standard-seed script from its own directory.

- [ ] **Step 3: Document learned seed variants**

Explain what is learned, what remains heuristic, expected input summaries, and why this route preceded the CNN classifiers.

- [ ] **Step 4: Commit learned-seed organization**

```powershell
git add methods/04_learned_seed
git commit -m "Preserve learned seed filtering methods"
```

## Task 4: Curate CNN v3

**Files:**
- Copy five v3 scripts into `methods/05_cnn_v3/`
- Create: `methods/05_cnn_v3/README.md`
- Modify: copied scripts for cross-method imports

- [ ] **Step 1: Copy v3 training, inference, and export scripts**

Copy:

```text
train_cnn_prototype_classifier.py
apply_cnn_prototype_classifier.py
export_combined_manual_cnn_dataset_upscaled_fast.py
export_v3_scale_normalized_dataset.py
```

from the snapshot tools directory into `methods/05_cnn_v3/`.

- [ ] **Step 2: Repair cross-method imports**

For scripts importing heuristic or learned-seed helpers, prepend these paths:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "methods" / "03_seed_heuristic"))
sys.path.insert(0, str(REPO_ROOT / "methods" / "04_learned_seed"))
```

Keep same-directory imports for v3 model and dataset helpers.

- [ ] **Step 3: Document the v3 route**

Include the six label classes, three-channel crop representation, manual-label dependency, export/train/apply commands, scale-normalized experiment, and limitations across magnification/background domains.

- [ ] **Step 4: Commit CNN v3**

```powershell
git add methods/05_cnn_v3
git commit -m "Organize CNN v3 prototype classification route"
```

## Task 5: Curate CNN v4 and v4.1

**Files:**
- Copy four v4/v4.1 scripts into `methods/06_cnn_v4_v41/`
- Create: `methods/06_cnn_v4_v41/README.md`
- Modify: application script import path for v3 overlay utilities

- [ ] **Step 1: Copy implemented v4.1 scripts**

Copy:

```text
train_cnn_scale_aware_classifier_v41.py
apply_cnn_scale_aware_classifier_v41.py
export_2k_manual_cnn_dataset_v4_from_candidates.py
export_split_upscale_cnn_inference_dataset.py
```

into `methods/06_cnn_v4_v41/`.

- [ ] **Step 2: Repair the v3 overlay import**

In `apply_cnn_scale_aware_classifier_v41.py`, prepend `methods/05_cnn_v3` to `sys.path` before importing `render_overlays` from `apply_cnn_prototype_classifier`.

- [ ] **Step 3: Document v4 history and v4.1 implementation**

The README must explicitly distinguish:

- v4 background-consistency motivation and experiments;
- v4.1 scale-aware implementation retained in source;
- 2K six-class labels;
- export, training, and inference commands;
- expected model/data resources and their pending Google Drive status.

- [ ] **Step 4: Commit CNN v4/v4.1**

```powershell
git add methods/06_cnn_v4_v41
git commit -m "Organize CNN v4 and v4.1 scale-aware route"
```

## Task 6: Curate Annotation Assets

**Files:**
- Create: `annotation/labels/`
- Create: `annotation/candidates/`
- Create: `annotation/viewers/`
- Copy: three manual-label CSVs
- Copy: candidate JSON/CSV metadata
- Copy: four viewer HTML files
- Copy: candidate builder and path normalization scripts
- Create: `annotation/README.md`

- [ ] **Step 1: Copy manual labels from the current root `demo/` files**

Copy:

```text
demo/a8638_double_ball_manual_labels_v2.csv
demo/a8638_split36_area_mid_manual_labels.csv
demo/ball2k_six_class_manual_labels.csv
```

to `annotation/labels/`.

- [ ] **Step 2: Copy candidate metadata and viewers from the snapshot**

Retain the three current candidate pairs and the legacy v3 prediction JSON because the manual labels are ID-coupled to this metadata.

- [ ] **Step 3: Copy annotation utilities**

Move the candidate builder and normalization script to `annotation/`. Update their documented default viewer/data paths to the new `annotation/` structure without changing candidate IDs.

- [ ] **Step 4: Document annotation invariants**

State prominently that old labels must not be attached to regenerated candidates unless `candidate_id`, `tile_id`, and `mask_index` equivalence has been proven.

- [ ] **Step 5: Commit annotation assets**

```powershell
git add annotation
git commit -m "Preserve manual labels and annotation metadata"
```

## Task 7: Curate Cross-Method Batch Scripts

**Files:**
- Create: `scripts/README.md`
- Copy: three batch scripts from the snapshot
- Modify: script defaults and invoked script paths

- [ ] **Step 1: Copy batch scripts into `scripts/`**

Copy:

```text
batch_process_new_membrane_cnn_v3_upscaled.py
batch_process_cnn_v3_upscaled_dynamic.py
batch_apply_v3_scale_normalized.py
```

- [ ] **Step 2: Replace author-machine defaults**

Remove `/mnt/csip-500/...` Python defaults. Use `sys.executable` when `--python` is omitted. Point invoked scripts at the new `methods/01_sam_generation`, `methods/05_cnn_v3`, and `methods/06_cnn_v4_v41` paths.

- [ ] **Step 3: Document orchestration scope**

Explain that these scripts compose methods and therefore do not belong to one method directory.

- [ ] **Step 4: Commit batch scripts**

```powershell
git add scripts
git commit -m "Add portable cross-method batch workflows"
```

## Task 8: Build Representative Examples

**Files:**
- Move/copy existing tracked demo assets into `examples/basic/`
- Create: `examples/cnn_v41/ball2k_result_viewer.html`
- Copy: sample 1 and sample 8 image triplets into `examples/cnn_v41/assets/`
- Modify: viewer sample manifest to contain only samples 1 and 8

- [ ] **Step 1: Move tracked basic examples with history**

Move the currently tracked original images and representative SAM/area/seed outputs from `demo/` into route-named subdirectories under `examples/basic/`.

- [ ] **Step 2: Copy two CNN v4.1 viewer samples**

Use sample 1 as a dense case and sample 8 as a sparse case. Copy each original/SAM/final triplet from `public_ball2k_viewer/assets/`.

- [ ] **Step 3: Reduce the viewer manifest**

Keep the existing viewer behavior but remove samples 2-7 from the JavaScript `samples` array. Update paths and titles only as required by the new location.

- [ ] **Step 4: Commit representative examples**

```powershell
git add examples
git commit -m "Add representative results across method generations"
```

## Task 9: Write Project-Level Documentation

**Files:**
- Rewrite: `README.md`
- Create: `docs/method_history.md`
- Create: `docs/resources.md`
- Create: `docs/reproduction.md`

- [ ] **Step 1: Rewrite the root README**

Keep it concise and link to method READMEs. Include a method table with columns: method, core idea, training required, retained entry points, and status.

- [ ] **Step 2: Write chronological method history**

Cover SAM generation, area/shape filtering, heuristic seed validation, learned seed variants, CNN v3, v4 background-consistency experiments, and v4.1 scale-aware classification. Preserve negative findings and limitations rather than presenting only successful outcomes.

- [ ] **Step 3: Write the external resource manifest**

Use a table with:

```text
Resource | Archive | Extract to | Size | SHA-256 | Google Drive | Used by
```

Set Google Drive cells to `Pending upload`. Record resources visible locally without moving or archiving them during this task.

- [ ] **Step 4: Write reproduction commands**

Translate `SOURCE_ONLY_COMMANDS.md` to the new paths. Separate commands by method and clearly mark commands requiring external data, SAM weights, CNN weights, or GPU execution.

- [ ] **Step 5: Commit project documentation**

```powershell
git add README.md docs methods/*/README.md annotation/README.md scripts/README.md
git commit -m "Document method history and reproduction resources"
```

## Task 10: Remove Obsolete Tracked Layout and Audit Scope

**Files:**
- Remove tracked empty `tools/` layout after moves
- Confirm snapshot, full viewer, and local data remain present but untracked/ignored

- [ ] **Step 1: Inspect the complete tracked file list**

```powershell
git ls-files
```

Expected: only curated root files, method sources, annotation assets, scripts, examples, and documentation.

- [ ] **Step 2: Confirm large local directories still exist**

```powershell
Test-Path v3_v41_source_manual_scripts
Test-Path public_ball2k_viewer
Test-Path demo/小球2K放大倍数
```

Expected: all return `True`.

- [ ] **Step 3: Review staged and unstaged scope**

```powershell
git status -sb
git diff --stat origin/main...HEAD
```

Expected: no accidentally tracked model weights, generated mask directories, full batch result trees, or all eight full viewer triplets.

- [ ] **Step 4: Commit any final scope corrections**

```powershell
git add .gitignore README.md docs methods annotation scripts examples pyproject.toml
git commit -m "Finalize minimal complete repository layout"
```

Skip the commit if there are no remaining changes.

## Task 11: Synchronize GitHub

**Files:** None.

- [ ] **Step 1: Confirm authentication and remote**

```powershell
gh auth status
git remote -v
```

Expected remote: `https://github.com/ly-rrrrr/ball_segmentation.git`.

- [ ] **Step 2: Push the reorganized `main` branch**

```powershell
git push origin main
```

- [ ] **Step 3: Confirm local and remote tracking refs match**

```powershell
git rev-parse main
git rev-parse origin/main
git status -sb
```

Expected: the two SHAs match and status shows `main...origin/main` with no curated changes pending.
