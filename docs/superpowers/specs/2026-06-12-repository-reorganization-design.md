# Repository Reorganization Design

## Goal

Reorganize `ball_segmentation` into a small but complete research repository that preserves the full technical evolution of the project. Historical methods remain reproducible and understandable instead of being replaced by the latest CNN route.

The Git repository will contain source code, method documentation, manual labels, essential candidate metadata, and a small set of representative results. Large datasets, generated masks, exported training sets, model weights, and complete inference outputs will be distributed through public Google Drive links.

## Method-Oriented Structure

```text
ball_segmentation/
|-- README.md
|-- LICENSE
|-- pyproject.toml
|-- common/
|-- methods/
|   |-- 01_sam_generation/
|   |-- 02_area_shape_filter/
|   |-- 03_seed_heuristic/
|   |-- 04_learned_seed/
|   |-- 05_cnn_v3/
|   `-- 06_cnn_v4_v41/
|-- annotation/
|-- scripts/
|-- examples/
`-- docs/
    |-- method_history.md
    |-- resources.md
    `-- reproduction.md
```

No `tests/` directory or mandatory minimum-validation framework is required for this repository reorganization. The organization will rely on source inspection, documented commands, and conservative file moves that preserve behavior.

## Method Boundaries

### 01 SAM Generation

Contains automatic SAM mask generation, tiled splitting, tile upscaling, mask stitching, union generation, and summary export. This is the common upstream source of candidate masks for all later routes.

### 02 Area and Shape Filter

Contains statistical area fitting and geometric mask filters such as circularity, aspect ratio, extent, solidity, convexity, circle IoU, and border contact. It documents both the value and limitations of geometry-only filtering.

### 03 Seed Heuristic

Contains image-processing-based ball seed detection and seed-supported SAM mask validation. It preserves the radial contrast, percentile threshold, non-maximum suppression, radial balance, tile-boundary, and high-score fallback logic.

### 04 Learned Seed

Contains learned-standard-seed and learned arc/shape routes. These methods remain separate from the handcrafted seed heuristic because they introduce learned reference statistics and additional classification logic.

### 05 CNN v3

Contains manual dataset export, prototype CNN training, inference, scale-normalized experiments, and v3 batch processing. The route uses manually labeled mask candidates and three-channel crop representations.

### 06 CNN v4 and v4.1

Contains the 2K six-class route, background-consistency design history, scale-aware v4.1 training, inference dataset export, classifier application, and associated batch commands. Documentation must distinguish experimental v4 ideas from the implemented v4.1 route.

## Shared Code Policy

Exact duplicate scripts will not be retained in multiple method directories. Shared image, mask, summary, path, overlay, and dataset utilities may move to `common/` only when this removes real duplication without obscuring a method's standalone flow.

Method-specific scripts stay within their method directory. Cross-method orchestration and batch processing belong in `scripts/`. Imports and documented commands will be updated to use repository-relative paths and must not depend on the original author's absolute filesystem paths.

## Annotation Assets

`annotation/` retains:

- annotation and result viewer HTML files;
- candidate-generation utilities;
- manual label CSV files;
- candidate JSON/CSV files that are required to preserve `candidate_id`, `tile_id`, and `mask_index` relationships.

Candidate metadata strongly coupled to existing labels is considered source research data and remains in Git even when it is several megabytes. Generated image crops and complete mask directories do not remain in Git.

## Examples

`examples/` contains a limited visual record of the project:

- a small number of representative input images;
- SAM output examples;
- area/shape filtering examples;
- seed heuristic examples;
- learned-seed examples when available;
- CNN v3 and v4.1 final overlays.

The existing 75 MB public viewer will be reduced to representative samples rather than publishing all eight original BMP, SAM, and final image triplets. Original scientific images should not be recompressed when fidelity is needed; otherwise preview PNG/JPEG files may be used for documentation.

## External Resources

Large resources will be grouped for future public Google Drive distribution:

1. original datasets;
2. SAM intermediate masks and summaries;
3. exported CNN training datasets;
4. trained model weights;
5. complete inference outputs;
6. optional full-resolution viewer assets.

Until uploads exist, `docs/resources.md` will use explicit `Pending upload` entries rather than fabricated links. Each entry records:

- archive filename;
- intended repository-relative extraction path;
- purpose and dependent methods;
- local size;
- SHA-256 checksum;
- public Google Drive URL once available.

All shared Google Drive files will use the permission setting "Anyone with the link can view/download."

## Documentation

The root README becomes a concise project entry point with:

- project purpose;
- method evolution overview;
- directory map;
- environment setup;
- quick links to each method;
- data and weight download status;
- representative results.

Each method directory gets a focused README covering principles, inputs, outputs, commands, applicability, limitations, and its relationship to earlier/later methods.

`docs/method_history.md` records the chronological progression and lessons learned. `docs/reproduction.md` provides end-to-end command sequences. `docs/resources.md` is the authoritative external-resource manifest.

## Dependency Scope

The existing Sa2VA-level `pyproject.toml`, `uv.lock`, and upstream README are not suitable as the root environment because they describe a much larger unrelated system. The reorganized repository will define only the dependencies directly used by retained scripts, primarily Python, NumPy, Pillow, PyTorch, and Transformers.

GPU/CUDA installation details will be documented rather than pinning a machine-specific build. Model weights remain external.

## Git Scope and Safety

Local scientific data will not be deleted during organization. It will either remain ignored in place or be left outside the new tracked structure. The `.gitignore` will explicitly exclude:

- model checkpoints and pretrained model directories;
- generated SAM masks and tiles;
- exported training datasets;
- full batch outputs and summaries where not essential metadata;
- caches, environments, logs, and temporary artifacts.

Only files intentionally selected for the minimal complete repository will be staged. Existing remote history will be updated with ordinary commits unless a later review demonstrates that history rewriting is necessary.

## Completion Criteria

The organization is complete when:

- every retained script has one clear method or shared ownership location;
- all historical method families listed above remain represented;
- manual labels and their required candidate metadata are retained;
- large data, weights, and generated outputs are excluded from Git;
- resource placeholders and future Google Drive placement instructions are documented;
- root and method READMEs explain how the pieces relate and how to run them;
- local data remains intact;
- the intended changes are committed and synchronized to `ly-rrrrr/ball_segmentation`.
