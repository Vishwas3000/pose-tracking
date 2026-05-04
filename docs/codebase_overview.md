# OnePose++ Codebase Overview

A deep walkthrough of the repo structure, what each component does, and how
the train → SfM → inference pipelines fit together. Use this as a reference
when navigating the code.

## What the project does

OnePose++ takes a short video sweep of an object → reconstructs a sparse 3D
point cloud of it (no CAD model, no manually labeled keypoints) → then for any
new image of the same object, predicts the 6-DoF pose by matching 2D pixels to
those 3D points and running PnP. The novelty is that everything is
**keypoint-free** (LoFTR-style detector-free matching) and **one-shot** (one
mapping sequence per object, no per-object training).

## Three pipelines, three entry points

| Stage | Entry | Hydra config group |
|---|---|---|
| Build 3D model from a mapping video | `run.py` | `+preprocess=...yaml` |
| Train the 2D-3D matching network | `train_onepose_plus.py` | `+experiment=train.yaml` |
| Inference on test sequences | `inference.py` (eval) / `demo.py` (custom data) | `+experiment=inference_*.yaml` |

`scripts/demo_pipeline.sh` wires Stage 1 + Stage 3 end-to-end for the
iOS-captured demo data.

The Hydra dispatch trick is in every entry point: `globals()[cfg.type](cfg)`
— `cfg.type` is set in the experiment YAML to e.g. `"sfm"` or `"inference"`,
which then calls the matching top-level function in the script.

---

## Stage 1 — Keypoint-Free SfM (`run.py` + `src/KeypointFreeSfM/`)

Goal: turn N RGB frames + COLMAP-friendly poses into a per-object semi-dense
point cloud where each 3D point already carries a **learned descriptor** that
the matching network expects.

### `run.py`
- `sfm()` (`run.py:21`): parses `cfg.dataset.data_dir`, optionally fans out
  across Ray workers.
- `sfm_core()` (`run.py:144`): the actual reconstruction.
- `postprocess()` (`run.py:300`): bbox/track-length filtering + descriptor
  averaging.

### `sfm_core` substeps

1. **Pair generation** — `src/sfm_utils/pairs_from_index.py`,
   `pairs_from_poses.py`, or `pairs_exhaustive_all.py`. Decides which image
   pairs to feed the matcher (covisibility heuristics).
2. **Coarse matching** —
   `src/KeypointFreeSfM/coarse_match/coarse_match.py:35`
   (`detector_free_coarse_matching`). Uses LoFTR to produce dense
   correspondences, then `Match2Pts2D` consolidates them into a per-image
   keypoint set, indexes matches, and writes `feats-*.h5` + `matches-*.h5`
   (HLoc format). The descriptors at this stage are placeholders — the
   keypoints are what matter.
3. **Triangulation + post-refinement**:
   - `src/sfm_utils/triangulation.py` calls COLMAP's mapper to produce a
     coarse model.
   - `src/KeypointFreeSfM/post_optimization/post_optimization.py:59`
     re-matches each track at fine resolution (`fine_match`), then runs a
     non-linear depth-only optimizer (`optimizer/optimizer.py`, with a DeepLM
     second-order solver or `first_order_solver.py` fallback). It refines 3D
     point depths so they are consistent with the fine matches, then writes
     the refined COLMAP model.

### `postprocess()` (`run.py:300`)
- Crops the cloud to the annotated 3D bbox (`filter_points.filter_bbox`)
- Caps the number of points by selecting a track-length threshold
  (`filter_tkl.get_tkl`)
- Merges nearby points
- Calls `feature_process.get_kpt_ann` **twice** — once for LoFTR-coarse
  features and once for fine features — to compute the **average descriptor
  per 3D point**.

Output: `anno_3d_average.npz` + `anno_3d_average_coarse.npz` per object.
These are what the matching network consumes.

---

## Stage 2 — The 2D-3D matching network (`src/models/OnePosePlus/`)

A LoFTR-derivative that matches a query image's CNN feature map against a set
of 3D points represented by descriptors. Defined in
`src/models/OnePosePlus/OnePosePlusModel.py:25`.

### Forward pass (`OnePosePlusModel.py:96`)

1. **Backbone** (`backbone/resnet.py`, ResNet-FPN) — extracts coarse (1/8)
   and fine (1/2) feature maps from the cropped query image.
2. **3D-side encoding** —
   `utils/position_encoding.py:KeypointEncoding_linear` is an MLP that fuses
   each 3D point's coordinate with its precomputed descriptor (the average
   descriptor from Stage 1). 2D features get sinusoidal
   `PositionEncodingSine`.
3. **Coarse Transformer** — `loftr_module/transformer.py`
   (`LocalFeatureTransformer`) does interleaved self/cross attention between
   the 3D-point sequence (`desc3d_db`) and the flattened 2D feature grid
   (`query_feat_c`). Linear attention (`linear_attention.py`) for memory.
4. **Coarse matching** — `utils/coarse_matching.py:CoarseMatching` uses
   dual-softmax over the similarity matrix to pick mutual-NN 3D↔2D pairs.
   Outputs `mkpts_3d_db`, `mkpts_query_c`, plus a confidence matrix used as
   training supervision.
5. **Fine refinement** — `loftr_module/fine_preprocess.py` crops a 5×5
   window around each coarse match in the fine feature map, runs a smaller
   fine transformer, then `utils/fine_matching.py` runs a sub-pixel **DSNT
   (spatial expectation)** to regress sub-pixel coords with an uncertainty
   std. Outputs `mkpts_query_f`.

### Training wrapper
`src/lightning_model/OnePosePlus_lightning_model.py:PL_OnePosePlus`.
`losses.py:Loss` combines focal-loss on the coarse confidence matrix +
L2-with-std regression for fine offsets (the std is used as inverse-variance
weight). Ground truth fine locations come from projecting GT 3D points with
the GT pose (`OnePosePlus_dataset.py:343`).

---

## Stage 3 — Inference / demo

### `inference.py` (benchmark eval)
Runs over OnePose / OnePose-LowTexture / LINEMOD test splits, parallelized
via Ray. Per object it calls:
- `src/inference/inference_OnePosePlus.py:inference_onepose_plus` — builds
  `OnePosePlusInferenceDataset` (loads `anno_3d_average.npz`, pads to
  `shape3d_val`), loads the trained checkpoint via `build_model`, then runs
  `inference_OnePosePlus_worker.py:extract_matches` per image.
  `compute_query_pose_errors` does PnP-RANSAC via
  `src/utils/metric_utils.py` and produces R/t errors, ADD, proj-2D.

### `demo.py` (custom video)
`inference_core` (`demo.py:67`):
- **Frame 0**: `LocalFeatureObjectDetector.detect`
  (`src/local_feature_object_detector/local_feature_2D_detector.py:169`)
  runs LoFTR between the query and ~15 reference frames from the SfM
  workspace, RANSAC-fits an affine to the matches, and uses the warped image
  corners as the 2D bbox.
- **Frames ≥ 1**: if the previous PnP had ≥20 inliers, project the
  annotated 3D bbox using the previous pose to get a 2D bbox
  (`previous_pose_detect`); otherwise fall back to local-feature detection.
- Crop+resize to 512×512, run the matcher, run `ransac_PnP` → render the 3D
  bbox onto the frame, write `demo_video.mp4`.

---

## Data layer

- `src/datasets/OnePosePlus_dataset.py:OnePosePlusDataset` — training set.
  Reads COCO-style JSON produced by `merge.py`. Per sample: pads 3D points
  to `shape3d`, loads avg coarse+fine descriptors, projects GT 3D points to
  2D using GT pose+intrinsics, and **builds the GT confidence matrix**
  (`build_assignmatrix`, line 174). Optional homography warping
  (`image_warp_adapt`) doubles the dataset by warping the query image and
  the projected GT.
- `src/datasets/OnePosePlus_inference_dataset.py` — same, minus GT-driven
  supervision; loads pose GT only for metric computation.
- `src/datamodules/OnePosePlus_datamodule.py` — Lightning DataModule
  wrapper; instantiated via Hydra (`configs/experiment/train.yaml:178`).
- `merge.py` — flattens per-object SfM annotations into one `train.json` /
  `val.json` (COCO format) for the training datamodule.

---

## Configs (Hydra)

- `configs/config.yaml` — root; `+experiment=...` or `+preprocess=...`
  overrides via Hydra's `@package _global_` mechanism.
- `configs/experiment/train.yaml` — full training spec (model arch,
  optimizer, datamodule, callbacks, loggers). Worth re-reading: it shows
  every arch hyperparameter.
- `configs/preprocess/sfm_*.yaml` — choose which dataset to reconstruct.
- `configs/experiment/inference_*.yaml` — eval splits.

---

## Suggested reading order to "get it"

1. `README.md` + `doc/demo.md` — the user-facing flow.
2. `scripts/demo_pipeline.sh` — see all three stages chained.
3. `run.py` then drill into
   `src/KeypointFreeSfM/coarse_match/coarse_match.py` and
   `post_optimization/post_optimization.py`.
4. `src/models/OnePosePlus/OnePosePlusModel.py` (forward pass) →
   `utils/coarse_matching.py` → `utils/fine_matching.py`.
5. `src/datasets/OnePosePlus_dataset.py` (especially `read_anno` and
   `build_assignmatrix`) — explains GT generation, which is the single
   hardest piece.
6. `src/lightning_model/OnePosePlus_lightning_model.py` + `losses.py` —
   training loop.
7. `demo.py` +
   `src/local_feature_object_detector/local_feature_2D_detector.py` —
   inference loop and the 2D bbox seeding trick.

---

## Module-by-module index

### Top-level scripts
| File | Purpose |
|---|---|
| `run.py` | SfM reconstruction entry point |
| `inference.py` | Benchmark inference (Ray-parallelized) |
| `demo.py` | Custom-video inference, writes `demo_video.mp4` |
| `train_onepose_plus.py` | Lightning training entry point |
| `merge.py` | Flatten per-object SfM annos → COCO JSON for training |
| `parse_scanned_data.py` | Convert OnePose Cap iOS scans → repo format |
| `parse_lm_real_data.py` | LINEMOD → OnePose format converter |

### `src/`
| Path | Purpose |
|---|---|
| `KeypointFreeSfM/` | Detector-free SfM: coarse match, triangulation, depth refinement |
| `KeypointFreeSfM/coarse_match/` | LoFTR-based coarse pair matching |
| `KeypointFreeSfM/post_optimization/` | Fine-match + non-linear depth optimization |
| `KeypointFreeSfM/loftr_for_sfm/` | LoFTR variant used in the SfM phase |
| `models/OnePosePlus/` | The 2D-3D matching network |
| `models/OnePosePlus/backbone/` | ResNet-FPN feature extractor |
| `models/OnePosePlus/loftr_module/` | Self/cross attention transformer |
| `models/OnePosePlus/utils/` | Coarse/fine matching heads, position encoding |
| `lightning_model/` | PL wrapper + loss functions |
| `datasets/` | Train + inference datasets |
| `datamodules/` | Lightning DataModule |
| `inference/` | Inference orchestration (Ray workers) |
| `local_feature_object_detector/` | First-frame 2D bbox seeding via LoFTR |
| `sfm_utils/` | Pair generation, triangulation, postprocess (filtering, feature averaging) |
| `utils/` | I/O, metrics, plotting, COLMAP I/O, ray helpers |
| `callbacks/` | PL training callbacks |
