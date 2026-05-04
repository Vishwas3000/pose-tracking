# OnePose++ Pipeline — Step-by-Step

A walkthrough of every step in OnePose++'s three pipelines: SfM
reconstruction (offline, per object), training (one-time), and inference
(per-frame, runtime). Use this as a runbook when reasoning about what's
happening at each stage and where to intervene.

Pairs with `doc/codebase_overview.md` (top-level architecture) and
`doc/sfm_and_descriptors.md` (SfM internals deep dive).

---

## Big Picture

OnePose++ is **three separate pipelines** that share data via files on disk:

```
PIPELINE 1: SfM RECONSTRUCTION (per object, ~5–30 min on workstation)
   mapping video + poses + bbox  →  anno_3d_average.npz

PIPELINE 2: TRAINING (one-time, ~1 day on 8 GPUs)
   per-object annos               →  models/checkpoints/onepose_plus/last.ckpt

PIPELINE 3: INFERENCE (per frame, ~30–50 ms on iPhone 15 Pro)
   camera frame + .npz + ckpt     →  6-DoF pose + 3D bbox overlay
```

---

## Pipeline 1 — SfM Reconstruction (`run.py`)

Goal: turn a mapping video of an object into a 3D point cloud + per-point
descriptors that the matcher can query at runtime.

### Step A — Capture mapping data
**Inputs**: video frames + camera poses + 3D bbox annotation.

- User scans the object with the **OnePose Cap iOS app** (or any
  SLAM-enabled tool that outputs ARKit-quality poses).
- Per-frame 6-DoF camera poses come from ARKit / SLAM.
- The app captures `box3d_corners.txt` — the object's 3D bounding box in
  world coordinates.

**Output structure**:
```
{obj_name}-annotate/
    color/*.png
    intrin/*.txt           (or intrin_ba — BA-refined intrinsics)
    poses/*.txt            (or poses_ba — BA-refined poses)
    box3d_corners.txt
```

### Step B — Pair generation
**File**: `src/sfm_utils/pairs_from_*.py`

Decides which image pairs to feed the matcher (avoids the O(N²) explosion
of running every-pair-vs-every-pair). Three strategies:

- `pose` — pick pairs with sufficient rotation difference (default ≥10°)
- `index` — temporal index gap heuristic
- `-1` — exhaustive (small datasets only)

**Output**: `pairs-covisN.txt` listing one `name0 name1` per line.

### Step C — Detector-free coarse matching (LoFTR)
**File**: `src/KeypointFreeSfM/coarse_match/coarse_match.py:35`

- For each pair, run **LoFTR** (the pretrained 2D matcher) to get dense
  correspondences `(mkpts0, mkpts1, mconf)` — sub-pixel matched 2D coords
  with confidence scores.
- Four reductions then turn matches into an HLoc-style keypoint database:
  1. `Match2Pts2D` gathers every matched coord per image
  2. `agg_groupby_2d` deduplicates exact integer coords, summing confidences
  3. Sort by confidence → assigns each unique 2D location a stable
     `keypoint_id`
  4. Rewrites every match list from `(x0, y0, x1, y1)` floats →
     `(kp_id_in_img0, kp_id_in_img1)` integers

**Outputs**:
- `feats-loftr.h5` — per-image keypoints
- `matches-loftr.h5` — per-pair index pairs

> **Important**: descriptors at this stage are zeros (placeholder). They
> get filled in much later, in step F.

### Step D — Empty COLMAP model
**File**: `src/sfm_utils/generate_empty.py:113`

- Reads per-frame `intrin_ba/*.txt` and `poses_ba/*.txt` (or
  `intrin/poses` fallback).
- Writes a binary COLMAP model: `cameras.bin`, `images.bin`, empty
  `points3D.bin`.
- This pre-loads the camera trajectory into COLMAP's data structures so we
  can hand off cleanly to triangulation.

### Step E — COLMAP triangulation
**File**: `src/sfm_utils/triangulation.py:213`

Five sub-steps:

1. `create_db_from_model` — opens `database.db`, registers cameras and
   images
2. `import_features` — pulls keypoints from `feats-loftr.h5` into the
   COLMAP database
3. `import_matches` — writes index-paired matches per image pair
4. `geometric_verification` — runs `colmap matches_importer` for RANSAC
   essential-matrix verification
5. `run_triangulation` — calls `colmap point_triangulator` with intrinsics
   and extrinsics **fixed** (poses come from SLAM, we trust them)

**Output**: `sfm_ws/model_coarse/{points3D, images, cameras}.bin` — each
3D point now has `xyz`, `image_ids`, and `point2D_idxs` (the track of
observations).

### Step F — Post-optimization (depth refinement + descriptor sampling)
**File**: `src/KeypointFreeSfM/post_optimization/post_optimization.py:59`

The most subtle step. For each 3D point's track:

- Re-run LoFTR at **fine resolution** between (query frame, ref frame)
  pairs.
- At each fine match, **sample LoFTR's backbone CNN feature map** at the
  keypoint coordinate. *This is the actual descriptor source* — what gets
  averaged later.
- Run **DeepLM (second-order)** or fallback Adam optimizer to refine *only*
  each point's `z` (depth) so observations agree across views.
- Cache the per-pair `feature0`, `feature1`, `feature_c0`, `feature_c1`
  (fine + coarse descriptors).

`feature_aggregation_and_update` then averages all collected descriptors
per (frame, keypoint_id) and writes them back into `feats-loftr.h5`.

`points3D.bin` is updated with refined depths.

### Step G — Postprocess (filter + average per 3D point)
**File**: `run.py:300` calls into `src/sfm_utils/postprocess/`

Four cleanup operations:

- **3D bbox crop** (`filter_points.filter_bbox`) — drop any 3D point
  outside the annotated bbox.
- **Track-length cap** (`filter_tkl.get_tkl`) — pick threshold T such that
  `points-with-track-length-≥-T` fits within `max_num_kp3d` (typically
  7000–15000).
- **Near-duplicate merge** (`filter_points.merge`) — cluster points within
  1 mm, replace with their centroid.
- **Average descriptors per 3D point** (`feature_process.get_kpt_ann`) —
  for each surviving 3D point, gather ALL descriptors that observed it
  across all frames, mean them → single 64-D fine + 256-D coarse vector.

**Final outputs** (per object):
```
anno/anno_3d_average.npz       ← fine descriptors (128-D), used by inference matcher
anno/anno_3d_average_coarse.npz ← coarse descriptors (256-D), used as positional encoding
anno/anno_2d.json              ← per-image 2D-3D index (training only)
```

---

## Pipeline 2 — Training (`train_onepose_plus.py`)

One-time. Trains the 2D-3D matching network that runs at inference.

### Step H — Merge per-object annotations into one COCO file
**File**: `merge.py`

- Each object has its own `anno_2d.json` from Step G.
- `merge.py` flattens these into one `train.json` + one `val.json` (COCO
  format).
- Per-image record:
  `{image_id, image_path, pose_path, anno_2d_path, avg_anno3d_path}`.

### Step I — Datamodule + Dataset
**File**: `src/datasets/OnePosePlus_dataset.py`

Per training sample:

- Loads cropped query image (typically 640×640).
- Loads the avg 3D descriptors for that object (`anno_3d_average.npz`).
- Pads to fixed `shape3d=7000` points (so batches have uniform shapes).
- **Builds the GT confidence matrix** (`build_assignmatrix`) — projects
  GT 3D points using GT pose + K to get the 2D correspondences the
  supervisor expects.
- Optional: random homography warp augmentation.

### Step J — The 2D-3D matcher network
**File**: `src/models/OnePosePlus/OnePosePlusModel.py:25`

This is **NOT LoFTR** — it's a LoFTR-shaped architecture but with 3D
points as one input sequence:

1. **ResNet-FPN backbone** extracts coarse (1/8) and fine (1/2) feature
   maps from the query image.
2. **3D-side encoding**: an MLP fuses each 3D point's `xyz` with its
   precomputed coarse descriptor → positional encoding for the point
   sequence.
3. **Coarse Transformer**: interleaved self/cross attention between the
   3D-point sequence and the flattened 2D feature grid (linear attention
   for memory).
4. **Coarse matching head**: dual-softmax over the similarity matrix →
   mutual-NN 3D↔2D pairs.
5. **Fine refinement**: 5×5 window crops at fine resolution, sub-pixel
   DSNT regression for sub-pixel accuracy.

### Step K — Loss
**File**: `src/lightning_model/losses.py`

- **Focal loss** on the coarse confidence matrix (binary correspondence GT).
- **L2-with-std regression** loss on fine sub-pixel offsets (uncertainty
  std as inverse-variance weight).

### Step L — Train loop
**File**: `train_onepose_plus.py` — PyTorch Lightning

- Default: 8 GPUs × 23 GB VRAM each.
- Backbone weights initialized from LoFTR's pretrained checkpoint.
- Saves to `models/checkpoints/{exp_name}/epoch=N.ckpt`.

---

## Pipeline 3 — Inference (`demo.py` / `inference.py`)

Per-frame runtime path. `demo.py` for custom user data; `inference.py`
for benchmark splits (OnePose, OnePose-LowTexture, LINEMOD).

### Step M — Build the model + load checkpoint
**File**: `src/inference/inference_OnePosePlus.py:build_model`

- Same `OnePosePlus_model` from Step J.
- Loads the trained `.ckpt` weights.

### Step N — Object detection (frame 0 / re-acquisition)
**File**: `src/local_feature_object_detector/local_feature_2D_detector.py`

Used on the first frame and whenever tracking is declared lost.

- Loads ~15 reference frames from the SfM workspace.
- Runs **LoFTR** between the current frame and each reference.
- RANSAC-fits an affine to the matches, uses warped image corners as 2D
  bbox.
- Picks the reference with the most inliers.

### Step O — Object detection (frames ≥ 1, fast path)
**File**: same file, `previous_pose_detect`

- If the previous frame had ≥20 PnP inliers → trust last pose.
- Project the annotated 3D bbox using last pose → axis-aligned 2D bbox in
  current frame.
- Free, no LoFTR call needed.

### Step P — Crop + resize + run matcher
**File**: `demo.py:67` (`inference_core`)

- Crop the query image to the 2D bbox, resize to 512×512.
- Build an `OnePosePlusInferenceDataset` sample with the preloaded
  `anno_3d_average.npz`.
- Run the 2D-3D matcher → outputs:
  - `mkpts_3d_db` — N×3 matched 3D points
  - `mkpts_query_f` — N×2 sub-pixel matched query pixels
  - `mconf` — match confidences

### Step Q — PnP-RANSAC for 6-DoF pose
**File**: `src/utils/metric_utils.py:ransac_PnP`

- Solves 6-DoF pose `(R, t)` from 2D-3D correspondences using
  `cv::solvePnPRansac`.
- Reprojection error threshold typically 5–7 px.
- Outputs: `pose_pred` (4×4), inlier indices.

### Step R — Render
- For each frame: project the 8 corners of `box3d_corners.txt` through
  `(K, R, t)` → 8 screen points.
- Draw the 3D bbox wireframe on the frame.
- Output: `demo_video.mp4`.

### Step S — State machine for tracking continuity
- If `len(inliers) > 20` → keep tracking; use last pose for next frame's
  detection.
- If `len(inliers) ≤ 20` → declared lost; fall back to Step N (full LoFTR
  detection).

---

## Data flow at a glance

```
SfM RECONSTRUCTION (per object, offline, ~5–30 min on workstation)
─────────────────────────────────────────────────────────────────
mapping video + poses + bbox
    ↓ pair generation (B)
    ↓ LoFTR pairwise matching (C)
    ↓ empty COLMAP model + triangulation (D, E)
    ↓ depth refinement + feature sampling (F)
    ↓ filter + average descriptors (G)
anno/anno_3d_average.npz (1–5 MB)


TRAINING (one-time, ~1 day on 8 GPUs)
─────────────────────────────────────
merged train.json + per-object .npz files
    ↓ Lightning training of 2D-3D matcher (J, K, L)
models/checkpoints/onepose_plus/last.ckpt


INFERENCE (per frame, ~30–50 ms on iPhone 15 Pro)
─────────────────────────────────────────────────
camera frame
    ↓ object detect (N or O)
    ↓ crop + resize
    ↓ 2D-3D matcher with .npz database (P)
    ↓ PnP-RANSAC (Q)
    ↓ render 3D bbox (R)
6-DoF pose drawn on frame
```

---

## What's reusable for cross-platform mobile

If you go the **OnePose++ route on mobile**:
- Steps A–G stay on a workstation. The `.npz` is the deliverable.
- Step J's matcher is the ML model you'd port to Core ML / ONNX / TFLite.
- Steps N, O, P, Q, R are the per-frame mobile pipeline. Step Q (PnP) is
  OpenCV — but EPnP-only is also pure-Swift / pure-Kotlin doable.

If you replace OnePose++'s matcher with **XFeat**:
- Steps A–F still apply (SfM is feature-extractor-agnostic at the COLMAP
  level).
- Step G needs a tweak: use XFeat to sample descriptors instead of
  LoFTR's CNN map.
- Step J's matcher becomes "cosine NN against the .npz" — much simpler.
- Steps N, O, P stay the same conceptually; only the matcher swap differs.

The tradeoff: OnePose++'s matcher is more accurate (the transformer
learns 2D-3D-specific features), but XFeat is simpler / lighter /
cross-platform faster.

---

## Where each step's "knobs" live

| Step | What you'd tune | Where |
|---|---|---|
| B | Pair generation strategy + min rotation | `configs/preprocess/sfm_*.yaml` (`sfm.gen_cov_from`, `sfm.min_rotation`) |
| C | LoFTR weights | `coarse_match.py:cfgs.matcher.model.weight_path` |
| F | Solver type, optimization LR | `cfg.post_optim.optimizer.solver_type` / `optimize_lr.depth` |
| G | Track-length cap, bbox filter, merge distance | `cfg.dataset.max_num_kp3d`, `cfg.post_process.skip_bbox_filter`, `merge` `dist_threshold` |
| I | Image size, 3D-point cap, augmentation | `configs/experiment/train.yaml: datamodule` |
| J | Model arch (channels, layers, attention) | `configs/experiment/train.yaml: model.OnePosePlus` |
| K | Loss weights | `configs/experiment/train.yaml: model.loss` |
| N | Number of reference views, RANSAC threshold | `local_feature_2D_detector.py:LocalFeatureObjectDetector.__init__` |
| Q | PnP reprojection threshold | `cfg.model.eval_metrics.pnp_reprojection_error` |
