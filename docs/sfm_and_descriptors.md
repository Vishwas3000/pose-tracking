# SfM Database & Descriptors — Deep Dive

How OnePose++ generates the COLMAP database, the 3D point cloud, and the
per-point descriptors that the inference network queries at runtime — plus
practical levers for getting a cleaner / more useful 3D database.

This is the "what actually happens between `your-obj-annotate/color/*.png`
and `anno_3d_average.npz`" reference.

---

## 1. The end-to-end pipeline at a glance

```
mapping video frames + camera poses (from OnePose Cap / ARKit / SLAM)
     │
     ▼
[A] Pair generation                           src/sfm_utils/pairs_*.py
     │   covisibility heuristics → pairs-covisN.txt
     ▼
[B] LoFTR coarse matching                     src/KeypointFreeSfM/coarse_match/coarse_match.py
     │   per-pair dense matches → raw_matches.h5
     │   per-image kept-keypoint set → feats-loftr.h5
     │   per-pair keypoint-index pairs → matches-loftr.h5
     ▼
[C] Build empty COLMAP model from known poses src/sfm_utils/generate_empty.py
     │   cameras.bin + images.bin (no points)
     ▼
[D] COLMAP triangulation                      src/sfm_utils/triangulation.py
     │   creates database.db, runs `colmap point_triangulator`
     │   → sfm_ws/model_coarse/{points3D, images, cameras}.bin
     ▼
[E] Post-optimization (depth refinement)      src/KeypointFreeSfM/post_optimization/post_optimization.py
     │   - re-match each track at fine LoFTR resolution
     │   - sample CNN features at refined 2D locations
     │   - DeepLM/first-order optimizer refines 3D depth (point z) per track
     │   - aggregate per-track features (avg) → updated feats-loftr.h5 (and _coarse)
     │   → sfm_ws/model/{points3D, images, cameras}.bin
     ▼
[F] Postprocess                               src/sfm_utils/postprocess/{filter_points,filter_tkl,feature_process}.py
     │   - bbox crop      (filter_points.filter_bbox)
     │   - track-length cap to ≤ max_num_kp3d (filter_tkl.get_tkl)
     │   - merge near-duplicate points (filter_points.merge)
     │   - average per-3D-point descriptor across all observing frames
     ▼
anno/anno_3d_average.npz       ← FINE descriptors (what inference uses)
anno/anno_3d_average_coarse.npz ← COARSE descriptors (positional encoding input)
anno/anno_2d.json              ← per-image 2D-3D index (training only)
```

`run.py:sfm_core` orchestrates A–E (`run.py:144`), `run.py:postprocess`
orchestrates F (`run.py:300`).

---

## 2. Detailed walkthrough

### [A] Pair generation
Three strategies, picked by `sfm.gen_cov_from`:
- `pose` — `pairs_from_poses.covis_from_pose`: pick pairs with sufficient
  rotation difference (`min_rotation`).
- `index` — `pairs_from_index.covis_from_index`: simple temporal index gap.
- `-1 covis_num` — `pairs_exhaustive_all.exhaustive_all_pairs`.

Output: `pairs-covis{N}.txt` (one `name0 name1` line per pair).

### [B] LoFTR coarse matching
`detector_free_coarse_matching` (`src/KeypointFreeSfM/coarse_match/coarse_match.py:35`).

For each pair, `match_worker` runs LoFTR (`LoFTR_for_OnePose_Plus`,
`fine_matching=False`) and gets dense `(mkpts0, mkpts1, mconf)` triplets at
**1/8 resolution rounded** (LoFTR's coarse grid). These are *not* keypoints
yet — they are pairwise matches.

Then four reductions turn matches into a HLoc-style keypoint database:
1. `Match2Pts2D` (`coarse_match/utils.py:20`) — per image, gather every
   matched coordinate that image participates in.
2. `points2D_worker` — `agg_groupby_2d` deduplicates exact integer
   coordinates and sums their match-confidences. The (coord, summed_conf)
   pairs are sorted by confidence, giving each unique 2D location a stable
   `keypoint_id`.
3. `update_matches` — rewrites every per-pair match list from
   `(x0,y0,x1,y1)` floats to `(kp_id_in_img0, kp_id_in_img1)` integer
   indices.
4. `transform_points2D` — packs the final per-image keypoint arrays.

Files written:
- `feats-loftr.h5`: per image — `keypoints` (N×2), `descriptors` (zeros,
  256×N — placeholders!), `scores` (ones).
- `matches-loftr.h5`: per pair — `matches`/`matches0` (M×2 indices),
  `matching_scores` (placeholder ones).
- `raw_matches.h5`: cached raw float-coord matches in case you re-run.

Important: **the descriptors at this stage are zeros**. They get filled in
after triangulation, by sampling the LoFTR CNN feature map at the final
keypoint locations (steps E–F).

### [C] Empty COLMAP model
`generate_empty.generate_model` (`src/sfm_utils/generate_empty.py:113`)
turns the per-frame `intrin_ba/*.txt` and `poses_ba/*.txt` (preferred — BA-
refined) or `intrin/*.txt` and `poses/*.txt` into a binary COLMAP model
with `cameras.bin` + `images.bin` and an empty `points3D.bin`. Each frame
has no associated 2D keypoints yet — those come from the H5 file in step D.

### [D] COLMAP triangulation
`triangulation.main` (`src/sfm_utils/triangulation.py:213`):

1. `create_db_from_model` — reads the empty model, opens
   `database.db`, registers cameras and images.
2. `import_features` — pulls `feats-loftr.h5` keypoints into the COLMAP DB
   (`+0.5` to convert from pixel-corner to pixel-center convention).
3. `import_matches` — for each pair, writes the integer-indexed match list
   into `two_view_geometries`. With `match_model="loftr"` the matches are
   already verified-style pairs (M×2), so `valid = all True`.
4. `geometric_verification` — runs `colmap matches_importer` to do RANSAC
   essential-matrix verification per pair (drops bad matches).
5. `run_triangulation` — runs `colmap point_triangulator` with **bundle
   adjustment of intrinsics & extrinsics disabled** (`--Mapper.ba_refine_*=0`):
   poses come from the iOS scanner / SLAM and we trust them. Triangulation
   produces `points3D.bin` where each point carries:
   - `xyz`
   - `image_ids` and `point2D_idxs` (the "track" — every (frame,
     keypoint_id) that observes it)

### [E] Post-optimization
`post_optimization.post_optimization`
(`src/KeypointFreeSfM/post_optimization/post_optimization.py:59`):

- **CoarseReconDataset** loads the coarse COLMAP model and assigns each
  3D point a "query" frame (greedy strategy in
  `coarse_recon_data.feature_track_assignment_strategy`). A track of length
  k yields k−1 pairs `(query_frame, ref_frame_i)` for re-matching.
- **MatchingPairData** + `fine_matcher` re-runs LoFTR for those pairs **at
  fine resolution** (the LoFTR fine head is enabled here). For each match
  the worker also samples and stores the **CNN feature vectors at the
  matched 2D location** on both sides — this is the source of truth for the
  per-3D-point descriptor.
  - Uses `sample_feature_from_featuremap`
    (`src/KeypointFreeSfM/loftr_for_sfm/utils/sample_feature_from_featuremap.py:28`)
    — `F.grid_sample` on the LoFTR backbone feature map at the keypoint
    coords, normalized to L2-unit-length if `norm_feature=True`.
  - Output: `fine_matches.pkl` with per-pair
    `{mkpts0_idx, mkpts1_idx, mkpts0_f, mkpts1_f, feature_c0, feature_c1, feature0, feature1}`.
    `*_c*` are coarse-stage descriptors (used as 3D-point inputs to the
    matcher's positional encoding); `*0`/`*1` are fine-stage descriptors.
- **Optimizer** (`optimizer/optimizer.py:17`) refines **only the depth (z)**
  of each 3D point in its query-frame camera coordinate. Variables:
  `depth` (per point). Constants: query/ref poses, intrinsics, coarse
  matches `mkpts0_c, mkpts1_c`, fine match `mkpts1_f`.
  Residual function: `optimizer/residual.py:depth_residual` —
  geometry/reprojection error between the query point lifted by `depth`
  and projected into ref frame, vs. `mkpts1_f`. Solver is
  `submodules/DeepLM` (Levenberg-Marquardt, second-order); falls back to
  `first_order_solver.FirstOrderSolve` (Adam, 1000 steps) if DeepLM fails
  to import.
- **`update_optimize_results_to_colmap`** writes refined depths back into
  the COLMAP model (`sfm_ws/model/`).
- **`feature_aggregation_and_update`**
  (`src/KeypointFreeSfM/post_optimization/feature_aggregation.py:10`) —
  for every 3D point, average all the per-pair feature vectors collected
  above into one descriptor per (frame, keypoint_id) observation, and
  overwrite the placeholder zeros in `feats-loftr.h5` (and the
  `_coarse` variant). After this, the H5 file's `descriptors` are real
  CNN features keyed to the COLMAP keypoint indices.

> Why depth-only? Poses come from SLAM and are trusted. With camera
> intrinsics + extrinsics fixed, refining one z per point is a vastly
> easier convex-ish problem than full BA, and it lets DeepLM/Adam converge
> reliably without drifting the global frame.

### [F] Postprocess (run.py:300)
This is where the runtime database is finalized:

1. **3D bbox crop** — `filter_points.filter_bbox`
   (`src/sfm_utils/postprocess/filter_points.py:172`). Reads
   `box3d_corners.txt` (the annotated object bbox in world coords), drops
   any 3D point outside it, and rewrites the COLMAP model to
   `model_filted_bbox/`. Skippable via `post_process.skip_bbox_filter`.

2. **Track-length cap** — `filter_tkl.get_tkl`
   (`src/sfm_utils/postprocess/filter_tkl.py:37`). Walks track-lengths
   from low → high; picks the smallest threshold T such that the count of
   points with track ≥ T is below `dataset.max_num_kp3d` (default 7000 for
   train, 15000 for the demo config). This is how OnePose++ keeps the 3D
   set bounded — the longer a track, the more views agree on it, the more
   likely it's a real, repeatable feature.

3. **`filter_points.filter_track_length`** — actually removes points below
   the threshold and returns `(xyzs, points_idxs)`.

4. **`filter_points.merge`** — pairwise distance matrix on remaining
   points; for any cluster within `dist_threshold = 1e-3` (1 mm assuming
   metric units), replace by their centroid and remember the mapping
   `new_idx → [old_idx, ...]`.

5. **`feature_process.get_kpt_ann`**
   (`src/sfm_utils/postprocess/feature_process.py:544`) — the descriptor
   averaging pass. This is run **twice**: once with `network.detection =
   loftr_coarse` against `feats-loftr_coarse.h5`, once with the fine
   descriptors. For each surviving 3D point:
   - `count_features` walks every image, finds 2D keypoints whose COLMAP
     `point3D_id` belongs to this 3D point (or any of the merged old ids),
     and pulls their descriptor vectors out of the H5 file.
   - `gather_3d_ann` concatenates all observing descriptors per point.
   - `mean_descriptors_and_scores` averages them → one D-dim vector per
     3D point.
   - Saves `anno_3d_average.npz` (`keypoints3d` N×3, `descriptors3d` D×N,
     `scores3d` N×1).
   - Plus per-image `anno_*.json` with the 2D-keypoints/descriptors and
     the 2-row `assign_matrix [kp2d_idx, kp3d_idx]` used for training
     supervision.

### What inference actually queries
At runtime (`demo.py` / `inference.py`), `OnePosePlusInferenceDataset`
reads `anno_3d_average.npz` (fine) and `anno_3d_average_coarse.npz`
(coarse). The matching network (`OnePosePlusModel.forward`) does:
- `kpt_3d_pos_encoding` mixes each point's xyz with its **coarse**
  descriptor — that's why both fine and coarse files exist.
- The transformer's **3D-side input is the fine descriptor**.
- The fine refinement window samples from the query image's CNN feature
  map and learns to land at the fine 3D descriptor's matching pixel.

So the answer to "where do the descriptors come from at inference?":
**they're the time-averaged LoFTR backbone features sampled at the 2D
projection of every 3D point across every mapping frame.** No new
descriptors are computed for the 3D side at runtime.

---

## 3. How to get a cleaner point cloud / 3D database

Tuning ranked by impact, with the file/config knob in each case.

### 3.1 Better mapping data (biggest lever, no code change)
- **Cover more views, more uniformly.** The track-length filter directly
  rewards points seen from many angles. Pan slowly, orbit fully, vary the
  elevation. Avoid stopping and rapid-rotating in place.
- **Stable lighting, low motion blur, sharp focus.** LoFTR matches degrade
  fast on blur — those matches survive RANSAC but produce drifty 3D
  points.
- **Check the iOS-scan poses.** Bad ARKit drift → bad triangulation. If
  available, prefer `poses_ba/` (BA-refined) over `poses/`. Currently the
  pipeline already prefers `poses_ba`/`intrin_ba` when present
  (`generate_empty.py:64-78`).
- **Tighten the annotated 3D bbox** in `box3d_corners.txt`. The bbox
  filter (`filter_points.filter_bbox`) is the single hardest cleanup
  step — every stray background point inside the bbox stays, every real
  point outside it dies. Re-tightening the bbox to the actual object
  silhouette is by far the cheapest cleanup.

### 3.2 Pair-generation knobs (`configs/preprocess/sfm_*.yaml`, `sfm:` block)
- `gen_cov_from`: switch from `index` to `pose` for pose-aware pairs.
  Demo already does (`sfm_demo.yaml:29`).
- `min_rotation` (default 10°): raise it to force more diverse pairs;
  lower it for slow scans.
- `covis_num`: higher → more pairs, denser tracks, slower SfM. The demo
  uses 10. 15–20 helps recall on textureless objects.
- `down_ratio` (default 5 in demo): every Nth frame is used. Lower N → use
  more frames → richer tracks but slower. If your scan is short, try 2–3.

### 3.3 LoFTR matcher (in `coarse_match/coarse_match.py:cfgs`)
- The default LoFTR weight is `weight/LoFTR_wsize9.ckpt`. If you have a
  domain-specialized LoFTR (e.g., low-texture, indoor) point
  `cfgs["matcher"]["model"]["weight_path"]` at it.
- `cfgs["data"]["img_resize"]: None` keeps the original resolution for
  matching — don't reduce it; it kills fine descriptor quality.

### 3.4 COLMAP triangulation (`src/sfm_utils/triangulation.py`)
- `min_match_score` — currently `None` (off). If you turn on geometric
  verification with a score threshold (see `import_matches`), more dubious
  matches get dropped before triangulation. Note OnePose++ already
  bypasses scoring for LoFTR matches because all LoFTR matches are
  treated as valid — you can post-filter by `mconf` in
  `coarse_match.py` instead.
- `Mapper.ba_refine_focal_length=0` keeps intrinsics fixed. Only flip
  this on if you don't trust the iOS intrinsics.

### 3.5 Post-optimization (`run.py: cfg.post_optim` and
`post_optimization.py:cfgs`)
- `solver_type: SecondOrder` (DeepLM) is more accurate than `FirstOrder`.
  Make sure DeepLM installed correctly per README; otherwise the code
  silently falls back.
- `optimizer.optimize_lr.depth: 0.03` for first-order. Lower it (1e-2)
  for fragile scenes.
- `feature_aggregation_method: "avg"` is the only option implemented. A
  trimmed mean would be more robust but you'd need to add it.

### 3.6 Postprocess (`run.py: cfg.dataset` and `cfg.post_process`)
- `dataset.max_num_kp3d` — caps point count. Lower it (3000–5000) →
  fewer, higher-quality (longer-track) points. Raises pose-estimation
  recall on small or low-texture objects but each remaining point is
  more reliable.
- `post_process.skip_bbox_filter: False` — never skip this in production.
- `filter_points.merge`'s `dist_threshold = 1e-3`. Increase (1e-2 = 1 cm)
  for chunkier objects, decrease for tiny precision parts. Tighter merge
  → more points; looser merge → more averaging per point.

### 3.7 Direct surgical cleanups (you'd modify code)
- **Reproj-error filter.** After triangulation COLMAP records each track's
  reprojection error. Add a filter in `postprocess()` that drops tracks
  with mean reproj > τ (e.g. 2 px). Pattern:
  ```python
  err_ok = [pt for pt in points3D.values() if pt.error < 2.0]
  ```
- **Per-track descriptor variance filter.** In
  `feature_process.gather_3d_ann`, also compute the std of the stacked
  descriptors. Drop or downweight points whose descriptors disagree
  across views (likely matched onto different surfaces) — those are the
  ones that hurt PnP at inference.
- **View-angle diversity filter.** A long track from cameras that all
  look from one angle is a cluster of duplicates, not a multi-view
  consensus. After `count_features`, compute the angular span of viewing
  rays per point and drop points with span < 20–30°.
- **Statistical outlier removal (Open3D).** After `merge`, run
  `pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)` and
  filter `xyzs / points_idxs` accordingly. Cheap, surprisingly effective
  on background bleed.
- **Downsample to a target density.** Voxel-downsample with Open3D before
  the track-length cap, so the cap doesn't preferentially keep one dense
  region over a sparse but valid region.

### 3.8 Inference-side levers
Even with the same database, at runtime you can:
- **Lower `cfg.datamodule.shape3d_val`** — random-pads to N points; if
  your real cloud is much smaller than 7000–15000, the pad is mostly
  noise. Match it to the cloud's actual size.
- **Tune `coarse_matching.thr`** in the model config — the default 0.1
  works for trained checkpoints. Raising 0.15 for noisy scenes drops
  weaker 2D-3D matches before PnP.
- **Tighter 2D detector.** `LocalFeatureObjectDetector` seeds the first
  frame's bbox via LoFTR-affine-fit. Raise `n_ref_view` (default 15) for
  more robust seeding on the first frame; bad seeding is the most common
  cause of demo-time failure.

---

## 4. What's where, quick reference

| Concern | File:line |
|---|---|
| Pair selection | `src/sfm_utils/pairs_from_*.py` |
| LoFTR coarse matching | `src/KeypointFreeSfM/coarse_match/coarse_match.py:35` |
| H5 keypoint/match writing | `src/KeypointFreeSfM/coarse_match/coarse_match.py:188-214` |
| Empty COLMAP model | `src/sfm_utils/generate_empty.py:113` |
| COLMAP DB + triangulation | `src/sfm_utils/triangulation.py:213` |
| Coarse-to-fine pair re-matching | `src/KeypointFreeSfM/post_optimization/post_optimization.py:108-122` |
| CNN feature sampling at keypoints | `src/KeypointFreeSfM/loftr_for_sfm/utils/sample_feature_from_featuremap.py:28` |
| Depth-only optimizer | `src/KeypointFreeSfM/post_optimization/optimizer/optimizer.py:17` |
| Residual function | `src/KeypointFreeSfM/post_optimization/optimizer/residual.py` |
| Feature averaging into H5 | `src/KeypointFreeSfM/post_optimization/feature_aggregation.py:10` |
| 3D bbox crop | `src/sfm_utils/postprocess/filter_points.py:172` |
| Track-length threshold | `src/sfm_utils/postprocess/filter_tkl.py:37` |
| Near-duplicate point merge | `src/sfm_utils/postprocess/filter_points.py:265` |
| Per-3D-point descriptor avg | `src/sfm_utils/postprocess/feature_process.py:544` |
| Final on-disk format | `anno/anno_3d_average.npz` (`keypoints3d`, `descriptors3d`, `scores3d`) |

## 5. Mental model in one paragraph

OnePose++ uses **LoFTR as a dense matcher** to invent keypoints (not
detected — emergent from where pairs agree), runs **COLMAP triangulation
with poses fixed** to lift those agreements to 3D points, then runs a
**fine-resolution re-match per track** to (a) refine each point's depth
and (b) collect the LoFTR-backbone CNN features at every observation. The
**average of those features per 3D point is the descriptor** the inference
matcher queries. Cleaning the cloud means cleaning *what gets considered a
real, repeatable, geometrically-consistent observation* — done via 3D
bbox, track-length cap, near-duplicate merge, and (highest leverage) the
quality of the input scan itself.
