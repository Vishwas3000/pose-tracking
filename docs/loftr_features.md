# LoFTR & "LoFTR features"

What LoFTR is, what people mean when they say "LoFTR features", and how
OnePose++ uses both. Pairs with `doc/sfm_and_descriptors.md` (which covers
how OnePose++ converts LoFTR matches into a 3D database).

---

## 1. What LoFTR is (the paper)

**LoFTR** = *Local Feature TRansformer* — a 2021 CVPR paper (Sun et al.,
ZJU3DV) for image feature matching. The headline idea: do **detector-free,
dense matching** between two images using a CNN + Transformer, instead of
the classical "detect keypoints → describe → nearest-neighbor match"
pipeline (SIFT, ORB, SuperPoint+SuperGlue).

### Why "detector-free" matters

Classical pipelines fail on **low-texture surfaces** (white walls, plain
mugs, polished metal) because there are no salient corners to detect. If
the detector finds nothing, the descriptor has nothing to describe.

LoFTR skips detection entirely. It produces a dense feature map for each
image, lets every position in image A attend to every position in image B
through a Transformer, and reads matches off the resulting similarity
matrix. Salience is *emergent* — match locations are wherever the network
decides two images agree, even on smooth surfaces.

This is exactly why OnePose++ adopted LoFTR: object-pose estimation needs
to work on textureless objects (the `OnePose_LowTexture` dataset is built
around this).

### Architecture (4 stages)

```
   image0           image1
     │                │
     ▼                ▼
 ResNet-FPN       ResNet-FPN          ← shared CNN backbone
  ↓     ↓          ↓     ↓
 fc0   ff0        fc1   ff1           ← coarse (1/8) + fine (1/2) features
  │     │          │     │
  └─────┴──┐    ┌──┴─────┘
           ▼    ▼
    Coarse Transformer                ← interleaved self/cross attention
    (linear attention, ~4 layers)        on flattened coarse features
           │
           ▼
  Coarse matching (dual-softmax)       ← mutual NN over similarity matrix
           │
           ▼  for each coarse match, crop a 5×5 fine patch on each side
    Fine Transformer
           │
           ▼
  Fine matching (DSNT)                 ← sub-pixel via spatial-expectation
           │
           ▼
   final correspondences (mkpts0_f, mkpts1_f, mconf)
```

Implementation in this repo: `submodules/LoFTR/src/loftr/` is the upstream
LoFTR; OnePose++ wraps it in
`src/KeypointFreeSfM/loftr_for_sfm/loftr.py:LoFTR_for_OnePose_Plus`.

### Output

For an image pair, LoFTR returns:
- `mkpts0_f`, `mkpts1_f` — matched 2D coordinates, sub-pixel, in the two
  images.
- `mconf` — match confidence per pair.

That's it. There are **no keypoint locations as a primary output** — the
matched coordinates *are* the keypoints. Different pairs of images
involving the same image will produce different "keypoints" because the
matcher is trained pairwise, not as a detector.

---

## 2. What "LoFTR features" usually means

Two different things, depending on context:

### Meaning A — "matches produced by LoFTR"
When a paper says "we use LoFTR features for SfM", they usually mean **the
matched 2D point pairs** that LoFTR outputs. In the OnePose++ pipeline,
this is what flows into COLMAP at triangulation time. The HLoc-style
`feats-*.h5` and `matches-*.h5` files are LoFTR matches reshaped into a
COLMAP-friendly database (see `doc/sfm_and_descriptors.md` step B).

### Meaning B — "CNN feature vectors at LoFTR-matched locations"
A more specific use: take LoFTR's CNN backbone feature map (the same one
the matcher attends over) and **sample its values at matched 2D
coordinates**. That gives you a per-match descriptor vector — typically
256-D for the coarse map, 128-D for the fine map.

This is what OnePose++ stores as the per-3D-point descriptor.
`src/KeypointFreeSfM/loftr_for_sfm/utils/sample_feature_from_featuremap.py:28`
is the function that does the sampling — `F.grid_sample` on the feature
map at the keypoint coordinates, optionally L2-normalized.

So in OnePose++ specifically, **"LoFTR features" = CNN-feature-map values
sampled at LoFTR-matched 2D pixels**. They are:

- 256-D **coarse** features → `descriptors3d` in
  `anno_3d_average_coarse.npz`. Used by the matcher's keypoint encoder
  (mixed with the 3D xyz via an MLP) as the *positional encoding* of each
  3D point.
- 128-D **fine** features → `descriptors3d` in `anno_3d_average.npz`. Used
  as the *content* descriptor that the inference matcher actually scores
  against the query image's CNN features.

---

## 3. How OnePose++ uses LoFTR features

The repo uses LoFTR in **three** distinct places, often confused:

### 3.1 SfM — building the database (`run.py`)
- Pretrained LoFTR (`weight/LoFTR_wsize9.ckpt`) matches every covisible
  pair of mapping frames (`coarse_match.py`).
- Same LoFTR re-matches each track at fine resolution during
  post-optimization (`post_optimization.py`).
- During fine re-match, the worker also samples the **256-D coarse** and
  **128-D fine** CNN features at every matched location and stores them in
  `fine_matches.pkl` (`feature_aggregation.py`). These are then averaged
  per 3D point and saved as `anno_3d_average*.npz`.
- This LoFTR is **frozen** — it is never trained inside OnePose++. It's
  just used as a feature extractor and matcher.

### 3.2 First-frame object detection (`demo.py`)
- `LocalFeatureObjectDetector` (`src/local_feature_object_detector/local_feature_2D_detector.py`)
  also runs LoFTR — between the live query frame and ~15 reference frames
  from the SfM workspace. Matches → affine RANSAC → 2D bbox for the
  cropping step. Same checkpoint, same features, totally separate purpose.

### 3.3 OnePose++ matching network — *inspired by* LoFTR
The actual 2D-3D matcher (`src/models/OnePosePlus/OnePosePlusModel.py`) is
**not** LoFTR. It's a LoFTR-shaped network (ResNet-FPN backbone + coarse
self/cross transformer + fine refinement window + DSNT) but it operates
between **3D points** (descriptors as one sequence) and **2D feature map
positions** (the other sequence). The backbone weights are initialized
from the same LoFTR checkpoint (see `train.yaml: loftr_backbone.pretrained`)
and then trained for the 2D-3D matching task.

**Important distinction:**
- LoFTR (3.1, 3.2) → **2D ↔ 2D** pretrained matcher used as a tool.
- OnePose++ matcher (3.3) → **3D ↔ 2D**, trained from scratch on the
  OnePose dataset (initialized from LoFTR's CNN weights).

---

## 4. Why LoFTR was the right pick for object pose

| Property | Classical (SIFT/SuperPoint) | LoFTR |
|---|---|---|
| Works on textureless surfaces | poorly | yes |
| Repeatable keypoints | yes (detector finds same point) | not really — emergent |
| Sub-pixel accuracy | yes | yes (via DSNT) |
| Speed | fast | slower (Transformer) |
| Good for SfM with fixed poses | yes | yes |
| Good for general SLAM (no priors) | yes | harder (no fixed keypoints across pairs) |

Object pose is a **closed-set** problem — the object's mapping video gives
you a fixed gallery of frames with known poses. LoFTR's lack of repeatable
keypoints across pairs is solved by COLMAP's triangulation: every
LoFTR-matched point becomes a track of (frame, pixel) observations, and
triangulation forces all those observations to agree on one 3D location.
After that, **the 3D point IS the stable identifier** that LoFTR couldn't
provide on its own — and the per-frame LoFTR descriptors get averaged
into a single per-3D-point descriptor. That averaging is what makes
inference-time matching tractable.

---

## 5. Quick reference

| Question | Answer |
|---|---|
| Which file is the LoFTR wrapper? | `src/KeypointFreeSfM/loftr_for_sfm/loftr.py` |
| LoFTR config? | `src/KeypointFreeSfM/loftr_for_sfm/utils/loftr_for_onepose_plus_cfg.py` |
| Which checkpoint is loaded? | `weight/LoFTR_wsize9.ckpt` (referenced from `coarse_match.py`, `local_feature_2D_detector.py`, `post_optimization.py`) |
| Who samples features from the map? | `src/KeypointFreeSfM/loftr_for_sfm/utils/sample_feature_from_featuremap.py:28` |
| Coarse feature dim? | 256 (LoFTR coarse `d_model`) |
| Fine feature dim? | 128 (LoFTR fine `d_model`) |
| Are they L2-normalized in the npz? | They are stored as raw averages; normalization happens inside the OnePose++ matcher (`feat_norm_method: sqrt_feat_dim`). |
| Are LoFTR weights trained during OnePose++ training? | Only if `loftr_backbone.pretrained_fix: False` (the default in `train.yaml`). Set `True` to freeze. |

---

## 6. One-paragraph mental model

LoFTR replaces "detect keypoints, then describe them" with "let a CNN
make dense feature maps and let a transformer decide where the maps
agree". The matched pixel pairs *are* the keypoints, and the CNN feature
vectors at those pixels are the descriptors. OnePose++ runs LoFTR
pairwise across a mapping video, lets COLMAP triangulate the matches into
3D points (each point's track is the set of frames that observed it),
samples LoFTR's CNN features at every observation, averages them per 3D
point, and stores the result. At inference, the OnePose++ matcher (a
LoFTR-shaped 3D↔2D network, not LoFTR itself) compares those stored
per-point feature vectors against a fresh CNN encoding of the query image
and lets PnP do the rest.
