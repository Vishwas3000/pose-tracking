# OnePose++ vs DINOv2 + ALIKED + LightGlue (Kaggle baseline)

A common Kaggle Image Matching Challenge baseline pipeline is
**DINOv2 + ALIKED + LightGlue**. People sometimes wonder if it's an
alternative to OnePose++ — it's not, they solve different problems at
different levels of the vision stack. Worth understanding both so you
can borrow pieces from either.

Reference notebook:
https://www.kaggle.com/code/octaviograu/baseline-dinov2-aliked-lightglue

---

## TL;DR

| | OnePose++ | DINOv2 + ALIKED + LightGlue |
|---|---|---|
| **Problem** | Where is THIS object in 3D? (6-DoF pose) | Which images match? Which pixels match? (2D matching) |
| **Output** | 4×4 pose matrix `R\|t` per frame | List of 2D↔2D pixel pairs per image pair |
| **3D involved?** | Yes — pre-built 3D database per object | No — purely 2D matching |
| **Reference data** | One scanned object's `.npz` with 3D points | Pool of unstructured images |
| **Use case** | AR overlays, robotics grasping, object tracking | SfM frontend, visual localization, image retrieval |
| **Layer of stack** | Object pose estimation | Feature matching / image correspondence |

**They are not competitors.** The Kaggle pipeline produces the *input*
to systems like SfM. OnePose++ *consumes* the output of an SfM (which
itself can use a Kaggle-style frontend) to do pose estimation.

---

## What the Kaggle pipeline does

Three-stage hierarchical pipeline for the IMC (Image Matching Challenge):

```
[ pool of N images, no structure ]
            │
            ▼
   ┌─────────────────────┐
   │ 1. DINOv2           │   Global descriptor per image (~768-D)
   │   (Meta, foundation │   Cluster similar images.
   │    visual model)    │   Output: pairs to match further.
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐
   │ 2. ALIKED           │   Detector-based: finds sparse keypoints,
   │   (lightweight CNN  │   computes per-keypoint descriptors (~128-D)
   │    detector +       │   Run on each image individually.
   │    descriptor)      │
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐
   │ 3. LightGlue        │   Graph attention matcher.
   │   (faster successor │   For each pair, matches ALIKED descriptors.
   │    to SuperGlue)    │   Output: 2D-2D pixel correspondences.
   └─────────────────────┘
            │
            ▼
   [ matched pixel pairs per image pair ]
            │
            │   (sometimes followed by COLMAP / hloc for SfM,
            │    or used directly to score matches)
            ▼
```

Each component:
- **DINOv2** — Meta's foundation model, used here to compute one global
  feature vector per image. Cosine similarity between vectors → pairs of
  similar images. Replaces hand-crafted "covisibility heuristics".
- **ALIKED** — Detector-based learned local features. Lightweight, fast.
  Outputs sparse keypoint coords + 128-D descriptors per keypoint.
- **LightGlue** — Faster, attention-based learned matcher. For two ALIKED
  feature sets, outputs which keypoints correspond.

Pipeline output: just 2D-2D matches. To get poses or 3D structure,
you'd plug the matches into COLMAP / hloc / your own bundler.

---

## What OnePose++ does

A **complete object pose system**, end-to-end:

```
[ mapping video of object + ARKit poses + 3D bbox ]
            │
            ▼
   SfM RECONSTRUCTION (offline, per object)
   ─────────────────────────────────────────
   • Pair generation (covisibility heuristic — could be DINOv2!)
   • LoFTR pairwise matching (detector-FREE)
   • COLMAP triangulation
   • Depth refinement + LoFTR feature sampling per 3D point
   • Per-3D-point descriptor averaging
            │
            ▼
   [ anno_3d_average.npz — 3D points + descriptors + bbox ]
            │
            │   (training is a separate one-time pipeline that
            │    teaches a custom 3D-2D matcher to use these npz files)
            ▼
   INFERENCE (per frame, runtime)
   ──────────────────────────────
   • Crop image to 2D bbox (from prev pose or LoFTR detection)
   • OnePose++ matcher (custom transformer, 3D-2D)
   • PnP-RANSAC → 6-DoF pose (R, t)
   • Project bbox corners → render
            │
            ▼
   [ R, t pose matrix; 3D bbox drawn on frame ]
```

OnePose++ is end-to-end pose estimation. The Kaggle pipeline is
just the "match pixels" stage that lives **inside** any system like
this.

---

## Where the two pipelines could overlap (component reuse)

You CAN swap pieces between them. Here's where:

### Step B in OnePose++ (pair generation) ↔ DINOv2 retrieval

OnePose++ today picks pairs via simple heuristics (`pose` / `index` /
`exhaustive`). DINOv2 retrieval would be a **strict upgrade** — global
visual similarity is more discriminative than pose-distance heuristics.

**When it matters**: large mapping videos with redundant views. DINOv2
clusters near-duplicates better than the temporal index gap, giving
LoFTR more diverse pairs to work with.

### Step C in OnePose++ (LoFTR) ↔ ALIKED + LightGlue

This is the interesting one. Two paradigms:

| | LoFTR (detector-free) | ALIKED+LightGlue (detector-based) |
|---|---|---|
| Speed | ~100 ms/pair | ~25 ms/pair |
| Density | Dense — matches at every coarse cell | Sparse — only at detected keypoints |
| Textureless content | Strong (it was designed for this) | Weak — no keypoints found, no matches |
| Repetitive content | Weak (similar features → false matches) | Decent — keypoints help disambiguate |
| Mobile-friendly | Hard — large model | Easier — much smaller |

**For the OnePose++ use case** (objects scanned for pose tracking):
- If your objects have **uniform texture / drawings / signs** → keep LoFTR (or its mobile-friendly successor, EfficientLoFTR)
- If your objects are **textured products with distinct features** → ALIKED+LightGlue is a faster swap

### Step Q in OnePose++ (PnP-RANSAC)

Both pipelines need this when used downstream. Kaggle stops at 2D
matches, so you'd add COLMAP / PnP yourself. OnePose++ already has
this baked in.

### Step P in OnePose++ (the custom 3D-2D matcher)

**Cannot be replaced by Kaggle's pipeline.** The Kaggle pipeline does
2D-2D matching only. OnePose++'s 3D-2D matcher is a different network
that takes 3D points as one of its inputs — there's no off-the-shelf
equivalent in the IMC ecosystem.

If you want to use Kaggle-style components only, your runtime would be:
1. Detect 2D keypoints in the query (ALIKED)
2. Match against 2D keypoints in stored reference views (LightGlue)
3. Each reference view's 2D keypoints have known 3D points (from SfM)
4. Use those 2D-3D correspondences for PnP

This is the **classic visual localization pipeline** (e.g., hloc), and
it's a viable alternative to OnePose++'s trained 3D-2D matcher.

---

## Could you build OnePose++ functionality with just Kaggle components?

Yes — this is essentially what classical visual localization
frameworks like **hloc** do:

```
                    Kaggle pipeline replaces:
                    ────────────────────────
SfM mapping  ──→    DINOv2 retrieval (B)
                    +
                    ALIKED + LightGlue (C)
                    +
                    COLMAP triangulation (D, E)

Inference    ──→    Image retrieval via DINOv2 (find closest reference)
                    +
                    ALIKED + LightGlue (match against retrieved)
                    +
                    Lookup 3D points from SfM
                    +
                    PnP-RANSAC
```

**Trade-offs vs OnePose++**:

| | hloc-style (DINOv2 + ALIKED + LightGlue + PnP) | OnePose++ |
|---|---|---|
| Components | All off-the-shelf | One trained network (the 3D-2D matcher) |
| Quality on textureless objects | Worse (sparse detector misses features) | Better (detector-free LoFTR) |
| Engineering effort | Lower (no custom training) | Higher (train the 3D-2D matcher) |
| Mobile deployment | Multiple models to ship + convert | One model + .npz |
| Database size | Larger (per-view keypoint sets, 1 set per ref view) | Smaller (averaged per 3D point) |
| Pose accuracy | Depends on reference view density | Often better (custom-trained) |

For **mobile cross-platform** specifically, the hloc-style approach
is a serious alternative because:
- DINOv2 is overkill on mobile (huge), but you can replace it with a
  smaller image-similarity model (or just brute-force match against
  all reference views)
- ALIKED is mobile-friendly (smaller than LoFTR or OnePose++ matcher)
- LightGlue is small and fast
- All components have open-source weights

---

## Practical recommendation for your situation

You've been building XFeat-based mobile tracking. Here's how the
Kaggle pipeline maps to your work:

| Your XFeat path | Kaggle equivalent |
|---|---|
| XFeat (detector-based, 64-D) | ALIKED + LightGlue |
| Cosine NN matching | LightGlue (better, smarter graph attention) |
| `.npz` from custom SfM | hloc-style database |

**XFeat is in the same family as ALIKED+LightGlue.** They're all
detector-based, mobile-friendly, learned local features. So your
existing XFeat work is closer to the Kaggle pipeline than to the
OnePose++ pipeline.

If you decide to migrate up the quality ladder:
- **Easy upgrade**: replace XFeat with **ALIKED + LightGlue** (more
  accurate, slightly slower). Drop-in for your existing matching
  funnel.
- **Different approach**: train a OnePose++-style 3D-2D matcher (better
  on textureless / repetitive content but ~30 MB model, requires
  training infrastructure).
- **Cloud retrieval**: add **DINOv2-based image retrieval** if you
  have many objects and need to identify which one is in frame before
  matching.

---

## Summary

Don't think of these as competing. They live at different layers:

- **DINOv2 + ALIKED + LightGlue** is a *2D feature matching pipeline*.
  It's a building block. It produces pixel correspondences.
- **OnePose++** is a *6-DoF object pose estimation system*. It uses
  feature matching internally (LoFTR) but has additional layers above
  it (3D structure, PnP, tracking state machine).

The Kaggle pipeline is closer to OnePose++'s **Step B + C** (pair
generation + coarse matching). It's not a replacement for the whole
OnePose++ system — it's an alternative for the matching frontend, with
COLMAP / PnP / etc. needed downstream to actually produce poses.

If your end goal is 6-DoF object tracking on mobile, you need a
*system* like OnePose++ or hloc — not just a matching pipeline like
the Kaggle baseline.
