# Architecture (this project)

For the *full* design rationale and 6-week roadmap, see
[`path_b_implementation_roadmap.md`](./path_b_implementation_roadmap.md).
This file is the **TL;DR for the current project's specific
architecture decisions**.

## Goal
Real-time (≥30 FPS) 6-DoF object pose tracking on iOS + Android.

## The pipeline at a glance

```
OFFLINE (Python, per object)              ONLINE (Swift/Kotlin, per frame)
─────────────────────────────             ────────────────────────────────
COLMAP/SfM workspace                      camera frame
   ↓                                          ↓
sample K=20-50 ref views                  preprocess (vImage → 640×480 RGB)
   ↓                                          ↓
ALIKED on each ref                        ALIKED inference (ONNX RT)
   ↓                                          ↓
map kpts → 3D pts                         retrieve top-K' refs (or skip if K≤6)
   ↓                                          ↓
(opt) MobileNet retrieval emb             LightGlue × K' (ONNX RT)
   ↓                                          ↓
.bundle file (5-20 MB)                    2D-3D pairs (lookup ref→3D)
                                              ↓
                                          EPnP-RANSAC (pure native)
                                              ↓
                                          pose smoothing + state machine
                                              ↓
                                          render 3D bbox
```

## Stack decisions

| Layer | Choice | Alternative considered |
|---|---|---|
| Inference engine | **ONNX Runtime Mobile** | Core ML (iOS-only); TFLite (worse on iOS) |
| Local features | **ALIKED** | XFeat (lighter but worse quality); LoFTR (heavier) |
| Matcher | **LightGlue** | SuperGlue (deprecated); raw cosine NN (worse) |
| Retrieval | **MobileNetV3 penultimate** | DINOv2 (heavier, only worth it at K>50); skip if K≤6 |
| PnP | **Pure native EPnP** | OpenCV solvePnPRansac (~200 MB binary) |
| 3D database | **Custom binary `.bundle`** | `.npz` (slow to load); FlatBuffers (overkill) |
| Camera/render | Native per platform | Cross-platform engines like Unity (overkill) |

See `comparison_kaggle_pipeline.md` for why ALIKED+LightGlue and not
OnePose++'s custom 2D-3D matcher.

## Cross-platform strategy

- **One ONNX model per matcher** (ALIKED.onnx, LightGlue.onnx, optionally
  MobileNet.onnx) shipped to both platforms
- **One `.bundle` file format** parsed by both Swift and Kotlin
  (custom binary, ~150 LOC parser per platform)
- **One algorithm spec per stage** (`shared/algorithms/*.md`) implemented
  twice in native code
- **Native UI per platform** (SwiftUI on iOS, Compose on Android) — UI
  is too platform-specific to share

## Performance budget

iPhone 15 Pro target (steady tracking):

| Stage | Time | Cum |
|---|---|---|
| Preprocess (vImage SIMD) | 2 ms | 2 |
| ALIKED ONNX (ANE) | 12 ms | 14 |
| LightGlue × 2 refs | 10 ms | 24 |
| 2D-3D pair assembly | <1 ms | 24 |
| EPnP-RANSAC adaptive | 3 ms | 27 |
| Smoothing + render | 2 ms | **29 ms = 34 FPS** |

Mid-range Android (Pixel 7a, no NPU): ~2× these numbers. Frame-skip
+ pose extrapolation still gives smooth on-screen 30 FPS visually.

## Decisions that surprised us during XFeat prototype

These shaped current decisions:

- **FP32 throughout, not FP16.** FP16 cosine similarity drift (~2-4%)
  pushed correct matches below threshold. Stay FP32.
- **Skip Core Image; use vImage directly** — way faster preprocessing.
- **PROSAC + adaptive RANSAC iterations** — single biggest perf win
  (319 ms → ~0 ms on easy cases) by stopping early on high inlier ratios.
- **`cblas_sgemm` for cosine sim** — `[[Float]]` array-of-arrays is
  shockingly slow in Swift (~3800 ms); flat arrays + Accelerate matrix
  multiply is ~5 ms.
- **Don't render via Core Animation transforms with Float16 inputs** —
  CATransform3D from FP16 homography lost precision. Compute in FP32,
  cast to CATransform3D at the end.

## Open architectural questions

These don't block Phase 1 but need answers before Phase 3:

1. **Single-object app or multi-object catalog?** Affects retrieval
   strategy and bundle storage.
2. **Bundle generation on device or cloud-only?** On-device requires
   shipping ALIKED + LightGlue weights AND running SfM on phone — much
   harder.
3. **iOS minimum target?** iOS 17+ for clean ANE FP16; iOS 15 lowest
   sane for ONNX Runtime.
4. **Camera intrinsics availability?** If targeting non-Pro iPhones,
   we need fallback intrinsics in the bundle's "default K" field.
