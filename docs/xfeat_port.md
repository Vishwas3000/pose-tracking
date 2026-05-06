# XFeat Port — How It Fits Together

This document explains the XFeat-based pose-tracking pipeline as it stands
on the `xfeat-switch` branch. Read this before touching any of the files
listed below.

## TL;DR

We replaced ALIKED with **XFeat** (`verlab/accelerated_features`, CVPR 2024)
because ALIKED's deformable convolutions don't lower to Core ML, leaving
iOS stuck on a 340 ms / frame CPU path. XFeat is a plain conv net with no
DCN, no `grid_sample`, no fancy ops — converts cleanly to a single
`mlprogram` graph and runs end-to-end on the Apple Neural Engine.

End-to-end latency on iPhone with the new pipeline:

| Stage | Cost |
|---|---|
| XFeat (Core ML, ANE) | ~5 ms |
| Decode + NMS + bicubic descriptor sample (Swift) | ~6 ms |
| Brute-force matcher (`Matcher.swift`, BLAS) | ~5–15 ms |
| EPnP + RANSAC + iterative refinement (`EPnP.swift`, `RANSAC.swift`) | ~3 ms |
| **Total** | **~20–30 ms** |

vs ~340 ms with ALIKED on CPU.

## The five things you must keep aligned

These are the parts where the iOS Swift side, the offline Python side, and
the Core ML model have to agree. A mismatch on any of them produces
**zero matches** at runtime even though detection looks fine.

### 1. Input canvas: 640×640 letterbox, RGB CHW, [0, 1]

- `offline/tools/xfeat_inference.py:_letterbox_for_xfeat` — proportional
  resize + zero-pad to `DEFAULT_TARGET_SIZE = 640`.
- `ios/objectCapture/Inference/XFeatRunner.swift:preprocessBGRAToCHW` —
  same vImage path, same letterbox math.
- `offline/tools/_export_xfeat_coreml.py` — `--input-size 640` (default).

If you change one, change all three. The bundle is built using whatever
canvas the offline pipeline produces, so iOS *must* match it.

### 2. Descriptor sampling: bicubic, NOT bilinear

Upstream's `XFeat.detectAndCompute` uses
`InterpolateSparse2d('bicubic')` on the descriptor field `M1`. Bilinear
in iOS lost ~5% precision per descriptor → cosine sim for true matches
dropped from ~0.85 to ~0.72 → with `simMin=0.82` we got **zero matches**.

- Swift: `XFeatRunner.bicubicSample` — Catmull-Rom kernel (a = -0.5),
  4×4 neighborhood, edge-clamp. Mirrors PyTorch
  `F.grid_sample(mode='bicubic', align_corners=False)`.
- Python: `offline/tools/_xfeat/xfeat.py:37` — keep
  `InterpolateSparse2d('bicubic')`.

Don't "simplify" to bilinear without changing both ends.

### 3. Channel ordering: `pixel_unshuffle` ≡ `_unfold2d`

The Core ML export wrapper (`_export_xfeat_coreml._XFeatForExport`)
replaces upstream's `_unfold2d` (which uses `Tensor.unfold` and chokes
coremltools 9 on the `_int` op) with `F.pixel_unshuffle(x, 8)`. The two
are mathematically identical for the input we care about — channel
`c = i*8 + j` of the 64-D output corresponds to pixel offset `(i, j)`
within the 8×8 block. **Verified** by parity check between traced
wrapper and `_unfold2d`.

If you change downsampling factor or input size, recompute.

### 4. Coordinate convention: `simd_float3x3 K[col, row]`

Apple's simd matrix subscript is `K[column, row]`, not `[row, column]`.
This bit us hard early on:

- `K[0, 0]` = fx (col 0, row 0)
- `K[1, 1]` = fy (col 1, row 1)
- `K[2, 0]` = cx (col 2, row 0)
- `K[2, 1]` = cy (col 2, row 1)

`PoseTracker.scaledK` documents this in line comments. If you write
`K[0, 2]` for cx (the natural-looking thing) you'll silently zero out
cx every time.

### 5. EXIF rotation for still-image and live preview

iOS auto-rotates `UIImage` for display via `imageOrientation`, and the
preview layer has `videoOrientation = .portrait`. The pose math runs in
the buffer's raw sensor-orientation pixel space. So bbox 2-D coords
must be rotated to display orientation **before** scaling to view.

`InferenceViewController.applyDisplayRotation` handles all 8
`UIImage.Orientation` cases. `displayOrientation` is set:

- `.up` for gallery video (CIImage→UIImage(cgImage:) shows raw buffer).
- `.right` for live camera (preview connection `.portrait` ⇒ 90° CW).
- Picked image's actual `imageOrientation` for still images.

## File map

### Offline (Python — runs on Linux dev machine)

| File | Responsibility |
|---|---|
| `offline/tools/_xfeat/xfeat.py` | Vendored upstream wrapper (the model class). Patched to use relative imports. **Don't touch the bicubic interpolator.** |
| `offline/tools/_xfeat/{model,interpolator}.py` | Vendored verbatim. |
| `offline/tools/xfeat_inference.py` | `XFeatRunner` with the `extract(image_path)` interface that `build_object_bundle.py` calls. Letterbox to 640×640. |
| `offline/tools/build_object_bundle.py` | Phase O orchestrator. Uses `XFeatRunner`, runs 2-D NN to COLMAP-tracked keypoints, packs into `.bundle`. CLI arg is `--xfeat-model`. |
| `offline/tools/bundle_writer.py` | Format definition. `desc_dim` is auto-detected from `ref.descriptors.shape[1]`, so XFeat (64) and legacy ALIKED (128) bundles parse the same. |
| `offline/tools/_export_xfeat_coreml.py` | Convert PyTorch checkpoint to `xfeat.mlpackage`. Requires `minimum_deployment_target=ct.target.iOS17` (without it, coremltools 9 hits a `_int` cast bug on 0-D shape tensors). |
| `shared/models/xfeat.pt` | Upstream PyTorch checkpoint (~6 MB). |
| `shared/models/xfeat.mlpackage` | Core ML model, FP16 weights. **Source of truth** — copy this into the iOS app bundle for inference. |
| `shared/objects/session_1777549127.bundle` | Per-object database, 30 refs, ~7k keypoints, 64-D XFeat descriptors. |

### Online Python reference pipeline (mirrors what iOS does)

| File | Notes |
|---|---|
| `online/tools/pipeline.py` | `PoseTracker` orchestrator. Defaults: `ratio=0.9`, `sim_min=0.82`, `top_k=4096` for XFeat. |
| `online/tools/matcher.py` | Brute-force cosine + Lowe ratio + mutual NN, **vectorized**. One big sgemm against all refs concatenated, then per-ref slicing for mutual-NN. ~30 ms per call vs 2.7 s before vectorization. |
| `online/tools/bundle_loader.py` | Reads `desc_dim` from header, so descriptor-dim-agnostic. |
| `online/demo/{single_image,test_images,sweep_session,video,overlay_bbox,cross_check_solver}.py` | All renamed `--aliked` → `--xfeat`. |

### iOS

| File | Notes |
|---|---|
| `ios/objectCapture/Inference/XFeatRunner.swift` | The runner. Loads the `.mlpackage` (Core ML / ANE), runs vImage preprocess, decodes the `(M, K, H)` outputs in pure Swift. |
| `ios/objectCapture/Inference/PoseTracker.swift` | Per-frame orchestrator. XFeat → Matcher → RANSAC(EPnP). XFeat-tuned matcher params: `ratio=0.95`, `simMin=0.65` (looser than desktop's 0.82 because ANE FP16 noise drops cossim by ~5–8 %). Reproj threshold 8 px. |
| `ios/objectCapture/Inference/Matcher.swift` | cblas_sgemm-based brute-force matcher. Descriptor-dim-agnostic — reads `bundle.descDim` at runtime. |
| `ios/objectCapture/Inference/BundleLoader.swift` | Defines `struct ObjectBundle` (NOT `Bundle` — that name shadows `Foundation.Bundle` in the auto-generated CoreML wrapper for `xfeat.mlpackage`, which writes `Bundle(for: xfeat.self)`). |
| `ios/objectCapture/Inference/EPnP.swift` | True Lepetit/Moreno-Noguer EPnP. Descriptor-agnostic, **don't change**. |
| `ios/objectCapture/Inference/RANSAC.swift` | Adaptive RANSAC + iterative consensus refinement. Descriptor-agnostic. |
| `ios/objectCapture/Inference/InferenceViewController.swift` | Capture / picker / display path. EXIF-aware bbox rotation via `applyDisplayRotation`. |
| `ios/objectCapture/{xfeat.mlpackage,default.bundle}` | App resources. `default.bundle` is a copy of `shared/objects/session_1777549127.bundle` — re-copy after every bundle rebuild. |

## Build / rebuild commands

```bash
# Activate the project venv (CUDA-12 wheels for ALIKED-era ALIKED-on-GPU
# still work; XFeat uses torch directly).
source .venv/bin/activate

# Re-export Core ML model from the PyTorch checkpoint (run on Linux is fine,
# only the validation step requires macOS).
python -m offline.tools._export_xfeat_coreml \
    --weights shared/models/xfeat.pt \
    --out     shared/models/xfeat.mlpackage

# Rebuild the per-object bundle. Adjust kpt-match-px / top-k to taste —
# higher kpt-match-px gives more bundle keypoints (better recall, looser
# 3-D linkage); higher top-k gives more candidates per frame (more
# inliers, slower matcher).
python -m offline.tools.build_object_bundle \
    --colmap-dir    offline/data/workspace_session_1777549127/sparse/cropped_bin \
    --source-images offline/data/workspace_session_1777549127/images \
    --bbox          offline/data/workspace_session_1777549127/bounds.json \
    --xfeat-model   shared/models/xfeat.pt \
    --num-refs      30 \
    --kpt-match-px  6.0 \
    --xfeat-top-k   4096 \
    --out           shared/objects/session_1777549127.bundle

# Copy bundle + model into iOS app resources.
cp shared/objects/session_1777549127.bundle  ios/objectCapture/default.bundle
cp -r shared/models/xfeat.mlpackage          ios/objectCapture/xfeat.mlpackage

# Sanity-check on the test images set.
python -m online.demo.test_images \
    --bundle  shared/objects/session_1777549127.bundle \
    --xfeat   shared/models/xfeat.pt \
    --in-dir  offline/data/test_images \
    --out-dir offline/data/test_images_infered_xfeat \
    --ref-size 1920x1440
```

## CRITICAL: Export the Core ML model with `compute_precision=FLOAT32`

The model is exported with FP32 compute. **Don't switch it to FP16** without
reading this section — both FP16 attempts on this graph collapsed iOS
descriptor cossim to <0.60 against the bundle (i.e., 0 matches at any
threshold). The next debugger will be tempted to "save 14 ms" by going
FP16; the experiments below show why that's a research project, not a knob
flip.

### FP16 ANE — both attempts failed

**Attempt 1: in-model InstanceNorm.** Just set
`compute_precision=ct.precision.FLOAT16`. Keep the upstream
`forward()` (RGB → mean → InstanceNorm → conv stack).
*Result:* matchDiag histogram showed `mean-best-cossim=0.40,
global-max=0.58, 94% of queries < 0.50`. Pose pipeline received zero
matches.
*Hypothesis at the time:* FP16 InstanceNorm overflows when reducing
640×640 = 409,600 pixel values in [0, 1] (sum reaches ~205,000 vs FP16
max 65,504). Plausible but unconfirmed.

**Attempt 2: move InstanceNorm out of the model into Swift FP32.** Export
wrapper changed to skip `x.mean(dim=1)` and `self.norm`; input becomes
(1, 1, H, W) already-normalized. Swift code computes grayscale + per-image
mean/var/normalization in FP32 before feeding the model.
*Result:* histogram **identical to attempt 1, four decimal places**.
`mean-best-cossim=0.398, global-max=0.582`. M output stats:
`min=-6.79, max=8.12, mean=0.047, std=1.69` — same as attempt 1.
*This rules out the InstanceNorm-overflow hypothesis.* Whatever the FP16
break is, it's somewhere in the conv stack itself, almost certainly
accumulated rounding drift across 15-20 conv layers in the descriptor
head. XFeat's L2-normalized 64-D descriptor matching is unusually
sensitive to per-element drift — other vision models (classification,
segmentation) tolerate FP16 because their downstream task does.

**Verdict:** ANE FP16 doesn't work for this model on this matching task.
Solving it would require mixed-precision export (force the descriptor
head to FP32, leave the rest FP16) or porting to a smaller XFeat variant
with shorter conv stacks. Both are research efforts, not engineering.

**For now, ship FP32 at ~19 ms/frame.** Still ~6× faster than ALIKED on
CPU. Total iPhone latency ~40 ms (~25 FPS) which is plenty for hand-held
single-object tracking.

## Pitfalls we've already hit (don't re-discover them)

- **Threshold tuning is per-extractor.** ALIKED's `sim_min=0.5` was fine.
  XFeat's tighter 64-D descriptors need `sim_min=0.82` desktop / `0.65`
  iPhone (FP16 noise). Cossim distributions are different — re-tune
  whenever you swap detectors.
- **`detection_threshold` is upstream's NMS floor (default 0.05).** Below
  that, the heatmap NMS rejects the candidate before we ever see it.
  Don't conflate this with the matcher's `sim_min`.
- **`kpt_match_px=3` was tuned for ALIKED keypoints.** XFeat detects on
  slightly different positions than COLMAP's SIFT, so 3 px loses too many
  matches. We raised it to 6 px. If you switch detectors, re-tune.
- **The CoreML auto-generated wrapper writes `Bundle(for: xfeat.self)`.**
  If your top-level `Bundle` type isn't `Foundation.Bundle`, the build
  errors with "Missing arguments / Extra argument 'for'". Don't name
  global structs `Bundle`.
- **`Tensor.unfold` doesn't lower to Core ML.** coremltools 9 chokes on
  the dynamic 0-D shape tensor. Replace with `F.pixel_unshuffle` for
  export — they're equivalent.
- **`F.interpolate(x, x3.shape[-2:])` doesn't lower either.** Same
  reason. Use `scale_factor=` for the export wrapper.
- **`minimum_deployment_target=ct.target.iOS17`** is required for the
  Core ML conversion to succeed at all on this XFeat graph (without it,
  even after the unfold fix, an internal `_int` cast trips up).
- **Vectorize the matcher.** A naïve Python per-ref matmul + per-query
  for-loop is 2.7 s on the desktop side for our match volume. The
  one-big-sgemm + vectorized mutual-NN version is ~30 ms.
- **Per-3-D-point dedup is essential.** A single 3-D point can match from
  multiple ref views; without dedup the same world-point appears ~10x in
  the input to RANSAC and skews the inlier set. `match_query_to_bundle`
  keeps the highest-cossim instance per pt3d.

## What we deferred

These were intentional cuts; they're easy to revisit if needed:

- **DINOv2 retrieval** (`bundle.ref_global_emb`) — currently OFF. Could
  cut matcher time ~5x by only matching against the top-N most similar
  refs. Wire `DinoV2Embedder` into `online/tools/pipeline.py` (already
  half-done in code) and feed `ref_subset=` to `match_query_to_bundle`.
- **iOS top-K tuning** — currently 1000. Upstream uses 4096. Higher
  improves recall, costs matcher CPU. Iterate after on-device numbers.
- **iOS matcher vectorization** — `Matcher.swift` does one cblas_sgemm
  per ref. The Python equivalent of "concatenate all refs, single big
  sgemm" would shave another ~30 % off iOS matcher time.
- **LightGlue** — there's an `lightglue_for_aliked.onnx` in
  `shared/models/` from the ALIKED era. We never wired it up; brute-
  force matcher is good enough for the current bundle sizes. Useful
  later if bundles grow >50k keypoints.
- **`verify_bundle.py`** — still has stub `_check_format` /
  `_check_reprojection` / `_check_round_trip` (NotImplementedError).
  Implement when you have time.
- **Android port** — the iOS code is platform-specific Swift. The
  algorithm spec in `shared/algorithms/epnp_spec.md` is enough to port,
  but no Android engineer has been lined up.

## Where to look first when something breaks

| Symptom | Most likely cause | First file to check |
|---|---|---|
| `kept=0` from XFeatRunner | NMS rejecting everything; heatmap is all zeros | `XFeatRunner.decodeKeypointHeatmap`; verify `K`/`H` outputs aren't NaN |
| `kept` looks right but `match=0` | Descriptor sampler mismatch (bilinear vs bicubic, or letterbox geometry off by half-pixel) | `XFeatRunner.bicubicSample`; compare against Python descriptors on same image |
| `match` reasonable but `inliers=0` | Coplanar 3-D points or pose-recovery bug | `EPnP.solve` (look for `eigenFail` tally), then `RANSAC.swift` thresholds |
| Bbox lands somewhere wrong on screen | EXIF rotation or `simd_float3x3` indexing | `InferenceViewController.projectBboxToView`; add a corner-by-corner dump |
| Inference works but bbox shape is weird (skewed cube) | Bundle/iOS canvas mismatch | Verify `DEFAULT_TARGET_SIZE` (Python) matches `inputSize` (Swift) and `--input-size` (export) |
| Build error: "Missing arguments for parameters 'points3d'..." | Our `Bundle` type is shadowing `Foundation.Bundle` in the auto-generated CoreML wrapper | Rename the struct (we did this already; see `BundleLoader.swift` for `ObjectBundle`) |

## Verification snapshot (taken at branch creation)

Desktop test_images set (9 photos), settings = bundle 640×640 letterbox /
4096 top-k / `kpt_match_px=6`, matcher `ratio=0.9 sim_min=0.82`:

| Image | Result | Inliers |
|---|---|---|
| IMG_9235 | LOST | — |
| IMG_9236 | OK | 6 |
| IMG_9237 | OK | 42 |
| IMG_9238 | OK | 10 |
| IMG_9239 | OK | 12 |
| IMG_9240 | OK | 16 |
| IMG_9241 | OK | 18 |
| IMG_9242 | LOST | — |
| IMG_9246 | OK | 34 |

7 / 9 OK, matching the ALIKED baseline (also 7 / 9). Inlier counts are
lower than ALIKED (which got 22–90) but pose quality is unchanged — bbox
visually encloses the figurine on every OK case.
