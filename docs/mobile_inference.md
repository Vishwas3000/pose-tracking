# Handling Inference on Mobile

How to architect the **runtime side** of OnePose++ (or an XFeat-derived
variant) on iOS + Android. This is the per-frame implementation detail
companion to `doc/mobile_deployment.md`'s feasibility analysis.

Read together with:
- `doc/pipeline_steps.md` — Steps M–S are the inference pipeline
- `doc/sfm_and_descriptors.md` — explains what's in the `.npz` you're consuming
- `doc/mobile_deployment.md` — high-level cross-platform deployment trade-offs

---

## TL;DR — The recommended cross-platform stack

| Layer | Pick | Reason |
|---|---|---|
| **Inference engine** | ONNX Runtime Mobile | Single model file works on both platforms; Core ML EP for ANE on iOS, NNAPI EP for NPU on Android. Mature, well-tested. |
| **Matcher model** | XFeat (small) for cross-platform; OnePose++ matcher only if accuracy demands | XFeat: ~11 MB, ~10 ms inference, 64-D descriptors. OnePose++ matcher: ~30 MB, ~25 ms, much higher accuracy on textureless/repetitive content. |
| **3D database** | `.npz` with descriptors compatible with the matcher you ship | If you ship XFeat → regenerate the .npz with XFeat-extracted descriptors. If you ship OnePose++ matcher → use its native .npz format unchanged. |
| **PnP solver** | Pure native EPnP (~400 lines per platform) | Avoids the ~200 MB OpenCV binary. EPnP + RANSAC framework is straightforward to port. |
| **Camera + render** | Native per platform | iOS: AVFoundation + RealityKit/Metal. Android: CameraX + Filament/SceneView. |
| **Intrinsics** | `AVCaptureDevice.activeFormat` (iOS) / `Camera2.CameraCharacteristics` (Android) | Real device intrinsics where available; reasonable defaults otherwise. |

---

## 1. Three inference architectures, choose one

### Option A — Full OnePose++ matcher on device

Ship the trained `PL_OnePosePlus` matcher to mobile. Load `.npz` unchanged.

**Pros**
- Highest accuracy. The transformer learns 2D-3D-specific features.
- Works on textureless / repetitive content where XFeat struggles.
- Uses your existing OnePose++ training and `.npz` files unchanged.

**Cons**
- Heavier model (~30 MB), longer inference (~25–40 ms on flagships).
- You need to convert it to ONNX or per-platform formats. Some ops
  (custom attention, einsums) may need rewrites for Core ML / TFLite.
- 128-D coarse + 256-D fine descriptors → larger memory footprint per
  3D point.

**When to pick this**
- Your target objects have repeating texture (drawings, fabrics, signs)
  and XFeat fails on them.
- You can tolerate ~25–30 ms inference budget.
- You have engineering bandwidth for non-trivial model conversion.

### Option B — XFeat-derived database + simple cosine matcher

Replace OnePose++'s 2D-3D matcher with XFeat (or any other detector-
free 2D matcher) and a plain cosine-NN against the `.npz`.

**Pros**
- Tiny matcher (~11 MB), fast inference (~8–15 ms on flagships).
- Cosine NN + RANSAC are 100% portable; no platform-specific quirks.
- Existing iOS XFeat scaffolding is mostly reusable.

**Cons**
- Lower accuracy than OnePose++'s trained 3D-2D matcher on hard scenes.
- Requires regenerating the `.npz` with XFeat-extracted descriptors
  (see step G in `pipeline_steps.md`).
- Repetitive content (rows of similar features) gives many descriptor
  false-matches that need geometric verification (RANSAC homography
  before PnP) to clean up.

**When to pick this**
- Your target objects are textured and unique enough for XFeat to find
  distinctive features.
- You want maximum cross-platform parity / minimum binary size.
- You're already invested in the XFeat path (existing iOS code).

### Option C — Cloud matching, device-side PnP + render

Stream cropped frames to a server. Server runs OnePose++ matcher,
returns 2D-3D correspondences. Device runs PnP + renders.

**Pros**
- No model on device → tiny app binary.
- Server can run the heaviest model. Easy to update.
- Device computes pose locally so latency is matching round-trip + PnP.

**Cons**
- Network dependency. Bad for offline / low-signal use.
- Privacy concerns (sending camera frames).
- Battery and data costs.
- 5G round-trip is ~50–100 ms minimum — not great for AR feeling.

**When to pick this**
- Tracking is occasional (snapshot pose, not continuous).
- You're already running cloud infra (e.g., AR-cloud platforms).
- Target devices are too weak for any on-device matching.

---

## 2. The runtime pipeline, per stage

Refer to Steps M–S in `pipeline_steps.md` for the algorithm. Below is
the **mobile implementation guidance** for each.

### Step M — Model loading (app start)

**iOS**
```swift
let cfg = MLModelConfiguration()
cfg.computeUnits = .all   // CPU + GPU + ANE
let model = try MLModel(contentsOf: bundleURL, configuration: cfg)
```

**Android (ONNX Runtime)**
```kotlin
val sessionOptions = OrtSession.SessionOptions().apply {
    addNnapi()           // hardware acceleration on Pixel/SD-equipped phones
    setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT)
}
val session = OrtEnvironment.getEnvironment()
    .createSession(modelBytes, sessionOptions)
```

**Best practices**
- Load **once** at app start; keep the session/model alive.
- **Warm up** with a dummy inference of the right shape before the user
  sees anything — first call cold-starts the GPU/NPU compiler (~200 ms).
- Use FP16 for ANE on iOS (3–5× faster than FP32) but verify descriptor
  quality doesn't drop too much. We saw FP16 drop XFeat sim by ~2-4%
  which moved correct matches under our 0.7 threshold — re-tuned by
  going to FP32.

### Step N — First-frame detection (object found)

**Why it exists**: you need an initial 2D bbox in the camera frame
before you can crop the image and feed the matcher.

**Original OnePose++ approach**: run LoFTR between query and ~15
reference frames from the SfM workspace, RANSAC-fit affine, use warped
image corners as bbox.

**Mobile considerations**
- LoFTR is *another* big model. Shipping both LoFTR and OnePose++ matcher
  doubles binary size.
- For mobile, prefer a **smaller detector** for this step:
  - Same XFeat / SuperPoint instance you use for matching (run it
    against 5–10 reference views, take the best)
  - Or a lightweight scene-category classifier
  - Or just instruct the user to "frame the object" — works in many
    consumer use cases
- This step is **expensive** (multiple inferences) — only run it on
  frame 0 and on tracking-loss recovery.

### Step O — Subsequent-frame detection (project last pose)

If frame N-1 had a valid pose with ≥20 PnP inliers:
- Project the 8 corners of `box3d_corners.txt` through `(K, R_{N-1}, t_{N-1})`
- Take their bounding rect as frame N's crop region
- Add 20–30% margin to absorb motion between frames

**Free** — no inference needed. This is the typical path for steady
tracking.

### Step P — Crop, resize, run matcher

The hot path. Per-frame stages:

#### P.1 — Pixel preprocessing
- Aspect-fill scale + center crop to model input (typically 640×480)
- Use **vImage** on iOS (`vImageScale_ARGB8888` + `vImageConvert_*toPlanarF`)
  or **RenderScript / Bitmap** on Android
- Avoid Swift/Kotlin pixel loops — they're 10–50× slower than SIMD

#### P.2 — Inference
- Forward pass through the matcher
- ONNX Runtime gives consistent ~10–25 ms across devices
- Float32 for accuracy; FP16 for speed if quality acceptable

#### P.3 — Decode model outputs
- For OnePose++ matcher: `mkpts_3d_db (M, 3)`, `mkpts_query_f (M, 2)`,
  `mconf (M)` — ready to feed PnP
- For XFeat path: extract top-K keypoints + descriptors, then mutual NN
  + ratio test against the `.npz` descriptors → 2D-3D correspondences

### Step Q — PnP-RANSAC

**The choice**: OpenCV's `cv::solvePnPRansac` (industry standard) or a
pure-native EPnP implementation (~400 lines, no dependency).

**Pure-native recommended** because:
- Adding OpenCV bloats iOS binary by ~200 MB
- EPnP + RANSAC is well-known math
- The RANSAC scaffolding (PROSAC sampling, adaptive iteration) you'd
  share with homography-RANSAC

**EPnP implementation outline**
```
solveEPnP(p3d: [SIMD3<Float>], p2d: [SIMD2<Float>], K) -> (R, t)?
  // 1. Compute control points (4 in 3D space, weighted with α coefficients)
  // 2. Build the linear system M·x = 0 (n correspondences → 2n equations)
  // 3. SVD to find null space (4 candidate solutions)
  // 4. Resolve scale ambiguity by minimizing reprojection error
  // 5. Refine with Gauss-Newton on rotation + translation

ransacEPnP(matches, K, iterations, inlierPxThreshold)
  // 1. Sample 4 random correspondences
  // 2. Solve EPnP for that minimal set
  // 3. Project all 3D points, count inliers within reprojection threshold
  // 4. Track best (R, t) with most inliers
  // 5. Adaptive iteration count: shrink budget as inlier ratio improves
  // 6. Final refinement with all inliers (LM or Gauss-Newton)
```

**Reuses framework from homography-RANSAC** — adaptive iteration math
(`log(1−C) / log(1−p^k)`), PROSAC sampling order (matches sorted by
descriptor confidence), early-exit on high inlier ratio.

**Performance budget**: 1–5 ms typical. Fast even at 1000 RANSAC
iterations because each EPnP minimal solve is ~10 µs.

### Step R — Render

You have the 6-DoF pose `(R, t)`. Render in screen space:

**iOS — Metal / RealityKit**
- Compute 8 screen points by projecting `box3d_corners.txt` through
  `(K, R, t)` then converting model-image coords to layer coords via
  `AVCaptureVideoPreviewLayer.layerPointConverted`
- Draw lines between them with `CAShapeLayer`
- For a 3D model overlay: anchor a `RealityKit.Entity` at the world-space
  pose (composed with ARKit's camera-in-world if you want stability)

**Android — Filament / OpenGL ES**
- Same projection math
- Draw via `MaterialInstance` + `RenderableManager` for textured quads
- Or pure GLES wireframe via `glDrawElements(GL_LINES, ...)`

### Step S — State machine for tracking continuity

Three states:

```
TRACKING ─(inliers < 20 for 1–3 frames)──→ COASTING (motion model only, hide overlay)
COASTING ─(inliers ≥ 20)──→ TRACKING
COASTING ─(>10 frames lost)──→ LOST
LOST ─(detection succeeds via Step N)──→ TRACKING
```

**Coasting matters**: without it, your overlay flickers off every time
the user's hand briefly occludes the object, or a quick camera tilt
loses descriptors momentarily. With coasting, you hold the last pose
for a few frames before declaring tracking lost.

---

## 3. Cross-platform implementation choices

### Inference engine

| Engine | iOS support | Android support | Notes |
|---|---|---|---|
| **ONNX Runtime Mobile** | ✅ Core ML EP for ANE | ✅ NNAPI / XNNPACK | Single model file. Recommended. |
| **Core ML** | ✅ Native | ❌ | iOS-only; cleanest on Apple. |
| **TFLite** | ⚠️ Slower than Core ML | ✅ Native | Android-friendly, painful on iOS. |
| **MNN / NCNN** | ✅ | ✅ | Chinese mobile inference engines. Very lean, mature in CN ecosystem. |
| **PyTorch Mobile** | ⚠️ | ⚠️ | Fading; PT Edge / ExecuTorch is the future but not mature yet. |
| **MediaPipe** | ✅ | ✅ | Cross-platform but locked to its own model format. |

**Pick ONNX Runtime** unless you have specific reasons (e.g., already
shipping with TFLite for other reasons).

### PnP solver

| Option | Effort | Binary cost | Notes |
|---|---|---|---|
| **Pure native EPnP** | ~400 lines per platform | 0 | Recommended. Fast, dependency-free. |
| **OpenCV `solvePnPRansac`** | Trivial integration | ~200 MB on iOS, ~30 MB on Android | Battle-tested. Use if you're already shipping OpenCV. |
| **Vision (iOS) / ML Kit (Android) built-ins** | None | 0 | Don't expose PnP. Useless for this. |
| **C++ shared library** (e.g., from Ceres/g2o) | Medium | ~1–10 MB | Good if you want one implementation across platforms. |

### Camera intrinsics

PnP needs a 3×3 K matrix per frame (focal length, principal point).

**iOS** — `AVCaptureDevice.activeFormat` exposes intrinsics on Pro
devices:
```swift
let intrinsicMatrix = sampleBuffer.attachment(forKey:
    kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix) as? Data
```

**Android** — `Camera2.CameraCharacteristics.LENS_INTRINSIC_CALIBRATION`
on devices that support it.

**Fallback**: hardcode reasonable defaults for the device (e.g.,
~485 px focal at 480p for iPhone 14 Pro back wide). PnP is fairly
robust to ±5% intrinsics error.

### Database (.npz) loading

**Format**: a `.npz` is just a zip of `.npy` files. To read on mobile:

**iOS / Swift**
```swift
import Foundation
import Compression

// 1. Unzip the .npz (zip32 with no compression for .npy entries — common)
// 2. Parse each .npy header (magic bytes, shape, dtype, byte order)
// 3. mmap the data section into a `[Float]` or similar
```
~150 lines of Swift. Or use a small library (`SwiftNumPy`, `npy_swift`).

**Android / Kotlin**
- `java.util.zip.ZipInputStream` for .npz unpacking
- Custom .npy header parser
- `FloatBuffer` / `ByteBuffer` for the data
- ~100 lines of Kotlin. Or `org.jetbrains.kotlinx:multik` for typed arrays.

**Tip**: pre-process the `.npz` to a flat custom binary format
(e.g., `[N: u32][D: u32][points3D: f32 × N×3][descriptors: f32 × N×D]`).
Loads in milliseconds without zip/header parsing.

---

## 4. Recommended file structure for a cross-platform implementation

```
mobile-tracker/                                 (shared assets)
├── models/
│   ├── matcher.onnx                            (single ONNX, ~11 MB for XFeat / ~30 MB for OnePose++)
│   └── README.md                               (training/conversion notes)
├── databases/                                  (per-object 3D databases)
│   ├── object1.bundle                          (custom binary: 3D points + descriptors + bbox)
│   └── object2.bundle
└── shared/                                     (algorithm specs, can be a doc or shared C++)
    ├── matcher_funnel.md                       (cosine NN + ratio + mutual NN spec)
    ├── ransac_pnp_spec.md                      (EPnP + adaptive RANSAC spec)
    └── (optional) algorithms.cpp               (shared C++ if you want)

ios-app/
├── XFeatApp/
│   ├── CameraModel.swift                       (AVCaptureSession + ONNX-RT + matcher orchestration)
│   ├── EPnP.swift                              (~400 lines; pure-Swift EPnP + RANSAC)
│   ├── NPZLoader.swift OR DatabaseBundle.swift (loads the per-object .bundle / .npz)
│   ├── ObjectRenderer.swift                    (3D bbox / model overlay via Metal/RealityKit)
│   └── Resources/
│       ├── matcher.onnx
│       └── object1.bundle

android-app/
├── app/src/main/kotlin/...
│   ├── CameraEngine.kt                         (CameraX + ONNX Runtime + matcher orchestration)
│   ├── EPnP.kt                                 (~500 lines; pure-Kotlin EPnP + RANSAC)
│   ├── DatabaseBundle.kt                       (loads the per-object .bundle / .npz)
│   ├── ObjectRenderer.kt                       (3D overlay via Filament/SceneView)
│   └── ...
└── assets/
    ├── matcher.onnx
    └── object1.bundle
```

---

## 5. Performance budget per stage

For **XFeat-based** path on iPhone 15 Pro:

| Stage | Time | Cumulative |
|---|---|---|
| Camera frame arrives | 0 | 0 |
| Aspect-fill crop + Float32 conversion (vImage) | ~2 ms | 2 ms |
| Matcher inference (ONNX Runtime, ANE) | ~10 ms | 12 ms |
| Top-K keypoints + descriptor sampling | ~5 ms | 17 ms |
| Cosine NN matrix multiply (vDSP) | ~3 ms | 20 ms |
| Funnel filter (threshold + ratio + mutual) | ~1 ms | 21 ms |
| EPnP-RANSAC (~50 iterations, adaptive) | ~3 ms | 24 ms |
| Project 8 bbox corners + render layer | ~1 ms | 25 ms |
| **Total** | **25 ms** | **40 FPS** |

For **OnePose++ matcher** path on iPhone 15 Pro:

| Stage | Time | Cumulative |
|---|---|---|
| Camera frame arrives | 0 | 0 |
| Crop + Float32 conversion | ~2 ms | 2 ms |
| Matcher inference (ANE, FP16) | ~25 ms | 27 ms |
| Output decoding | ~1 ms | 28 ms |
| EPnP-RANSAC | ~3 ms | 31 ms |
| Render | ~1 ms | 32 ms |
| **Total** | **32 ms** | **31 FPS** |

For **mid-range Android** (Pixel 7a, no NPU):

| Stage | XFeat path | OnePose++ path |
|---|---|---|
| Inference | ~50 ms | ~80 ms |
| Everything else | ~10 ms | ~10 ms |
| **Total** | **60 ms (~16 FPS)** | **90 ms (~11 FPS)** |

If sub-30 FPS is unacceptable on weak devices, **frame-skip + project
last pose between matched frames** (Step O) gets visible smoothness back
without extra ML cost.

---

## 6. Failure modes and recovery

### Bad inliers in some frames (descriptor noise, motion blur)
- **Symptom**: PnP returns reasonable pose but reprojection error high
- **Mitigation**: Kalman filter on `(R, t)` over time; reject outlier
  frames; coast on last good pose

### Tracking lost (object out of frame, occluded)
- **Symptom**: `inliers < 20` for several frames
- **Mitigation**: state-machine fallback to Step N (full re-detection);
  hide overlay during LOST state

### Drifting pose (slow accumulation of error)
- **Symptom**: tracking nominally OK but overlay slowly drifts off
  object
- **Mitigation**: every N frames, re-anchor by running detection from
  scratch and comparing. If drift > threshold, replace with re-detected
  pose

### Battery / thermal throttling
- **Symptom**: frame rate halves after ~3 minutes of use
- **Mitigation**: drop to 15 FPS sustained, frame-skip to fill in;
  reduce model resolution; offload to NPU (avoids CPU/GPU thermal)

### First-frame latency
- **Symptom**: 200–500 ms hitch when user first points at the object
- **Mitigation**: pre-warm the session with dummy inference; show a
  "scanning..." UI; the pre-warm during app launch hides the cold start

### Repetitive content false matches
- **Symptom**: many "matches" but they don't fit any single homography
  / pose (RANSAC inlier count stays low even though descriptor matches
  are abundant)
- **Mitigation**: ratio test (`Lowe`); RANSAC with PROSAC sampling
  (matches sorted by descriptor strength, sample top first); switch to
  OnePose++ matcher if XFeat falsely matches similar structures

---

## 7. What to NOT do on mobile

- **Don't run inference on the main/UI thread** — kills frame rate and
  can ANR on Android. Always background queue (`captureQueue` in our
  iOS code; `Executor` on Android).
- **Don't allocate a new MLMultiArray / OrtValue per frame** — re-use
  pre-allocated buffers. Allocation is surprisingly expensive (1–5 ms).
- **Don't decode JPEGs in-loop** — if you're processing a list of test
  images, decode once at startup.
- **Don't write per-pixel Swift / Kotlin loops** — always vImage / RS /
  vector intrinsics. Save 10–50× factor.
- **Don't use `[[Float]]` in Swift** for descriptors — use a flat
  `[Float]` (count × dim contiguous) and feed `cblas_sgemm` directly.
  Array-of-arrays is shockingly slow. (We saw this firsthand: ~3800 ms
  → ~5 ms by switching.)
- **Don't ship the model as FP32 if FP16 quality is acceptable** —
  doubles inference time on ANE.

---

## 8. Where each implementation piece lives

If we follow the recommended XFeat-based architecture, the actual
implementation maps to:

| Concept | OnePose++ source (reference) | Mobile counterpart (to write) |
|---|---|---|
| Step F descriptor sampling | `KeypointFreeSfM/loftr_for_sfm/utils/sample_feature_from_featuremap.py:28` | Python preprocessing script (Mac, one-time per object) |
| Step G filter + average | `sfm_utils/postprocess/feature_process.py:get_kpt_ann` | Same script |
| Step P matcher inference | `models/OnePosePlus/OnePosePlusModel.py:25` | ONNX Runtime call site in iOS/Android |
| Step P top-K extraction | (none — OnePose++ matcher outputs them directly) | Custom Swift/Kotlin (~150 lines each, with vImage/Accelerate) |
| Step P cosine NN + funnel | (none, OnePose++ does this end-to-end in the network) | `cblas_sgemm` (iOS) / `nd4j` or hand SIMD (Android), plus filter loop |
| Step Q PnP-RANSAC | `utils/metric_utils.py:ransac_PnP` (calls `cv::solvePnPRansac`) | Pure native EPnP + RANSAC (~400 lines per platform) |
| Step R 3D bbox render | `utils/vis_utils.py:save_demo_image` | Metal CALayer / Filament |
| Step S state machine | `demo.py` orchestration loop | Native per platform |

---

## Summary

The cleanest architecture is **ONNX Runtime + XFeat + native EPnP**.
That's a model file, a database file, ~1500 lines of Swift, ~1500 lines
of Kotlin. Hits 30+ FPS on flagships, 15+ on mid-range Android.

Ship OnePose++'s matcher only if descriptor quality is the bottleneck
on your specific objects. For repetitive / drawn / textureless content
it's noticeably better. For most textured products / faces / unique
objects, XFeat is plenty.

The non-negotiable parts: **adaptive RANSAC** (huge speedup), **vImage
preprocessing** (10–50× factor), **frame-skip + pose extrapolation**
(smooth UX on slow devices), and a **state machine for tracking
continuity** (otherwise the overlay flickers).
