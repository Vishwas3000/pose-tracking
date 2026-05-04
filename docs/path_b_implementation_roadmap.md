# Path B Implementation Roadmap

How to actually build the **hloc-style 6-DoF object tracker** (ALIKED +
LightGlue + EPnP) for iOS and Android. Covers both offline (one-time per
object) and online (per-frame, real-time) pipelines, plus the technical
topics you need to research.

Read together with:
- `doc/comparison_kaggle_pipeline.md` — why Path B vs. OnePose++
- `doc/mobile_inference.md` — general mobile-inference patterns
- `doc/pipeline_steps.md` — what OnePose++'s steps map to in this path

---

## 1. Architecture overview

```
┌─ OFFLINE (per object, ~10 min on workstation) ────────────────────┐
│                                                                    │
│   COLMAP/SfM workspace + source images + 3D bbox                   │
│             ↓                                                       │
│   1. Sample K reference views (viewpoint-sphere)                   │
│             ↓                                                       │
│   2. ALIKED per ref → per-view 2D keypoints + descriptors          │
│             ↓                                                       │
│   3. Map ALIKED keypoints to COLMAP 3D points via nearest-2D       │
│             ↓                                                       │
│   4. (Opt) DINOv2 per ref → global retrieval embedding             │
│             ↓                                                       │
│   5. Pack into per-object bundle (~5-20 MB)                        │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘

┌─ ONLINE (per frame, mobile, ~30 ms target) ───────────────────────┐
│                                                                    │
│   Camera frame                                                     │
│             ↓                                                       │
│   1. Preprocess (640×480 RGB Float32 via vImage / RenderScript)    │
│             ↓                                                       │
│   2. ALIKED inference  → query keypoints + descriptors             │
│             ↓                                                       │
│   3. Reference view retrieval (DINOv2 or simpler) → top-K refs     │
│             ↓                                                       │
│   4. LightGlue × K → per-ref query↔ref keypoint matches            │
│             ↓                                                       │
│   5. 2D-3D correspondence assembly (lookup ref kp → 3D point)      │
│             ↓                                                       │
│   6. EPnP-RANSAC → R, t (6-DoF pose)                               │
│             ↓                                                       │
│   7. Pose smoothing (EKF on SE(3) or velocity model)               │
│             ↓                                                       │
│   8. Tracking state machine (TRACKING / COASTING / LOST)           │
│             ↓                                                       │
│   9. Project 3D bbox + render overlay                              │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. OFFLINE pipeline — implementation details

Run once per object, typically on a workstation with a GPU.

### Phase O1 — Inputs you need

```
sfm_workspace/
├── cameras.bin              ← COLMAP camera intrinsics
├── images.bin               ← per-image: pose + 2D keypoints + 3D point IDs
├── points3D.bin             ← per-3D-point: xyz + observations
├── source_images/
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
└── box3d_corners.txt        ← 8 corners of the 3D bbox in object frame
```

If you only have OnePose++'s `anno_3d_average.npz`, **you cannot do
Path B from it alone** — you need the full COLMAP workspace + source
images. Re-run SfM if you've lost these.

### Phase O2 — Reference view sampling

You don't want all source frames as references — most are redundant.
Pick K representative views (typical K = 20-50).

**Strategies** (pick one, in order of preference):

1. **Viewpoint-sphere sampling** (recommended)
   - Project each source frame's camera position onto the unit sphere
     centered at the object centroid
   - Use farthest-point sampling on this sphere to pick K diverse views
   - Guarantees rotation coverage; good for full 360° tracking

2. **DINOv2-diversity sampling**
   - Compute DINOv2 embeddings for all source frames
   - Greedy selection: pick the frame least similar to already-picked ones
   - Captures appearance diversity, not just geometric diversity

3. **Uniform temporal subset**
   - Just pick every Nth frame
   - Cheapest, works fine if the original capture was already a
     uniform orbit

### Phase O3 — ALIKED feature extraction per reference view

For each chosen reference image, run ALIKED to get keypoints + descriptors.

**Suggested settings**:
- `top_k = 1024` (1000 keypoints per reference is plenty)
- `detection_threshold = 0.005`
- Input resolution: 640×480 (matches mobile model input)

**Output per reference view**:
```python
{
  'keypoints': np.ndarray,      # (N, 2) pixel coords in 640×480 space
  'descriptors': np.ndarray,    # (N, 128) float32
  'scores': np.ndarray,         # (N,) detection confidence
}
```

### Phase O4 — Map ALIKED keypoints to COLMAP 3D points

This is the bridge between detector-based ALIKED keypoints and the
known 3D structure from SfM.

**Algorithm**:
```python
for each ALIKED keypoint kp_a in reference view:
    # Find the nearest COLMAP-tracked 2D point in this image
    candidates = images_bin[ref_image_id].xys  # (M, 2) COLMAP keypoints
    distances = np.linalg.norm(candidates - kp_a.xy, axis=1)
    nearest_idx = distances.argmin()

    if distances[nearest_idx] < 3.0:  # within 3 pixels
        # Inherit that COLMAP keypoint's 3D point ID
        point3d_id = images_bin[ref_image_id].point3D_ids[nearest_idx]
        if point3d_id != -1:  # COLMAP marks unmatched as -1
            kp_a.point3d_id = point3d_id
        else:
            discard kp_a
    else:
        discard kp_a  # no nearby triangulated point
```

**Result**: for each reference view, a filtered set of ALIKED keypoints
where every survivor has a known 3D point ID.

**Tip**: store the 3D points in object-frame coordinates (centered on
the bbox centroid, scaled to mm). This makes PnP results consistent
across objects.

### Phase O5 — Global retrieval features (optional but recommended)

For each reference view, compute one global descriptor for fast retrieval
at runtime.

**Options**:
- **DINOv2 (ViT-B/14)** — 768-D, ~85 MB model. Highest quality. Worth it
  if your reference set is small (<100 views) so you only run it K times.
- **MobileViT-XS or MobileNetV3 penultimate layer** — 128-D, ~5 MB.
  Good mobile compromise. Has to be the SAME model used at runtime.
- **Avg-pooled ALIKED features** — free (you already have ALIKED). Works
  poorly compared to DINOv2 for retrieval but acceptable for closed-set
  matching against one object.

If K is small (≤10 reference views), skip retrieval entirely — just
match against all references every frame.

### Phase O6 — Pack into per-object bundle

Final on-disk format. Keep it simple, optimize for fast load on mobile.

**Recommended layout** — a single zipped `.bundle` file:
```
object1.bundle (zip with no compression for fast random access)
├── meta.json                  # version, K, descriptor_dim, intrinsics_used_for_sfm
├── points3D.bin               # (M, 3) float32, all 3D points
├── bbox3d.bin                 # (8, 3) float32, bbox corners
├── ref_global.bin             # (K, D_global) float32, global retrieval descs
├── ref_thumbnails.jpg         # (K thumbnails for debug visualization)
└── per_ref/
    ├── ref_000.bin            # custom struct: keypoints + descriptors + point3D_indices + pose
    ├── ref_001.bin
    └── ...
```

**Per-reference binary struct**:
```
[uint32 num_keypoints]
[uint32 desc_dim]                                  # always 128 for ALIKED
[float32 × num_keypoints × 2]   keypoints xy
[float32 × num_keypoints × 128] descriptors
[uint32 × num_keypoints]        point3D_indices    # index into points3D.bin
[float32 × 16]                  pose_4x4           # camera pose for visualization
```

**Why custom binary**: parses in milliseconds on mobile vs ~50-100 ms
for unzipping a `.npz`. Worth the engineering effort.

**Final size**: ~5-20 MB per object depending on K. Ships with the app
or downloaded on demand.

---

## 3. ONLINE pipeline — per-frame implementation

The hot path. Target ~30 ms per frame on flagship phones.

### Phase R1 — Capture + preprocess

**iOS**: AVCaptureSession with `.vga640x480` preset. BGRA pixel buffer →
vImage SIMD conversion to Float32 RGB CHW (already implemented in your
existing `pixelBufferToMLArrayFast`).

**Android**: CameraX with target resolution 640×480. RenderScript or
Halide for byte → Float32 conversion.

**Cost**: ~2 ms.

### Phase R2 — ALIKED inference on query

ONNX Runtime call with the same model used during offline preprocessing.

**Important**: descriptor space MUST match between offline and online
inference. Use the SAME ALIKED checkpoint, SAME input resolution, SAME
preprocessing.

**Cost on iPhone 15 Pro ANE**: ~10-15 ms.
**Cost on Pixel 9 NPU**: ~12-18 ms.

### Phase R3 — Reference view retrieval

```python
def select_references(query_global_emb, ref_global_embs, k=2):
    sims = ref_global_embs @ query_global_emb.T   # (K, 1)
    top_k_idx = np.argsort(-sims)[:k]
    return top_k_idx
```

**Steady-tracking shortcut**: if the previous frame had ≥20 PnP inliers,
reuse the same K references plus their immediate viewpoint neighbors.
Skip retrieval entirely. Saves ~5-10 ms per frame.

### Phase R4 — LightGlue matching

For each retrieved reference k, run LightGlue on (query keypoints +
descriptors, reference k keypoints + descriptors).

**ONNX export gotcha**: LightGlue's official ONNX export has dynamic
shape handling. Use `cvg/LightGlue-ONNX` repo for the cleanest exported
model.

**Output per pair**: `[(q_idx, r_idx, score), ...]`.

**Cost on iPhone 15 Pro ANE**: ~5 ms per pair × 2 refs = ~10 ms total.

### Phase R5 — Build 2D-3D correspondences

```swift
var pairs2D3D: [(p2d: SIMD2<Float>, p3d: SIMD3<Float>, score: Float)] = []
for refIdx in retrieved {
    let matches = lightGlueMatches[refIdx]   // [(q_idx, r_idx, score)]
    for m in matches {
        let queryPx = queryKeypoints[m.q_idx]
        let pt3DIdx = perRef[refIdx].point3DIndices[m.r_idx]
        let pt3D = allPoints3D[pt3DIdx]
        pairs2D3D.append((queryPx, pt3D, m.score))
    }
}
// Optional: deduplicate (multiple refs may match the same 3D point)
// Keep highest-score match per 3D point
```

**Cost**: <1 ms.

### Phase R6 — EPnP-RANSAC

This is the geometry stage. Pure native, ~400 lines per platform.

**Algorithm outline**:
```
ransacEPnP(pairs, K_intrinsics, threshold_px, max_iter):
    sort pairs by descriptor score (PROSAC sampling)
    bestInliers = []

    for iteration in 0..max_iter:
        sample 4 distinct pairs (biased toward top-scoring early)
        (R, t) = solveEPnP_4points(sample, K)

        # Score: project all 3D points, count inliers
        inliers = []
        for (p2d, p3d) in pairs:
            projected = K · (R · p3d + t)
            projected.xy /= projected.z
            if ||projected.xy - p2d|| < threshold_px:
                inliers.append(idx)

        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            (bestR, bestT) = (R, t)
            # Adaptive: shrink iteration budget based on inlier ratio
            p = len(inliers) / len(pairs)
            max_iter = min(max_iter, log(1-0.999) / log(1-p^4))

    # Refine with all inliers
    (finalR, finalT) = solveEPnP_LM(bestInliers, K)
    return (finalR, finalT, bestInliers)
```

**Cost**: 1-5 ms with adaptive iterations. EPnP minimal solve is ~10 µs;
inlier counting dominates for large pair sets.

### Phase R7 — Pose smoothing

Without smoothing, the pose jitters frame-to-frame. Two simple options:

1. **Velocity-based prediction** — assume constant velocity in 6-DoF.
   Estimate `dR/dt`, `dt/dt` from the last two frames; predict next; if
   measured is close to predicted, use measured; else blend.

2. **EKF on SE(3)** — proper Kalman filter with rotation in Lie algebra
   (so₃ tangent space). More math (~100 lines) but smoother results.

Either works. Start with velocity prediction; upgrade to EKF if jitter
is visible.

### Phase R8 — Tracking state machine

```
state: TRACKING
on each frame:
    inliers = current frame's PnP inlier count
    if inliers >= 20:
        state = TRACKING
        save current pose as last_good
    elif inliers >= 8 OR coasting < 5:
        state = COASTING
        coasting += 1
        use velocity model to extrapolate from last_good
    else:
        state = LOST
        coasting = 0
        next frame: re-run retrieval from scratch (R3 with full DINOv2)

on render:
    if state in [TRACKING, COASTING]: show overlay
    else: hide overlay
```

### Phase R9 — Project 3D bbox + render

```swift
let bbox3D = bundle.bbox3D    // 8 corners
let projected: [SIMD2<Float>] = bbox3D.map { corner in
    let camCoord = R · corner + t
    let imgCoord = K · camCoord
    return SIMD2(imgCoord.x / imgCoord.z, imgCoord.y / imgCoord.z)
}
// Convert image coords (640×480 model space) to screen coords
// via AVCaptureVideoPreviewLayer.layerPointConverted
```

Draw the 12 wireframe edges between projected corners (or anchor a 3D
model via SceneKit / Filament for fancier overlays).

**Cost**: <1 ms.

---

## 4. Implementation roadmap

Realistic timeline assuming one developer, 8 hours/day. Adjust for your team.

### Week 1 — Offline pipeline

**Goal**: produce a per-object `.bundle` from a COLMAP workspace.

Tasks:
- [ ] Set up Python env (PyTorch, ALIKED, LightGlue, DINOv2, COLMAP I/O)
- [ ] Reuse `OnePose_Plus_Plus/src/utils/colmap/read_write_model.py` to load COLMAP outputs
- [ ] Implement reference view sampling (viewpoint-sphere FPS)
- [ ] Implement ALIKED inference + keypoint→3D-point mapping
- [ ] (Optional) DINOv2 global descriptor extraction
- [ ] Pack into custom binary `.bundle` format
- [ ] Write a Python verifier that loads a `.bundle`, picks a reference,
      simulates query=reference, and verifies the bundle is round-trip
      correct (matching against itself returns ~all keypoints as inliers)

**Deliverable**: `tools/build_object_bundle.py` script that takes a
COLMAP workspace and outputs `object.bundle`.

### Week 2 — Mobile ML inference

**Goal**: ALIKED + LightGlue + DINOv2 running on iOS and Android via
ONNX Runtime, producing identical outputs to the Python reference.

Tasks:
- [ ] ONNX export of ALIKED — start with `cvg/LightGlue-ONNX` repo's
      provided ALIKED export, or use `Shiaoming/ALIKED` directly
- [ ] ONNX export of LightGlue — use `cvg/LightGlue-ONNX`
- [ ] ONNX export of DINOv2 (or chosen embedder)
- [ ] Verify Python ↔ ONNX outputs match (max abs diff <1e-3 for FP32)
- [ ] Set up ONNX Runtime in iOS project (Swift Package Manager: `microsoft/onnxruntime-swift-package-manager`)
- [ ] Set up ONNX Runtime in Android project (Gradle: `com.microsoft.onnxruntime:onnxruntime-android`)
- [ ] Run a single test image through the ONNX model on each platform
      and compare to Python output

**Deliverable**: working `aliked.onnx`, `lightglue.onnx`, `embedder.onnx`
files plus Swift + Kotlin snippets that load and run them.

### Week 3 — iOS app integration

**Goal**: extend your existing XFeat iOS app to use ALIKED + LightGlue +
new bundle format.

Tasks:
- [ ] Add bundle loader (`BundleLoader.swift`) — parses your custom binary format
- [ ] Replace `pixelBufferToMLArrayFast` calls' model with ALIKED ONNX
- [ ] Add LightGlue inference call
- [ ] Replace `mutualNearestNeighbours` cosine-NN with LightGlue match results
- [ ] Implement DINOv2 retrieval (or skip if K is small)
- [ ] Verify on test images that 2D-3D pairs are sane (project 3D pts
      with known pose, compare to query pixel matches)

**Deliverable**: pick-test-image flow shows green dots correctly matched
to 3D points (validate by overlaying ground-truth-projected 3D points).

### Week 4 — PnP + rendering

**Goal**: 6-DoF pose estimation working on iOS, with 3D bbox rendered.

Tasks:
- [ ] Implement EPnP solver in Swift (~400 lines). Reference:
      Lepetit et al. 2008 paper + OpenCV's `solvePnP_EPNP` for sanity
- [ ] Implement RANSAC wrapper with adaptive iterations (reuse pattern
      from existing `ransacHomographyInliers`)
- [ ] Camera intrinsics: read from `AVCaptureDevice.activeFormat`, fall
      back to hardcoded per-device defaults
- [ ] Replace `homographyToCATransform3D` rendering with 3D bbox
      projection: project 8 corners → screen → draw 12 wireframe edges
- [ ] Implement tracking state machine
- [ ] Add velocity-based pose smoothing

**Deliverable**: live camera shows 3D bbox locked to the printed/scanned
object. Pose stable when phone is held still.

### Week 5 — Android sister app

**Goal**: mirror the iOS app with Kotlin + CameraX + ONNX Runtime.

Tasks:
- [ ] New project: `xFeat-android/` with CameraX + Compose UI
- [ ] Port `BundleLoader` to Kotlin
- [ ] Port ALIKED + LightGlue + retrieval inference calls
- [ ] Port EPnP + RANSAC to Kotlin (~500 lines)
- [ ] Render 3D bbox via Filament or Camera2 + Canvas
- [ ] Verify same `.bundle` works on both platforms (the bundle is
      platform-agnostic by design)

**Deliverable**: same test image flow works on a Pixel device with
similar inlier count and bbox alignment.

### Week 6 — Polish

- [ ] Frame-skip on slower devices (process every Nth frame, extrapolate
      pose between)
- [ ] Battery / thermal profiling (sustained 5-min session)
- [ ] Multi-orientation reference templates if needed for rotation
- [ ] Pose smoothing upgrade to EKF on SE(3) if velocity model jitters
- [ ] Tracking-loss recovery UX (show "scanning..." during LOST state)

---

## 5. Technical topics to research

In rough priority order.

### 5.1 ML model sourcing & ONNX export

| Component | Source | Notes |
|---|---|---|
| ALIKED | `github.com/Shiaoming/ALIKED` (original) | Lightweight, ~5MB FP32 |
| ALIKED ONNX | `github.com/cvg/LightGlue-ONNX` | Pre-exported variants |
| LightGlue | `github.com/cvg/LightGlue` | Active, well-maintained |
| LightGlue ONNX | `github.com/cvg/LightGlue-ONNX` | Use this — handles dynamic shapes |
| DINOv2 | `huggingface.co/facebook/dinov2-base` | ViT-B/14 best balance for retrieval |
| MobileViT-XS (alternative) | `huggingface.co/apple/mobilevit-xx-small` | Lightweight retrieval embedder |

**Read first**: the LightGlue-ONNX repo's README — it has working ONNX
exports for both ALIKED and LightGlue with sample mobile inference code.

### 5.2 Mobile inference engines

- **ONNX Runtime Mobile**: cross-platform, recommended.
  - iOS: Swift Package Manager `microsoft/onnxruntime-swift-package-manager`
  - Android: Gradle `com.microsoft.onnxruntime:onnxruntime-android`
- **iOS Core ML EP** (Execution Provider): enable for ANE acceleration
  on iPhone 15+ via `addCoreML()` in session options
- **Android NNAPI EP**: enable for Pixel/Snapdragon NPU via `addNnapi()`
- **Android XNNPACK EP**: faster CPU fallback when NPU unavailable

**Read**: ONNX Runtime Execution Providers docs.

### 5.3 EPnP algorithm

EPnP (Efficient PnP) is the standard algorithm for the 4-point minimal
solve in PnP-RANSAC. Mathematical background:

- Original paper: **Lepetit, Moreno-Noguer, Fua. "EPnP: Efficient
  Perspective-n-Point Camera Pose Estimation." IJCV 2008.**
- 4 control points expressed as barycentric coordinates of the 3D points
- Builds a linear system M·x = 0, solves via SVD
- Resolves scale ambiguity by minimizing reprojection error
- Final refinement via Gauss-Newton on rotation+translation

**Reference implementations to study**:
- OpenCV's `solvePnP` with `SOLVEPNP_EPNP` flag
  (`opencv/modules/calib3d/src/epnp.cpp`)
- `colmap/src/colmap/estimators/epnp.cc`
- Several pure-Python ports on GitHub (search "epnp pure python")

For the Swift / Kotlin port, you'll need:
- 3×3 SVD (use Accelerate's `LAPACKE_sgesvd` on iOS, JBLAS on Android)
- 8×8 linear system solve (Gaussian elimination is fine; you've already
  written this for the homography case)
- Quaternion / rotation matrix utilities

### 5.4 Image retrieval alternatives

Trade-off: model size vs retrieval quality.

| Embedder | Size | Retrieval quality | Mobile-ready? |
|---|---|---|---|
| DINOv2 ViT-B/14 | 85 MB | Excellent | Hard (ViT ops on ANE/NPU are slow) |
| MobileViT-XS | 5 MB | Good | Yes |
| MobileNetV3-Large penultimate | 6 MB | OK | Yes |
| SimCLR / MoCo distilled | varies | Good | Varies |
| Avg-pooled ALIKED (free) | 0 (already running ALIKED) | Poor | Yes |

**Practical rule**: if K ≤ 10, skip retrieval. If K = 20-50, use the
mobile embedder. If K > 100, use DINOv2 (and bake retrieval index into
the bundle for fast top-K).

### 5.5 Reference view sampling strategies

- **Farthest point sampling** on the viewpoint sphere — coverage over
  entire orbit
- **Pose-clustering** (k-means on camera positions) — densely covers
  popular viewpoints
- **DINOv2-diversity sampling** — appearance-driven, may miss
  geometrically-spread views

For most AR-style use cases (object viewed from a hemisphere), 30-50
viewpoint-sphere-sampled views is plenty.

### 5.6 Bundle file format

Three options:

| Format | Load speed | Tooling | Recommended? |
|---|---|---|---|
| Custom binary | Fastest (~1-5 ms) | DIY parser per platform | ✅ Recommended |
| `.npz` (zipped .npy) | Slow (~50-100 ms) | Easy in Python; medium in Swift/Kotlin | OK for prototyping |
| FlatBuffers / Protobuf | Fast | Well-supported across platforms | If you already use them |

Custom binary is straightforward — see the layout in §2.6. The total
parsing code is ~200 lines per platform.

### 5.7 Camera intrinsics

PnP needs a 3×3 K matrix. Sources:

- **iOS**: `AVCaptureDevice.activeFormat.deviceCalibrationData` (only
  available on Pro devices; ARKit also exposes intrinsics)
- **Android**: `Camera2.CameraCharacteristics.LENS_INTRINSIC_CALIBRATION`
  (only on devices that support it, typically Pixel 4+ and Galaxy S20+)

**Fallback** when intrinsics unavailable:
```
focalPx = max(width, height) * 0.85     // empirical for phone wide cams
cx = width / 2
cy = height / 2
K = [[focalPx, 0, cx],
     [0, focalPx, cy],
     [0, 0, 1]]
```

PnP is fairly robust to ±5% intrinsic error.

### 5.8 SE(3) pose representation & EKF

For pose smoothing across frames:
- Rotation: store as quaternion or rotation matrix; tangent space
  updates in so₃ (3D angular velocity)
- Translation: 3D vector; standard
- EKF state: `[qx, qy, qz, qw, tx, ty, tz, ωx, ωy, ωz, vx, vy, vz]`
- Process model: constant velocity in tangent space
- Measurement: PnP-estimated `(R, t)` per frame

**Read**: "A Tutorial on SE(3) Transformation Parameterizations and
On-Manifold Optimization" (Blanco-Claraco). For implementation,
Sophus C++ lib is the reference; SwiftSophus / similar exists for
mobile platforms.

### 5.9 Reference implementations to study

- **hloc** (`cvg/Hierarchical-Localization`) — the gold-standard
  Python implementation of this exact pipeline. Read the inference
  code in `hloc/match_features.py` and `hloc/extract_features.py`.
- **pixloc** (`cvg/pixloc`) — pixel-perfect pose refinement on top of
  hloc
- **OpenCV iOS samples** — for camera + GLES rendering boilerplate
- **AlikedC++** — sometimes faster than ONNX Runtime for very tight
  inference loops

### 5.10 Testing & verification

For each phase:

- **Phase O3 (ALIKED)**: pick the same image as both reference AND
  query, run the full match pipeline. Should get ~all keypoints as
  matches, RANSAC should give identity pose.
- **Phase O4 (3D mapping)**: project 3D points back through known SfM
  pose to image, verify they fall on/near ALIKED keypoints (mean
  reprojection error <2 px).
- **Phase R6 (EPnP)**: synthetic test with known (R, t), random 3D
  points, project to 2D, add small noise, verify EPnP recovers (R, t)
  within 1° / 5 mm.
- **End-to-end**: pick a test photo with known ground-truth pose
  (capture with ARKit's `currentFrame.camera.transform`), verify
  Path B's estimated pose matches within angular tolerance.

---

## 6. Critical files to create

### Python (offline)
| File | Purpose |
|---|---|
| `tools/build_object_bundle.py` | Phase O — main entry point |
| `tools/aliked_inference.py` | ALIKED extraction wrapper |
| `tools/lightglue_export.py` | LightGlue ONNX export |
| `tools/colmap_io.py` | Wrapper around COLMAP outputs (reuses upstream `read_write_model.py`) |
| `tools/verify_bundle.py` | Round-trip + reprojection checks |

### iOS
| File | Purpose |
|---|---|
| `XFeatApp/BundleLoader.swift` | Parse custom binary `.bundle` format |
| `XFeatApp/AlikedRunner.swift` | ONNX Runtime call site for ALIKED |
| `XFeatApp/LightGlueRunner.swift` | ONNX Runtime call site for LightGlue |
| `XFeatApp/Retrieval.swift` | DINOv2 / embedder + top-K nearest |
| `XFeatApp/EPnP.swift` | Pure-Swift EPnP + RANSAC (~400 lines) |
| `XFeatApp/PoseTracker.swift` | State machine + smoothing |
| `XFeatApp/BBoxRenderer.swift` | Project 3D bbox to screen + draw |

### Android
Mirror the iOS files in Kotlin under `app/src/main/kotlin/com/example/xfeat/`.

---

## 7. Performance budget on iPhone 15 Pro (target)

| Stage | Steady tracking | Re-acquisition |
|---|---|---|
| Preprocess | 2 ms | 2 ms |
| ALIKED inference | 12 ms | 12 ms |
| Retrieval | (skipped) | 8 ms |
| LightGlue × K | 10 ms (K=2) | 25 ms (K=5) |
| 2D-3D assembly | 1 ms | 1 ms |
| EPnP-RANSAC | 3 ms | 4 ms |
| Pose smoothing | 1 ms | 1 ms |
| Render | 1 ms | 1 ms |
| **Total** | **~30 ms (33 FPS)** | **~54 ms (one-time)** |

Mid-range Android (Pixel 7a, no NPU): roughly 2× these numbers, so
~60 ms steady. Frame-skip + pose extrapolation hides this on screen
(visible 30 FPS even with 15 FPS inference).

---

## 8. Known risks & mitigations

| Risk | Mitigation |
|---|---|
| ALIKED descriptors don't match across query/reference scale | Train-time augmentation; choose `top_k` at inference to capture multi-scale features |
| LightGlue ONNX export breaks on ANE | Fall back to GPU EP on iOS; both produce same outputs |
| EPnP fails on degenerate point configurations (e.g., all 3D points coplanar) | RANSAC discards bad samples; ensure reference views sample >2 distinct planes |
| Repetitive content (palm trees, brick walls) | Multi-orientation references + tighter RANSAC threshold; consider OnePose++ matcher as fallback for these specific objects |
| Bundle file too large for low-end devices | Reduce K; quantize descriptors to int8 (loses ~1% accuracy) |
| iOS / Android pose math diverges (different conventions) | Standardize on column-major right-handed coords; document clearly in `BundleLoader` |

---

## 9. Open questions to resolve before starting

These will affect your design — answer them up front:

1. **What objects?** Textured products vs uniform-texture content (drawings, etc.)? This changes whether you need OnePose++ matcher fallback for hard cases.
2. **Single object or many?** If many (e.g., 500-product catalog), bundle storage and retrieval indexing matter much more.
3. **Online or downloadable bundles?** If users scan their own objects on-device, the offline pipeline needs to run on phone too (much harder).
4. **iOS minimum target?** ANE access requires iOS 17+. Older devices fall back to GPU/CPU and inference is 2-3× slower.
5. **Camera intrinsics availability?** If targeting non-Pro iPhones / mid-range Android, expect to use fallback intrinsics most of the time.

---

## 10. Summary

Path B is a **complete 6-DoF object tracking system** built from
off-the-shelf ML components plus your own glue code. The offline
pipeline (Week 1) takes a COLMAP workspace and produces a per-object
binary bundle. The online pipeline (Weeks 2-5) is a per-frame loop:
ALIKED → retrieve refs → LightGlue × K → 2D-3D → EPnP-RANSAC →
project bbox → render.

Total scope: **~6 weeks for cross-platform MVP**, faster if iOS-first
and you skip Android initially.

The hardest parts are: ONNX exports of ALIKED + LightGlue (Week 2 — has
been done by others, study `cvg/LightGlue-ONNX`); EPnP solver (Week 4 —
straightforward with the paper as guide); and the tracking state
machine (Week 6 — needs real device testing to tune thresholds).

The trade-off vs OnePose++: **easier to ship, slightly worse on
textureless/repetitive content**. For most consumer AR use cases
(product visualization, AR characters anchored to objects, etc.), Path B
quality is sufficient and the simpler engineering pays for itself in
maintainability.
