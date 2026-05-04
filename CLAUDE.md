# Pose Tracker — Claude Operating Guide

You are working on **pose-tracker**, a cross-platform (iOS + Android)
real-time **6-DoF object pose tracking** application. This file tells you
the project's goals, decisions already made, conventions to follow, and
what to work on next.

---

## Project goal

Real-time (≥30 FPS) **6-DoF object pose tracking** on mobile, given a
**pre-built per-object 3D database**. Architecture is "Path B" — the
hloc-style pipeline using ALIKED + LightGlue + EPnP-RANSAC.

**End user experience**:
1. User scans an object once on a workstation (offline pipeline) →
   produces a `.bundle` file (~5–20 MB per object)
2. App ships with one or more bundles
3. At runtime: camera frame → detect features → match to database → 6-DoF
   pose → render 3D bbox/model anchored to the object

---

## Architecture (already decided — don't re-debate)

```
┌─ OFFLINE (Python, per object, ~10 min on workstation) ─┐
│   COLMAP/SfM workspace + source images + 3D bbox       │
│       ↓ sample reference views (~20-50)                 │
│       ↓ ALIKED → per-view keypoints + 128-D descriptors │
│       ↓ map ALIKED keypoints to COLMAP 3D points        │
│       ↓ (optional) DINOv2/MobileNet → global retrieval  │
│       ↓ pack into custom binary .bundle                 │
└────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─ ONLINE (Swift/Kotlin, per frame, ~30ms target) ───────┐
│   camera frame                                          │
│       ↓ vImage/RenderScript preprocess to 640×480 RGB   │
│       ↓ ONNX Runtime: ALIKED → query keypoints + descs  │
│       ↓ retrieve top-K reference views (or skip if K≤6) │
│       ↓ ONNX Runtime: LightGlue × K → 2D-2D matches     │
│       ↓ lookup ref keypoint → 3D point ID → 2D-3D pairs │
│       ↓ EPnP-RANSAC (pure native, ~400 LOC)             │
│       ↓ pose smoothing (velocity model or EKF on SE(3)) │
│       ↓ tracking state machine (TRACKING/COASTING/LOST) │
│       ↓ project 3D bbox + render overlay                │
└────────────────────────────────────────────────────────┘
```

---

## Tech stack (decided — don't change without strong reason)

| Layer | Choice | Why |
|---|---|---|
| Inference engine | **ONNX Runtime Mobile** | Single model file works on both platforms; CoreML EP for ANE on iOS, NNAPI EP for NPU on Android |
| Local features | **ALIKED** (~5 MB) | Mobile-friendly detector + descriptor; 128-D L2-normalizable |
| Matcher | **LightGlue** (~3 MB) | Faster successor to SuperGlue; clean ONNX export available via `cvg/LightGlue-ONNX` |
| Retrieval (optional) | **MobileNetV3-Large penultimate** OR **DINOv2 ViT-S/14 INT8** | Skip entirely if K≤6 reference views; use MobileNet for K=10-30; DINOv2 only if multi-object catalog |
| 3D database format | **Custom binary `.bundle`** (zip-of-binary entries) | Loads in <5 ms vs ~50 ms for `.npz`; flexible per-platform parser |
| PnP solver | **Pure-native EPnP + RANSAC** (~400 LOC each platform) | Avoids ~200 MB OpenCV binary; reuses adaptive RANSAC framework from earlier work |
| Camera/render | Native: AVFoundation+RealityKit (iOS), CameraX+Filament (Android) | Best performance per platform |
| Pixel preprocessing | **vImage SIMD** (iOS), **RenderScript** (Android) | 10–50× faster than per-pixel loops |
| Descriptor precision | **Float32** end-to-end | FP16 lost ~2-4% cosine similarity in earlier XFeat experiments — pushed correct matches below threshold |
| Model input resolution | **640×480** RGB CHW | 4:3 matches iPhone VGA preset; multiple of 14 for ALIKED stride |

---

## Project layout

```
pose-tracker/
├── CLAUDE.md                          ← this file (you're reading it)
├── README.md                          ← human-facing top-level doc
├── .gitignore
│
├── offline/                           ← Python, runs once per object
│   ├── README.md
│   ├── requirements.txt
│   ├── tools/
│   │   ├── build_object_bundle.py     ← main entry — Phase O orchestrator
│   │   ├── colmap_io.py               ← reads COLMAP .bin files
│   │   ├── sample_reference_views.py  ← viewpoint-sphere FPS
│   │   ├── aliked_inference.py        ← ALIKED extraction wrapper
│   │   ├── retrieval_features.py      ← MobileNet/DINOv2 global descs
│   │   ├── bundle_writer.py           ← custom binary serialization
│   │   └── verify_bundle.py           ← round-trip + reprojection tests
│   ├── tests/
│   └── data/                          ← scratch (gitignored)
│
├── ios/                               ← Swift app (Phase R)
│   ├── README.md
│   └── (Xcode project added during Phase R)
│
├── android/                           ← Kotlin app (Phase R)
│   ├── README.md
│   └── (Gradle project added during Phase R)
│
├── shared/                            ← Algorithm specs + ONNX models
│   ├── bundle_format.md               ← canonical bundle byte layout
│   ├── algorithms/
│   │   ├── epnp_spec.md               ← EPnP solver pseudocode
│   │   ├── ransac_spec.md             ← adaptive RANSAC details
│   │   └── matcher_funnel.md          ← descriptor matching stages
│   └── models/                        ← ONNX models (gitignored, large)
│
└── docs/                              ← Reference notes & decisions
    ├── architecture.md
    └── decisions/
        └── 2026-05-04_initial.md
```

---

## Key reference paths (DO NOT modify these — they're external)

- **`/Users/sudeepsharma/Documents/GitHub/OnePose_Plus_Plus/`**
  Reference research repo. Has rich `doc/` folder with deep-dive
  explanations. **Read but don't edit.** Particularly useful:
  - `doc/path_b_implementation_roadmap.md` — the 6-week roadmap this
    project implements
  - `doc/sfm_and_descriptors.md` — what's in COLMAP outputs
  - `doc/comparison_kaggle_pipeline.md` — why ALIKED+LightGlue
  - `doc/mobile_inference.md` — runtime patterns
  - `doc/loftr_features.md` — LoFTR vs ALIKED conceptual
  - `src/utils/colmap/read_write_model.py` — COLMAP file parsers
    (we'll re-use this directly)

- **`/Users/sudeepsharma/Documents/GitHub/xFeat/`**
  Earlier prototype (XFeat-based). Same iOS app skeleton we'll evolve
  for Path B. Useful patterns:
  - `XFeatApp/CameraModel.swift` — vImage preprocessing, ONNX-equivalent
    Core ML loading, cblas_sgemm matching, adaptive homography RANSAC
  - `XFeatApp/CameraPreview.swift` — AVCaptureVideoPreviewLayer overlay
    rendering
  - `python_compare/test_xfeat.py` — pattern for verifying Python ↔
    on-device parity
  - `python_compare/test_images/` — test images for verification

- **`/Users/sudeepsharma/Documents/GitHub/xfeat_ios/accelerated_features/.venv/`**
  Existing Python 3.12 venv with PyTorch + opencv + numpy already
  installed. **Reuse this** when running offline scripts to avoid
  re-installing PyTorch (~80 MB).

---

## Operating conventions

### Always
- **Save deep-dive explanations to `OnePose_Plus_Plus/doc/*.md`** when
  the user asks for deep technical explanations. The user has
  reiterated this preference throughout the project.
- **Use `TaskCreate`** to track multi-step work. Mark tasks completed
  as soon as they're done, don't batch.
- **Verify Python ↔ ONNX ↔ on-device parity** at every conversion step.
  Max abs diff <1e-3 for FP32 export.
- **Reuse existing assets** — see "Key reference paths" above. Don't
  re-implement what already exists in `xFeat/` or `OnePose_Plus_Plus/`.

### Never
- **Don't auto-commit or push.** User will ask explicitly.
- **Don't add OpenCV.** Adds ~200 MB to iOS binary; we have native EPnP.
- **Don't add ARKit-only paths** without an Android equivalent. This
  project is cross-platform first.
- **Don't introduce Float16 throughout the pipeline.** FP16 quantization
  cost real match quality in the earlier XFeat work (~2-4% cosine sim
  loss → matches below threshold). Stay FP32 unless explicitly asked.
- **Don't create the Xcode/Gradle projects until explicitly starting
  Phase R3/R4.** Premature project skeletons get out of sync with code.

### Style
- Python: PEP 8, type hints, docstrings on public functions.
  `pyproject.toml` over `setup.py`.
- Swift: 2-space indent, `//` comments, `@MainActor` for UI updates,
  background queues for inference.
- Kotlin: 4-space indent, idiomatic coroutines for async.
- Comments: explain *why*, not *what*. Code already says what.

---

## Phase plan & current status

**Status (last updated 2026-05-04)**: Phase 0 — scaffolding complete.
Next: Phase 1 (offline bundle generator).

### Phase 0 — Scaffolding (DONE)
Project structure created, this CLAUDE.md written.

### Phase 1 — Offline pipeline (Week 1)
Goal: produce `object.bundle` from a COLMAP workspace.

- [ ] Set up `offline/` Python env (reuse xfeat_ios venv or create new)
- [ ] Verify `colmap_io.py` can read a `cameras.bin` / `images.bin` /
      `points3D.bin` triple (test on data from
      `OnePose_Plus_Plus/data/...` if available)
- [ ] Implement `sample_reference_views.py` — farthest-point sampling
      on viewpoint sphere
- [ ] Implement `aliked_inference.py` — wrap official ALIKED
      (`github.com/Shiaoming/ALIKED`); emit per-image `(kpts, descs)`
- [ ] Implement keypoint→3D-point mapping (~3 px nearest-neighbor)
- [ ] Implement `bundle_writer.py` — see `shared/bundle_format.md`
- [ ] Implement `verify_bundle.py` — load bundle, simulate query=ref,
      verify ~all keypoints become inliers under identity pose
- [ ] Run end-to-end on one test object

**Deliverable**: `python -m tools.build_object_bundle --colmap-dir X
--source-images Y --bbox bbox.txt --out object.bundle` produces a
verified bundle.

### Phase 2 — Mobile ML inference (Week 2)
Goal: ALIKED + LightGlue running on iOS + Android via ONNX Runtime.

- [ ] Use `cvg/LightGlue-ONNX` repo's pre-exported ONNX models
- [ ] Verify Python ↔ ONNX outputs match (max abs diff <1e-3)
- [ ] Place ONNX files in `shared/models/` (gitignored)
- [ ] iOS: ONNX Runtime via SwiftPM
- [ ] Android: ONNX Runtime via Gradle
- [ ] Test single-image inference parity on each platform

### Phase 3 — iOS app integration (Week 3)
Goal: extend an Xcode project to run ALIKED+LightGlue+bundle loading.

- [ ] Create `ios/PoseTracker.xcodeproj` (xcodegen)
- [ ] Port preprocessing from `xFeat/XFeatApp/CameraModel.swift`
      (vImage SIMD)
- [ ] `BundleLoader.swift` — parse custom binary
- [ ] `AlikedRunner.swift`, `LightGlueRunner.swift`
- [ ] Verify dot-overlay matches Python expectations on a test image

### Phase 4 — PnP + rendering (Week 4)
Goal: 6-DoF pose with 3D bbox rendered.

- [ ] `EPnP.swift` — ~400 LOC pure Swift
- [ ] Adaptive RANSAC reusing the pattern from
      `xFeat/XFeatApp/CameraModel.swift:ransacHomographyInliers`
- [ ] Camera intrinsics from `AVCaptureDevice.activeFormat`
- [ ] 3D bbox projection rendering
- [ ] Tracking state machine

### Phase 5 — Android sister app (Week 5)
Mirror iOS in Kotlin. Same `.bundle` format, same algorithms.

### Phase 6 — Polish (Week 6)
Frame-skip, EKF smoothing, multi-orientation references, battery
profiling.

---

## Verification per phase

| Phase | How to verify |
|---|---|
| 1 | `verify_bundle.py` round-trip: load bundle, project 3D points using stored ref pose, verify they fall on stored ref keypoints (mean reproj err <2 px) |
| 2 | `python_parity_test.py`: run ALIKED+LightGlue in Python and via ONNX Runtime, compare outputs (max abs diff <1e-3) |
| 3 | Pick test image of scanned object → green dots cluster on real features (visualize via overlay) |
| 4 | Pick test image with known ARKit pose → estimated pose within 5° rotation, 5cm translation |
| 5 | Same test image flow on Pixel device → pose within 50% accuracy of iOS result |
| 6 | Live camera at 30+ FPS sustained, smooth bbox tracking under hand motion, recovers from tracking loss |

---

## Open questions to resolve before each phase

Phase 1 — Inputs:
- Does the user have the FULL COLMAP workspace (cameras.bin / images.bin
  / points3D.bin + source_images) or only the post-processed `.npz`?
  Path B requires the full workspace.
- What's K (number of reference views)? K≤6 → skip retrieval entirely.

Phase 2 — Models:
- Use `cvg/LightGlue-ONNX` exports as-is, or re-export with our own
  preprocessing?

Phase 3 — iOS target:
- iOS minimum version? iOS 17+ for clean ANE FP16, iOS 15 lowest sane.
- Single-object app or multi-object catalog?

---

## Common workflows

### Run offline pipeline on a test object
```bash
cd /Users/sudeepsharma/Documents/GitHub/pose-tracker/offline
source /Users/sudeepsharma/Documents/GitHub/xfeat_ios/accelerated_features/.venv/bin/activate

# (after first time, install ALIKED + LightGlue if not already)
pip install -r requirements.txt

python -m tools.build_object_bundle \
    --colmap-dir /path/to/sfm_workspace \
    --source-images /path/to/source_images \
    --bbox /path/to/bbox3d_corners.txt \
    --out ../shared/objects/test_object.bundle

python -m tools.verify_bundle ../shared/objects/test_object.bundle
```

### Inspect a bundle
```bash
cd offline
python -c "
from tools.bundle_writer import load_bundle
b = load_bundle('../shared/objects/test_object.bundle')
print(f'K={b.num_refs}  M={b.num_3d_points}  desc_dim={b.desc_dim}')
for i, ref in enumerate(b.refs[:3]):
    print(f'  ref {i}: {ref.num_keypoints} kpts, pose={ref.pose}')"
```

### Verify ONNX vs PyTorch parity
```bash
cd offline/tests
python test_aliked_parity.py
python test_lightglue_parity.py
```

---

## Things that lived in earlier prototypes — reusable patterns

These exist in `/Users/sudeepsharma/Documents/GitHub/xFeat/XFeatApp/` and
should inform our iOS implementation:

- **`pixelBufferToMLArrayFast`** (vImage SIMD BGRA→Float32 RGB CHW) —
  port directly to Phase 3
- **`ransacHomographyInliers`** with adaptive iteration count
  (PROSAC sampling, early-exit on high inlier ratio) — port the
  framework, swap homography → EPnP solve
- **`mutualNearestNeighbours`** with `cblas_sgemm` matrix multiply for
  cosine similarity — *might* still be useful for the optional
  retrieval step against MobileNet features
- **AVCapture preview overlay pattern** (`CameraPreview.swift`) — same
  pattern, render 3D bbox instead of homography-warped image

---

## When in doubt

1. Read `OnePose_Plus_Plus/doc/path_b_implementation_roadmap.md` first
2. Check `xFeat/XFeatApp/` for existing Swift patterns to reuse
3. Ask the user; they've thought a lot about the architecture and have
   strong opinions about FP16 vs FP32, OpenCV vs native, etc.
4. Don't over-design — ship the minimum to get to the next verification
   gate, then iterate

The user has been working on this iteratively for a while and prefers
**concrete progress over architecture astronaut work**. Bias toward
shipping a working artifact at each phase.
