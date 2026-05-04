# Running OnePose++ / LoFTR on Mobile (iOS & Android)

Is a transformer-based feature matcher like LoFTR realistic for real-time
pose estimation on phones? **Yes — with the right architecture choices and
a clear split of what runs offline vs on-device.** This doc covers what
"transformer on edge" really means in 2025, what to keep on a server, and
a concrete recipe for shipping OnePose++-style pose estimation on a phone.

Pairs with `doc/loftr_features.md` (LoFTR explained) and
`doc/sfm_and_descriptors.md` (how the offline DB is built).

---

## 1. The two myths to clear up first

### Myth A: "Transformers are too heavy for phones"
Half-true ten years ago, false now.

- **Linear attention** is the key. Vanilla self-attention is O(N²) which
  blows up on dense feature maps. LoFTR (and the OnePose++ matcher) use
  **linear attention** (`src/models/OnePosePlus/loftr_module/linear_attention.py`)
  → O(N) cost. A 480×640 image at 1/8 coarse resolution = 4800 tokens,
  and linear attention handles that comfortably on a phone NPU.
- Modern phones run **multi-billion-parameter transformers** locally
  (LLMs on iPhone 16 Pro, Pixel 9, S24). LoFTR is ~11M params. It's
  smaller than the tokenizer of a typical mobile LLM.
- Reference numbers: LoFTR at 480p ≈ 50–100 ms on a desktop GPU,
  ≈ 30–60 ms on Apple Neural Engine (A17 Pro / M-series), ≈ 80–150 ms on
  flagship Snapdragon NPUs. That's already 6–20 FPS without any
  optimization.

### Myth B: "OnePose++ is one big model that has to ship together"
False. OnePose++ has a **strict offline / runtime split**:

| Stage | When | Where it can run |
|---|---|---|
| Capture mapping video | offline, once per object | phone (already exists: OnePose Cap iOS app) |
| Keypoint-Free SfM (COLMAP + DeepLM) | offline, once per object | desktop / cloud GPU only |
| Build `anno_3d_average.npz` | offline, once per object | desktop / cloud GPU only |
| **Runtime pose estimation** | every frame | **phone** ← this is what we ship |

So the question isn't "can LoFTR's full SfM pipeline run on a phone" —
it's "can the **runtime 2D-3D matcher** run on a phone". That matcher
(`src/models/OnePosePlus/OnePosePlusModel.py`) is the only thing that
needs to be fast on-device. SfM stays in the cloud or on a workstation.

---

## 2. What's already mobile-feasible vs. what isn't

### ✅ Mobile-feasible (the hot path)
- **OnePose++ matcher** — ResNet-FPN backbone + linear-attention coarse
  transformer + small fine refinement window. Pure feedforward PyTorch,
  exports cleanly to Core ML / ONNX / TFLite.
- **PnP-RANSAC** — `ransac_PnP` is OpenCV; iOS and Android have OpenCV
  builds. Sub-millisecond on phone CPU.
- **First-frame 2D detection** — `LocalFeatureObjectDetector` runs
  LoFTR against ~15 reference frames. Slightly heavier (15 forward
  passes) but only on frame 0 / when tracking is lost. On steady-state
  frames, the previous-pose-projection path (`previous_pose_detect`) is
  free — just a 3D bbox reprojection.
- **Loading `anno_3d_average.npz`** — it's a numpy file with N×3 + N×D
  arrays. Trivially deserializes on device.

### ❌ NOT mobile-feasible (and doesn't need to be)
- **COLMAP** — 100MB+ C++ binary, expects a workstation. Stays in the
  cloud.
- **DeepLM second-order optimizer** — CUDA-only, requires a GPU. Stays
  in the cloud. The first-order Adam fallback also assumes desktop
  training-style compute.
- **LoFTR pairwise matching for SfM** — would need to run on every
  pair (potentially thousands). Too slow even with optimization. Stays
  in the cloud.

### Architecture you'd actually ship
```
   Phone                                          Cloud / Workstation
 ┌─────────────────────────┐                  ┌──────────────────────────┐
 │ Capture mapping video   │  upload video →  │ Keypoint-Free SfM        │
 │ (existing OnePose Cap)  │                  │ (LoFTR + COLMAP + DeepLM)│
 └─────────────────────────┘                  │                          │
                                              │ → anno_3d_average.npz    │
                                              │   (~1–5 MB per object)   │
                                              └──────────────┬───────────┘
                                                             │ download
                                                             ▼
 ┌─────────────────────────┐                  ┌──────────────────────────┐
 │ RUNTIME (every frame):  │  ← npz cached    │                          │
 │  - CNN encode crop      │     on device    │                          │
 │  - Match vs N 3D points │                  │                          │
 │  - PnP-RANSAC → 6-DoF   │                  │                          │
 │  - Render 3D bbox       │                  │                          │
 └─────────────────────────┘                  └──────────────────────────┘
```

---

## 3. Realistic frame-rate targets

Based on published numbers from related work (LoFTR, EfficientLoFTR,
XFeat) and typical mobile-NPU performance for transformer-CNN hybrids:

| Device class | Architecture choice | Expected FPS |
|---|---|---|
| iPhone 15 Pro / 16 Pro (A17/A18, ANE) | LoFTR-as-is | 12–20 |
| iPhone 15 Pro / 16 Pro (ANE) | EfficientLoFTR / distilled | 30–60 |
| Pixel 9 / S24 (Tensor G4 / Snapdragon 8 Gen 3) | LoFTR-as-is | 8–15 |
| Pixel 9 / S24 (NPU) | EfficientLoFTR / distilled | 20–40 |
| Mid-range Android (Snapdragon 7-series) | LoFTR-as-is | 3–6 |
| Mid-range Android | distilled + INT8 | 10–20 |
| Apple Vision Pro | EfficientLoFTR | 60+ (passthrough budget) |
| AR glasses / wearables | needs aggressive distillation | 5–15 |

For OnePose++-style pose estimation specifically, you usually only need
**the 3D-2D matcher**, not full LoFTR. The matcher is **lighter** than
LoFTR because the 3D-side input is N descriptors (1500–7000 vectors), not
a full feature map. Add ~5 ms to the per-frame budget for that.

**Rule of thumb:** if you can hit 15 FPS with the matcher, tracking
between frames (Kalman / optical flow on the 3D bbox) gets you to a
smooth ≥30 FPS visually.

---

## 4. The concrete optimization levers

Ranked by impact:

### 4.1 Use a faster matcher than vanilla LoFTR
- **EfficientLoFTR** (CVPR 2024, ZJU3DV — same lab) — ~2.5× faster than
  LoFTR with comparable accuracy. Aggregates redundant tokens, uses a
  RepVGG-style backbone. **This is the obvious upgrade path.**
- **XFeat** (CVPR 2024, VeRLab) — explicitly mobile-targeted, runs >100
  FPS on CPU. Lighter than LoFTR but slightly less accurate on
  low-texture surfaces.
- **ALIKED, DISK, MatchFormer** — variations on the same idea. ALIKED is
  particularly small.

You'd retrain the OnePose++ pipeline with one of these as the backbone
matcher. The architecture in `OnePosePlusModel.py` doesn't care which
upstream LoFTR variant produced the descriptors, as long as the same
variant is used at SfM-time and inference-time.

### 4.2 Quantization
- **FP16** — universal 2× speedup, near-zero accuracy loss. Just export
  with FP16 weights. Both Core ML (`compute_precision = .float16`) and
  TFLite support it natively.
- **INT8** — another 1.5–2× speedup, requires a small calibration set
  (~100 images). Some accuracy loss on the fine refinement; do INT8 on
  the backbone, keep transformer FP16.
- **4-bit weight-only** — only worth it if you're memory-bound (rare for
  this size of model).

### 4.3 Resolution + 3D-point budget
- Drop input resolution from 512×512 to **256×256** → 4× compute
  reduction, ~1° of pose-error increase typically.
- Reduce `shape3d_val` from 7000 → **1500–2000**. Inference compute is
  dominated by N (3D points) × M (image tokens) attention. Halving N
  halves attention cost and memory.

### 4.4 Architectural distillation
- Replace the ResNet-FPN backbone with **MobileNetV3-Large** or
  **EfficientNet-Lite**. Lose 1–2 dB on feature quality, gain 3–5×
  speed.
- Reduce coarse transformer layers from 4 → 2.
- Skip fine refinement entirely on intermediate frames; only run it
  every 3rd frame and interpolate.

### 4.5 Frame-rate amortization (the cheapest trick)
- Run the full matcher every Nth frame (e.g., every 3rd).
- Between matched frames, propagate pose with **2D KLT optical flow** on
  the previous bbox corners + EKF on pose. This is what every AR app
  does — it's standard practice.
- Net: matcher at 10 FPS becomes a smooth 30 FPS user experience.

### 4.6 Tracking-mode object detection
- `previous_pose_detect` (`local_feature_2D_detector.py:200`) is already
  this trick — projects the 3D bbox with the last frame's pose for the
  cropping bbox. Free.
- Only fall back to LoFTR-based detection when PnP fails (< 20 inliers).

---

## 5. Deployment toolchains

### iOS (Apple Neural Engine via Core ML) — usually best
```
PyTorch model → coremltools.convert(...) → .mlpackage → ANE
```
- `coremltools` 7+ supports linear-attention transformers natively.
- Use `compute_units=.cpuAndNeuralEngine` (or `.all`) — you want ANE.
- ANE is 5–15× faster than CPU and ~2× faster than GPU on iPhone for
  this workload, plus near-zero battery drain.
- Pin `compute_precision=.float16` for free speedup.

### Android (NNAPI / Vendor NPU) — pickier
- **TFLite + GPU/NNAPI delegate** is the safe default. Convert via
  `torch → onnx → tf → tflite`. Some custom ops may need rewriting.
- **Qualcomm SNPE / Hexagon SDK** — biggest perf on Snapdragon devices,
  but requires vendor-specific build.
- **MediaPipe** has a feature-matching pipeline you can adapt.
- **ONNX Runtime Mobile** with NNAPI EP — most portable, slightly
  slower than vendor-specific paths.

### Cross-platform
- **ONNX Runtime Mobile** — same model artifact runs on both platforms.
  Slower than ANE/Hexagon-native, simpler CI.
- **MNN** (Alibaba) and **NCNN** (Tencent) — very lean inference
  engines, mature transformer support, popular in CN mobile apps. Good
  fallback if you hit issues with TFLite.

### Server-assisted (hybrid)
- Run the matcher in the cloud, the phone streams crops + pose from
  device IMU. Latency budget: ~80 ms total round-trip on 5G is feasible.
- Useful when the device is too weak (smartwatch, very-low-end Android).
- AR-cloud architectures (Niantic Lightship, Google ARCore Geospatial)
  do exactly this.

---

## 6. Practical shipping recipe

If you wanted to ship "real-time AR pose for known objects" tomorrow:

1. **Train OnePose++ with EfficientLoFTR** as the backbone matcher and
   shape3d=2000. Use 256×256 crops.
2. **Keep SfM in the cloud.** User scans an object with the phone, video
   uploads, server returns `anno_3d_average.npz` (~1–3 MB) within a
   minute.
3. **Cache the npz on device** so subsequent uses are offline.
4. **Convert the matcher to Core ML (iOS) and TFLite (Android)** with
   FP16. INT8 the backbone if you need extra speed.
5. **Run the matcher every 3rd frame.** Use ARKit/ARCore world tracking
   to propagate pose between matched frames — they expose 6-DoF camera
   pose at 60 FPS for free.
6. **Object detection on frame 0** via the LoFTR-affine method. After
   that, use `previous_pose_detect` until tracking is lost.
7. **PnP-RANSAC** on device with OpenCV. ~1 ms.
8. **Render** the 3D bbox / model with Metal (iOS) or OpenGL ES /
   Vulkan (Android), or Unity / Unreal if you're already in an engine.

End-to-end on iPhone 15 Pro: budget ~25 ms per matched frame, 5 ms for
PnP, plus frame-skip and tracking → smooth 60 FPS AR experience.

---

## 7. Honest caveats

- **First-frame latency** is real. The detection step needs to crop and
  match against ~15 references — that's a noticeable hitch (~150–300 ms
  on phone). Mitigations: ship fewer reference views (5 instead of 15),
  use a tiny dedicated detector network instead of LoFTR for frame 0.
- **Object-specific data ships per object.** A 3D database is ~1–5 MB.
  Fine for ~hundreds of objects on device, painful for thousands.
- **Low-light / motion blur** still hurts LoFTR-class matchers. Real-
  time on a moving phone in a dim room is the failure case.
- **No CAD model** is the project's whole point, but it means you need
  an SfM-quality scan per object. That's fine for product placement,
  museum AR, manuals — less fine for "any object you point at".
- **Evaluation parity:** if you swap LoFTR for EfficientLoFTR or XFeat,
  re-run the SfM and re-train. Mixing matchers across SfM-time and
  inference-time will silently degrade quality.

---

## 8. References worth reading

- **LoFTR**: Sun et al., CVPR 2021. Original paper.
- **EfficientLoFTR**: Wang et al., CVPR 2024. The mobile-friendly
  successor from the same lab.
- **XFeat**: Potje et al., CVPR 2024. Mobile-first detector-free matcher.
- **MobileSAM / FastSAM**: not directly related, but the same playbook
  (distill a heavy transformer to mobile) is now well-trodden.
- **OnePose Cap**: existing iOS app from the OnePose team. Confirms the
  authors do mobile capture; the inference path is what's missing from
  the open-source release.

---

## 9. Bottom line

Transformer-based dense matchers run **comfortably real-time on flagship
phones today** (15–30 FPS unoptimized, 30–60 FPS with EfficientLoFTR-
class architectures). The real engineering work isn't "make a transformer
fit on a phone" — that's solved. It's:

1. Splitting offline (SfM, in cloud) from runtime (matcher, on device).
2. Picking a faster matcher than vanilla LoFTR (EfficientLoFTR / XFeat).
3. Quantizing, dropping resolution, shrinking the 3D point set.
4. Amortizing with frame-skip + native AR tracking between matches.

OnePose++ as published is **research-grade desktop code**. None of the
authors' choices block mobile deployment — they just didn't ship the
mobile inference path open-source. Building it is a 2–4 engineer-month
project, not a research problem.
