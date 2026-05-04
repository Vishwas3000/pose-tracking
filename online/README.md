# online — desktop reference for the on-device pose pipeline

This folder runs the **runtime** side of pose-tracking on Linux/macOS so we
can validate the algorithm end-to-end before porting to Swift / Kotlin.

It consumes a `.bundle` produced by `offline/tools/build_object_bundle.py`
and estimates a 6-DoF pose from a single image:

```
camera frame
  │
  ▼  ALIKED via ONNX Runtime  (tools/aliked_inference.py — shared with offline)
query keypoints + descriptors
  │
  ▼  brute-force mutual-NN + Lowe ratio  (tools/matcher.py)
2D-3D correspondences  (q_kpt → world_xyz via bundle.points3d[ref.pt3d_indices[ri]])
  │
  ▼  cv2.solvePnPRansac (EPnP)  (tools/pose_solver.py)
4×4 W→C pose
```

## Differences from the eventual mobile pipeline

| Step | Desktop (this folder)                | Mobile (Swift / Kotlin)                       |
| --- | ------------------------------------ | --------------------------------------------- |
| Inference | ONNX Runtime (CPU EP)         | ONNX Runtime Mobile (CoreML EP / NNAPI EP)    |
| Matcher   | brute-force mutual-NN          | LightGlue ONNX (shared/models/lightglue_for_aliked.onnx) |
| PnP       | cv2.solvePnPRansac (SOLVEPNP_EPNP) | pure-native EPnP (shared/algorithms/epnp_spec.md, ~400 LOC) |
| Render    | none — prints the pose         | RealityKit (iOS) / Filament (Android)         |

OpenCV is **not** allowed on mobile (per CLAUDE.md). When the iOS/Android
ports happen, the pose solver gets reimplemented natively against the spec
in `shared/algorithms/`.

## Quick start

```bash
# Bundle was generated earlier:
ls shared/objects/session_1777549127.bundle

# Run on one frame from the session
python -m online.demo.single_image \
    --bundle   shared/objects/session_1777549127.bundle \
    --aliked   shared/models/aliked-n16rot-top1k-640.onnx \
    --image    offline/data/session_1777549127/frames/frame_0210.jpg \
    --metadata offline/data/session_1777549127/metadata/metadata_0210.json
```

When `--metadata` is provided, the demo reports rotation/translation error
against the ARKit ground-truth pose embedded in the session export.

## Files

- `tools/bundle_loader.py` — thin wrapper around `offline.tools.bundle_writer.load_bundle`
- `tools/matcher.py` — descriptor matching (brute-force; LightGlue swap-in TODO)
- `tools/pose_solver.py` — RANSAC + EPnP via OpenCV
- `tools/pipeline.py` — `PoseTracker` orchestrator
- `demo/single_image.py` — run on one JPEG, compare to ARKit pose if available
