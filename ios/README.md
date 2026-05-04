# ios — Swift app (Phase 3 onwards)

Real-time per-frame pose tracking on iOS. **Not implemented yet** —
created during Phase 3 of the roadmap.

## Plan

When Phase 3 starts:

1. Create Xcode project via xcodegen (`project.yml` will live here)
2. Port the proven preprocessing pipeline from
   `/Users/sudeepsharma/Documents/GitHub/xFeat/XFeatApp/CameraModel.swift`:
   - `pixelBufferToMLArrayFast` (vImage SIMD BGRA→Float32 RGB CHW)
   - `imageToPixelBufferFast` (Core Image scale + center-crop)
3. Replace XFeat Core ML with ALIKED ONNX + LightGlue ONNX (via ONNX Runtime)
4. Add `BundleLoader.swift` to parse the `.bundle` format
5. Add `EPnP.swift` (~400 lines, pure Swift)
6. Replace homography overlay with 3D bbox projection

## Tech stack

- ONNX Runtime via SwiftPM (`microsoft/onnxruntime-swift-package-manager`)
- AVFoundation + AVCaptureVideoDataOutput for camera
- vImage (Accelerate framework) for pixel preprocessing
- RealityKit/Metal for 3D bbox / model rendering
- xcodegen for project file generation (so the .xcodeproj isn't checked in)

## Reusable patterns from the XFeat prototype

The earlier XFeat-based prototype solved a lot of mobile-specific problems
that we'll reuse directly:

| Pattern | Source | What we get |
|---|---|---|
| vImage preprocessing | `xFeat/XFeatApp/CameraModel.swift:pixelBufferToMLArrayFast` | ~2 ms BGRA→Float32 RGB |
| Adaptive RANSAC | `xFeat/XFeatApp/CameraModel.swift:ransacHomographyInliers` | Iteration budget that shrinks on high inlier ratio |
| Camera preview overlay | `xFeat/XFeatApp/CameraPreview.swift` | AVCaptureVideoPreviewLayer w/ overlay layers |
| Dev signing via xcconfig | `xFeat/XFeatApp/Config/Developer.xcconfig` | Per-developer team ID, gitignored |
| xcodegen project generation | `xFeat/project.yml` | Reproducible .xcodeproj |

## Performance target

iPhone 15 Pro: 30 ms/frame (steady tracking) → 33 FPS.
See `docs/path_b_implementation_roadmap.md` §7 for the full budget.
