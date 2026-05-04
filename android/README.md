# android — Kotlin app (Phase 5)

Sister of the iOS app. **Not implemented yet** — created during Phase 5
of the roadmap, after iOS is working.

## Plan

When Phase 5 starts:

1. Create new Android Studio project (Kotlin + Jetpack Compose UI)
2. Port `BundleLoader` from Swift to Kotlin (~150 LOC)
3. ONNX Runtime via Gradle dep `com.microsoft.onnxruntime:onnxruntime-android`
4. CameraX for camera capture
5. RenderScript or Halide for pixel preprocessing
6. Port `EPnP.kt` (~500 LOC) and the adaptive RANSAC framework
7. Filament or SceneView for 3D bbox rendering

## Tech stack

- Kotlin + Jetpack Compose
- ONNX Runtime Android
- CameraX (target API 24+ for broad device support)
- Filament for 3D rendering
- KotlinX serialization for bundle parsing (or pure ByteBuffer parser)

## Cross-platform parity

The same `.bundle` file should produce equivalent results on iOS and
Android. Verify by:
- Loading the same bundle on both platforms, dumping the parsed contents,
  asserting byte-for-byte parity in points3D + descriptors
- Running both apps on a fixed test image (AirDropped/sideloaded), checking
  that pose error vs ground truth is comparable across platforms

## Performance target

Pixel 9 / Snapdragon 8 Gen 3: ~25 ms/frame (steady tracking).
Mid-range (Pixel 7a, no NPU): ~60 ms/frame — frame-skip + ARCore camera
pose extrapolation hides this on screen.
