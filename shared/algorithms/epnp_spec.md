# EPnP — Pure-Native Spec

Algorithm: **Efficient Perspective-n-Point** (Lepetit, Moreno-Noguer,
Fua, IJCV 2008). Solves 6-DoF pose `(R, t)` from N ≥ 4 2D-3D
correspondences.

We implement it twice — once in Swift (~400 LOC), once in Kotlin
(~500 LOC). Both follow this spec to ensure cross-platform parity.

## Why pure-native instead of OpenCV

OpenCV's `solvePnPRansac` works fine but adds ~200 MB to the iOS binary.
EPnP's math is well-defined and stable; ~400 LOC is a tractable port.

## Algorithm

### Inputs
- `p3d: [SIMD3<Float>]` (N) — 3D points in object frame
- `p2d: [SIMD2<Float>]` (N) — corresponding 2D pixels in image frame
- `K: 3×3 Matrix` — camera intrinsics

### Outputs
- `R: 3×3 Matrix` — rotation
- `t: 3×1 Vector` — translation
- (or `nil` if degenerate input)

### Steps

1. **Pick 4 control points** — typically 4 non-coplanar 3D points spanning
   the object (one centroid + 3 PCA axes is the recommended choice).
   Express each input 3D point as a barycentric combination of the 4
   control points (4 weights `α_i` per input point).

2. **Build the linear system M·x = 0** where:
   - x is a 12-vector of the 4 control points' 2D-image-plane coords
   - Each correspondence contributes 2 rows of M (one from x-projection,
     one from y-projection)
   - M is `(2N × 12)`

3. **Solve via SVD**. The null-space of M (the smallest singular vector)
   gives the control points in the camera coordinate frame.

4. **Resolve scale ambiguity**. The SVD gives a 1-parameter family of
   solutions (because of homogeneous scaling). Pick the scale that
   minimizes reprojection error using the original 3D-2D pairs.

5. **Recover (R, t)** from the control points in camera frame +
   the same control points in object frame using **Procrustes / Horn's
   method** (3D-3D rigid alignment).

6. **Refine** via Gauss-Newton or Levenberg-Marquardt minimizing
   reprojection error.

## Reference implementations to study

- OpenCV: `opencv/modules/calib3d/src/epnp.cpp`
- COLMAP: `colmap/src/colmap/estimators/epnp.cc`
- LearnOpenCV blog post on EPnP

## Pseudocode

```
solveEPnP(p3d, p2d, K):
    # 1. Pick control points
    cw = compute_control_points(p3d)        # 4 points in object frame
    alphas = compute_barycentric_coords(p3d, cw)  # (N, 4)

    # 2. Build M and solve null space
    M = build_M(alphas, p2d, K)              # (2N, 12)
    _, _, Vt = svd(M)
    x = Vt[-1]                               # 12-vector
    cc = x.reshape(4, 3)                     # control points in camera frame (up to scale)

    # 3. Resolve scale
    scale = solve_scale(cc, cw)
    cc *= scale

    # 4. Recover (R, t) via Procrustes
    R, t = procrustes(cw, cc)

    # 5. Refine
    R, t = gauss_newton_refine(p3d, p2d, K, R, t)

    return R, t
```

## Edge cases

- **N < 4**: return nil
- **Coplanar 3D points**: M is rank-deficient; use the planar EPnP
  variant (different control-point selection)
- **Near-degenerate point configuration**: SVD's smallest two singular
  values nearly equal → ambiguity. Pick the solution with lower
  reprojection error.

## Tests

For RANSAC use, the inner solve must be **fast** (target <100 µs). Test
with:
- Synthetic: random (R, t), random 3D points, project to 2D, recover
  pose. Expect <0.001 rad rotation error, <0.001 m translation error.
- Noise: add Gaussian noise to 2D points; pose error should grow
  smoothly.
- Degenerate cases: 3 collinear points → handled gracefully.
