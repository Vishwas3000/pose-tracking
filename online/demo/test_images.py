"""Run the online pipeline on a folder of arbitrary test images.

Differs from sweep_session.py in two ways:
  - No ARKit metadata, so no ground-truth pose error reporting.
  - Test images are usually a different resolution than the bundle's reference
    images (e.g. 4032×3024 photos vs the 1920×1440 ARKit video frames the
    bundle was built from). We scale the camera intrinsics proportionally so
    the same K applies to whatever resolution the test image has — assumes the
    same field of view, which is true for photo / video on the same iPhone
    main camera.

Usage:
    python -m online.demo.test_images \\
        --bundle  shared/objects/session_1777549127.bundle \\
        --xfeat   shared/models/xfeat.pt \\
        --in-dir  offline/data/test_images \\
        --out-dir offline/data/test_images_infered \\
        --ref-size 1920x1440
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from online.tools.pipeline import PoseTracker             # noqa: E402
from online.demo.overlay_bbox import (                     # noqa: E402
    project_points, draw_bbox, annotate_status,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def scale_K(K_ref: np.ndarray, ref_size: tuple[int, int],
            tgt_size: tuple[int, int]) -> np.ndarray:
    """Scale the camera matrix from ref_size (W,H) to tgt_size (W,H).

    Same aspect ratio  -> simple proportional scaling.
    Different aspect   -> assume square pixels + center-vertical-crop
                          (matches iPhone main-cam behaviour going from
                           4:3 ARKit video to 16:9 photo/video modes).
    """
    rw, rh = ref_size
    tw, th = tgt_size
    sx = tw / rw
    K = K_ref.astype(np.float64).copy()
    K[0, 0] *= sx          # fx
    K[0, 2] *= sx          # cx

    same_aspect = abs((rw * th) - (rh * tw)) < max(rw, rh) * 0.01
    if same_aspect:
        sy = th / rh
        K[1, 1] *= sy
        K[1, 2] *= sy
    else:
        # Square pixels -> fy after scaling equals fx after scaling.
        K[1, 1] = K[0, 0]
        # Imagine the source extended to the bundle's width: at that width
        # it would be `h_inferred` tall; the visible area is a center crop.
        h_inferred = th * (rw / tw)
        cy_at_inferred = K_ref[1, 2] - (rh - h_inferred) / 2.0
        K[1, 2] = cy_at_inferred * (th / h_inferred)
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle",   required=True, type=Path)
    ap.add_argument("--xfeat",    required=True, type=Path)
    ap.add_argument("--in-dir",   required=True, type=Path)
    ap.add_argument("--out-dir",  required=True, type=Path)
    ap.add_argument("--ref-size", default="1920x1440",
                    help="Resolution the bundle's K was calibrated for (W x H).")
    ap.add_argument("--dinov2",   default=None, type=Path,
                    help="DINOv2 ONNX path; enables top-N retrieval (no-op if bundle has no embeddings).")
    ap.add_argument("--top-n",    type=int, default=5,
                    help="Number of refs to keep after DINOv2 retrieval.")
    args = ap.parse_args()

    rw, rh = (int(x) for x in args.ref_size.lower().split("x"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading bundle {args.bundle.name} ...")
    tracker = PoseTracker(args.bundle, args.xfeat,
                          dinov2_onnx=args.dinov2,
                          retrieval_top_n=args.top_n)
    print(f"  M={len(tracker.bundle.points3d):,}  K_refs={len(tracker.bundle.refs)}")
    print(f"  retrieval     : {'DINOv2 top-' + str(args.top_n) if tracker.embedder else 'OFF (matching all refs)'}")
    print(f"  ref K (for {rw}x{rh}):  fx={tracker.K[0,0]:.1f}  cx={tracker.K[0,2]:.1f}  cy={tracker.K[1,2]:.1f}")

    images = sorted(p for p in args.in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"\nFound {len(images)} images in {args.in_dir}\n")

    for i, fp in enumerate(images):
        # Use PIL to read RAW pixel data (no EXIF auto-rotation), which matches
        # how the session frames were captured + how the bundle was built.
        with Image.open(fp) as im:
            tgt_w, tgt_h = im.size
        K = scale_K(tracker.K, (rw, rh), (tgt_w, tgt_h))

        # The PoseTracker's runner reads via PIL/raw; its keypoints will be in
        # the test image's pixel space. Patch the K used by solve_pose.
        prev_K = tracker.K
        try:
            tracker.K = K
            t0 = time.perf_counter()
            result = tracker.process(fp)
            ms = 1000 * (time.perf_counter() - t0)
        finally:
            tracker.K = prev_K

        # CRITICAL: cv2.imread DOES apply EXIF rotation by default; our pipeline
        # (via PIL) does NOT, so the keypoints and the projected bbox live in the
        # RAW (sensor-orientation) pixel space. We must read the raw bytes here,
        # otherwise the bbox draws on a rotated canvas and lands off-target.
        img_bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if result.pose is not None:
            corners_3d = tracker.bundle.bbox3d.astype(np.float64)
            corners_2d, z = project_points(corners_3d, result.pose, K)
            in_front = z > 0
            img_bgr = draw_bbox(img_bgr, corners_2d, in_front)
            status = [
                f"{fp.name}  {tgt_w}x{tgt_h}",
                f"matches={result.n_matches}  inliers={result.n_inliers}  "
                f"kpts={result.n_kpts}  ({ms:.0f} ms)",
            ]
        else:
            status = [
                f"{fp.name}  {tgt_w}x{tgt_h}",
                f"[LOST]  matches={result.n_matches}  inliers={result.n_inliers}",
            ]

        out = annotate_status(img_bgr, status)
        out_path = args.out_dir / f"{fp.stem}_bbox{fp.suffix}"
        cv2.imwrite(str(out_path), out)
        flag = "OK" if result.pose is not None else "LOST"
        print(f"  [{i+1}/{len(images)}] {fp.name:30s} -> {flag:4s}  "
              f"matches={result.n_matches:4d}  inliers={result.n_inliers:4d}  "
              f"({ms:.0f} ms)  -> {out_path}")


if __name__ == "__main__":
    main()
