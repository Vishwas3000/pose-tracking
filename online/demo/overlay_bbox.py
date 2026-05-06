"""Draw the projected 3D bbox on a tracked image and save the result.

Usage:
    python -m online.demo.overlay_bbox \\
        --bundle  shared/objects/session_1777549127.bundle \\
        --xfeat   shared/models/xfeat.pt \\
        --image   offline/data/session_1777549127/frames/frame_0210.jpg \\
        --out     offline/data/overlays/frame_0210_bbox.jpg

Pose: estimated by the online pipeline (ALIKED -> match -> EPnP-RANSAC).
Bbox: the 8 corners stored in the bundle, in COLMAP/ARKit world space.
Edges drawn (12): connect corners that differ in exactly one bit (X/Y/Z axis).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from online.tools.pipeline import PoseTracker   # noqa: E402

# Edges connecting corner indices that differ in exactly one of bits {0,1,2}.
# Matches the corner ordering produced by offline.tools.bounds.corners().
EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),   # along X
    (0, 2), (1, 3), (4, 6), (5, 7),   # along Y
    (0, 4), (1, 5), (2, 6), (3, 7),   # along Z
]


def project_points(world_pts: np.ndarray, pose_w2c: np.ndarray, K: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Project (N,3) world points through (4,4) W2C pose and (3,3) intrinsics.

    Returns ((N,2) image points, (N,) z-in-camera). Points with z<=0 are behind
    the camera and should not be drawn.
    """
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    cam = (R @ world_pts.T + t[:, None]).T   # (N, 3)
    z = cam[:, 2]
    img_h = (K @ (cam / z[:, None]).T).T     # (N, 3)
    return img_h[:, :2], z


def draw_bbox(image: np.ndarray, corners_2d: np.ndarray, in_front: np.ndarray,
              color=(0, 255, 0), thickness: int = 3) -> np.ndarray:
    out = image.copy()
    for a, b in EDGES:
        if not (in_front[a] and in_front[b]):
            continue
        pa = tuple(corners_2d[a].round().astype(int))
        pb = tuple(corners_2d[b].round().astype(int))
        cv2.line(out, pa, pb, color, thickness, lineType=cv2.LINE_AA)

    # Color the 8 corner dots (helps spot orientation flips visually)
    for i, (p, fz) in enumerate(zip(corners_2d, in_front)):
        if not fz:
            continue
        cv2.circle(out, tuple(p.round().astype(int)), 6, (0, 200, 255), -1)
        cv2.putText(out, str(i), tuple((p + np.array([8, -8])).round().astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def annotate_status(image: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image.copy()
    pad = 10
    h_line = 32
    h_box = pad * 2 + h_line * len(lines)
    cv2.rectangle(out, (0, 0), (760, h_box), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (pad, pad + h_line * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--xfeat",  required=True, type=Path)
    ap.add_argument("--image",  required=True, type=Path)
    ap.add_argument("--out",    required=True, type=Path)
    args = ap.parse_args()

    tracker = PoseTracker(args.bundle, args.xfeat)

    # Match the pipeline's PIL-based loader: ignore EXIF so we draw on the raw
    # sensor-orientation pixel array that ALIKED keypoints live in.
    img_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img_bgr is None:
        raise FileNotFoundError(args.image)

    t0 = time.perf_counter()
    result = tracker.process(args.image)
    elapsed_ms = 1000 * (time.perf_counter() - t0)

    if result.pose is None:
        out = annotate_status(img_bgr, [
            f"{args.image.name}",
            "[LOST] no pose",
            f"matches={result.n_matches}  inliers={result.n_inliers}",
        ])
    else:
        # Bundle.bbox3d is (8,3) float32 in world space.
        corners_3d = tracker.bundle.bbox3d.astype(np.float64)
        corners_2d, z = project_points(corners_3d, result.pose, tracker.K)
        in_front = z > 0
        out = draw_bbox(img_bgr, corners_2d, in_front)
        out = annotate_status(out, [
            f"{args.image.name}",
            f"matches={result.n_matches}  inliers={result.n_inliers}  "
            f"kpts={result.n_kpts}  ({elapsed_ms:.0f} ms)",
        ])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), out)
    print(f"wrote {args.out}  ({out.shape[1]}x{out.shape[0]})")
    if result.pose is None:
        print("  pose: LOST")
    else:
        print(f"  pose: rot {result.pose[:3,3].tolist()}  inliers={result.n_inliers}/{result.n_matches}")


if __name__ == "__main__":
    main()
