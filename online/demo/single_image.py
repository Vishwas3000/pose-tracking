"""Demo: run the online pipeline on a single image and compare to ARKit ground truth.

Usage (from repo root):
    python -m online.demo.single_image \\
        --bundle    shared/objects/session_1777549127.bundle \\
        --aliked    shared/models/aliked-n16rot-top1k-640.onnx \\
        --image     offline/data/session_1777549127/frames/frame_0210.jpg \\
        --metadata  offline/data/session_1777549127/metadata/metadata_0210.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reuse ARKit -> COLMAP conversion that was used at bundle-build time.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from online.tools.pipeline import PoseTracker   # noqa: E402


def _arkit_ground_truth_w2c(metadata_path: Path) -> np.ndarray:
    meta = json.loads(metadata_path.read_text())
    c2w = np.array(meta["camera_pose_c2w"], dtype=np.float64)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(c2w @ flip)


def _pose_error(P_est: np.ndarray, P_gt: np.ndarray) -> tuple[float, float]:
    """Returns (rotation_deg, translation_m)."""
    R_err = P_est[:3, :3] @ P_gt[:3, :3].T
    cos_t = (np.trace(R_err) - 1) / 2
    rot_deg = float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
    t_err = float(np.linalg.norm(P_est[:3, 3] - P_gt[:3, 3]))
    return rot_deg, t_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle",   required=True, type=Path)
    ap.add_argument("--aliked",   required=True, type=Path)
    ap.add_argument("--image",    required=True, type=Path)
    ap.add_argument("--metadata", default=None, type=Path,
                    help="Optional: ARKit metadata JSON for ground-truth comparison.")
    args = ap.parse_args()

    print(f"Loading bundle  : {args.bundle}")
    print(f"Loading model   : {args.aliked}")
    tracker = PoseTracker(args.bundle, args.aliked)
    print(f"  M (3D points) : {len(tracker.bundle.points3d)}")
    print(f"  K (refs)      : {len(tracker.bundle.refs)}")
    print(f"  K_intrinsics  : fx={tracker.K[0,0]:.1f}  cx={tracker.K[0,2]:.1f}  cy={tracker.K[1,2]:.1f}")
    print()
    print(f"Processing frame: {args.image.name}")
    t0 = time.perf_counter()
    result = tracker.process(args.image)
    elapsed_ms = 1000 * (time.perf_counter() - t0)
    print(f"  ALIKED kpts   : {result.n_aliked_kpts}")
    print(f"  matched 2D-3D : {result.n_matches}")
    print(f"  RANSAC inliers: {result.n_inliers}")
    print(f"  elapsed       : {elapsed_ms:.1f} ms")

    if result.pose is None:
        print("\n[LOST] Could not estimate pose.")
        return

    print("\nEstimated W->C pose:")
    np.set_printoptions(precision=4, suppress=True)
    print(result.pose)

    if args.metadata is not None and args.metadata.exists():
        gt = _arkit_ground_truth_w2c(args.metadata)
        rot_deg, t_m = _pose_error(result.pose, gt)
        print("\nGround-truth W->C pose (ARKit):")
        print(gt)
        print(f"\nError vs ARKit:  rotation = {rot_deg:.2f} deg   translation = {t_m*100:.2f} cm")


if __name__ == "__main__":
    main()
