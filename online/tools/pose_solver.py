"""6-DoF pose from 2D-3D correspondences via EPnP + RANSAC.

Desktop reference uses cv2.solvePnPRansac (SOLVEPNP_EPNP). The iOS/Android
ports MUST replace this with the pure-native EPnP described in
shared/algorithms/epnp_spec.md — the project doesn't ship OpenCV on mobile.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PoseEstimate:
    pose: np.ndarray          # (4, 4) float64 — world->camera
    inliers: np.ndarray       # (M,) int64 indices into the input correspondences
    num_input: int
    rvec: np.ndarray          # (3,)
    tvec: np.ndarray          # (3,)


def solve_pose(world_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray,
               reproj_err_px: float = 4.0,
               iterations: int = 200,
               confidence: float = 0.999,
               min_inliers: int = 6,
               ) -> PoseEstimate | None:
    """Solve PnP with RANSAC. Returns None if the solver fails or
    inliers < min_inliers."""
    if len(world_pts) < 4 or len(img_pts) < 4:
        return None

    obj = world_pts.reshape(-1, 1, 3).astype(np.float64)
    img = img_pts.reshape(-1, 1, 2).astype(np.float64)
    Kf  = K.astype(np.float64)
    dist = np.zeros(5, dtype=np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, Kf, dist,
        iterationsCount=iterations,
        reprojectionError=reproj_err_px,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None

    R, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3]  = tvec.flatten()

    return PoseEstimate(
        pose=pose,
        inliers=inliers.flatten().astype(np.int64),
        num_input=len(world_pts),
        rvec=rvec.flatten(),
        tvec=tvec.flatten(),
    )
