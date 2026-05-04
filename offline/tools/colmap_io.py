"""Read/write COLMAP binary model files.

COLMAP outputs three files:
  cameras.bin   — camera intrinsics
  images.bin    — per-image: pose (qvec, tvec) + 2D keypoints + 3D point IDs
  points3D.bin  — per-3D-point: xyz + which images observed it + at what 2D coord

We don't reimplement the parsers — the upstream OnePose++ repo has a clean
pure-Python implementation we can reuse directly:

    /Users/sudeepsharma/Documents/GitHub/OnePose_Plus_Plus/src/utils/colmap/read_write_model.py

This module is a thin adapter: it imports from there and exposes a single
read_model() function that returns the three dicts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Vendored from colmap/colmap@3.9.1: scripts/python/read_write_model.py.
# Keeps Phase 1 self-contained — no dependency on a separate COLMAP source tree.
from . import _colmap_upstream as _upstream  # noqa: E402


class ColmapModel(NamedTuple):
    cameras: dict
    images: dict
    points3D: dict


def read_model(workspace_dir: Path) -> ColmapModel:
    """Read cameras.bin, images.bin, points3D.bin from a workspace directory.

    Returns a ColmapModel namedtuple with three dicts keyed by COLMAP IDs.
    """
    workspace_dir = Path(workspace_dir)
    cameras, images, points3D = _upstream.read_model(str(workspace_dir), ext=".bin")
    return ColmapModel(cameras=cameras, images=images, points3D=points3D)


def pose_from_qvec_tvec(qvec, tvec) -> np.ndarray:
    """Convert COLMAP's (quaternion, translation) to a 4×4 pose matrix.

    COLMAP stores qvec as (qw, qx, qy, qz). The pose maps WORLD → CAMERA:
        x_cam = R · x_world + t
    """
    R = _upstream.qvec2rotmat(qvec)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = tvec
    return pose


def camera_intrinsics(camera) -> np.ndarray:
    """Build a 3×3 K matrix from a COLMAP Camera struct.

    Supports PINHOLE, SIMPLE_PINHOLE, OPENCV models — the most common ones.
    """
    K = np.eye(3, dtype=np.float32)
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        f, cx, cy = camera.params[:3]
        K[0, 0] = K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
    elif camera.model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = camera.params[:4]
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
    else:
        raise NotImplementedError(f"Unsupported COLMAP camera model: {camera.model}")
    return K
