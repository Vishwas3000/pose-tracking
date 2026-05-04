"""Oriented 3D bounding box — load / membership test / corner derivation.

Ported verbatim from the upstream ColmapReconstruction repo
(`crop.py:load_bounds`, `crop.py:inside_box`, `validate.py:make_box_lineset`).
That repo is the producer of the `bounds.json` we consume; keeping the logic
byte-identical avoids subtle frame/normalization drift between the two repos.

bounds.json schema (ARKit world space — same frame as COLMAP points3D):

    {
      "center":   [x, y, z],
      "extents":  [w, h, d],            # full dimensions
      "rotation": [[r00,...],[...],[...]]  # columns = box axes in world space
    }

A point xyz (in world space) is inside the box iff
    local = (xyz - center) @ R
    |local_i| <= extents_i / 2  for i = 0,1,2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_bounds(path: Path) -> dict:
    """Parse bounds.json into numpy arrays. No normalization (R must be orthonormal)."""
    with open(path) as f:
        b = json.load(f)
    return {
        "center":  np.array(b["center"],   dtype=np.float64),
        "extents": np.array(b["extents"],  dtype=np.float64),
        "R":       np.array(b["rotation"], dtype=np.float64),
    }


def inside_box(xyz_array: np.ndarray, bounds: dict) -> np.ndarray:
    """Vectorized oriented-AABB membership test. Returns (N,) bool array."""
    c, e, R = bounds["center"], bounds["extents"], bounds["R"]
    # R columns = box axes in world space.
    # For row vectors: world->local = v @ R  (equiv to R.T @ v for column vectors).
    local = (xyz_array - c) @ R
    half = e / 2.0
    return (
        (np.abs(local[:, 0]) <= half[0])
        & (np.abs(local[:, 1]) <= half[1])
        & (np.abs(local[:, 2]) <= half[2])
    )


def corners(bounds: dict) -> np.ndarray:
    """8 corner points of the oriented box in world space, shape (8, 3) float32.

    Ordering matches make_box_lineset in upstream validate.py:
      bit 0 -> X, bit 1 -> Y, bit 2 -> Z   (0 = -extent, 1 = +extent)
    """
    c = bounds["center"]
    e = bounds["extents"] / 2.0
    R = bounds["R"]
    corners_local = np.array([
        [-e[0], -e[1], -e[2]],
        [ e[0], -e[1], -e[2]],
        [-e[0],  e[1], -e[2]],
        [ e[0],  e[1], -e[2]],
        [-e[0], -e[1],  e[2]],
        [ e[0], -e[1],  e[2]],
        [-e[0],  e[1],  e[2]],
        [ e[0],  e[1],  e[2]],
    ])
    return (corners_local @ R.T + c).astype(np.float32)
