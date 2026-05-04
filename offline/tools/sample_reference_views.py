"""Pick K representative reference views from the SfM source frames.

Strategy: viewpoint-sphere farthest-point-sampling (FPS).

  1. Project each frame's camera position onto the unit sphere centered at
     the object's centroid (mean of its 3D points).
  2. Greedy FPS: start with one view, then iteratively add the view whose
     viewpoint direction is farthest (largest angular distance) from any
     already-selected view.

This guarantees rotational coverage of the object's viewable hemisphere
and avoids picking redundant near-duplicate views.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .colmap_io import ColmapModel


def farthest_point_sphere(model: ColmapModel, k: int = 30) -> list[int]:
    """Return list of K image_ids that span the viewpoint sphere.

    Args:
        model: COLMAP outputs (cameras, images, points3D)
        k: number of reference views to pick (typically 20-50)

    Returns:
        List of length min(k, len(images)) with image_ids in selection order.
        First image_id is the "canonical" view (highest 3D coverage).
    """
    raise NotImplementedError(
        "Phase 1 task: implement viewpoint-sphere FPS. "
        "See docs/path_b_implementation_roadmap.md §2.2 for the algorithm."
    )


def _viewpoint_directions(model: ColmapModel) -> tuple[np.ndarray, np.ndarray]:
    """Compute camera-to-object direction unit vectors for every image.

    Returns:
        image_ids: (N,) int — image IDs in deterministic order
        directions: (N, 3) float32 — unit vectors from object centroid to camera
    """
    raise NotImplementedError
