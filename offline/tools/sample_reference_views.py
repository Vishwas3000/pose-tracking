"""Pick K representative reference views from the SfM source frames.

Strategy: viewpoint-sphere farthest-point-sampling (FPS).

  1. Camera position in world space = -R.T @ t for each image's W2C pose.
  2. Project onto the unit sphere centered at the object centroid (mean of
     points3D — for bbox-cropped models this is essentially the bbox center).
  3. Greedy FPS by angular distance: start at index 0, then iteratively
     append the view whose minimum angle to any selected view is largest.

This guarantees rotational coverage of the object's viewable hemisphere
and avoids picking redundant near-duplicate views.
"""

from __future__ import annotations

import numpy as np

from .colmap_io import ColmapModel


def _camera_position(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """COLMAP stores world->camera (qvec, tvec). Camera center in world = -R.T @ t."""
    from . import _colmap_upstream as _u
    R = _u.qvec2rotmat(qvec)
    return -R.T @ tvec


def _viewpoint_directions(model: ColmapModel) -> tuple[np.ndarray, np.ndarray]:
    """Return (image_ids, unit_vectors_from_centroid_to_camera).

    image_ids ordered ascending so output is deterministic.
    """
    centroid = np.mean(np.stack([p.xyz for p in model.points3D.values()]), axis=0)
    image_ids = np.array(sorted(model.images.keys()), dtype=np.int64)
    dirs = np.zeros((len(image_ids), 3), dtype=np.float64)
    for i, iid in enumerate(image_ids):
        img = model.images[iid]
        cam_pos = _camera_position(img.qvec, img.tvec)
        v = cam_pos - centroid
        n = np.linalg.norm(v)
        dirs[i] = v / n if n > 0 else v
    return image_ids, dirs


def farthest_point_sphere(model: ColmapModel, k: int = 30) -> list[int]:
    """Return up to k image_ids that span the viewpoint sphere.

    First image is the one with smallest ID (deterministic seed).
    Subsequent picks maximize the minimum angular distance to anything
    already chosen.
    """
    image_ids, dirs = _viewpoint_directions(model)
    n = len(image_ids)
    k = min(k, n)
    if k == 0:
        return []

    selected = [0]
    # Cosine sim between every direction and currently-selected set;
    # we'll track per-frame "max similarity to any selected" and pick
    # whichever has the smallest max-sim (= largest min-angle).
    sims_to_set = dirs @ dirs[0]
    for _ in range(1, k):
        next_idx = int(np.argmin(sims_to_set))
        selected.append(next_idx)
        new_sims = dirs @ dirs[next_idx]
        sims_to_set = np.maximum(sims_to_set, new_sims)

    return [int(image_ids[i]) for i in selected]
