"""Custom binary `.bundle` format reader/writer.

Why custom binary instead of .npz/zip?
    Loads in <5 ms on mobile vs ~50 ms for .npz unpacking. Saves a noticeable
    chunk of the first-frame budget.

See `shared/bundle_format.md` for the canonical byte layout.

The high-level structure:
    [HEADER]                  fixed-size, version + counts + offsets
    [GLOBAL points3D]         (M, 3) float32
    [GLOBAL bbox3d]           (8, 3) float32
    [GLOBAL ref_global_emb]   (K, D_global) float32   (optional)
    [PER REF block × K]       per-reference keypoints/descriptors/pose/mapping
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

BUNDLE_MAGIC = b"PBND"   # "Pose Bundle"
BUNDLE_VERSION = 1


@dataclass
class ReferenceView:
    image_id: int                   # original COLMAP image ID (debug use)
    keypoints: np.ndarray           # (N, 2) float32
    descriptors: np.ndarray         # (N, 128) float32, L2-normalized
    pt3d_indices: np.ndarray        # (N,) int32 — index into global points3D
    pose: np.ndarray                # (4, 4) float32 — world→camera
    K: np.ndarray                   # (3, 3) float32 — camera intrinsics


@dataclass
class Bundle:
    points3d: np.ndarray            # (M, 3) float32
    bbox3d: np.ndarray              # (8, 3) float32
    ref_global_emb: np.ndarray | None   # (K, D_global) float32 or None
    refs: list[ReferenceView]


def write(out_path: Path, bundle: Bundle) -> None:
    """Serialize a Bundle to a custom binary file."""
    raise NotImplementedError(
        "Phase 1 task: implement binary writer. "
        "See ../../shared/bundle_format.md for layout."
    )


def load_bundle(path: Path) -> Bundle:
    """Deserialize a Bundle from a custom binary file."""
    raise NotImplementedError


def inspect(path: Path) -> None:
    """Print a human-readable summary of a bundle (CLI use)."""
    b = load_bundle(path)
    print(f"=== {path} ===")
    print(f"K (refs)        : {len(b.refs)}")
    print(f"M (3D points)   : {len(b.points3d)}")
    print(f"3D bbox         : {b.bbox3d.tolist()}")
    if b.ref_global_emb is not None:
        print(f"Global emb dim  : {b.ref_global_emb.shape[1]}")
    for i, ref in enumerate(b.refs[:5]):
        print(f"  ref {i}: image_id={ref.image_id}  N={len(ref.keypoints)}  "
              f"desc_dim={ref.descriptors.shape[1]}")
    if len(b.refs) > 5:
        print(f"  ... +{len(b.refs) - 5} more")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--inspect":
        inspect(Path(sys.argv[2]))
    else:
        print("Usage: python -m tools.bundle_writer --inspect path/to/object.bundle")
