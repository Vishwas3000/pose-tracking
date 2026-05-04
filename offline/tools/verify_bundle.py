"""Sanity-check a generated .bundle.

Verifies:
  1. The bundle parses cleanly (basic format check)
  2. For each reference view, projecting its 3D points through the stored
     pose lands within ~2 px of the stored 2D keypoints (reprojection check)
  3. Round-trip: use one reference image as both query and database;
     run XFeat-style matching against itself; expect >100 inliers
     (validates that the descriptors are intact and matchable)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .bundle_writer import Bundle, load_bundle


def verify(path: Path, max_reproj_px: float = 2.0) -> bool:
    """Run all checks. Returns True if everything passes."""
    print(f"=== verify_bundle: {path} ===")
    b = load_bundle(path)

    ok = True
    ok &= _check_format(b)
    ok &= _check_reprojection(b, max_reproj_px)
    # ok &= _check_round_trip(b)   # requires aliked_inference + matcher to be implemented

    if ok:
        print("✓ bundle is valid")
    else:
        print("✗ bundle has issues — see above")
    return ok


def _check_format(b: Bundle) -> bool:
    """Basic shape/dtype consistency."""
    raise NotImplementedError("Phase 1 task")


def _check_reprojection(b: Bundle, max_reproj_px: float) -> bool:
    """For each reference view, project 3D points → 2D and compare to stored keypoints."""
    raise NotImplementedError("Phase 1 task")


def _check_round_trip(b: Bundle) -> bool:
    """Use one ref image as a query against the bundle; verify high inlier count."""
    raise NotImplementedError("Phase 1 task — requires aliked_inference")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tools.verify_bundle path/to/object.bundle")
        sys.exit(1)
    success = verify(Path(sys.argv[1]))
    sys.exit(0 if success else 1)
