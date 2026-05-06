"""Frame -> pose orchestrator. Desktop reference for the on-device pipeline.

Per-frame flow:
  1. XFeat on the query image -> kpts + 64-D descriptors (PyTorch)
  2. Brute-force descriptor matching against every reference view in the bundle
     -> a deduped set of (query_kpt, world_xyz) pairs
  3. cv2.solvePnPRansac (EPnP) -> pose + inlier mask

The mobile pipeline replaces step 1 with a Core ML XFeat runner and step 3
with the native Swift EPnP in `ios/objectCapture/Inference/EPnP.swift`; the
algorithmic flow is identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Re-use the offline XFeat runner so we don't drift between train/test paths.
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from offline.tools.xfeat_inference import XFeatRunner    # noqa: E402

from .bundle_loader import Bundle, load                  # noqa: E402
from .matcher import match_query_to_bundle               # noqa: E402
from .pose_solver import PoseEstimate, solve_pose        # noqa: E402


@dataclass
class FrameResult:
    pose: np.ndarray | None             # (4, 4) world->camera, or None if lost
    n_matches: int
    n_inliers: int
    n_kpts: int
    ref_subset: list[int] | None = None     # ref indices used after DINOv2 filter


class PoseTracker:
    def __init__(self, bundle_path: Path | str, xfeat_model: Path | str,
                 dinov2_onnx: Path | str | None = None,
                 # XFeat descriptors are 64-D and tighter than ALIKED's 128-D:
                 # upstream's recommended cossim threshold is 0.82. Below that
                 # we get hundreds of false-positive matches that overwhelm
                 # RANSAC.
                 ratio: float = 0.9, sim_min: float = 0.82,
                 reproj_err_px: float = 8.0,
                 retrieval_top_n: int = 5,
                 top_k: int = 4096):
        self.bundle: Bundle = load(bundle_path)
        # XFeat runs on CUDA via PyTorch (nvidia-cu12 pip wheels in the
        # venv supply the runtime libs).
        self.runner = XFeatRunner(xfeat_model, top_k=top_k)

        # DINOv2 retrieval — enabled only if (a) caller passed a model and
        # (b) the bundle actually has retrieval embeddings.
        self.embedder = None
        self.retrieval_top_n = retrieval_top_n
        if dinov2_onnx is not None and self.bundle.ref_global_emb is not None:
            from .retrieval import DinoV2Embedder
            self.embedder = DinoV2Embedder(dinov2_onnx)

        self.ratio = ratio
        self.sim_min = sim_min
        self.reproj_err_px = reproj_err_px
        self.K = self.bundle.refs[0].K.astype(np.float64)

    def process(self, image_path: Path | str) -> FrameResult:
        feats = self.runner.extract(image_path)
        q_kpts = feats["keypoints"]
        q_desc = feats["descriptors"]

        ref_subset = None
        if self.embedder is not None:
            from .retrieval import select_top_refs
            q_emb = self.embedder.embed(image_path)
            idx, _ = select_top_refs(q_emb, self.bundle.ref_global_emb,
                                     top_n=self.retrieval_top_n)
            ref_subset = idx

        img_pts, world_pts = match_query_to_bundle(
            q_kpts, q_desc, self.bundle,
            ratio=self.ratio, sim_min=self.sim_min,
            ref_subset=ref_subset,
        )
        sub_list = ref_subset.tolist() if ref_subset is not None else None
        if len(img_pts) == 0:
            return FrameResult(pose=None, n_matches=0, n_inliers=0,
                               n_kpts=len(q_kpts),
                               ref_subset=sub_list)

        est = solve_pose(world_pts, img_pts, self.K,
                         reproj_err_px=self.reproj_err_px)
        if est is None:
            return FrameResult(pose=None, n_matches=len(img_pts), n_inliers=0,
                               n_kpts=len(q_kpts),
                               ref_subset=sub_list)

        return FrameResult(pose=est.pose, n_matches=len(img_pts),
                           n_inliers=len(est.inliers),
                           n_kpts=len(q_kpts),
                           ref_subset=sub_list)
