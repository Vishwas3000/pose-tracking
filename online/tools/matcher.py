"""Match query ALIKED descriptors against per-reference descriptors.

Strategy for the desktop reference: brute-force cosine similarity (descriptors
are L2-normalized) + Lowe ratio test + mutual-NN. This is much simpler than
LightGlue but produces enough good 2D-3D correspondences to validate the rest
of the pipeline. We'll swap in a LightGlue ONNX session as a follow-up.

The mobile pipeline is expected to use LightGlue; the spec lives at
shared/algorithms/matcher_funnel.md.
"""

from __future__ import annotations

import numpy as np

from .bundle_loader import Bundle, ReferenceView


def _mutual_nn_lowe(q_desc: np.ndarray, r_desc: np.ndarray,
                    ratio: float = 0.85, sim_min: float = 0.5
                    ) -> np.ndarray:
    """Return (M, 2) array of (q_idx, r_idx) pairs that pass mutual-NN + ratio."""
    sim = q_desc @ r_desc.T   # (Nq, Nr) in [-1, 1]
    if sim.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    # q -> r: top-1 + top-2 for ratio test
    q_sort = np.argsort(-sim, axis=1)
    q_best = q_sort[:, 0]
    if sim.shape[1] >= 2:
        q_top1 = sim[np.arange(sim.shape[0]), q_sort[:, 0]]
        q_top2 = sim[np.arange(sim.shape[0]), q_sort[:, 1]]
        # Lowe's ratio test in *distance* space; for cosine sim the analogue is
        # 1 - top1 < ratio * (1 - top2)  (smaller "distance" by ratio margin).
        q_pass = (1 - q_top1) < ratio * (1 - q_top2)
    else:
        q_top1 = sim[:, 0]
        q_pass = np.ones(sim.shape[0], dtype=bool)
    q_pass &= (q_top1 >= sim_min)

    # r -> q: mutual-NN check
    r_best = sim.argmax(axis=0)            # for each ref kpt, the best query
    matches = []
    for qi in np.where(q_pass)[0]:
        ri = q_best[qi]
        if r_best[ri] == qi:
            matches.append((qi, ri))
    return np.array(matches, dtype=np.int64).reshape(-1, 2)


def match_query_to_bundle(q_kpts: np.ndarray, q_desc: np.ndarray, bundle: Bundle,
                          ratio: float = 0.85, sim_min: float = 0.5,
                          ref_subset: np.ndarray | None = None,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate matches across reference views.

    Args:
        ref_subset: optional indices into bundle.refs — if provided, only those
            references are matched (used by DINOv2 retrieval to skip ~83% of
            refs when N=5 of K=30).

    Returns (img_pts (M, 2) query pixel coords, world_pts (M, 3)).
    Per-3D-point dedup: if a single 3D point is matched from multiple refs,
    keep the match with highest cosine similarity.
    """
    best_per_pt3d: dict[int, tuple[float, int]] = {}   # pt3d_idx -> (sim, q_kpt_idx)
    refs_iter = (bundle.refs if ref_subset is None
                 else [bundle.refs[i] for i in ref_subset])
    for ref in refs_iter:
        if len(ref.descriptors) == 0:
            continue
        pairs = _mutual_nn_lowe(q_desc, ref.descriptors, ratio=ratio, sim_min=sim_min)
        for qi, ri in pairs:
            pt3d_idx = int(ref.pt3d_indices[ri])
            sim = float(q_desc[qi] @ ref.descriptors[ri])
            prev = best_per_pt3d.get(pt3d_idx)
            if prev is None or sim > prev[0]:
                best_per_pt3d[pt3d_idx] = (sim, int(qi))

    if not best_per_pt3d:
        return (np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32))

    pt3d_idxs = np.array(list(best_per_pt3d.keys()), dtype=np.int64)
    q_idxs    = np.array([best_per_pt3d[k][1] for k in pt3d_idxs], dtype=np.int64)
    img_pts   = q_kpts[q_idxs].astype(np.float32)
    world_pts = bundle.points3d[pt3d_idxs].astype(np.float32)
    return img_pts, world_pts
