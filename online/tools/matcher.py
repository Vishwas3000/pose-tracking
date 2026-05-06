"""Match query descriptors against per-reference descriptors.

Strategy for the desktop reference: brute-force cosine similarity (descriptors
are L2-normalized) + Lowe ratio test + mutual-NN. This is much simpler than
LightGlue but produces enough good 2D-3D correspondences to validate the rest
of the pipeline.

Implementation notes:
  - Single big matmul of q_desc against ALL refs concatenated, then per-ref
    sub-blocks for the mutual-NN step. With XFeat's 4096 query keypoints
    against ~15k bundle keypoints across 30 refs, this drops from ~2.7s to
    ~30 ms on CPU thanks to BLAS amortization.
  - Vectorized mutual-NN check: avoids the per-passing-query Python loop the
    original implementation had.
"""

from __future__ import annotations

import numpy as np

from .bundle_loader import Bundle, ReferenceView


def _per_ref_mutual_nn_vectorized(sim_block: np.ndarray,
                                  ratio: float, sim_min: float
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized mutual-NN + Lowe ratio test on a single (Nq, Nr) sim block.

    Returns (q_idx, r_idx, sim) arrays of length M (matches that survived).
    """
    if sim_block.size == 0:
        return (np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32))

    Nq, Nr = sim_block.shape
    # Top-2 per query row via argpartition (O(Nq*Nr) vs argsort's Nq*Nr*log Nr).
    if Nr >= 2:
        # argpartition picks indices of top-2 (unsorted).
        top2_idx = np.argpartition(-sim_block, 1, axis=1)[:, :2]
        rows = np.arange(Nq)[:, None]
        top2_vals = sim_block[rows, top2_idx]                    # (Nq, 2)
        # Order so column 0 = top1.
        order = np.argsort(-top2_vals, axis=1)
        top2_idx = np.take_along_axis(top2_idx, order, axis=1)
        top2_vals = np.take_along_axis(top2_vals, order, axis=1)
        q_best  = top2_idx[:, 0]
        q_top1  = top2_vals[:, 0]
        q_top2  = top2_vals[:, 1]
        q_pass  = (1 - q_top1) < ratio * (1 - q_top2)
    else:
        q_best = np.zeros(Nq, dtype=np.int64)
        q_top1 = sim_block[:, 0]
        q_pass = np.ones(Nq, dtype=bool)
    q_pass &= (q_top1 >= sim_min)

    # r -> q: best query for each ref kpt.
    r_best = sim_block.argmax(axis=0)                            # (Nr,)
    # Mutual: r_best[q_best[qi]] == qi
    qi_all = np.arange(Nq)
    mutual = r_best[q_best] == qi_all
    keep = q_pass & mutual
    return qi_all[keep], q_best[keep], q_top1[keep]


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
    refs_iter = (list(bundle.refs) if ref_subset is None
                 else [bundle.refs[i] for i in ref_subset])
    refs_iter = [r for r in refs_iter if len(r.descriptors) > 0]
    if not refs_iter:
        return (np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32))

    # Concatenate refs + record per-ref slice boundaries so we can split
    # the big sim matrix back into per-ref blocks for mutual-NN.
    sizes      = np.array([len(r.descriptors) for r in refs_iter], dtype=np.int64)
    boundaries = np.concatenate([[0], np.cumsum(sizes)])
    all_descs  = np.concatenate([r.descriptors for r in refs_iter], axis=0)
    all_pt3d   = np.concatenate([r.pt3d_indices for r in refs_iter], axis=0)

    # Single big matmul — BLAS amortizes much better than 30 small matmuls.
    sim_full = q_desc @ all_descs.T                              # (Nq, sum_Nr)

    # best_per_pt3d: pt3d_idx -> (sim, q_idx)
    best_per_pt3d: dict[int, tuple[float, int]] = {}
    for ki, ref in enumerate(refs_iter):
        s, e = int(boundaries[ki]), int(boundaries[ki + 1])
        sim_block = sim_full[:, s:e]
        qi_arr, ri_arr, sim_arr = _per_ref_mutual_nn_vectorized(
            sim_block, ratio=ratio, sim_min=sim_min,
        )
        if qi_arr.size == 0:
            continue
        pt3d_arr = ref.pt3d_indices[ri_arr].astype(np.int64)
        for qi, sim_v, pt3d in zip(qi_arr, sim_arr, pt3d_arr):
            prev = best_per_pt3d.get(int(pt3d))
            if prev is None or float(sim_v) > prev[0]:
                best_per_pt3d[int(pt3d)] = (float(sim_v), int(qi))

    if not best_per_pt3d:
        return (np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32))

    pt3d_idxs = np.array(list(best_per_pt3d.keys()), dtype=np.int64)
    q_idxs    = np.array([best_per_pt3d[k][1] for k in pt3d_idxs], dtype=np.int64)
    img_pts   = q_kpts[q_idxs].astype(np.float32)
    world_pts = bundle.points3d[pt3d_idxs].astype(np.float32)
    return img_pts, world_pts
