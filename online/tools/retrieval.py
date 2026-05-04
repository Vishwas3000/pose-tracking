"""Top-N reference selection via DINOv2 cosine retrieval.

Re-uses the same DinoV2Embedder used at bundle-build time so embeddings live
in identical model weights / normalization.

Cost on CPU: DINOv2-S/14 INT8 forward at 224×224 ≈ 80 ms/frame on this box.
Saving: matcher work scales linearly with the ref subset size — going from
K=30 to top-N=5 cuts the brute-force matching step from ~500 ms to ~80 ms.
For larger K this gap widens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from offline.tools.retrieval_features import DinoV2Embedder  # noqa: E402

__all__ = ["DinoV2Embedder", "select_top_refs"]


def select_top_refs(query_emb: np.ndarray, ref_embs: np.ndarray,
                    top_n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices (N,), cosine_sim (N,)) for the top-N refs by cosine.

    Both query_emb and ref_embs are assumed to be L2-normalized so the dot
    product is the cosine similarity in [-1, 1].
    """
    sims = ref_embs @ query_emb            # (K,)
    n = min(top_n, len(sims))
    idx = np.argsort(-sims)[:n]            # descending
    return idx, sims[idx]
