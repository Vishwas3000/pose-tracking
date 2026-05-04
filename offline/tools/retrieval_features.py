"""Optional: compute global retrieval features per reference view.

When K (number of reference views) > ~10, a global retrieval embedder
dramatically speeds up the runtime per-frame matching: instead of running
LightGlue against ALL K references each frame, we cosine-NN-rank
references first and only match against the top 2-5.

Three options, in order of (size, quality):
  - MobileNetV3-Large penultimate layer  — ~6 MB,  80% retrieval acc
  - MobileViT-XS                         — ~5 MB,  85% retrieval acc
  - DINOv2 ViT-S/14 INT8                 — ~22 MB, 95% retrieval acc

For single-object apps with K ≤ 10, **skip this step entirely** — match
against all K every frame; it's faster than DINOv2 forward pass + match
× 2.

See `docs/path_b_implementation_roadmap.md` §5.4 for the full comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class GlobalEmbedder(Protocol):
    """Common interface for retrieval embedders."""

    embedding_dim: int

    def embed(self, image_path: Path | str) -> np.ndarray:
        """Returns an L2-normalized 1D embedding (embedding_dim,)."""
        ...


class MobileNetEmbedder:
    """Fast, mobile-friendly. Uses MobileNetV3-Large penultimate layer.

    Recommended default for K = 10-30 reference views per object.
    """
    embedding_dim = 1280  # MobileNetV3-Large penultimate

    def __init__(self):
        raise NotImplementedError("Phase 1 task: wrap torchvision MobileNetV3")

    def embed(self, image_path: Path | str) -> np.ndarray:
        raise NotImplementedError


class DinoV2Embedder:
    """Highest-quality retrieval. Use only if K > 50 OR multi-object catalog.

    ViT-S/14 quantized to INT8 is the mobile-deployable variant.
    """
    embedding_dim = 384  # ViT-S/14

    def __init__(self):
        raise NotImplementedError("Phase 1 task: wrap HF transformers DINOv2")

    def embed(self, image_path: Path | str) -> np.ndarray:
        raise NotImplementedError
