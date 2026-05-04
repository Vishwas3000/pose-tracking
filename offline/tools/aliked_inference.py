"""Run ALIKED on reference images.

ALIKED produces sparse keypoints + 128-D L2-normalized descriptors per
image. It's the detector-based feature extractor that replaces XFeat in
this project.

Repo: https://github.com/Shiaoming/ALIKED
ONNX exports: https://github.com/cvg/LightGlue-ONNX

This module wraps the official PyTorch implementation. We'll add an ONNX
runtime path later in Phase 2 for parity testing on mobile.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import torch


class AlikedFeatures(TypedDict):
    keypoints: np.ndarray   # (N, 2) pixel coords in input image space
    descriptors: np.ndarray # (N, 128) L2-normalized float32
    scores: np.ndarray      # (N,) detection confidence


class AlikedRunner:
    """Wraps the official ALIKED model for batched inference on reference images.

    Usage:
        aliked = AlikedRunner(top_k=1024, threshold=0.005)
        feats = aliked.extract("path/to/image.png")
        # feats["keypoints"]: (N, 2)
        # feats["descriptors"]: (N, 128)  L2-normalized
    """

    def __init__(
        self,
        model_name: str = "aliked-n16",       # 'aliked-n16' is the standard mobile variant
        top_k: int = 1024,
        threshold: float = 0.005,
        device: str = "auto",
    ):
        self.top_k = top_k
        self.threshold = threshold
        self.device = self._resolve_device(device)
        # TODO: import from the installed `aliked` package once requirements.txt installs work
        # from aliked import ALIKED
        # self.model = ALIKED(model_name=model_name, top_k=top_k,
        #                     detection_threshold=threshold, device=self.device)
        raise NotImplementedError("Phase 1 task: wire up the ALIKED model")

    def extract(self, image_path: Path | str) -> AlikedFeatures:
        """Run ALIKED on one image and return its features."""
        raise NotImplementedError

    def extract_batch(self, image_paths: list[Path]) -> list[AlikedFeatures]:
        """Run ALIKED on multiple images. Default impl: loop. Override for batched ANE/GPU."""
        return [self.extract(p) for p in image_paths]

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
