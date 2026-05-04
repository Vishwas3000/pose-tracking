"""Run ALIKED on reference images via ONNX Runtime.

Model: bukuroo/ALIKED-LightGlue-ONNX  (aliked-n16rot-top1k-640.onnx)
  input:   image  [1, 3, 640, 640] float32 — letterboxed RGB CHW, [0, 1]
  outputs: keypoints   [1000, 2]   float32 — NORMALIZED [-1, 1] in input tensor space
           descriptors [1000, 128] float32 — L2-normalized
           scores      [1000]      float32 — detection confidence in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import onnxruntime as ort
from PIL import Image


MODEL_INPUT = 640


class AlikedFeatures(TypedDict):
    keypoints: np.ndarray   # (N, 2) float32 — pixel coords in ORIGINAL image space
    descriptors: np.ndarray # (N, 128) float32 — L2-normalized
    scores: np.ndarray      # (N,) float32


@dataclass
class _Letterbox:
    """Maps original-image pixel coords -> letterboxed canvas pixel coords:
        x_canvas = x_orig * scale + offset_x
        y_canvas = y_orig * scale + offset_y
    The ONNX model outputs normalized coords in [-1, 1] over the canvas.
    """
    scale: float
    offset_x: float
    offset_y: float
    canvas: int = MODEL_INPUT

    def normalized_to_original(self, kpts_norm: np.ndarray) -> np.ndarray:
        """ALIKED [-1, 1] coords -> original-image pixel coords."""
        half = self.canvas / 2.0
        x_canvas = (kpts_norm[:, 0] + 1.0) * half
        y_canvas = (kpts_norm[:, 1] + 1.0) * half
        out = np.empty_like(kpts_norm, dtype=np.float32)
        out[:, 0] = (x_canvas - self.offset_x) / self.scale
        out[:, 1] = (y_canvas - self.offset_y) / self.scale
        return out


def _letterbox(img: Image.Image, target: int = MODEL_INPUT) -> tuple[np.ndarray, _Letterbox]:
    """Resize preserving aspect ratio, pad to target×target with zeros."""
    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), (0, 0, 0))
    off_x = (target - new_w) // 2
    off_y = (target - new_h) // 2
    canvas.paste(img_resized, (off_x, off_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0     # (H, W, 3) in [0, 1]
    arr = arr.transpose(2, 0, 1)[None, ...]                 # (1, 3, H, W)
    return arr, _Letterbox(scale=scale, offset_x=off_x, offset_y=off_y)


class AlikedRunner:
    """ONNX Runtime wrapper for ALIKED. Returns keypoints in ORIGINAL image space."""

    def __init__(
        self,
        onnx_path: Path | str,
        score_threshold: float = 0.0,
        providers: list[str] | None = None,
    ):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.score_threshold = score_threshold
        self.input_name = self.session.get_inputs()[0].name

    def extract(self, image_path: Path | str) -> AlikedFeatures:
        with Image.open(str(image_path)) as im:
            img = im.convert("RGB")
            orig_size = img.size  # (W, H)
            tensor, lb = _letterbox(img)
        kpts, descs, scores = self.session.run(None, {self.input_name: tensor})
        kpts = lb.normalized_to_original(kpts)

        # Filter: keep only kpts within original-image bounds and above threshold.
        # ALIKED pads to a fixed top-K (1000); padding rows have score≈0.
        w, h = orig_size
        in_bounds = (
            (kpts[:, 0] >= 0) & (kpts[:, 0] < w)
            & (kpts[:, 1] >= 0) & (kpts[:, 1] < h)
        )
        keep = in_bounds & (scores >= self.score_threshold)
        return {
            "keypoints":   kpts[keep].astype(np.float32),
            "descriptors": descs[keep].astype(np.float32),
            "scores":      scores[keep].astype(np.float32),
        }
