"""Run XFeat (verlab/accelerated_features, CVPR 2024) on reference images.

Drop-in replacement for the previous `AlikedRunner`. The public interface is
identical so `build_object_bundle.py` and `online/tools/pipeline.py` only need
an import + class-name swap.

Why XFeat instead of ALIKED:
- ALIKED uses deformable convolutions which don't lower to Core ML. ORT's
  Core ML EP fragments the ALIKED graph into 38 NeuralNetwork partitions and
  blew iOS memory to 1.9 GB → jetsam. XFeat is a plain ResNet-style net with
  no DCN, no grid_sample — converts cleanly to .mlpackage.
- 64-D descriptors instead of 128-D (half the bundle size, ~equivalent
  matching quality on the figurine dataset per upstream benchmarks).

Why PyTorch and not ONNX in the offline pipeline:
- XFeat's NMS produces a dynamic-length keypoint list, which trips ONNX
  shape inference. Upstream's official path is PyTorch; we vendor the
  three required modules under `_xfeat/` to keep this file self-contained.

Model: `verlab/accelerated_features` — checkpoint at `shared/models/xfeat.pt`.
  input:   image  [1, 3, H, W] float32 — RGB CHW, [0, 1] (H, W any size; XFeat
           internally resizes to nearest multiple of 32)
  outputs: keypoints   [N, 2]   float32 — pixel coords on the (resized) input
           descriptors [N, 64]  float32 — L2-normalized
           scores      [N]      float32 — detection confidence
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image

# Vendored upstream XFeat — see offline/tools/_xfeat/.
from ._xfeat.xfeat import XFeat


# Match the iOS Core ML model's fixed 640x640 input. Although XFeat itself
# handles arbitrary shapes, Apple's mlpackage we ship to iOS is exported at
# a fixed 640x640 input — so the offline bundle MUST be built with the same
# letterbox preprocessing for descriptor distribution parity. Cross-resolution
# matching across XFeat's L2-normalized 64-D descriptors is lossy, and any
# mismatch shows up as missing inliers downstream.
DEFAULT_TARGET_SIZE = 640


class XFeatFeatures(TypedDict):
    keypoints:   np.ndarray   # (N, 2) float32 — pixel coords in ORIGINAL image
    descriptors: np.ndarray   # (N, 64) float32 — L2-normalized
    scores:      np.ndarray   # (N,)   float32


@dataclass
class _Letterbox:
    """Maps original-image pixel coords -> letterboxed canvas coords:
        canvas_x = orig_x * scale + offset_x
        canvas_y = orig_y * scale + offset_y
    """
    scale:    float
    offset_x: float
    offset_y: float
    canvas:   int = DEFAULT_TARGET_SIZE

    def canvas_to_original(self, kpts: np.ndarray) -> np.ndarray:
        out = np.empty_like(kpts, dtype=np.float32)
        out[:, 0] = (kpts[:, 0] - self.offset_x) / self.scale
        out[:, 1] = (kpts[:, 1] - self.offset_y) / self.scale
        return out


def _letterbox_for_xfeat(img: Image.Image,
                         target: int = DEFAULT_TARGET_SIZE
                         ) -> tuple[np.ndarray, _Letterbox]:
    """Proportional resize + zero-pad to (target, target). Mirrors the
    AlikedRunner.swift preprocessing so bundle and iOS see identical canvases.
    Returns (1, 3, target, target) float32 tensor in [0, 1] + letterbox record.
    """
    w, h = img.size
    scale = target / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR).convert("RGB")
    canvas = Image.new("RGB", (target, target), (0, 0, 0))
    off_x = (target - new_w) // 2
    off_y = (target - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0     # (target, target, 3)
    arr = arr.transpose(2, 0, 1)[None, ...]                # (1, 3, target, target)
    return arr, _Letterbox(scale=scale, offset_x=float(off_x),
                            offset_y=float(off_y), canvas=target)


class XFeatRunner:
    """Drop-in replacement for the previous AlikedRunner.

    Same constructor signature so build_object_bundle.py + online pipeline
    can swap the type without touching call sites. `onnx_path` is renamed
    `model_path` since we now load a PyTorch checkpoint (`.pt`) — but the
    arg position is unchanged.
    """

    def __init__(
        self,
        model_path: Path | str,
        score_threshold: float = 0.0,
        top_k: int = 1000,
        target_size: int = DEFAULT_TARGET_SIZE,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # NOTE upstream constructor wires its own self.dev internally; we
        # pass `weights=` and let it pick CUDA if available.
        self.model = XFeat(weights=str(model_path), top_k=top_k)
        self.model.dev = torch.device(device)
        self.model.net = self.model.net.to(self.model.dev)
        self.model.net.eval()

        self.score_threshold = score_threshold
        self.top_k = top_k
        self.target_size = target_size

    @torch.inference_mode()
    def extract(self, image_path: Path | str) -> XFeatFeatures:
        with Image.open(str(image_path)) as im:
            img = im.convert("RGB")
            orig_w, orig_h = img.size
            arr, lb = _letterbox_for_xfeat(img, target=self.target_size)

        x = torch.from_numpy(arr).to(self.model.dev)
        out = self.model.detectAndCompute(x, top_k=self.top_k)[0]

        kpts_canvas = out["keypoints"].cpu().numpy().astype(np.float32)    # (N, 2)
        descs       = out["descriptors"].cpu().numpy().astype(np.float32)  # (N, 64)
        scores      = out["scores"].cpu().numpy().astype(np.float32)       # (N,)

        # Map keypoints from letterbox canvas back to original image space.
        kpts = lb.canvas_to_original(kpts_canvas)

        # Filter: in-bounds + above score threshold. (XFeat already filters
        # score>0 in detectAndCompute, but we also enforce the user's higher
        # score_threshold here for parity with the ALIKED runner.)
        in_bounds = (
            (kpts[:, 0] >= 0) & (kpts[:, 0] < orig_w)
            & (kpts[:, 1] >= 0) & (kpts[:, 1] < orig_h)
        )
        keep = in_bounds & (scores >= self.score_threshold)
        return {
            "keypoints":   kpts[keep],
            "descriptors": descs[keep],
            "scores":      scores[keep],
        }
