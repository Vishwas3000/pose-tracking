"""Global retrieval embeddings (DINOv2 ViT-S/14 INT8).

Used by the online pipeline to filter the bundle's K reference views down
to the top-N nearest-by-cosine before brute-force descriptor matching:

    query frame -> DINOv2 (384-D, L2-normalized)
    -> cosine vs bundle.ref_global_emb (K × 384)
    -> top-N indices (typically N=5)
    -> only run brute-force ALIKED matcher against those N refs

Same module is used at bundle-build time (over reference frames) and at
inference time (over query frames) so the embedding space stays identical.

Model: onnx-community/dinov2-small / onnx/model_int8.onnx (24 MB)
  input  pixel_values  [B, 3, 224, 224] float — ImageNet-normalized RGB
  output last_hidden_state [B, 257, 384] — token 0 (CLS) is the global embedding
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE    = 224         # center crop
_RESIZE_SHORT  = 256         # resize shortest edge first
RETRIEVAL_DIM  = 384


def _preprocess(img: Image.Image) -> np.ndarray:
    """PIL -> (1, 3, 224, 224) float32, ImageNet-normalized.

    Matches the preprocessor_config.json from onnx-community/dinov2-small:
    resize shortest edge to 256 (bicubic), center-crop 224, mean/std normalize.
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = _RESIZE_SHORT / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - _INPUT_SIZE) // 2
    top  = (new_h - _INPUT_SIZE) // 2
    img = img.crop((left, top, left + _INPUT_SIZE, top + _INPUT_SIZE))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr.astype(np.float32)


class DinoV2Embedder:
    """ONNX wrapper. Returns L2-normalized 384-D global embeddings."""
    embedding_dim = RETRIEVAL_DIM

    def __init__(self, onnx_path: Path | str,
                 providers: list[str] | None = None):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, image_path: Path | str) -> np.ndarray:
        with Image.open(str(image_path)) as im:
            tensor = _preprocess(im)
        out = self.session.run(None, {self.input_name: tensor})[0]   # (1, 257, 384)
        cls = out[0, 0, :].astype(np.float32)
        n = np.linalg.norm(cls)
        return cls / n if n > 0 else cls

    def embed_batch(self, image_paths: list[Path]) -> np.ndarray:
        """(K, 384) float32 L2-normalized."""
        return np.stack([self.embed(p) for p in image_paths]).astype(np.float32)
