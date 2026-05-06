"""Global retrieval embeddings (DINOv2 ViT-S/14).

Used by the online pipeline to filter the bundle's K reference views down
to the top-N nearest-by-cosine before brute-force descriptor matching:

    query frame -> DINOv2 (384-D, L2-normalized)
    -> cosine vs bundle.ref_global_emb (K × 384)
    -> top-N indices (typically N=5–10)
    -> only run brute-force XFeat matcher against those N refs

The same module is used at bundle-build time AND at inference time so the
embedding space stays identical.

Two backends:
- `DinoV2Embedder` (legacy, INT8 ONNX from onnx-community/dinov2-small).
  Kept around for older bundles that were built against it.
- `DinoV2EmbedderPT` (current, FP32 PyTorch from torch.hub).
  Required for parity with the iOS Core ML model exported by
  `_export_dinov2_coreml.py` (which traces the same PyTorch source).

Model: ViT-S/14, trained at 518x518.
  input  pixel_values  [B, 3, 224, 224] float32 — ImageNet-normalized RGB
  output CLS token     [B, 384]                 — what we L2-normalize and store
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
except ImportError:
    ort = None  # PT-only deployments don't need ORT


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
    """ONNX wrapper. Returns L2-normalized 384-D global embeddings.

    LEGACY: kept for bundles that were built against the INT8 ONNX. The
    iOS Core ML mlpackage is exported from PyTorch (see
    `_export_dinov2_coreml.py`), so for new bundles prefer
    `DinoV2EmbedderPT` to keep embedding spaces consistent across Python
    desktop ↔ iOS.
    """
    embedding_dim = RETRIEVAL_DIM

    def __init__(self, onnx_path: Path | str,
                 providers: list[str] | None = None):
        if ort is None:
            raise RuntimeError("onnxruntime not installed — pip install it or use DinoV2EmbedderPT")
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


class DinoV2EmbedderPT:
    """PyTorch (torch.hub) wrapper. Returns L2-normalized 384-D embeddings.

    Required for parity with the iOS Core ML model exported by
    `offline/tools/_export_dinov2_coreml.py`. Both go through the same
    `dinov2_vits14` weights from torch.hub; iOS runs them at FP16 on the
    ANE, desktop here runs FP32 on whatever device torch picks. Cosine
    similarity space is robust to the FP32 vs FP16 drift (~1% per element
    → cosine sim stays >0.99 for matched pairs).
    """
    embedding_dim = RETRIEVAL_DIM

    def __init__(self, device: str | None = None):
        import torch                                                # imported lazily so we don't pay the cost on the legacy path
        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                     verbose=False, trust_repo=True).to(device).eval()

    def embed(self, image_path: Path | str) -> np.ndarray:
        torch = self._torch
        with Image.open(str(image_path)) as im:
            tensor = _preprocess(im)                                # (1, 3, 224, 224)
        x = torch.from_numpy(tensor).to(self.device)
        with torch.no_grad():
            cls = self.model(x).cpu().numpy()[0]                    # (384,)
        n = np.linalg.norm(cls)
        return (cls / n if n > 0 else cls).astype(np.float32)

    def embed_batch(self, image_paths: list[Path]) -> np.ndarray:
        return np.stack([self.embed(p) for p in image_paths]).astype(np.float32)
