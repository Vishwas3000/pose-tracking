"""Convert DINOv2-small (ViT-S/14) PyTorch model -> Core ML .mlpackage.

We use this model on iOS for *retrieval*: given a query frame, embed it
into a 384-D L2-normalized vector, then cosine-compare against the K
reference embeddings stored in the bundle. Top-N refs become the matcher's
candidate set — drops matcher cost from O(K) to O(N) at K >= 100.

To keep iOS embeddings consistent with the bundle's reference embeddings,
the bundle MUST be built with the same PyTorch DINOv2 model (see
`offline/tools/retrieval_features_pt.py` and the build_object_bundle
`--dinov2-pt` switch). Mixing INT8 ONNX (old bundle path) with FP16
Core ML (new iOS path) introduces a small cosine-space drift; small enough
that retrieval usually still works, but consistency is cheap insurance.

Inputs:
    image: (1, 3, 224, 224) float32, ImageNet-normalized RGB
           (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

Outputs:
    embedding: (1, 384) float32  — CLS token, NOT yet L2-normalized.
                                   Swift normalizes per call.

USAGE (any platform with torch + coremltools):
    python -m offline.tools._export_dinov2_coreml \\
        --out shared/models/dinov2.mlpackage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import coremltools as ct


class _CLSTokenWrapper(torch.nn.Module):
    """Trim DINOv2 to a fixed-input → CLS-token forward.

    DINOv2 ViT-S/14 trains at 518x518 (37x37 patch grid + CLS = 1370 pos
    embeds). Upstream's `interpolate_pos_encoding` resamples the position
    embedding to match arbitrary input sizes at runtime — but it does
    `w0 = w // self.patch_size` on a shape-derived int, which coremltools
    9 can't lower (`TypeError: only 0-dimensional arrays can be
    converted to Python scalars`).

    Fix: pre-interpolate pos_embed ONCE for the export's fixed
    `input_size`, replace the model's pos_embed buffer with the
    interpolated version (now matching the 16x16 grid for 224x224 input),
    and forward through the model bypassing `interpolate_pos_encoding`
    entirely.

    For dinov2-small, `register_tokens is None`, so the register-token
    branch in upstream's `prepare_tokens_with_masks` is also skipped.
    """

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.base
        x = b.patch_embed(x)                                    # (1, N, 384)
        cls = b.cls_token.expand(x.shape[0], -1, -1)            # (1, 1, 384)
        x = torch.cat((cls, x), dim=1)                          # (1, N+1, 384)
        x = x + b.pos_embed                                     # uses pre-interpolated buffer
        if b.register_tokens is not None:
            x = torch.cat(
                (x[:, :1],
                 b.register_tokens.expand(x.shape[0], -1, -1),
                 x[:, 1:]),
                dim=1,
            )
        for blk in b.blocks:
            x = blk(x)
        x = b.norm(x)
        return x[:, 0]                                          # (B, 384)


def _bake_pos_embed_for_input_size(model: torch.nn.Module, input_size: int) -> None:
    """Resample `model.pos_embed` to match `input_size`. Mutates model.

    Uses the SAME bicubic interpolation upstream does at runtime — only
    runs ONCE here, then the result is a static buffer the trace sees
    as a plain constant.
    """
    import math
    pe = model.pos_embed                                        # (1, N+1, 384)
    cls_pe = pe[:, :1]
    patch_pe = pe[:, 1:]
    N = patch_pe.shape[1]
    M = int(round(math.sqrt(N)))
    assert M * M == N, f"pos_embed has {N} patch tokens — not a square grid"
    target_grid = input_size // model.patch_size                # e.g. 16 for 224x224
    if target_grid == M:
        # Already the right size — nothing to do.
        return
    print(f"  Resampling pos_embed: grid {M}x{M} (training) -> {target_grid}x{target_grid} (export at {input_size}x{input_size})")
    dim = patch_pe.shape[-1]
    # (1, M, M, dim) -> (1, dim, M, M) for interpolate
    grid = patch_pe.reshape(1, M, M, dim).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(
        grid, size=(target_grid, target_grid),
        mode="bicubic", align_corners=False,
    )
    # Back to (1, target_grid*target_grid, dim) and prepend CLS pe.
    grid = grid.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, dim)
    new_pe = torch.cat([cls_pe, grid], dim=1)
    model.pos_embed = torch.nn.Parameter(new_pe, requires_grad=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path,
                    help="Output .mlpackage path")
    ap.add_argument("--input-size", type=int, default=224,
                    help="Square input H = W (default 224 — matches DINOv2 patch grid)")
    args = ap.parse_args()
    assert args.input_size % 14 == 0, "DINOv2 patch size is 14 — input must be multiple of 14"

    print("Loading dinov2_vits14 from torch.hub ...")
    base = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                          verbose=False, trust_repo=True)
    base.eval()
    _bake_pos_embed_for_input_size(base, args.input_size)
    wrapper = _CLSTokenWrapper(base).eval()

    # Trace at the fixed input size we'll use on iOS.
    example = torch.randn(1, 3, args.input_size, args.input_size)
    print("Tracing wrapper (torch.export — newer graph mode handles transformer\n"
          "shape arithmetic that torch.jit.trace + coremltools 9 chokes on) ...")
    with torch.no_grad():
        # `torch.export` produces a graph that's typed and shape-frozen,
        # so coremltools' newer `_torch_export.load` frontend can lower
        # the transformer's shape int extractions cleanly.
        traced = torch.export.export(wrapper, (example,))
        out_ref = wrapper(example)
        out_t   = traced.module()(example)
        err = (out_ref - out_t).abs().max().item()
        print(f"  export max-abs-err: {err:.2e}  output shape={tuple(out_ref.shape)}")

    print("\nConverting to Core ML (mlpackage, FP16, iOS17 target) ...")
    # FP16 is fine here — DINOv2 doesn't have the 640x640 InstanceNorm
    # reduction that broke XFeat at FP16. CLS-token output is robust to
    # ~1% per-element drift from FP16 quantization (cosine similarity
    # space stays >0.99 for matching pairs).
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image",
                              shape=(1, 3, args.input_size, args.input_size),
                              dtype=np.float32)],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))
    print(f"\n✓ wrote {args.out}")


if __name__ == "__main__":
    main()
