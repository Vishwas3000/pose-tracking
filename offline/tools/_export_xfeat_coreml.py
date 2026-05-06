"""Convert vendored XFeat PyTorch checkpoint -> Core ML .mlpackage.

We export ONLY the convolutional backbone (`XFeatModel`), not the upstream
`XFeat` wrapper. The wrapper does NMS + interpolation + top-K selection in
Python with dynamic shapes that don't trace cleanly. Those steps move into
`XFeatRunner.swift` so the Core ML model is a pure conv graph that the
Apple Neural Engine can run end-to-end.

Inputs:
    image: (1, 3, 640, 640) float32, RGB CHW, [0, 1]

Outputs:
    M:  (1, 64, 80, 80) float32 — descriptor field (NOT yet L2-normalized;
                                  Swift does the per-keypoint norm after
                                  bilinear interpolation)
    K:  (1, 65, 80, 80) float32 — keypoint logits (8x8 + 1 dustbin per cell)
    H:  (1, 1,  80, 80) float32 — reliability heatmap

USAGE (run on any platform with torch + coremltools):
    python -m offline.tools._export_xfeat_coreml \\
        --weights shared/models/xfeat.pt \\
        --out     shared/models/xfeat.mlpackage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import coremltools as ct

from ._xfeat.model import XFeatModel


class _XFeatForExport(torch.nn.Module):
    """Wraps XFeatModel for Core ML export.

    TWO divergences from upstream `XFeatModel.forward`:

    1. **`F.interpolate(x, x3.shape[-2:])` -> `scale_factor=2.0`/`4.0`.**
       coremltools 9 can't lower the dynamic 0-D shape tensor. For our
       fixed 640x640 input, x3 is /8, x4 /16, x5 /32 — so 2x / 4x scale
       factors give the right output shape without runtime shape arithmetic.

    2. **`_unfold2d(x, 8)` -> `F.pixel_unshuffle(x, 8)`.** Same coremltools
       issue with `Tensor.unfold`. The two ops are mathematically identical
       for our case: channel `c = i*8 + j` maps to spatial offset `(i, j)`
       within the 8x8 block in both.

    Input is (1, 3, H, W) RGB float32 in [0, 1] — same as upstream — and the
    model handles RGB-mean + InstanceNorm internally. We tried moving those
    out into Swift FP32 to enable FP16 ANE; it didn't help (see
    docs/xfeat_port.md: "FP16 ANE — both attempts failed"). We're back at
    FP32 for the whole graph.
    """
    def __init__(self, base: XFeatModel):
        super().__init__()
        self.base = base

    def forward(self, x):
        # Upstream's preprocessing: RGB→grayscale, InstanceNorm. Kept inside
        # the model since at FP32 the reduction is stable.
        b = self.base
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = b.norm(x)
        x1 = b.block1(x)
        x2 = b.block2(x1 + b.skip1(x))
        x3 = b.block3(x2)
        x4 = b.block4(x3)
        x5 = b.block5(x4)
        x4 = F.interpolate(x4, scale_factor=2.0, mode='bilinear', align_corners=False)
        x5 = F.interpolate(x5, scale_factor=4.0, mode='bilinear', align_corners=False)
        feats = b.block_fusion(x3 + x4 + x5)
        heatmap = b.heatmap_head(feats)
        x_unfolded = F.pixel_unshuffle(x, downscale_factor=8)
        keypoints = b.keypoint_head(x_unfolded)
        return feats, keypoints, heatmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True,
                    help="Path to xfeat.pt checkpoint")
    ap.add_argument("--out",     type=Path, required=True,
                    help="Output .mlpackage path")
    ap.add_argument("--input-size", type=int, default=640,
                    help="Fixed square input H = W (default 640, must be /32)")
    args = ap.parse_args()
    assert args.input_size % 32 == 0, "input-size must be multiple of 32"

    print(f"Loading XFeat weights from {args.weights}")
    net = XFeatModel().eval()
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    net.load_state_dict(state)

    wrapper = _XFeatForExport(net).eval()
    # Input is (1, 3, H, W) RGB FP32 [0, 1]. Use a non-zero example so the
    # trace exercises the full graph (zeros short-circuit some lowering
    # paths).
    example = torch.rand(1, 3, args.input_size, args.input_size)
    print("Tracing the export wrapper (3-channel RGB input) ...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)

    # Sanity-check the trace matches the wrapper.
    with torch.no_grad():
        m_ref, k_ref, h_ref = wrapper(example)
        m_t, k_t, h_t = traced(example)
        for name, ref, got in [("M", m_ref, m_t), ("K", k_ref, k_t), ("H", h_ref, h_t)]:
            err = (ref - got).abs().max().item()
            print(f"  {name} traced max-abs-err: {err:.2e}  shape={tuple(ref.shape)}")

    print("\nConverting to Core ML (mlpackage, FP32 compute, iOS 17 target) ...")
    import numpy as np
    # FP32 compute is required for descriptor quality. FP16 was tried twice
    # and produced descriptors that collapsed to <0.60 max cossim against
    # the bundle (full reasoning + experiments in `docs/xfeat_port.md`).
    # FP32 falls back from the ANE to GPU/CPU on iPhone — ~19 ms / frame
    # instead of the ~5 ms ANE could deliver, but still ~6× faster than
    # ALIKED on CPU.
    #
    # iOS 17 deployment target is required: without it coremltools 9 hits
    # an `_int` cast bug somewhere in the conv stack.
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image",
                              shape=(1, 3, args.input_size, args.input_size),
                              dtype=np.float32)],
        outputs=[ct.TensorType(name="M"),
                 ct.TensorType(name="K"),
                 ct.TensorType(name="H")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))
    print(f"\n✓ wrote {args.out}")


if __name__ == "__main__":
    main()
