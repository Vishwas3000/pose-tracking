"""Custom binary `.bundle` format reader/writer.

See `shared/bundle_format.md` for the canonical byte layout.

  HEADER 64 B  | points3D (M×3 f32) | bbox3d (8×3 f32)
  | [retrieval (K×D f32) — optional]
  | ref_offsets (K × u64) | ref_block × K (variable)

Per ref_block:
  image_id u32  num_kpts u32  pose 16×f32  K_intr 9×f32
  kpts (N×2 f32)  descs (N×D f32)  pt3d_indices (N×u32)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BUNDLE_MAGIC = b"PBND"
BUNDLE_VERSION = 1
HEADER_SIZE = 64
HAS_RETRIEVAL_BIT = 1


@dataclass
class ReferenceView:
    image_id: int
    keypoints: np.ndarray           # (N, 2) float32
    descriptors: np.ndarray         # (N, D) float32 (D = 128 for ALIKED)
    pt3d_indices: np.ndarray        # (N,) uint32 — index into Bundle.points3d
    pose: np.ndarray                # (4, 4) float32 — world->camera
    K: np.ndarray                   # (3, 3) float32 — camera intrinsics


@dataclass
class Bundle:
    points3d: np.ndarray            # (M, 3) float32
    bbox3d: np.ndarray              # (8, 3) float32
    ref_global_emb: np.ndarray | None   # (K, D_global) float32 or None
    refs: list[ReferenceView]


def _pack_header(num_points3d: int, num_refs: int, desc_dim: int,
                 retrieval_dim: int, has_retrieval: bool) -> bytes:
    flags = HAS_RETRIEVAL_BIT if has_retrieval else 0
    head = struct.pack(
        "<4s6I",
        BUNDLE_MAGIC, BUNDLE_VERSION, flags,
        num_points3d, num_refs, desc_dim, retrieval_dim,
    )
    # Pad to HEADER_SIZE (the struct above is 4 + 6*4 = 28 bytes)
    return head + b"\x00" * (HEADER_SIZE - len(head))


def _ref_block_bytes(ref: ReferenceView, desc_dim: int) -> bytes:
    n = len(ref.keypoints)
    assert ref.descriptors.shape == (n, desc_dim), \
        f"descriptors shape {ref.descriptors.shape} != ({n}, {desc_dim})"
    assert ref.pt3d_indices.shape == (n,)
    assert ref.pose.shape == (4, 4) and ref.K.shape == (3, 3)
    parts = [
        struct.pack("<II", int(ref.image_id), n),
        ref.pose.astype(np.float32, copy=False).tobytes(order="C"),
        ref.K.astype(np.float32, copy=False).tobytes(order="C"),
        ref.keypoints.astype(np.float32, copy=False).tobytes(order="C"),
        ref.descriptors.astype(np.float32, copy=False).tobytes(order="C"),
        ref.pt3d_indices.astype(np.uint32, copy=False).tobytes(order="C"),
    ]
    return b"".join(parts)


def write(out_path: Path, bundle: Bundle) -> None:
    """Serialize a Bundle to the canonical binary format."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    M = len(bundle.points3d)
    K = len(bundle.refs)
    desc_dim = bundle.refs[0].descriptors.shape[1] if K else 0
    has_retrieval = bundle.ref_global_emb is not None
    retrieval_dim = bundle.ref_global_emb.shape[1] if has_retrieval else 0

    header = _pack_header(M, K, desc_dim, retrieval_dim, has_retrieval)
    pts_bytes  = bundle.points3d.astype(np.float32, copy=False).tobytes(order="C")
    bbox_bytes = bundle.bbox3d.astype(np.float32, copy=False).tobytes(order="C")
    retrieval_bytes = (bundle.ref_global_emb.astype(np.float32, copy=False).tobytes(order="C")
                       if has_retrieval else b"")

    # Compute reference offsets *before* writing so the offsets table is correct.
    fixed_size = (
        HEADER_SIZE + len(pts_bytes) + len(bbox_bytes)
        + len(retrieval_bytes) + (K * 8)
    )
    ref_blocks = [_ref_block_bytes(r, desc_dim) for r in bundle.refs]
    offsets = []
    cursor = fixed_size
    for blk in ref_blocks:
        offsets.append(cursor)
        cursor += len(blk)

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(pts_bytes)
        f.write(bbox_bytes)
        f.write(retrieval_bytes)
        f.write(np.array(offsets, dtype=np.uint64).tobytes(order="C"))
        for blk in ref_blocks:
            f.write(blk)


def load_bundle(path: Path) -> Bundle:
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(HEADER_SIZE)
        magic, version, flags, num_pts, num_refs, desc_dim, retrieval_dim = \
            struct.unpack("<4s6I", head[:28])
        if magic != BUNDLE_MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        if version != BUNDLE_VERSION:
            raise ValueError(f"unsupported version: {version}")

        pts = np.frombuffer(f.read(num_pts * 3 * 4), dtype="<f4").reshape(num_pts, 3).copy()
        bbox = np.frombuffer(f.read(8 * 3 * 4), dtype="<f4").reshape(8, 3).copy()
        if flags & HAS_RETRIEVAL_BIT:
            emb = np.frombuffer(f.read(num_refs * retrieval_dim * 4), dtype="<f4") \
                    .reshape(num_refs, retrieval_dim).copy()
        else:
            emb = None

        offsets = np.frombuffer(f.read(num_refs * 8), dtype="<u8").copy()

        refs = []
        for off in offsets:
            f.seek(int(off))
            image_id, n = struct.unpack("<II", f.read(8))
            pose = np.frombuffer(f.read(16 * 4), dtype="<f4").reshape(4, 4).copy()
            K = np.frombuffer(f.read(9 * 4), dtype="<f4").reshape(3, 3).copy()
            kpts  = np.frombuffer(f.read(n * 2 * 4), dtype="<f4").reshape(n, 2).copy()
            descs = np.frombuffer(f.read(n * desc_dim * 4), dtype="<f4") \
                      .reshape(n, desc_dim).copy()
            ids   = np.frombuffer(f.read(n * 4), dtype="<u4").copy()
            refs.append(ReferenceView(
                image_id=int(image_id), keypoints=kpts, descriptors=descs,
                pt3d_indices=ids, pose=pose, K=K,
            ))

    return Bundle(points3d=pts, bbox3d=bbox, ref_global_emb=emb, refs=refs)


def inspect(path: Path) -> None:
    b = load_bundle(path)
    print(f"=== {path} ===")
    print(f"K (refs)        : {len(b.refs)}")
    print(f"M (3D points)   : {len(b.points3d)}")
    print(f"3D bbox corners : min={b.bbox3d.min(axis=0).tolist()}  max={b.bbox3d.max(axis=0).tolist()}")
    if b.ref_global_emb is not None:
        print(f"Global emb dim  : {b.ref_global_emb.shape[1]}")
    n_kpts = sum(len(r.keypoints) for r in b.refs)
    print(f"Total keypoints : {n_kpts:,}")
    for i, ref in enumerate(b.refs[:5]):
        print(f"  ref {i}: image_id={ref.image_id}  N={len(ref.keypoints)}  "
              f"desc_dim={ref.descriptors.shape[1]}  "
              f"unique_pt3d={len(np.unique(ref.pt3d_indices))}")
    if len(b.refs) > 5:
        print(f"  ... +{len(b.refs) - 5} more")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--inspect":
        inspect(Path(sys.argv[2]))
    else:
        print("Usage: python -m tools.bundle_writer --inspect path/to/object.bundle")
