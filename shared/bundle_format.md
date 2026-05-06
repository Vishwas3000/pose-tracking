# `.bundle` Custom Binary Format

Per-object database format consumed by both iOS and Android apps.
Produced by `offline/tools/bundle_writer.py`.

## Why custom binary instead of `.npz` or zip

- Loads in <5 ms on mobile (random-access mmap-friendly) vs ~50 ms for
  zip-of-npy parsing
- Single file is easy to ship as an app resource OR download on demand
- Simple enough to write manual parsers for Swift / Kotlin in ~150 LOC
  each — no extra dependencies

## File layout (version 1)

All numeric fields are **little-endian**. Float fields are `float32`.
Integer fields are `uint32` unless noted.

```
┌─ HEADER (64 bytes, fixed) ─────────────────────────────────────────┐
│  magic            : 4 bytes  = b"PBND"                              │
│  version          : uint32   = 1                                    │
│  flags            : uint32   = bitfield (bit 0 = has retrieval embs)│
│  num_points3d     : uint32   = M                                    │
│  num_refs         : uint32   = K                                    │
│  desc_dim         : uint32   = 64 (XFeat) or 128 (legacy ALIKED)    │
│  retrieval_dim    : uint32   = 0 if no embs, else D_global          │
│  reserved         : 32 bytes = zeros                                │
└────────────────────────────────────────────────────────────────────┘

┌─ GLOBAL POINTS3D ───────────────────────────────────────────────────┐
│  M × 3 × float32   (12 × M bytes)                                   │
└────────────────────────────────────────────────────────────────────┘

┌─ GLOBAL BBOX3D ─────────────────────────────────────────────────────┐
│  8 × 3 × float32   (96 bytes)                                       │
└────────────────────────────────────────────────────────────────────┘

┌─ RETRIEVAL EMBEDDINGS (only if flags bit 0 set) ───────────────────┐
│  K × retrieval_dim × float32                                        │
└────────────────────────────────────────────────────────────────────┘

┌─ REFERENCE OFFSETS TABLE ──────────────────────────────────────────┐
│  K × uint64   = byte offset of each reference block, from file start│
└────────────────────────────────────────────────────────────────────┘

┌─ REFERENCE BLOCK ×K (variable-size) ───────────────────────────────┐
│ For each reference k = 0..K-1:                                       │
│   image_id        : uint32                                           │
│   num_keypoints   : uint32   = N_k                                   │
│   pose_4x4        : 16 × float32                                     │
│   K_intrinsics    : 9 × float32                                      │
│   keypoints       : N_k × 2 × float32                                │
│   descriptors     : N_k × desc_dim × float32                         │
│   pt3d_indices    : N_k × uint32   (index into global points3D)     │
└────────────────────────────────────────────────────────────────────┘
```

## Read order on mobile

1. Read header (64 bytes). Validate magic + version.
2. Compute static section offsets from `num_points3d`, `num_refs`,
   `retrieval_dim`. Memory-map the file.
3. Lazily slice each section directly off the mapped bytes — no copies.

## Writer code reference

`offline/tools/bundle_writer.py` is the canonical Python writer. It uses
`numpy.tofile` after serializing the header to ensure binary parity with
the Swift/Kotlin readers.

## Limits

- `M` (3D points): up to 4 billion in theory; practically 5,000–50,000
- `K` (reference views): up to 4 billion; practically 20–100
- `N_k` (keypoints per ref): up to 4 billion; practically 500–2,000
- `desc_dim`: 64 for XFeat, 128 for legacy ALIKED — version unchanged
  since the reader is dim-aware (reads `desc_dim` from the header)

## Future extensions

If we ever need them:

- **version 2**: variable descriptor type (FP16 quantized)
- **version 3**: optional per-reference covariance (uncertainty per kpt)
- **version 4**: octree spatial index for 3D points (faster outlier
  rejection in PnP)

Each new version requires: increment `version`, document changes here,
support old versions in mobile readers (or refuse to load with a clear
error).
