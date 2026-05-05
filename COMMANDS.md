# Commands — pose-tracker desktop reference

Copy-paste-ready commands for the offline (bundle producer) and online
(reference pipeline) work on this Linux box. All commands assume the
working directory is the repo root: `/davidRavi/pose-tracking/`.

---

## 0. One-time setup (already done on this machine)

```bash
# COLMAP 4.0.4 from source with CUDA (sm_89 for RTX 4090)
# Installed to /usr/local/bin/colmap. Verify:
colmap -h | head -2          # → "COLMAP 4.0.4 ... with CUDA"

# Python venv (lives at repo root, gitignored)
# IMPORTANT: build against /venv/main/bin/python, not the system /usr/bin/python3
# — the system binary on this container has ABI-mismatched extensions that
# break ctypes / ORT preload_dlls.
/venv/main/bin/python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# requirements.txt pins: pillow, numpy, opencv, onnxruntime-gpu, typer, tqdm,
# and CUDA 12 runtime libs as nvidia-* pip wheels. ORT auto-finds them via
# preload_dlls() so CUDAExecutionProvider works on the RTX 4090 without
# touching the system CUDA 13 toolkit COLMAP needs. ~1.5 GB extra in .venv.

# Pre-trained models (already in shared/models/)
ls shared/models/
# aliked-n16rot-top1k-640.onnx        local features + descriptors
# lightglue_for_aliked.onnx           descriptor matcher (not yet wired in)
# dinov2-small-int8.onnx              global retrieval (DINOv2 ViT-S/14 INT8)
```

### Each new terminal

```bash
cd /davidRavi/pose-tracking && source .venv/bin/activate
```

---

## 1. Offline — build a `.bundle` from an ARKit session

### 1a. Unzip a session export

```bash
# Replace SESSION with your zip name. The repo ignores offline/data/.
unzip offline/session_1777549127.zip -d offline/data/
ls offline/data/session_1777549127/{frames,metadata}/ | head
cat offline/data/session_1777549127/bounds.json
```

### 1b. ARKit poses → COLMAP sparse SfM

Uses a patched copy of ColmapReconstruction's `inject_poses.py` (the COLMAP
path is rewritten to `/usr/local/bin/colmap` and the Windows-only Qt env
var is stripped):

```bash
cd offline/data
python3 inject_poses.py \
    --session   /davidRavi/pose-tracking/offline/data/session_1777549127 \
    --workspace /davidRavi/pose-tracking/offline/data/workspace_session_1777549127
cd ../..
```

Output: `workspace_session_1777549127/sparse/0/{cameras,images,points3D}.bin`.

### 1c. Crop the sparse model to the user-drawn 3-D bbox

```bash
cd offline/data
python3 crop.py \
    --workspace /davidRavi/pose-tracking/offline/data/workspace_session_1777549127 \
    --sparse-only
cd ../..

# Optional: export a PLY to view the cropped points in MeshLab / CloudCompare
colmap model_converter \
    --input_path  offline/data/workspace_session_1777549127/sparse/cropped_bin \
    --output_path offline/data/workspace_session_1777549127/sparse/cropped_sparse.ply \
    --output_type PLY
```

### 1d. Pack the `.bundle`

```bash
cd offline
python3 -m tools.build_object_bundle \
    --colmap-dir    data/workspace_session_1777549127/sparse/cropped_bin \
    --source-images data/workspace_session_1777549127/images \
    --bbox          data/workspace_session_1777549127/bounds.json \
    --aliked-onnx   ../shared/models/aliked-n16rot-top1k-640.onnx \
    --dinov2-onnx   ../shared/models/dinov2-small-int8.onnx \
    --num-refs      30 \
    --kpt-match-px  3 \
    --out           ../shared/objects/session_1777549127.bundle
cd ..
# --dinov2-onnx is optional. With it, the bundle includes a (K, 384)
# float32 retrieval-embedding section that the online pipeline can use
# to skip ~80% of the brute-force matcher work. Cost: +45 KB per bundle.
```

Inspect what was written:

```bash
python3 -m offline.tools.bundle_writer --inspect \
    shared/objects/session_1777549127.bundle
```

Tunables to lift the LOST rate (re-run `build_object_bundle`):
- `--kpt-match-px 6` — looser ALIKED↔SIFT 2-D NN at build time
- `--num-refs 60`   — denser viewpoint coverage
- swap to the `aliked-n16rot-top2k-640.onnx` model for more candidate kpts

---

## 2. Online — verify the pipeline against the bundle

### 2a. Single image, with optional ARKit ground-truth comparison

```bash
python3 -m online.demo.single_image \
    --bundle   shared/objects/session_1777549127.bundle \
    --aliked   shared/models/aliked-n16rot-top1k-640.onnx \
    --image    offline/data/session_1777549127/frames/frame_0210.jpg \
    --metadata offline/data/session_1777549127/metadata/metadata_0210.json
```

### 2b. Single image with a drawn 3-D bbox overlay

```bash
python3 -m online.demo.overlay_bbox \
    --bundle shared/objects/session_1777549127.bundle \
    --aliked shared/models/aliked-n16rot-top1k-640.onnx \
    --image  offline/data/session_1777549127/frames/frame_0210.jpg \
    --out    offline/data/overlays/frame_0210_bbox.jpg
```

### 2c. Sweep an entire session at a stride

```bash
python3 -m online.demo.sweep_session \
    --bundle  shared/objects/session_1777549127.bundle \
    --aliked  shared/models/aliked-n16rot-top1k-640.onnx \
    --session offline/data/session_1777549127 \
    --out     offline/data/overlays \
    --stride  10                 # every 10th of 430 frames -> 43 outputs
```

Outputs `offline/data/overlays/frame_NNNN_bbox.jpg` and a `sweep_summary.csv`
with per-frame match/inlier counts and rotation/translation error vs ARKit GT.

### 2d. Run on arbitrary test images

```bash
# Step 1 — downsize to bundle resolution (so the K applies directly + smaller files)
python3 -m online.demo.reduce_images \
    --in-dir  offline/data/test_images \
    --out-dir offline/data/test_images_reduced \
    --target-size 1920x1440 \
    --quality 85

# Step 2 — run the pipeline on the reduced folder
python3 -m online.demo.test_images \
    --bundle   shared/objects/session_1777549127.bundle \
    --aliked   shared/models/aliked-n16rot-top1k-640.onnx \
    --in-dir   offline/data/test_images_reduced \
    --out-dir  offline/data/test_images_infered \
    --ref-size 1920x1440

# Step 2b — same, but with DINOv2 top-N retrieval (no-op if bundle has
# no embeddings). Useful when K is large (50+) or for multi-object catalogs.
python3 -m online.demo.test_images \
    --bundle   shared/objects/session_1777549127.bundle \
    --aliked   shared/models/aliked-n16rot-top1k-640.onnx \
    --dinov2   shared/models/dinov2-small-int8.onnx \
    --top-n    5 \
    --in-dir   offline/data/test_images_reduced \
    --out-dir  offline/data/test_images_infered_dino \
    --ref-size 1920x1440
```

### 2e. Run on a video file

```bash
# Build a test clip from session frames (skip if you already have a video)
ffmpeg -y -framerate 30 -i offline/data/session_1777549127/frames/frame_%04d.jpg \
    -vframes 200 -c:v libx264 -pix_fmt yuv420p -preset veryfast \
    offline/data/session_1777549127.mp4

# Process it. --stride 2 halves compute and carries the last drawn pose
# forward on skipped frames so the output stays at full source fps.
python3 -m online.demo.video \
    --bundle  shared/objects/session_1777549127.bundle \
    --aliked  shared/models/aliked-n16rot-top1k-640.onnx \
    --video   offline/data/session_1777549127.mp4 \
    --out     offline/data/session_1777549127_tracked.mp4 \
    --stride  2

# With DINOv2 retrieval:
#   --dinov2 shared/models/dinov2-small-int8.onnx --top-n 5
# Cap input frames during testing:
#   --max-frames 100
```

Outputs an annotated MP4 plus a `.csv` with per-frame match/inlier counts.
Reads via `cv2.VideoCapture`, writes via `mp4v` codec.

Reads with `cv2.IMREAD_IGNORE_ORIENTATION` so EXIF rotation is **not**
applied — keeps drawing in the same coordinate frame ALIKED operates in.

---

## 3. Quick checks

```bash
# Bundle sanity
python3 -m offline.tools.bundle_writer --inspect shared/objects/session_1777549127.bundle

# COLMAP sparse model summary
python3 -c "
import sys; sys.path.insert(0, 'offline')
from tools import colmap_io
m = colmap_io.read_model('offline/data/workspace_session_1777549127/sparse/cropped_bin')
print(f'cameras={len(m.cameras)}  images={len(m.images)}  points3D={len(m.points3D)}')
"

# Per-point bbox containment check
python3 -c "
import sys; sys.path.insert(0, 'offline')
from tools import bounds, colmap_io
import numpy as np
m   = colmap_io.read_model('offline/data/workspace_session_1777549127/sparse/cropped_bin')
bnd = bounds.load_bounds('offline/data/session_1777549127/bounds.json')
xyz = np.stack([p.xyz for p in m.points3D.values()])
inside = bounds.inside_box(xyz, bnd).sum()
print(f'inside-bbox: {inside}/{len(xyz)} ({100*inside/len(xyz):.1f}%)')
"
```

---

## 4. Common gotchas (saved as memory)

- **EXIF orientation**: `cv2.imread` auto-rotates portrait photos; PIL does not.
  All online demos pass `cv2.IMREAD_IGNORE_ORIENTATION` to stay in raw sensor
  orientation (matching the session's landscape capture).
- **ALIKED ONNX kpt coords are NORMALIZED [-1, 1]**, not pixels — see
  `offline/tools/aliked_inference.py:_Letterbox.normalized_to_original`.
- **CUDA EP fails silently**: pip's `onnxruntime-gpu` wants CUDA 12 / cuDNN 9;
  this box has CUDA 13.1. ALIKED runs on CPU (~2.4 s/frame). COLMAP itself
  is fine — we built it with CUDA 13 from source.
- **COLMAP world frame == ARKit world frame**. The `c2w @ flip` step in
  `inject_poses.py` only flips camera-local axes, not world axes. So
  `points3D.bin`, `bounds.json`, and ARKit metadata `camera_pose_c2w` all
  share one frame.

---

## 5. Cleanup

```bash
# Wipe overlay outputs (keep originals)
rm -f offline/data/overlays/*.jpg offline/data/overlays/*.csv
rm -f offline/data/test_images_infered/*.JPG offline/data/test_images_infered/*.jpg

# Wipe an entire session workspace + bundle
rm -rf offline/data/session_1777549127 \
       offline/data/workspace_session_1777549127 \
       shared/objects/session_1777549127.bundle
```
