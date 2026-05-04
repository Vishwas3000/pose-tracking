# offline — Python pipeline (per-object setup)

Runs once per object, on a workstation with a GPU. Takes a COLMAP/SfM
workspace + source images + 3D bbox; outputs a `.bundle` file consumable
by the iOS and Android apps.

## Inputs

```
sfm_workspace/
├── cameras.bin              # COLMAP camera intrinsics
├── images.bin               # per-image pose + 2D keypoints + 3D point IDs
├── points3D.bin             # per-3D-point: xyz + observations
├── source_images/           # the actual photos used in SfM
│   ├── frame_001.png
│   └── ...
└── box3d_corners.txt        # 8 corners of object's 3D bbox
```

If you only have OnePose++'s `anno_3d_average.npz`, you cannot run this
pipeline — re-run SfM from your source images to get the COLMAP outputs.

## Outputs

```
test_object.bundle           # custom binary, ~5-20 MB per object
```

See [`../shared/bundle_format.md`](../shared/bundle_format.md) for the
on-disk byte layout.

## Setup

Reuses the existing PyTorch venv from the sister repo to avoid
re-installing PyTorch:

```bash
cd /Users/sudeepsharma/Documents/GitHub/pose-tracker/offline
source /Users/sudeepsharma/Documents/GitHub/xfeat_ios/accelerated_features/.venv/bin/activate

pip install -r requirements.txt   # adds ALIKED, LightGlue, etc.
```

If you'd rather have a fresh venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision   # ~80 MB, takes a couple minutes
```

## Running the pipeline

```bash
# Build a bundle from a COLMAP workspace
python -m tools.build_object_bundle \
    --colmap-dir   /path/to/sfm_workspace \
    --source-images /path/to/source_images \
    --bbox         /path/to/bbox3d_corners.txt \
    --num-refs     30 \
    --out          ../shared/objects/test_object.bundle

# Verify the bundle (round-trip parity check)
python -m tools.verify_bundle ../shared/objects/test_object.bundle

# Inspect a bundle's contents
python -m tools.bundle_writer --inspect ../shared/objects/test_object.bundle
```

## Files

| File | Purpose |
|---|---|
| `tools/build_object_bundle.py` | Main entry — orchestrates Phase O1 → O6 |
| `tools/colmap_io.py` | Read COLMAP `.bin` files (wraps upstream `read_write_model.py`) |
| `tools/sample_reference_views.py` | Pick K reference views via viewpoint-sphere FPS |
| `tools/aliked_inference.py` | Run ALIKED on each reference view |
| `tools/retrieval_features.py` | Optional: MobileNet/DINOv2 global descriptors |
| `tools/bundle_writer.py` | Custom binary serialization |
| `tools/verify_bundle.py` | Round-trip + reprojection sanity check |

## Verification

The pipeline is correct if `verify_bundle.py` reports:

- All references load and parse cleanly
- For each reference view, projecting its 3D points back through the
  stored pose lands within ~2 px of the stored 2D keypoints
- A round-trip (use the bundle's own ref images as queries) gives near-
  identity homographies and >100 PnP inliers per query
