# pose-tracker

Cross-platform real-time **6-DoF object pose tracking** for iOS + Android.

Architecture: hloc-style pipeline (ALIKED + LightGlue + EPnP-RANSAC),
with offline per-object database generation in Python and on-device
inference via ONNX Runtime Mobile.

## Quick orientation

| What | Where |
|---|---|
| Master plan + conventions for Claude | [`CLAUDE.md`](./CLAUDE.md) |
| 6-week implementation roadmap | [`docs/path_b_implementation_roadmap.md`](./docs/path_b_implementation_roadmap.md) |
| Why ALIKED+LightGlue (vs alternatives) | [`docs/comparison_kaggle_pipeline.md`](./docs/comparison_kaggle_pipeline.md) |
| Documentation index | [`docs/INDEX.md`](./docs/INDEX.md) |

## Project structure

```
pose-tracker/
├── CLAUDE.md                          # operating guide for Claude
├── README.md                          # this file
├── offline/                           # Python: per-object bundle generator
├── ios/                               # Swift app (Phase 3+)
├── android/                           # Kotlin app (Phase 5)
├── shared/                            # Bundle format, algorithm specs
└── docs/                              # Design docs (copied from OnePose_Plus_Plus/doc/)
```

## Status

**Phase 0 — Scaffolding (complete).** Next: Phase 1, the Python offline
pipeline that turns a COLMAP/SfM workspace into a per-object `.bundle`.

See `CLAUDE.md` for the full phase plan and what to do next.

## Quick start (offline, once Phase 1 is implemented)

```bash
cd offline
source /Users/sudeepsharma/Documents/GitHub/xfeat_ios/accelerated_features/.venv/bin/activate
pip install -r requirements.txt

python -m tools.build_object_bundle \
    --colmap-dir /path/to/sfm_workspace \
    --source-images /path/to/source_images \
    --bbox /path/to/bbox3d_corners.txt \
    --out ../shared/objects/test_object.bundle
```
