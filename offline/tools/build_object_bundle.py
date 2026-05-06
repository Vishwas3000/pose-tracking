"""Phase O orchestrator — builds an object .bundle from a COLMAP workspace.

Pipeline:
  O1: load COLMAP outputs + bbox + source images
  O2: filter points3D inside the oriented bbox; sample K reference views
  O3: XFeat inference per reference view  (PyTorch, vendored under _xfeat/)
  O4: map XFeat keypoints -> COLMAP 3D point IDs (2D NN within max-px)
  O5: optional global retrieval features  (skipped — K is small)
  O6: pack everything into custom .bundle

USAGE:
    python -m tools.build_object_bundle \\
        --colmap-dir    /path/to/workspace/sparse/cropped_bin \\
        --source-images /path/to/workspace/images \\
        --bbox          /path/to/workspace/bounds.json \\
        --xfeat-model   ../shared/models/xfeat.pt \\
        --num-refs      30 \\
        --out           ../shared/objects/test_object.bundle
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from . import bounds, bundle_writer, colmap_io, sample_reference_views
from .xfeat_inference import XFeatRunner
from .retrieval_features import DinoV2Embedder


app = typer.Typer(add_completion=False)


@app.command()
def main(
    colmap_dir: Path = typer.Option(..., help="COLMAP sparse model dir (cameras.bin / images.bin / points3D.bin)"),
    source_images: Path = typer.Option(..., help="Folder of source images used during SfM"),
    bbox: Path = typer.Option(..., help="bounds.json — oriented AABB (center / extents / rotation)"),
    xfeat_model: Path = typer.Option(..., help="Path to XFeat PyTorch checkpoint (xfeat.pt)"),
    out: Path = typer.Option(..., help="Output .bundle path"),
    num_refs: int = typer.Option(30, help="Number of reference views to sample"),
    kpt_match_px: float = typer.Option(3.0, help="Max distance (orig px) XFeat kpt -> COLMAP-tracked kpt"),
    xfeat_score_min: float = typer.Option(0.0, help="XFeat score floor (0 = keep all in-bounds)"),
    xfeat_top_k: int = typer.Option(1000, help="Top-K XFeat keypoints per image (matches ALIKED's 1000 for parity)"),
    dinov2_onnx: Path = typer.Option(None, help="Path to DINOv2 ONNX model (enables retrieval embeddings)"),
):
    """Build an object .bundle from a COLMAP/SfM workspace."""
    typer.echo(f"=== build_object_bundle ===")

    # --- O1 / O2 --- load + bbox filter -----------------------------------
    model = colmap_io.read_model(colmap_dir)
    bnd = bounds.load_bounds(bbox)
    typer.echo(f"Loaded COLMAP: {len(model.images)} images, {len(model.points3D)} 3D points")

    pt3d_keys = sorted(model.points3D.keys())
    pt3d_xyz = np.stack([model.points3D[k].xyz for k in pt3d_keys])
    inside = bounds.inside_box(pt3d_xyz, bnd)
    pt3d_keys = [k for k, m in zip(pt3d_keys, inside) if m]
    pt3d_xyz = pt3d_xyz[inside].astype(np.float32)
    # Map COLMAP point3D_id -> dense [0, M) index (used in ref blocks).
    pt3d_id_to_idx = {pid: i for i, pid in enumerate(pt3d_keys)}
    typer.echo(f"After bbox filter: {len(pt3d_keys)} / {inside.size} 3D points")

    ref_image_ids = sample_reference_views.farthest_point_sphere(model, k=num_refs)
    typer.echo(f"Sampled {len(ref_image_ids)} reference views via viewpoint-sphere FPS")

    # --- O3 / O4 --- XFeat + 2D NN to COLMAP-tracked keypoints ------------
    runner = XFeatRunner(xfeat_model, score_threshold=xfeat_score_min, top_k=xfeat_top_k)
    refs: list[bundle_writer.ReferenceView] = []
    cam = next(iter(model.cameras.values()))
    K_intr = colmap_io.camera_intrinsics(cam)

    for img_id in ref_image_ids:
        img = model.images[img_id]
        feats = runner.extract(source_images / img.name)
        kpts_xfeat = feats["keypoints"]               # (Na, 2)
        descs      = feats["descriptors"]             # (Na, 64)

        # COLMAP-tracked 2D points for THIS image, with a valid 3D association
        # AND that 3D point survived the bbox crop.
        col_xy   = img.xys                             # (Nc, 2)
        col_p3d  = img.point3D_ids                     # (Nc,) int — -1 = no 3D
        valid_c  = np.array([(p != -1 and p in pt3d_id_to_idx) for p in col_p3d])
        col_xy_v   = col_xy[valid_c]
        col_p3d_v  = col_p3d[valid_c]
        if len(col_xy_v) == 0:
            continue

        # Brute-force 2D NN: distances Na × Nc — Nc is small here (~hundreds)
        d2 = ((kpts_xfeat[:, None, :] - col_xy_v[None, :, :]) ** 2).sum(-1)
        nn_idx = d2.argmin(axis=1)
        nn_d   = np.sqrt(d2[np.arange(len(kpts_xfeat)), nn_idx])
        keep   = nn_d <= kpt_match_px
        if keep.sum() == 0:
            continue

        kept_kpts  = kpts_xfeat[keep]
        kept_descs = descs[keep]
        kept_pt3d_dense = np.array([pt3d_id_to_idx[col_p3d_v[i]] for i in nn_idx[keep]],
                                    dtype=np.uint32)

        # 4×4 world->camera pose
        from . import _colmap_upstream as _u
        R = _u.qvec2rotmat(img.qvec)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3]  = img.tvec

        refs.append(bundle_writer.ReferenceView(
            image_id=int(img_id),
            keypoints=kept_kpts.astype(np.float32),
            descriptors=kept_descs.astype(np.float32),
            pt3d_indices=kept_pt3d_dense,
            pose=pose,
            K=K_intr.astype(np.float32),
        ))
        typer.echo(f"  ref img {img_id} ({img.name}): {len(kpts_xfeat):4d} XFeat → {keep.sum():4d} matched 3D")

    if not refs:
        raise RuntimeError("No reference views produced any 3D matches — check kpt_match_px or bbox.")

    # --- O5 --- DINOv2 retrieval embeddings (optional) ---------------------
    ref_global_emb = None
    if dinov2_onnx is not None:
        typer.echo(f"\nDINOv2 retrieval embeddings ({dinov2_onnx.name}) ...")
        embedder = DinoV2Embedder(dinov2_onnx)
        emb_paths = [source_images / model.images[r.image_id].name for r in refs]
        ref_global_emb = embedder.embed_batch(emb_paths)
        typer.echo(f"  shape={ref_global_emb.shape}  dtype={ref_global_emb.dtype}  "
                   f"L2-norm avg={np.linalg.norm(ref_global_emb, axis=1).mean():.4f}")

    # --- O6 --- pack -------------------------------------------------------
    bundle = bundle_writer.Bundle(
        points3d=pt3d_xyz,
        bbox3d=bounds.corners(bnd),
        ref_global_emb=ref_global_emb,
        refs=refs,
    )
    bundle_writer.write(out, bundle)
    typer.echo(f"\n✓ wrote bundle: {out}")
    typer.echo(f"  size: {Path(out).stat().st_size / 1024:.1f} KB")
    bundle_writer.inspect(out)


if __name__ == "__main__":
    app()
