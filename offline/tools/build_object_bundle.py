"""Phase O orchestrator — builds an object .bundle from a COLMAP workspace.

Pipeline:
  O1: load COLMAP outputs + source images           (colmap_io)
  O2: sample K reference views                      (sample_reference_views)
  O3: ALIKED inference per reference view           (aliked_inference)
  O4: map ALIKED keypoints → COLMAP 3D point IDs    (this file, _associate_kpts_to_3d)
  O5: optional global retrieval features            (retrieval_features)
  O6: pack everything into custom .bundle           (bundle_writer)

USAGE:
    python -m tools.build_object_bundle \
        --colmap-dir    /path/to/sfm_workspace \
        --source-images /path/to/source_images \
        --bbox          /path/to/bbox3d_corners.txt \
        --num-refs      30 \
        --out           ../shared/objects/test_object.bundle

Reads the FULL COLMAP workspace (cameras.bin, images.bin, points3D.bin).
If you only have OnePose++'s post-processed `anno_3d_average.npz`, this
pipeline cannot run — re-run SfM to get the COLMAP outputs.
"""

from __future__ import annotations

from pathlib import Path

import typer

# TODO: import siblings once implemented
# from . import colmap_io
# from . import sample_reference_views
# from . import aliked_inference
# from . import retrieval_features
# from . import bundle_writer


app = typer.Typer(add_completion=False)


@app.command()
def main(
    colmap_dir: Path = typer.Option(..., help="COLMAP workspace (cameras.bin, images.bin, points3D.bin)"),
    source_images: Path = typer.Option(..., help="Folder of source images used during SfM"),
    bbox: Path = typer.Option(..., help="bbox3d_corners.txt — 8 lines × 3 floats"),
    out: Path = typer.Option(..., help="Output .bundle path"),
    num_refs: int = typer.Option(30, help="Number of reference views to sample"),
    skip_retrieval: bool = typer.Option(False, help="Skip global retrieval features"),
    aliked_top_k: int = typer.Option(1024, help="Max keypoints per reference view"),
    aliked_threshold: float = typer.Option(0.005, help="ALIKED detection threshold"),
):
    """Build an object .bundle from a COLMAP/SfM workspace."""
    typer.echo(f"=== build_object_bundle ===")
    typer.echo(f"colmap_dir   : {colmap_dir}")
    typer.echo(f"source_images: {source_images}")
    typer.echo(f"bbox         : {bbox}")
    typer.echo(f"out          : {out}")
    typer.echo(f"num_refs     : {num_refs}")

    # --- O1: Load COLMAP workspace ---
    # cameras, images, points3D = colmap_io.read_model(colmap_dir)
    # bbox3d = colmap_io.read_bbox(bbox)
    # typer.echo(f"Loaded COLMAP: {len(images)} frames, {len(points3D)} 3D points")
    raise NotImplementedError("Phase 1 — implement colmap_io first")

    # --- O2: Sample reference views ---
    # ref_image_ids = sample_reference_views.farthest_point_sphere(
    #     images, points3D, k=num_refs
    # )

    # --- O3: ALIKED inference per reference ---
    # aliked = aliked_inference.AlikedRunner(top_k=aliked_top_k, threshold=aliked_threshold)
    # ref_features = {}  # image_id → {'keypoints': (N,2), 'descriptors': (N,128)}
    # for img_id in ref_image_ids:
    #     img_path = source_images / images[img_id].name
    #     ref_features[img_id] = aliked.extract(img_path)

    # --- O4: Map ALIKED keypoints to COLMAP 3D point IDs ---
    # ref_pkg = []
    # for img_id in ref_image_ids:
    #     ak = ref_features[img_id]
    #     ck = images[img_id]   # COLMAP-tracked 2D keypoints + 3D point IDs
    #     pt3d_indices = _associate_kpts_to_3d(ak["keypoints"], ck.xys, ck.point3D_ids)
    #     # Filter ALIKED keypoints with no nearby 3D match
    #     ref_pkg.append({"image_id": img_id, "keypoints": ..., "descriptors": ...,
    #                     "pt3d_indices": ..., "pose": _pose_from_qvec_tvec(ck.qvec, ck.tvec)})

    # --- O5: Optional global retrieval features ---
    # if not skip_retrieval:
    #     embedder = retrieval_features.MobileNetEmbedder()  # or DINOv2
    #     for ref in ref_pkg:
    #         ref["global_emb"] = embedder.embed(...)

    # --- O6: Pack bundle ---
    # bundle_writer.write(
    #     out_path=out,
    #     points3d=np.stack([p.xyz for p in points3D.values()]),
    #     bbox3d=bbox3d,
    #     ref_pkg=ref_pkg,
    # )
    # typer.echo(f"✓ wrote bundle: {out}")


def _associate_kpts_to_3d(aliked_xy, colmap_xy, colmap_pt3d_ids, max_dist_px: float = 3.0):
    """For each ALIKED keypoint, find the nearest COLMAP-tracked 2D point and
    inherit its 3D point ID. Returns array of point3D indices (or -1 for no match)."""
    raise NotImplementedError


def _pose_from_qvec_tvec(qvec, tvec):
    """Convert COLMAP's (quaternion, translation) to a 4×4 pose matrix."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
