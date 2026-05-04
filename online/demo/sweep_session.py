"""Sweep the online pipeline over a session and write per-frame bbox overlays.

For every Nth frame in the session, run ALIKED -> match -> EPnP-RANSAC, project
the bundle's 3D bbox using the estimated pose, draw it on the image, and save
to an output folder. Also writes a CSV summary with per-frame stats and the
pose error against ARKit ground-truth metadata.

Usage:
    python -m online.demo.sweep_session \\
        --bundle  shared/objects/session_1777549127.bundle \\
        --aliked  shared/models/aliked-n16rot-top1k-640.onnx \\
        --session offline/data/session_1777549127 \\
        --out     offline/data/overlays \\
        --stride  5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from online.tools.pipeline import PoseTracker             # noqa: E402
from online.demo.overlay_bbox import (                     # noqa: E402
    project_points, draw_bbox, annotate_status,
)


def gt_w2c_from_metadata(meta_path: Path) -> np.ndarray | None:
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    c2w = np.array(meta["camera_pose_c2w"], dtype=np.float64)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(c2w @ flip)


def pose_error(P_est: np.ndarray, P_gt: np.ndarray) -> tuple[float, float]:
    R_err = P_est[:3, :3] @ P_gt[:3, :3].T
    cos_t = (np.trace(R_err) - 1) / 2
    rot_deg = float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
    t_err = float(np.linalg.norm(P_est[:3, 3] - P_gt[:3, 3]))
    return rot_deg, t_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle",  required=True, type=Path)
    ap.add_argument("--aliked",  required=True, type=Path)
    ap.add_argument("--session", required=True, type=Path,
                    help="iOS session folder with frames/ + metadata/")
    ap.add_argument("--out",     required=True, type=Path)
    ap.add_argument("--stride",  type=int, default=5,
                    help="Process every Nth frame (default 5)")
    ap.add_argument("--limit",   type=int, default=0,
                    help="Optional cap on number of frames processed (0 = no cap)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading bundle {args.bundle.name} + model {args.aliked.name} ...")
    tracker = PoseTracker(args.bundle, args.aliked)
    print(f"  M={len(tracker.bundle.points3d):,}  K_refs={len(tracker.bundle.refs)}")

    frames_dir   = args.session / "frames"
    metadata_dir = args.session / "metadata"
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    todo = frames[::args.stride]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Processing {len(todo)} / {len(frames)} frames (stride={args.stride})\n")

    summary_rows = []
    n_ok = 0
    n_lost = 0
    t_start = time.perf_counter()

    for i, fp in enumerate(todo):
        frame_id = fp.stem.replace("frame_", "")
        meta_path = metadata_dir / f"metadata_{frame_id}.json"
        gt = gt_w2c_from_metadata(meta_path)

        t0 = time.perf_counter()
        result = tracker.process(fp)
        ms = 1000 * (time.perf_counter() - t0)

        # Read raw pixels (no EXIF auto-rotate) so the canvas matches the
        # coordinate frame ALIKED + bbox projection are in.
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if result.pose is not None:
            corners_3d = tracker.bundle.bbox3d.astype(np.float64)
            corners_2d, z = project_points(corners_3d, result.pose, tracker.K)
            in_front = z > 0
            img = draw_bbox(img, corners_2d, in_front)
            rot_deg = t_cm = None
            if gt is not None:
                rd, tm = pose_error(result.pose, gt)
                rot_deg, t_cm = rd, tm * 100
            status = [
                f"{fp.name}  inliers={result.n_inliers}/{result.n_matches}  "
                f"kpts={result.n_aliked_kpts}  ({ms:.0f} ms)",
                (f"err vs ARKit: rot={rot_deg:.2f} deg  t={t_cm:.2f} cm"
                 if rot_deg is not None else "err vs ARKit: (no metadata)"),
            ]
            n_ok += 1
        else:
            status = [f"{fp.name}  [LOST]",
                      f"matches={result.n_matches}  inliers={result.n_inliers}"]
            rot_deg = t_cm = None
            n_lost += 1

        out_img = annotate_status(img, status)
        cv2.imwrite(str(args.out / f"frame_{frame_id}_bbox.jpg"), out_img)

        summary_rows.append({
            "frame": fp.name,
            "frame_id": frame_id,
            "ok": result.pose is not None,
            "matches": result.n_matches,
            "inliers": result.n_inliers,
            "kpts": result.n_aliked_kpts,
            "rot_deg": f"{rot_deg:.3f}" if rot_deg is not None else "",
            "t_cm":    f"{t_cm:.3f}"    if t_cm    is not None else "",
            "elapsed_ms": f"{ms:.1f}",
        })
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (len(todo) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1:3d}/{len(todo)}]  ok={n_ok}  lost={n_lost}  "
                  f"{rate:.2f} fps  ETA {eta:.0f}s")

    csv_path = args.out / "sweep_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    # Aggregate stats over rows that produced a pose AND have GT metadata
    valid = [r for r in summary_rows if r["t_cm"]]
    if valid:
        rs = np.array([float(r["rot_deg"]) for r in valid])
        ts = np.array([float(r["t_cm"])    for r in valid])
        print(f"\n=== aggregate (over {len(valid)} frames with GT) ===")
        print(f"rotation err [deg]    mean={rs.mean():.2f}  median={np.median(rs):.2f}  p95={np.percentile(rs, 95):.2f}")
        print(f"translation err [cm]  mean={ts.mean():.2f}  median={np.median(ts):.2f}  p95={np.percentile(ts, 95):.2f}")

    print(f"\n=== sweep ===")
    print(f"  total processed : {len(summary_rows)}")
    print(f"  pose recovered  : {n_ok}  ({100*n_ok/len(summary_rows):.1f}%)")
    print(f"  LOST            : {n_lost}")
    print(f"  overlays        : {args.out}")
    print(f"  summary csv     : {csv_path}")


if __name__ == "__main__":
    main()
