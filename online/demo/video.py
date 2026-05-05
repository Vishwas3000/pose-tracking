"""Run the online pipeline on a video and write an annotated output video.

Same pipeline as `test_images.py` (ALIKED → match → EPnP-RANSAC → bbox draw),
just with cv2.VideoCapture / VideoWriter instead of folder-of-images I/O.

Usage:
    python -m online.demo.video \\
        --bundle  shared/objects/session_1777549127.bundle \\
        --aliked  shared/models/aliked-n16rot-top1k-640.onnx \\
        --video   /path/to/input.mp4 \\
        --out     offline/data/video_out.mp4

Optional:
    --dinov2 shared/models/dinov2-small-int8.onnx --top-n 5    DINOv2 retrieval
    --stride 2                                                 process every 2nd frame
    --max-frames 300                                           cap at 300 frames
    --ref-size 1920x1440                                       resolution the bundle's K was calibrated for
"""

from __future__ import annotations

import argparse
import csv
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
from online.demo.test_images import scale_K                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle",   required=True, type=Path)
    ap.add_argument("--aliked",   required=True, type=Path)
    ap.add_argument("--video",    required=True, type=Path)
    ap.add_argument("--out",      required=True, type=Path)
    ap.add_argument("--dinov2",   default=None, type=Path)
    ap.add_argument("--top-n",    type=int, default=5)
    ap.add_argument("--stride",   type=int, default=1,
                    help="Process every Nth frame; intermediate frames hold the last drawn pose.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Cap on number of source frames (0 = no cap).")
    ap.add_argument("--ref-size", default="1920x1440",
                    help="Resolution the bundle's K was calibrated for.")
    ap.add_argument("--rotate-output", choices=["0", "cw", "ccw", "180"],
                    default="0",
                    help="Rotate each rendered frame before writing — useful "
                         "when the source was shot in portrait but the bundle "
                         "(and thus our K) is in landscape sensor orientation.")
    args = ap.parse_args()

    rw, rh = (int(x) for x in args.ref_size.lower().split("x"))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading bundle  : {args.bundle.name}")
    tracker = PoseTracker(args.bundle, args.aliked,
                          dinov2_onnx=args.dinov2,
                          retrieval_top_n=args.top_n)
    print(f"  M={len(tracker.bundle.points3d):,}  K_refs={len(tracker.bundle.refs)}")
    print(f"  retrieval     : {'DINOv2 top-' + str(args.top_n) if tracker.embedder else 'OFF'}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    # Stop cv2 from auto-applying any displaymatrix rotation tag that's
    # leftover from the source. We want raw sensor-orientation pixel data
    # so the bundle's landscape K applies as-is; --rotate-output handles
    # the display orientation at write time.
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        # Re-read after disabling, dimensions can change.
        ok, probe = cap.read()
        if ok:
            src_h, src_w = probe.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo            : {args.video.name}  {src_w}x{src_h}  {src_fps:.2f} fps  ~{src_n} frames")

    K = scale_K(tracker.K, (rw, rh), (src_w, src_h))
    print(f"  ref K @ {rw}x{rh}: fx={tracker.K[0,0]:.1f}  cx={tracker.K[0,2]:.1f}  cy={tracker.K[1,2]:.1f}")
    print(f"  scaled K        : fx={K[0,0]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = src_fps / max(1, args.stride)
    rot_map = {
        "0":   None,
        "cw":  cv2.ROTATE_90_CLOCKWISE,
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }
    rot = rot_map[args.rotate_output]
    out_w, out_h = (src_w, src_h) if rot is None or rot == cv2.ROTATE_180 else (src_h, src_w)
    writer = cv2.VideoWriter(str(args.out), fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {args.out}")

    summary = []
    last_pose = None             # carry pose forward on stride-skipped frames
    last_drawn_status = ["", ""]

    n_in     = 0
    n_proc   = 0
    n_ok     = 0
    n_lost   = 0
    t_start  = time.perf_counter()
    recent_ms: list[float] = []   # rolling window for displayed fps
    tmp_jpg  = args.out.parent / ".video_frame.jpg"   # disk hop into AlikedRunner's PIL loader

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and n_in >= args.max_frames:
            break

        if n_in % args.stride == 0:
            cv2.imwrite(str(tmp_jpg), frame)
            t0 = time.perf_counter()
            try:
                tracker.K = K
                result = tracker.process(tmp_jpg)
            finally:
                pass
            ms = 1000 * (time.perf_counter() - t0)
            n_proc += 1
            recent_ms.append(ms)
            if len(recent_ms) > 30:
                recent_ms.pop(0)
            avg_ms = sum(recent_ms) / len(recent_ms)
            fps_now = 1000.0 / avg_ms if avg_ms > 0 else 0
            if result.pose is not None:
                last_pose = result.pose
                n_ok += 1
                last_drawn_status = [
                    f"{args.video.name}  frame {n_in:>5d}/{src_n}",
                    f"matches={result.n_matches}  inliers={result.n_inliers}  "
                    f"kpts={result.n_aliked_kpts}  ({ms:.0f} ms)",
                    f"fps {fps_now:5.1f}  (rolling 30-frame avg)",
                ]
            else:
                n_lost += 1
                last_pose = None
                last_drawn_status = [
                    f"{args.video.name}  frame {n_in:>5d}/{src_n}",
                    f"[LOST]  matches={result.n_matches}  inliers={result.n_inliers}",
                    f"fps {fps_now:5.1f}  (rolling 30-frame avg)",
                ]
            summary.append({
                "frame_idx": n_in,
                "ok": result.pose is not None,
                "matches": result.n_matches,
                "inliers": result.n_inliers,
                "kpts": result.n_aliked_kpts,
                "elapsed_ms": f"{ms:.1f}",
            })

        # Render whichever pose is current onto this frame
        rendered = frame
        if last_pose is not None:
            corners_3d = tracker.bundle.bbox3d.astype(np.float64)
            corners_2d, z = project_points(corners_3d, last_pose, K)
            in_front = z > 0
            rendered = draw_bbox(rendered, corners_2d, in_front)
        rendered = annotate_status(rendered, last_drawn_status)
        if rot is not None:
            rendered = cv2.rotate(rendered, rot)
        writer.write(rendered)

        n_in += 1
        if n_in % 30 == 0:
            elapsed = time.perf_counter() - t_start
            rate = n_in / elapsed if elapsed > 0 else 0
            eta = (src_n - n_in) / rate if rate > 0 else 0
            print(f"  [{n_in:>5d}/{src_n}]  ok={n_ok} lost={n_lost}  src {rate:.1f} fps  ETA {eta:.0f}s")

    cap.release()
    writer.release()
    if tmp_jpg.exists():
        tmp_jpg.unlink()

    csv_path = args.out.with_suffix(".csv")
    if summary:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)

    elapsed = time.perf_counter() - t_start
    print(f"\n=== done ===")
    print(f"  source frames   : {n_in}")
    print(f"  processed       : {n_proc}  (stride={args.stride})")
    print(f"  pose recovered  : {n_ok}  ({100*n_ok/max(1, n_proc):.1f}%)")
    print(f"  LOST            : {n_lost}")
    print(f"  wall            : {elapsed:.1f} s   ({n_proc/elapsed:.1f} processed fps)")
    print(f"  output video    : {args.out}")
    print(f"  csv             : {csv_path}")


if __name__ == "__main__":
    main()
