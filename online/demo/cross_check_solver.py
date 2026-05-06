"""Cross-check: feed the SAME 2D-3D matches into cv2 EPnP vs the iOS DLT-PnP
clone, and report which one recovers a usable pose.

This isolates "solver" from "matcher" — we know the matcher is identical
between Python and iOS (Matcher.swift mirrors matcher.py), so if cv2 EPnP
finds a pose where DLT-PnP fails, the iOS bug is the solver.

Usage (from repo root, with .venv active):
    python -m online.demo.cross_check_solver \\
        --bundle shared/objects/session_1777549127.bundle \\
        --xfeat  shared/models/xfeat.pt \\
        --image  offline/data/test_images/IMG_9237.JPG
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from online.tools.bundle_loader import load                # noqa: E402
from online.tools.matcher import match_query_to_bundle     # noqa: E402
from offline.tools.xfeat_inference import XFeatRunner      # noqa: E402
from online.demo.test_images import scale_K                # noqa: E402


# ---------------------------------------------------------------------------
# Pure-Python clone of the iOS DLT-PnP — mirrors EPnP.swift step-for-step so
# we can compare cv2's TRUE EPnP against the same algorithm iOS runs.
# ---------------------------------------------------------------------------

def _normalize_2d(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = pts.mean(axis=0)
    md = np.linalg.norm(pts - mean, axis=1).mean()
    s = np.sqrt(2) / md if md > 1e-9 else 1.0
    T = np.array([[s, 0, -s * mean[0]],
                  [0, s, -s * mean[1]],
                  [0, 0, 1]], dtype=np.float64)
    pts_n = (T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :2]
    return pts_n, T


def _normalize_3d(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = pts.mean(axis=0)
    md = np.linalg.norm(pts - mean, axis=1).mean()
    s = np.sqrt(3) / md if md > 1e-9 else 1.0
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = T[1, 1] = T[2, 2] = s
    T[:3, 3] = -s * mean
    pts_n = (T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]
    return pts_n, T


def dlt_pnp_solve(world: np.ndarray, image: np.ndarray, K: np.ndarray):
    """Mirror of iOS EPnP.swift's solve(...). Returns (R, t) or None.

    Returns the same kind of degenerate solution iOS would return for the
    same input: None on rank-deficient/coplanar samples.
    """
    n = len(world)
    if n < 6:
        return None, "tooFew"

    # 1. Strip K
    Kinv = np.linalg.inv(K)
    img_n = (Kinv @ np.hstack([image, np.ones((n, 1))]).T).T
    img_n = img_n[:, :2] / img_n[:, 2:3]

    # 2. Hartley normalize
    world_n, Tworld = _normalize_3d(world)
    img_n2, Timg   = _normalize_2d(img_n)

    # 3. Build A (2N x 12)
    A = np.zeros((2 * n, 12))
    for i in range(n):
        X, Y, Z = world_n[i]
        x, y    = img_n2[i]
        A[2*i,   :4]   = [X, Y, Z, 1]
        A[2*i,   8:12] = [-x*X, -x*Y, -x*Z, -x]
        A[2*i+1, 4:8]  = [X, Y, Z, 1]
        A[2*i+1, 8:12] = [-y*X, -y*Y, -y*Z, -y]

    # 4. Smallest right singular vector via SVD (= smallest eigenvector of A^T A)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    p = Vt[-1].reshape(3, 4)

    # 5. Denormalize
    P = np.linalg.inv(Timg) @ p @ Tworld

    r1 = P[0, :3]
    scale = np.linalg.norm(r1)
    if scale < 1e-9:
        return None, "scaleZero"

    Rraw = P[:, :3] / scale
    traw = P[:, 3]  / scale

    detR = np.linalg.det(Rraw)
    if detR < 0:
        Rraw = -Rraw
        traw = -traw
    if abs(detR) < 0.1:
        return None, f"planar (|det|={abs(detR):.4f})"

    # 6. Gram-Schmidt
    e1 = Rraw[:, 0] / np.linalg.norm(Rraw[:, 0])
    e2 = Rraw[:, 1] - np.dot(Rraw[:, 1], e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-9:
        return None, "gramFailed"
    e2 /= n2
    e3 = np.cross(e1, e2)
    R = np.column_stack([e1, e2, e3])

    # 7. Cheirality check
    cam = (R @ world.T + traw[:, None]).T
    n_front = (cam[:, 2] > 0).sum()
    if n_front < n // 2:
        return None, f"cheirality (front={n_front}/{n})"

    return (R, traw), "ok"


def dlt_ransac(world: np.ndarray, image: np.ndarray, K: np.ndarray,
               sample_size: int = 8,
               reproj_thresh_px: float = 8.0,
               max_iters: int = 200,
               min_inliers: int = 6):
    """Adaptive-iteration RANSAC around dlt_pnp_solve, mirroring RANSAC.swift."""
    n = len(world)
    rng = np.random.default_rng(0)
    sq_thresh = reproj_thresh_px ** 2

    fail_buckets = {"tooFew": 0, "eigenFailed": 0, "scaleZero": 0,
                    "gramFailed": 0, "ok": 0, "planar": 0, "cheirality": 0}
    best_inliers = []
    best_pose = None
    iter_cap = max_iters
    it = 0
    while it < iter_cap:
        it += 1
        idx = rng.choice(n, sample_size, replace=False)
        sol, why = dlt_pnp_solve(world[idx], image[idx], K)
        # Tally — collapse "planar" / "cheirality" into eigenFailed bucket
        # (matches what iOS prints) but also count separately.
        if why == "ok":
            fail_buckets["ok"] += 1
        elif why.startswith("planar"):
            fail_buckets["planar"] += 1
            fail_buckets["eigenFailed"] += 1
        elif why.startswith("cheirality"):
            fail_buckets["cheirality"] += 1
            fail_buckets["eigenFailed"] += 1
        elif why in fail_buckets:
            fail_buckets[why] += 1

        if sol is None:
            continue
        R, t = sol
        cam = (R @ world.T + t[:, None]).T
        valid = cam[:, 2] > 0
        proj = np.zeros((n, 2))
        proj[valid, 0] = K[0, 0] * cam[valid, 0] / cam[valid, 2] + K[0, 2]
        proj[valid, 1] = K[1, 1] * cam[valid, 1] / cam[valid, 2] + K[1, 2]
        d2 = ((proj - image) ** 2).sum(axis=1)
        d2[~valid] = np.inf
        inliers = np.where(d2 <= sq_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_pose = (R.copy(), t.copy())
            eps = len(inliers) / n
            if eps > 0:
                p1 = eps ** sample_size
                if 0 < p1 < 1:
                    denom = np.log(1 - p1)
                    if denom < -1e-12:
                        new_cap = int(np.ceil(np.log(1 - 0.999) / denom))
                        iter_cap = min(iter_cap, max(new_cap, 1))
            if eps > 0.95:
                break

    return best_pose, best_inliers, fail_buckets, it


# ---------------------------------------------------------------------------
# Coplanarity diagnostic
# ---------------------------------------------------------------------------

def coplanarity_score(pts3d: np.ndarray) -> dict:
    """Return PCA singular values + the 'thinnest dim ratio' score.

    A score near 0 means the points lie on a 2-D plane (rank-2 cov);
    near 1 means a fully 3-D distribution. iOS's DLT-PnP fails when
    each random 8-sample has a low score.
    """
    if len(pts3d) < 4:
        return {"score": 0.0, "sv": [0, 0, 0]}
    centered = pts3d - pts3d.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    score = float(sv[2] / max(sv[0], 1e-9))
    return {"score": score, "sv": sv.tolist()}


def coplanarity_per_sample(pts3d: np.ndarray, sample_size: int = 8,
                            n_samples: int = 200, seed: int = 0) -> dict:
    """How often does a random `sample_size`-subset look coplanar?"""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_samples):
        idx = rng.choice(len(pts3d), sample_size, replace=False)
        scores.append(coplanarity_score(pts3d[idx])["score"])
    scores = np.asarray(scores)
    return {
        "mean_score":     float(scores.mean()),
        "median_score":   float(np.median(scores)),
        "frac_planar_lt_0.05": float((scores < 0.05).mean()),
        "frac_planar_lt_0.10": float((scores < 0.10).mean()),
        "frac_planar_lt_0.20": float((scores < 0.20).mean()),
    }


# ---------------------------------------------------------------------------
# Main: run pipeline on one image, then compare cv2 EPnP vs DLT-PnP RANSAC.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--xfeat",  required=True, type=Path)
    ap.add_argument("--image",  required=True, type=Path)
    ap.add_argument("--ref-size", default="1920x1440")
    ap.add_argument("--reproj-thresh", type=float, default=8.0)
    ap.add_argument("--max-iters",     type=int,   default=200)
    args = ap.parse_args()

    rw, rh = (int(x) for x in args.ref_size.lower().split("x"))

    print(f"Loading bundle: {args.bundle}")
    bundle = load(args.bundle)
    runner = XFeatRunner(args.xfeat)

    from PIL import Image
    with Image.open(args.image) as im:
        tgt_w, tgt_h = im.size
    K_ref = bundle.refs[0].K.astype(np.float64)
    K = scale_K(K_ref, (rw, rh), (tgt_w, tgt_h))
    print(f"Image: {args.image.name}  size={tgt_w}x{tgt_h}")
    print(f"K(scaled): fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    feats = runner.extract(args.image)
    img_pts, world_pts = match_query_to_bundle(
        feats["keypoints"], feats["descriptors"], bundle,
        ratio=0.85, sim_min=0.5,
    )
    print(f"\nMatching: kpts={len(feats['keypoints'])}  matches={len(img_pts)}")
    if len(img_pts) == 0:
        print("[abort] no matches"); return

    # ---- A. cv2 SOLVEPNP_EPNP (the 'ground truth' for what should work) ----
    obj = world_pts.reshape(-1, 1, 3).astype(np.float64)
    pix = img_pts.reshape(-1, 1, 2).astype(np.float64)
    dist = np.zeros(5, dtype=np.float64)

    t0 = time.perf_counter()
    ok, rvec, tvec, inliers_cv = cv2.solvePnPRansac(
        obj, pix, K, dist,
        iterationsCount=args.max_iters,
        reprojectionError=args.reproj_thresh,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    t_cv = (time.perf_counter() - t0) * 1000
    if ok and inliers_cv is not None:
        R_cv, _ = cv2.Rodrigues(rvec)
        t_cv_vec = tvec.flatten()
        n_in_cv = len(inliers_cv)
    else:
        R_cv = None; t_cv_vec = None; n_in_cv = 0

    # ---- B. iOS DLT-PnP clone ----
    t0 = time.perf_counter()
    sol_dlt, in_dlt, buckets, n_iters = dlt_ransac(
        world_pts.astype(np.float64), img_pts.astype(np.float64), K,
        sample_size=8,
        reproj_thresh_px=args.reproj_thresh,
        max_iters=args.max_iters,
    )
    t_dlt = (time.perf_counter() - t0) * 1000

    # ---- C. coplanarity diagnostic on the matched 3-D points ----
    cop_overall = coplanarity_score(world_pts)
    cop_sample  = coplanarity_per_sample(world_pts, sample_size=8, n_samples=400)

    # ---- Report ----
    print("\n" + "=" * 70)
    print(" SOLVER COMPARISON ".center(70, "="))
    print("=" * 70)
    print(f"\n[cv2 SOLVEPNP_EPNP]   ok={ok}  inliers={n_in_cv}/{len(img_pts)}  ({t_cv:.0f} ms)")
    if R_cv is not None:
        print(f"  t = ({t_cv_vec[0]:+.3f}, {t_cv_vec[1]:+.3f}, {t_cv_vec[2]:+.3f})")
        print(f"  R[:,0] = ({R_cv[0,0]:+.3f}, {R_cv[1,0]:+.3f}, {R_cv[2,0]:+.3f})")

    print(f"\n[iOS DLT-PnP clone]   inliers={len(in_dlt)}/{len(img_pts)}  iters={n_iters}/{args.max_iters}  ({t_dlt:.0f} ms)")
    print(f"  failure buckets: {buckets}")
    if sol_dlt is not None:
        R_d, t_d = sol_dlt
        print(f"  t = ({t_d[0]:+.3f}, {t_d[1]:+.3f}, {t_d[2]:+.3f})")
        print(f"  R[:,0] = ({R_d[0,0]:+.3f}, {R_d[1,0]:+.3f}, {R_d[2,0]:+.3f})")
    else:
        print("  -> NO POSE RECOVERED")

    print("\n" + " 3-D POINT GEOMETRY ".center(70, "="))
    sv = cop_overall["sv"]
    print(f"\nFull match set (n={len(world_pts)}):")
    print(f"  PCA singular values: [{sv[0]:.4f}, {sv[1]:.4f}, {sv[2]:.4f}]")
    print(f"  thinnest/largest = {cop_overall['score']:.4f}  "
          f"({'PLANAR' if cop_overall['score'] < 0.05 else '3-D'})")
    print(f"\nRandom 8-sample planarity (400 random samples):")
    print(f"  mean score   = {cop_sample['mean_score']:.4f}")
    print(f"  median score = {cop_sample['median_score']:.4f}")
    print(f"  fraction with score < 0.05 (very planar): {cop_sample['frac_planar_lt_0.05']*100:.1f}%")
    print(f"  fraction with score < 0.10 (near-planar): {cop_sample['frac_planar_lt_0.10']*100:.1f}%")
    print(f"  fraction with score < 0.20 (thin):       {cop_sample['frac_planar_lt_0.20']*100:.1f}%")

    print("\n" + " VERDICT ".center(70, "="))
    if R_cv is not None and sol_dlt is None:
        print("\n>>> cv2 EPnP recovered a pose; DLT-PnP could not.")
        print(">>> The iOS solver is the bottleneck. Need true EPnP.")
    elif R_cv is not None and sol_dlt is not None:
        # Compare the two poses
        R_err = R_cv @ R_d.T
        cos_t = (np.trace(R_err) - 1) / 2
        rot_deg = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
        t_err  = np.linalg.norm(t_cv_vec - t_d)
        print(f"\nBoth solvers recovered a pose; cv2 vs DLT diff:")
        print(f"  rotation diff   = {rot_deg:.2f} deg")
        print(f"  translation diff = {t_err*100:.2f} cm")
        if rot_deg > 5 or t_err > 0.05:
            print(f"  >>> Solvers disagree. cv2 EPnP got {n_in_cv} inliers, DLT got {len(in_dlt)}.")
            print(f"  >>> Likely the DLT pose is the degenerate one.")
        else:
            print("  >>> Solvers agree. DLT-PnP got lucky on this image.")
    elif R_cv is None:
        print("\n>>> Even cv2 EPnP failed. Bundle/matching/K is probably the issue.")


if __name__ == "__main__":
    main()
