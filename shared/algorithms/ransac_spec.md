# Adaptive PROSAC RANSAC for PnP

Spec for the RANSAC wrapper around `solveEPnP`. Same framework on iOS
and Android.

## Why adaptive iterations

Standard RANSAC: fixed iteration count (e.g., 1000). Wastes time on
easy cases (high inlier ratio → 5 iterations is enough), under-shoots
on hard cases.

Adaptive: shrink the iteration budget as soon as we find a high inlier
ratio. Standard formula:

```
iterations_needed(p, k=4, confidence=0.999) = log(1 - confidence) / log(1 - p^k)
```

For `p` (inlier ratio):

| p (inlier ratio) | iterations needed |
|---|---|
| 0.95 (e.g., scanned object very visible) | 4 |
| 0.80 | 12 |
| 0.50 | 100 |
| 0.30 | 1000 |
| 0.10 | 70,000 (give up) |

We start with a max budget (e.g., 5000) and shrink it on every
improvement. Result: sub-millisecond RANSAC on easy frames, capped
budget on hard frames.

## Why PROSAC sampling

Standard RANSAC samples 4 correspondences uniformly at random. For
pipelines like ours where matches have a confidence score (e.g.,
LightGlue's match score), we can do better:

> Sort matches by confidence (descending). Early iterations sample 4
> correspondences from the **top of the list**; gradually expand to the
> full list as iterations progress.

If the top of the list is enriched for true matches (which it is, in
LightGlue's output), this dramatically increases the per-iteration
success probability.

## Pseudocode

```
ransacEPnP(matches, K_intrinsics, threshold_px=4.0, max_iter=5000, conf=0.999):
    matches.sort(by=confidence, descending=True)
    n = len(matches)
    if n < 4: return None

    best_inliers = []
    iterations = max_iter
    it = 0

    while it < iterations:
        # PROSAC: shrink the sample pool as iterations progress
        progress = it / max_iter
        top_n = max(8, int(8 + (n - 8) * progress))
        sample_idxs = random_sample_distinct(0..top_n, 4)
        sample = [matches[i] for i in sample_idxs]

        (R, t) = solveEPnP_4points(sample, K)
        if (R, t) is None: it += 1; continue   # degenerate

        # Score: count inliers across ALL matches
        inliers = []
        for i, m in enumerate(matches):
            projected = K · (R · m.p3d + t)
            projected.xy /= projected.z
            if dist(projected.xy, m.p2d) < threshold_px:
                inliers.append(i)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R, best_t = R, t

            # Shrink iteration budget based on new inlier ratio
            p = len(inliers) / n
            if 0 < p < 1:
                p4 = p ** 4
                needed = log(1 - conf) / log(1 - p4)
                iterations = min(iterations, max(it + 3, ceil(needed)))

        it += 1

    if len(best_inliers) < 8:   # require minimum quality
        return None

    # Final refinement using all inliers
    R, t = solveEPnP_LM([matches[i] for i in best_inliers], K)
    return R, t, best_inliers
```

## Parameters

| Param | Default | Notes |
|---|---|---|
| `threshold_px` | 4.0 | Max reprojection error to count as inlier; tune per camera resolution |
| `max_iter` | 5000 | Hard cap; adaptive logic usually exits much earlier |
| `confidence` | 0.999 | Probability we find the right model — paired with the iteration formula |
| `min_inliers` | 8 | Below this, declare RANSAC failed |

## Reuse from XFeat prototype

The pattern is already proven in
`/Users/sudeepsharma/Documents/GitHub/xFeat/XFeatApp/CameraModel.swift`
(`ransacHomographyInliers`), where adaptive iterations dropped RANSAC
time from 319 ms → ~0 ms on 95 % inlier ratios. Same exact framework;
just swap `solveHomography` → `solveEPnP`.

## Test cases

Synthetic — verify on a scenario where you know the answer:

| Test | Setup | Expected |
|---|---|---|
| 100 % inliers | All matches correct | ~3-5 iterations, recover exact (R, t) |
| 50 % inliers | Half random outliers | ~50-100 iterations, recover (R, t) within 1° |
| 10 % inliers | 90 % outliers | At cap (5000), inlier set ≈ true 10 % |
| Degenerate | 4 colinear points | Returns None |
