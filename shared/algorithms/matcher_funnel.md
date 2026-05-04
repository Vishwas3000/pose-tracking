# Match Filter Funnel

Spec for the descriptor-matching → 2D-3D-pair stage. Sits between
"LightGlue gave us pairs" and "PnP-RANSAC takes 2D-3D pairs".

This is shared between iOS Swift and Android Kotlin implementations to
ensure parity.

## Inputs

For each frame:
- `query_keypoints: [SIMD2<Float>]` — N query keypoints from ALIKED
- `query_descriptors: [SIMD<128, Float>]` — corresponding descriptors

For each retrieved reference view k (typically K' = 2-5):
- `ref_keypoints[k]: [SIMD2<Float>]`
- `ref_descriptors[k]: [SIMD<128, Float>]`
- `ref_pt3d_indices[k]: [Int]` — index into global points3D array
- (`points3d` global is shared)

## Stages

### Stage 1: LightGlue pairwise matching

For each retrieved reference k:
```
matches[k] = lightGlue(query_descs, ref_descs[k])
            # [(query_idx, ref_idx, score)]
```

LightGlue handles all the descriptor matching — we don't do explicit
mutual NN or ratio test on the raw descriptors. LightGlue's output is
already filtered by its learned graph attention.

### Stage 2: Score thresholding

Drop matches below `match_threshold` (default 0.5). LightGlue's score
is a calibrated confidence in [0, 1].

### Stage 3: 2D-3D conversion

For each surviving match `(q_idx, r_idx, score)` from reference k:
```
p2d = query_keypoints[q_idx]
pt3d_idx = ref_pt3d_indices[k][r_idx]
p3d = global_points3d[pt3d_idx]
matches_2d3d.append((p2d, p3d, score, pt3d_idx))
```

### Stage 4: Cross-reference deduplication (optional)

If multiple reference views match the same 3D point, keep only the
match with the highest score. Reduces redundant constraints in PnP and
can prevent the same outlier from biasing the solver multiple times.

```
# Group by pt3d_idx, keep max score
seen = {}
for m in matches_2d3d:
    if m.pt3d_idx not in seen or m.score > seen[m.pt3d_idx].score:
        seen[m.pt3d_idx] = m
matches_2d3d = list(seen.values())
```

### Stage 5: Sort by score

PROSAC sampling in RANSAC needs the matches sorted by descriptor
confidence:

```
matches_2d3d.sort(by=score, descending=True)
```

## Output

`matches_2d3d: [(p2d, p3d, score, pt3d_idx)]` — sorted by score, ready
for `ransacEPnP` (see `ransac_spec.md`).

## Tunable thresholds

| Param | Default | Notes |
|---|---|---|
| `match_threshold` | 0.5 | Lower → more matches, more outliers |
| `max_matches_in` | 1000 | Cap pre-PnP to bound RANSAC inlier-counting cost |
| `dedup_strategy` | "max_score" | "max_score" or "first" |

## Stages-funnel logging

Like the existing XFeat prototype, log a "funnel" line every N frames
so you can debug match degradation:

```
frame=120 query_kpts=487  ref_views=2
  lg_total=412 → thr=288 → dedup=187 → ransac=141
  maxScore=0.94 meanScore=0.71 inlier_ratio=0.75
```

This makes match degradation visible at a glance:
- `lg_total` low → LightGlue isn't finding matches (descriptor problem)
- `thr` low → matches exist but scores are weak (image quality / motion)
- `ransac` low → matches exist but don't fit a single pose (wrong refs)
