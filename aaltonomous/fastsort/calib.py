"""Head-camera self-calibration from the bins (fixed, known positions in scene.yaml).

The head extrinsics are never exact: the held-out sim variant has camera jitter, the DR config uses +-1 cm / +-1 deg,
and on the robot the URDF pose is unverified. In DR scenes that put object detections 12.5 mm off on average
(median 14, p90 19.5) - enough for the pinch grasp to slip. The three bins are always in view: their wall points,
seen through the same (wrong) extrinsics, are shifted by about the same amount as the objects next to them.

estimate_xy_offset(): depth pixels -> pelvis-frame points, keep those at bin-wall height near the bin footprints,
then a coarse-to-fine grid search for the xy shift that puts them onto the known wall outlines (truncated
point-to-segment distance, robust to objects / hands in the crop). Returns the shift observed - true (to subtract)
and a quality summary; None when there aren't enough wall points.
"""
from __future__ import annotations

import numpy as np


def _bin_rects(scene):
    """Pelvis-frame (cx, cy, hx, hy) of each bin's outer footprint."""
    b = scene.cfg["bins"]
    sx, sy = np.asarray(b["outer_size"], float)  # mat x (= pelvis -y), mat y (= pelvis x)
    out = []
    for c in b["centers"].values():
        p = scene.mat_to_pelvis(np.r_[np.asarray(c, float), 0.0])
        out.append((p[0], p[1], sy / 2, sx / 2))
    return out


def _segments(rects):
    segs = []
    for cx, cy, hx, hy in rects:
        c = np.array([[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy]])
        for i in range(4):
            segs.append((c[i], c[(i + 1) % 4]))
    A = np.array([s[0] for s in segs])
    B = np.array([s[1] for s in segs])
    return A, B


def _dist_to_segments(Q, A, B):
    AB = B - A                                     # (S, 2)
    AP = Q[:, None, :] - A[None]                   # (N, S, 2)
    t = np.clip((AP * AB[None]).sum(-1) / (AB * AB).sum(-1)[None], 0.0, 1.0)
    return np.linalg.norm(AP - t[..., None] * AB[None], axis=-1).min(axis=1)


def estimate_xy_offset(perception, depth, search=0.03, trunc=0.015, max_pts=4000, seed=0):
    if depth is None:
        return None
    sc, cal, cfg = perception.scene, perception.cal, perception.cfg
    d = np.asarray(depth, float)
    dlo, dhi = cfg["depth_range_mm"]
    vv, uu = np.nonzero((d >= dlo) & (d <= dhi))
    if len(uu) < 1000:
        return None
    rng = np.random.default_rng(seed)
    if len(uu) > 80000:
        k = rng.choice(len(uu), 80000, replace=False)
        vv, uu = vv[k], uu[k]
    P = cal.pixel_to_pelvis(perception.cam, uu, vv, d[vv, uu])
    h = P[:, 2] - sc.counter_z
    wall = float(sc.cfg["bins"]["wall_height"])
    rects = _bin_rects(sc)
    near = np.zeros(len(P), bool)
    for cx, cy, hx, hy in rects:
        near |= (np.abs(P[:, 0] - cx) <= hx + search + 0.01) & (np.abs(P[:, 1] - cy) <= hy + search + 0.01)
    Q = P[(h > 0.015) & (h < wall + 0.03) & near][:, :2]
    if len(Q) < 300:
        return None
    if len(Q) > max_pts:
        Q = Q[rng.choice(len(Q), max_pts, replace=False)]
    A, B = _segments(rects)

    def cost(off):
        return float(np.minimum(_dist_to_segments(Q - off, A, B), trunc).mean())

    best, bc = np.zeros(2), cost(np.zeros(2))
    c0 = bc
    for step, rad in ((0.005, search), (0.001, 0.005)):
        ctr = best.copy()
        g = np.arange(-rad, rad + 1e-9, step)
        for dx in g:
            for dy in g:
                o = ctr + (dx, dy)
                c = cost(o)
                if c < bc:
                    best, bc = o, c
    inl = float((_dist_to_segments(Q - best, A, B) < 0.006).mean())
    return {"offset": best, "cost": bc, "cost0": c0, "inliers": inl, "n": int(len(Q))}
