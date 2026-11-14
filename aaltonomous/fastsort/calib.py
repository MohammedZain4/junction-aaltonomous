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


def estimate_xy_offset(perception, depth, search=0.045, trunc=0.015, max_pts=4000, seed=0):
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


# ------------------------------------------------------------------------------------------------ table levelling
# Real robot (drop 3, 15 episodes, 2026-11-14): through the URDF head pose the counter plane came out tilted +14 deg
# (rising away from the robot) and 2.7 cm high, every episode, plane rms 2 mm; objects measured 5 cm above that plane.
# The bin fit then failed (14 % inliers). Rotating the camera about its origin so the counter is level (and shifting it
# so the counter is at counter_z) brought the bin fit to 55-60 % inliers with a consistent residual of ~2 cm in xy,
# which estimate_xy_offset() then removes. In sim the correction is ~0 (the camera sits at the URDF pose).

def fit_table_plane(perception, depth, iters=300, tol=0.006, seed=0, x_range=(0.15, 0.42), y_abs=0.30):
    """RANSAC plane z = z0 + sx (x - 0.30) + sy y through the depth points in front of the bins (pelvis frame,
    current extrinsics). None if there is too little depth."""
    d = np.asarray(depth, float)
    dlo, dhi = perception.cfg["depth_range_mm"]
    vv, uu = np.nonzero((d >= dlo) & (d <= dhi))
    if len(uu) < 2000:
        return None
    rng = np.random.default_rng(seed)
    k = rng.choice(len(uu), min(30000, len(uu)), replace=False)
    vv, uu = vv[k], uu[k]
    P = perception.cal.pixel_to_pelvis(perception.cam, uu, vv, d[vv, uu])
    P = P[(P[:, 0] > x_range[0]) & (P[:, 0] < x_range[1]) & (np.abs(P[:, 1]) < y_abs)]
    if len(P) < 1000:
        return None
    X = np.c_[np.ones(len(P)), P[:, 0] - 0.30, P[:, 1]]
    best, bn = None, -1
    for _ in range(iters):
        s = rng.choice(len(P), 3, replace=False)
        try:
            w = np.linalg.solve(X[s], P[s, 2])
        except np.linalg.LinAlgError:
            continue
        n = int((np.abs(X @ w - P[:, 2]) < tol).sum())
        if n > bn:
            best, bn = w, n
    if best is None:
        return None
    inl = np.abs(X @ best - P[:, 2]) < tol
    w, *_ = np.linalg.lstsq(X[inl], P[inl, 2], rcond=None)
    r = X[inl] @ w - P[inl, 2]
    return {"z0": float(w[0]), "sx": float(w[1]), "sy": float(w[2]), "inl": float(inl.mean()), "rms": float(r.std())}


def level_extrinsics(T, plane, counter_z):
    """T_pelvis_optical rotated about the camera origin so the fitted plane is level, then shifted in z so it lies at
    counter_z. Returns (T_new, tilt_deg)."""
    n = np.array([-plane["sx"], -plane["sy"], 1.0])
    n /= np.linalg.norm(n)
    v, c = np.cross(n, [0.0, 0.0, 1.0]), float(n[2])
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    C = np.eye(3) + V + V @ V / (1 + c)
    o = T[:3, 3]
    z_after = float((C @ (np.array([0.30, 0.0, plane["z0"]]) - o) + o)[2])
    T2 = np.asarray(T, float).copy()
    T2[:3, :3] = C @ T2[:3, :3]
    T2[2, 3] += counter_z - z_after
    return T2, float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def first_look_calibration(perception, depth, T0, xy_base, xy_bias=(0.0, 0.0), level=True, max_tilt_deg=25.0,
                           min_lean_deg=5.0):
    """Reset the head extrinsics to T0 / the xy offset to xy_base, then
      (1) fit the counter plane. A tilt >= min_lean_deg is BODY LEAN (real rig, drop 3: 14 deg; the URDF camera pose
          agreed with arm FK there, so the camera is right and the robot is pitched relative to the counter): the
          planning frame becomes a levelled 'world' frame W (pelvis -> world) for the camera AND the arms. Smaller
          tilts (sim: camera jitter only, upright robot) are left alone, exactly the behaviour tested in sim.
      (2) fit the bins in xy (+ method-bias correction): with lean, the shift moves the world frame (bins placed
          differently relative to the robot); without, it corrects the camera only (sim DR camera jitter).
    Mutates perception.cal / perception.xy_offset; returns (notes, W 4x4) - apply W to the kinematics."""
    cam = perception.cam
    T0 = np.asarray(T0, float)
    perception.cal.cams[cam].T_pelvis_optical = T0.copy()
    perception.xy_offset = np.asarray(xy_base, float).copy()
    notes, W, lean = [], np.eye(4), False
    if level:
        T = T0.copy()
        for _ in range(2):   # refit in the corrected frame: the plane crop moves with the correction
            perception.cal.cams[cam].T_pelvis_optical = T
            pl = fit_table_plane(perception, depth)
            if pl is None or pl["inl"] < 0.25 or pl["rms"] > 0.006:
                break
            T, _ = level_extrinsics(T, pl, perception.scene.counter_z)
        perception.cal.cams[cam].T_pelvis_optical = T
        chk = fit_table_plane(perception, depth)
        R_rel = T[:3, :3] @ T0[:3, :3].T
        tilt = float(np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))))
        ok = (chk is not None and chk["inl"] >= 0.25 and abs(chk["sx"]) < 0.02 and abs(chk["sy"]) < 0.02
              and tilt <= max_tilt_deg and abs(T[2, 3] - T0[2, 3]) <= 0.06)
        lean = ok and tilt >= min_lean_deg
        if lean:
            W = T @ np.linalg.inv(T0)
        else:
            perception.cal.cams[cam].T_pelvis_optical = T0.copy()
        notes.append(f"table plane: tilt {tilt:.1f} deg, height {100 * (T[2, 3] - T0[2, 3]):+.1f} cm"
                     + (f", inliers {chk['inl']:.2f} rms {chk['rms'] * 1000:.1f} mm" if chk else "")
                     + (": BODY LEAN, levelled world frame for camera + arms" if lean else
                        (": below lean threshold, unchanged" if ok else ": fit not trusted, unchanged")))
    try:
        est = estimate_xy_offset(perception, depth)
    except Exception as e:   # noqa: BLE001
        notes.append(f"self-calibration failed: {e!r}")
        return notes, W
    if est is None:
        notes.append("self-calibration: not enough bin wall points, using the configured extrinsics")
        return notes, W
    off = est["offset"]
    good = est["cost"] < 0.8 * est["cost0"] and np.linalg.norm(off) <= 0.045
    if good:
        s = -np.asarray(off, float) + np.asarray(xy_bias, float)
        if lean:
            S = np.eye(4)
            S[:2, 3] = s
            W = S @ W
            perception.cal.cams[cam].T_pelvis_optical = W @ T0
        else:
            perception.xy_offset = np.asarray(xy_base, float) + s
    notes.append(f"self-calibration: bins seen {np.round(off * 1000, 1)} mm off, fit {est['cost0'] * 1000:.1f}"
                 f" -> {est['cost'] * 1000:.1f} mm, inliers {est['inliers']:.2f}: "
                 + (("applied to the world frame" if lean else "applied") if good else "NOT applied"))
    return notes, W


def set_kin_world(kin, W):
    """Re-anchor the arm kinematics (pinocchio model) in the frame W (4x4, pelvis -> world): every FK point, Jacobian
    and the IK 'hand down' axis are then in world coordinates. Identity restores the URDF pelvis frame."""
    pin = kin.pin
    m = kin.model
    if not hasattr(kin, "_root0"):
        kin._root0 = {j: pin.SE3(m.jointPlacements[j]) for j in range(1, m.njoints) if m.parents[j] == 0}
        kin._frame0 = {}
        for f in range(m.nframes):
            fr = m.frames[f]
            pj = getattr(fr, "parentJoint", None)
            pj = getattr(fr, "parent", 0) if pj is None else pj
            if pj == 0:
                kin._frame0[f] = pin.SE3(fr.placement)
    Wse3 = pin.SE3(np.asarray(W, float)[:3, :3].copy(), np.asarray(W, float)[:3, 3].copy())
    for j, M in kin._root0.items():
        m.jointPlacements[j] = Wse3 * M
    for f, M in kin._frame0.items():
        fr = m.frames[f]
        fr.placement = Wse3 * M
        m.frames[f] = fr
    kin.world = np.asarray(W, float).copy()
