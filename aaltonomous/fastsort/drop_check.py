"""Run fastsort perception + bin self-calibration on real LeRobot drops (head RGB + head depth videos).

    python -m aaltonomous.fastsort.drop_check /tmp/drop3 --out /tmp/drop3_check [--frames 0,0.5]

Per episode: the frames at the given fractions of the episode (0 = first frame, arms parked as in our first look).
Prints depth units / validity, the calibration offset + fit quality, every detection (colour, shape, height above the
counter, xy) and saves an overlay PNG (detections, projected bin outlines before/after calibration, depth edges) so the
RGB-depth alignment and the extrinsics can be checked by eye. No labels needed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np


def _frames(path, idx, w=640, h=480, depth=False):
    """Decode the given frame indices of one video file (ffmpeg select), RGB uint8 or depth uint16.
    Without ffmpeg (sim container): read <video>.frames/<n>.png, extracted on the host (PNG, 16 bit for depth)."""
    import shutil
    if shutil.which("ffmpeg") is None:
        import cv2
        out = []
        for i in idx:
            im = cv2.imread(f"{path}.frames/{i:05d}.png", cv2.IMREAD_UNCHANGED)
            if im is None:
                raise FileNotFoundError(f"{path}.frames/{i:05d}.png")
            out.append(im if depth else im[..., ::-1])
        return np.stack(out)
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    fmt, ch, dt = ("gray16le", 1, np.uint16) if depth else ("rgb24", 3, np.uint8)
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select={sel}", "-vsync", "0",
                          "-f", "rawvideo", "-pix_fmt", fmt, "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(out, dt)
    n = a.size // (w * h * ch)
    return a[: n * w * h * ch].reshape(n, h, w, ch) if ch == 3 else a[: n * w * h].reshape(n, h, w)


def table_plane(per, depth, rgb=None, iters=300, tol=0.006, seed=0):
    """RANSAC plane z = z0 + sx (x - 0.30) + sy y through the depth points in front of the bins, in the pelvis frame
    seen through the configured extrinsics. A correct camera pose gives z0 = counter_z and zero slopes."""
    sc = per.scene
    d = np.asarray(depth, float)
    dlo, dhi = per.cfg["depth_range_mm"]
    vv, uu = np.nonzero((d >= dlo) & (d <= dhi))
    if len(uu) < 2000:
        return None
    rng = np.random.default_rng(seed)
    k = rng.choice(len(uu), min(30000, len(uu)), replace=False)
    vv, uu = vv[k], uu[k]
    P = per.cal.pixel_to_pelvis(per.cam, uu, vv, d[vv, uu])
    m = (P[:, 0] > 0.15) & (P[:, 0] < 0.42) & (np.abs(P[:, 1]) < 0.30)
    P = P[m]
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
    inl = np.abs(X @ best - P[:, 2]) < tol
    w, *_ = np.linalg.lstsq(X[inl], P[inl, 2], rcond=None)
    r = X[inl] @ w - P[inl, 2]
    return {"z0": float(w[0]), "sx": float(w[1]), "sy": float(w[2]), "inl": float(inl.mean()), "rms": float(r.std())}


def level_correction(per, pl):
    """Rotation about the camera origin (pelvis frame) that makes the fitted table plane level: returns the corrected
    T_pelvis_optical and the plane height after the rotation."""
    T = per.cal._T(per.cam).copy()
    n = np.array([-pl["sx"], -pl["sy"], 1.0])
    n /= np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(n, z), float(n @ z)
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    C = np.eye(3) + V + V @ V / (1 + c)          # rotates n onto z
    o = T[:3, 3]
    p0 = np.array([0.30, 0.0, pl["z0"]])          # a point on the fitted plane
    z_after = float((C @ (p0 - o) + o)[2])
    T2 = T.copy()
    T2[:3, :3] = C @ T[:3, :3]
    return T2, z_after, C


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="/tmp/drop_check")
    ap.add_argument("--frames", default="0", help="comma list of episode fractions, e.g. 0,0.5")
    ap.add_argument("--episodes", default=None, help="comma list (default all)")
    ap.add_argument("--level", action="store_true", help="also try the table-plane tilt correction")
    ap.add_argument("--fk", action="store_true", help="with --level: project the hands (FK of the recorded joint "
                    "angles) through each camera pose and compare with the depth image")
    ap.add_argument("--depth-shift", type=int, default=4, help="right-shift of decoded depth: gray12le video decoded "
                    "to 16 bit is scaled x16 (drop 3), so 4 gives the sensor's 12-bit value (mm)")
    a = ap.parse_args(argv)
    import pyarrow.parquet as pq
    import cv2
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.scene.scene import Scene
    from .calib import estimate_xy_offset, _bin_rects
    from .grasp import ShapePerception
    from .policy import FAST_OVERRIDES, FastSortPolicy

    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(None, dict(FAST_OVERRIDES))
    scene, cal = Scene.load(), Calibration.load()
    per = ShapePerception(cfg, scene, cal)
    base = np.asarray(per.xy_offset, float).copy()
    E = pq.read_table(glob.glob(os.path.join(a.root, "meta/episodes/*/*.parquet"))[0]).to_pydict()
    eps = [int(x) for x in a.episodes.split(",")] if a.episodes else list(range(len(E["episode_index"])))
    fr = [float(x) for x in a.frames.split(",")]
    k_rgb, k_d = "observation.images.cam_high", "observation.images.head_depth"
    STATE, kin = None, None
    if a.fk:
        from aaltonomous.baselines.classical.kinematics import ArmKinematics
        from aaltonomous.interface.safety import JointSpec
        kin = ArmKinematics(cfg, JointSpec())
        tbl = pq.read_table(glob.glob(os.path.join(a.root, "data/*/*.parquet"))[0], columns=["observation.state"])
        STATE = tbl.column("observation.state").to_pylist()
    rows = []
    for e in eps:
        L = E["length"][e]
        vid = {}
        for k in (k_rgb, k_d):
            ci, fi = E[f"videos/{k}/chunk_index"][e], E[f"videos/{k}/file_index"][e]
            t0 = E[f"videos/{k}/from_timestamp"][e]
            vid[k] = (os.path.join(a.root, f"videos/{k}/chunk-{ci:03d}/file-{fi:03d}.mp4"), int(round(t0 * 30)))
        idx = sorted({min(L - 1, int(f * L)) for f in fr})
        rgbs = _frames(vid[k_rgb][0], [vid[k_rgb][1] + i for i in idx])
        deps = _frames(vid[k_d][0], [vid[k_d][1] + i for i in idx], depth=True)
        for j, i in enumerate(idx):
            rgb, depth = rgbs[j], (deps[j] >> a.depth_shift).astype(np.uint16)
            v = depth[depth > 0]
            per.xy_offset = base.copy()
            est = estimate_xy_offset(per, depth)
            raw = per.detect(rgb, depth)
            line = (f"ep {e:2d} f {i:3d}: depth valid {100 * (depth > 0).mean():.0f}% median {np.median(v) if v.size else 0:.0f}"
                    f" (p5 {np.percentile(v, 5) if v.size else 0:.0f}, p95 {np.percentile(v, 95) if v.size else 0:.0f})")
            good = False
            if est is not None:
                good = est["cost"] < 0.8 * est["cost0"] and np.linalg.norm(est["offset"]) <= 0.03
                line += (f" | calib {np.round(est['offset'] * 1000, 1).tolist()} mm, fit {est['cost0'] * 1000:.1f}->"
                         f"{est['cost'] * 1000:.1f}, inl {est['inliers']:.2f}, n {est['n']} {'OK' if good else 'REJECT'}")
                if good:
                    per.xy_offset = base - est["offset"] + FastSortPolicy.calib_bias(FastSortPolicy)
            else:
                line += " | calib: none"
            dets = per.detect(rgb, depth)
            pl = table_plane(per, depth, rgb)
            if pl is not None:
                line += (f"\n     table plane: z {pl['z0'] * 100:+.1f} cm vs counter_z {scene.counter_z * 100:+.1f} at x=0.30,"
                         f" slope dz/dx {pl['sx']:+.3f} ({np.degrees(np.arctan(pl['sx'])):+.1f} deg), dz/dy {pl['sy']:+.3f}"
                         f" ({np.degrees(np.arctan(pl['sy'])):+.1f} deg), inliers {pl['inl']:.2f}, rms {pl['rms'] * 1000:.1f} mm")
                for d in dets:
                    d.h_plane = float(d.top[2] - (pl["z0"] + pl["sx"] * (d.top[0] - 0.30) + pl["sy"] * d.top[1]))
                if a.level:   # the policy's own first-look routine
                    from .calib import first_look_calibration
                    T_orig = per.cal.cams[per.cam].T_pelvis_optical.copy()
                    notes, W = first_look_calibration(per, depth, T_orig, base, FastSortPolicy.calib_bias(FastSortPolicy))
                    line += "\n     [first look] " + " | ".join(notes)
                    for d in per.detect(rgb, depth):
                        line += (f"\n       {d.color:<6} {str(getattr(d, 'shape', None)):<8} "
                                 f"h={(d.top[2] - scene.counter_z) * 100:4.1f}cm xy=({d.top[0]:+.3f},{d.top[1]:+.3f})")
                    if a.fk:   # hands from the recorded joint angles vs the depth image, per camera pose
                        T_lv = per.cal.cams[per.cam].T_pelvis_optical.copy()
                        T_noz = T_lv.copy()
                        T_noz[2, 3] = T_orig[2, 3]
                        st = np.asarray(STATE[E["dataset_from_index"][e] + i], float)
                        arm, hand = st[:14], st[14:28]
                        pts = np.vstack([kin.hand_points(sd, arm, hand) for sd in ("left", "right")])
                        img = cv2.imread(os.path.join(a.out, f"ep{e:02d}_f{i:03d}.jpg")) if False else rgb[..., ::-1].copy()
                        for tag, TT, col in (("urdf", T_orig, (0, 0, 255)), ("level", T_noz, (255, 0, 255)),
                                             ("level+z", T_lv, (0, 255, 0))):
                            per.cal.cams[per.cam].T_pelvis_optical = TT
                            uvz = per.cal.project(per.cam, pts)
                            dd, pix = [], []
                            for (u, v_, z) in uvz:
                                if z > 0 and 0 <= u < 640 and 0 <= v_ < 480:
                                    cv2.circle(img, (int(u), int(v_)), 4, col, -1)
                                    patch = depth[max(0, int(v_) - 3):int(v_) + 4, max(0, int(u) - 3):int(u) + 4]
                                    patch = patch[patch > 0]
                                    if patch.size:
                                        dd.append(float(np.median(patch)) - 1000 * z)
                            line += (f"\n     [fk {tag}] {len(dd)}/{len(pts)} hand points in view, depth - fk z: median "
                                     f"{np.median(dd) if dd else float('nan'):+.0f} mm (p25 {np.percentile(dd, 25) if dd else 0:+.0f},"
                                     f" p75 {np.percentile(dd, 75) if dd else 0:+.0f})")
                        # world-anchored kinematics (what the policy plans with) through the levelled camera
                        from .calib import set_kin_world
                        set_kin_world(kin, W)
                        ptw = np.vstack([kin.hand_points(sd, arm, hand) for sd in ("left", "right")])
                        per.cal.cams[per.cam].T_pelvis_optical = T_lv
                        dd = []
                        for (u, v_, z) in per.cal.project(per.cam, ptw):
                            if z > 0 and 0 <= u < 640 and 0 <= v_ < 480:
                                patch = depth[max(0, int(v_) - 3):int(v_) + 4, max(0, int(u) - 3):int(u) + 4]
                                patch = patch[patch > 0]
                                if patch.size:
                                    dd.append(float(np.median(patch)) - 1000 * z)
                        line += (f"\n     [fk world-kin + level cam] depth - fk z median "
                                 f"{np.median(dd) if dd else float('nan'):+.0f} mm over {len(dd)} pts")
                        # lowest fingertip of the whole episode in the world frame: ~counter_z + finger radius
                        lo = []
                        for gi in range(E["dataset_from_index"][e], E["dataset_to_index"][e], 3):
                            st2 = np.asarray(STATE[gi], float)
                            lo.append(min(kin.hand_points(sd, st2[:14], st2[14:28])[:, 2].min() for sd in ("left", "right")))
                        line += (f"\n     [teleop] lowest hand point in world frame: {min(lo) * 100:+.1f} cm "
                                 f"(p5 {np.percentile(lo, 5) * 100:+.1f}) vs counter_z {scene.counter_z * 100:+.1f}")
                        set_kin_world(kin, np.eye(4))
                        cv2.imwrite(os.path.join(a.out, f"fk_ep{e:02d}_f{i:03d}.jpg"), img)
                    per.cal.cams[per.cam].T_pelvis_optical = T_orig
                    per.xy_offset = base.copy()
            print(line)
            for d in dets:
                h = (d.top[2] - scene.counter_z) * 100
                print(f"     {d.color:<6} {str(getattr(d, 'shape', None)):<8} h={h:4.1f}cm xy=({d.top[0]:+.3f},{d.top[1]:+.3f})"
                      f" px={d.n_px} dpx={d.n_depth} {d.method} ratio={getattr(d, 'shape_ratio', None) or 0:.2f}"
                      + (f" h_above_plane={d.h_plane * 100:4.1f}cm" if hasattr(d, "h_plane") else ""))
            rows.append({"ep": e, "frame": i, "calib": None if est is None else np.round(est["offset"] * 1000, 1).tolist(),
                         "calib_ok": bool(good), "n_det": len(dets),
                         "dets": [{"color": d.color, "shape": getattr(d, "shape", None),
                                   "h_cm": round(float((d.top[2] - scene.counter_z) * 100), 1),
                                   "xy": np.round(d.top[:2], 3).tolist()} for d in dets]})
            # overlay: depth edges (cyan), bin outlines at counter + rim height before (red) / after (green) calib
            img = rgb[..., ::-1].copy()
            dz = depth.astype(np.float32)
            ed = cv2.Canny(cv2.convertScaleAbs(dz, alpha=255.0 / max(1.0, float(np.percentile(v, 99)) if v.size else 1.0)), 30, 90)
            img[ed > 0] = (255, 255, 0)
            for off, col in ((np.zeros(2), (0, 0, 255)),
                             (None if not good else est["offset"] - FastSortPolicy.calib_bias(FastSortPolicy), (0, 255, 0))):
                if off is None:
                    continue
                for cx, cy, hx, hy in _bin_rects(scene):
                    for z in (scene.counter_z, scene.counter_z + float(scene.cfg["bins"]["wall_height"])):
                        P = np.array([[cx - hx, cy - hy, z], [cx + hx, cy - hy, z], [cx + hx, cy + hy, z], [cx - hx, cy + hy, z]])
                        P[:, :2] += off   # where the walls should appear through the uncorrected extrinsics
                        uvz = cal.project(per.cam, P)
                        if np.all(uvz[:, 2] > 0):
                            cv2.polylines(img, [np.round(uvz[:, :2]).astype(np.int32)], True, col, 1)
            for d in dets:
                cv2.circle(img, (int(d.uv[0]), int(d.uv[1])), 9, (255, 255, 255), 2)
                cv2.putText(img, f"{d.color[0]}{(getattr(d, 'shape', None) or '?')[:3]}", (int(d.uv[0]) + 10, int(d.uv[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(a.out, f"ep{e:02d}_f{i:03d}.jpg"), img)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    offs = np.array([r["calib"] for r in rows if r["calib_ok"]])
    if len(offs):
        print(f"calib accepted {len(offs)}/{len(rows)}: mean {offs.mean(0).round(1).tolist()} mm, std {offs.std(0).round(1).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
