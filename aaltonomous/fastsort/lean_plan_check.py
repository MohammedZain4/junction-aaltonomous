"""Dry planning on real-rig frames (LeRobot drop 3: head RGB + depth, robot leaning ~14 deg towards the table).

    python -m aaltonomous.fastsort.lean_plan_check /tmp/drop3 [--episodes 0,5,10]

For each episode's first frame the fastsort policy self-calibrates and plans every object it would sort, twice:
  lean   : body-lean correction (camera + arm kinematics in the levelled world frame), the default
  nolean : AALTO_LEVEL=0 behaviour (URDF pelvis frame = world, i.e. what the robot would do without the fix)
Both plans are then measured in the TRUE geometry = the lean-corrected world frame (the table plane it fits has
~2 mm rms and agrees with arm FK and the teleop fingertips, drop_check --fk): lowest hand point over the plan vs the
counter, and where the grasp point ends up (end of descend) vs the object (lean detection). No robot motion.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--episodes", default="0,5,10")
    ap.add_argument("--depth-shift", type=int, default=4)
    a = ap.parse_args(argv)

    import pyarrow.parquet as pq
    from .calib import set_kin_world
    from .drop_check import _frames
    from .policy import FastSortPolicy

    E = pq.read_table(glob.glob(os.path.join(a.root, "meta/episodes/*/*.parquet"))[0]).to_pydict()
    k_rgb, k_d = "observation.images.cam_high", "observation.images.head_depth"
    pols = {}
    for mode in ("lean", "nolean"):
        os.environ["AALTO_LEVEL"] = "1" if mode == "lean" else "0"
        p = FastSortPolicy(log=lambda *x, **k: None)
        p.reset({"instruction": ""})
        pols[mode] = p
    cz = pols["lean"].scene.counter_z
    tot = {m: dict(n=0, ok=0, below=0, xy=[]) for m in pols}
    for e in [int(x) for x in a.episodes.split(",")]:
        img = {}
        for k in (k_rgb, k_d):
            ci, fi = E[f"videos/{k}/chunk_index"][e], E[f"videos/{k}/file_index"][e]
            t0 = int(round(E[f"videos/{k}/from_timestamp"][e] * 30))
            img[k] = _frames(os.path.join(a.root, f"videos/{k}/chunk-{ci:03d}/file-{fi:03d}.mp4"), [t0], depth=k == k_d)[0]
        rgb, depth = img[k_rgb], (img[k_d] >> a.depth_shift).astype(np.uint16)
        W_true, ref = None, None
        for mode, pol in pols.items():
            os.environ["AALTO_LEVEL"] = "1" if mode == "lean" else "0"
            pol.events = []
            pol._self_calibrate(depth)
            W = getattr(pol.kin, "world", np.eye(4))
            if mode == "lean":
                W_true = W.copy()
            to_true = W_true @ np.linalg.inv(W)   # this mode's planning frame -> true world
            dets = pol.perception.detect(rgb, depth)
            if mode == "lean":
                ref = [(d.color, d.top.copy()) for d in dets]
            print(f"ep {e:2d} [{mode}] " + " | ".join(str(n) for _, n in pol.events[-2:]))
            for d in dets:
                b = pol.planner.mapping.get(d.color)
                if b is None:
                    continue
                by = pol.planner.bin_xy(b)[1]
                xy = d.top[:2]
                side = ("left" if by > 0 else "right") if abs(by) > 0.1 else ("left" if xy[1] > 0 else "right")
                plan = pol.planner._plan_side_hints(d, side, b, pol.home.copy(), pol.post["open"].copy())
                t = tot[mode]
                t["n"] += 1
                if not plan.ok:
                    print(f"   {d.color:<6} {str(getattr(d, 'shape', None)):<8} -> {b:<6} {side:<5} NO PLAN: {plan.reason}")
                    continue
                t["ok"] += 1
                sl = slice(0, 7) if side == "left" else slice(7, 14)
                lo, g = 9.0, None
                for k in range(0, len(plan.arm), 2):
                    P = pol.kin.hand_points(side, plan.arm[k], plan.hand[k])
                    Pt = P @ to_true[:3, :3].T + to_true[:3, 3]
                    lo = min(lo, float(Pt[:, 2].min()))
                de = next(s_e for name, *s_e in plan.phases if name == "descend")[1] - 1
                P = pol.kin.hand_points(side, plan.arm[de], plan.hand[de])
                g = P[1] @ to_true[:3, :3].T + to_true[:3, 3]
                obj = min((r for r in ref if r[0] == d.color), key=lambda r: np.linalg.norm(r[1][:2] - g[:2]), default=None)
                exy = float(np.linalg.norm(obj[1][:2] - g[:2])) if obj is not None else float("nan")
                clip = pol.planner.dry_run_safety(pol.home.copy(), pol.post["open"].copy(), plan.arm, plan.hand)
                t["clip"] = t.get("clip", 0) + int(bool(clip))
                t["below"] += int(lo < cz - 0.002)
                t["xy"].append(exy)
                print(f"   {d.color:<6} {str(getattr(d, 'shape', None)):<8} -> {b:<6} {side:<5} lowest hand point "
                      f"{(lo - cz) * 100:+5.1f} cm vs real counter, grasp point z {(g[2] - cz) * 100:4.1f} cm, "
                      f"xy {exy * 1000:4.0f} mm from object (true frame), safety filter clips {clip} steps")
            set_kin_world(pol.kin, np.eye(4))
    for m, t in tot.items():
        xy = np.array(t["xy"]) * 1000
        print(f"[{m}] {t['ok']}/{t['n']} planned, {t['below']} plans put the hand >2 mm into the real counter, "
              f"{t.get('clip', 0)} clipped by the safety filter, grasp xy error median {np.nanmedian(xy) if xy.size else float('nan'):.0f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
