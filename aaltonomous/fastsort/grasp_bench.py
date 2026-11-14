"""Grasp benchmark against sim ground truth: ONE pick-and-place attempt per object, one arm at a time.

    python -m aaltonomous.fastsort.grasp_bench --seeds 1 2 3 [--closed 0,-0.1,1.2,-1.0,-0.6,-1.0,-0.6] [--acc 10]

Per seed (same sequence as sim/evaluate.py, with DR if the sim runs with AALTO_DR): arms to the park pose, one
look, then for each detected object (nearest-bin order) a single fastsort plan to its fixed-mapping bin, executed
open-loop through AaltoEnv. Scored at the end by the scene service (the official checker). Isolates grasp physics
(posture, squeeze, accelerations) from scheduling. Ground truth is used only for scoring.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from aaltonomous.interface.env import AaltoEnv
from aaltonomous.interface.tasks import FIXED_MAPPING
from aaltonomous.sim.evaluate import home_arms
from aaltonomous.sim.scene_client import SceneClient

from .policy import FastSortPolicy


def run(env, A, H, chunk=15, snap=None):
    """Execute open-loop. Returns (obs, end_reason or None). snap: {name: tick} -> obs right after that tick
    (stored back into snap as name -> obs), e.g. the lift start / end for the grip check."""
    obs = None
    want = dict(snap or {})
    for k in range(0, len(A), chunk):
        obs, info = env.step({"arm_q_target": A[k:k + chunk], "hand_q_target": H[k:k + chunk]})
        for name, t in list(want.items()):
            if k + chunk >= t:
                snap[name] = obs
                del want[name]
        if info.get("episode_over"):
            return obs, f"{info.get('end_reason')} (clipped {info.get('clipped_steps')}, {info.get('clip_reasons')})"
    return obs, None


def _n_correct(sc):
    return sum(o["outcome"] == "correct" for o in sc.evaluate("sort_fixed", dict(FIXED_MAPPING))["outcomes"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--closed", default=None, help="hand.closed posture, 7 values, LEFT order")
    ap.add_argument("--acc", type=float, default=None, help="motion.arm_max_acc")
    ap.add_argument("--close-hold", type=float, default=None, help="motion.close_hold_s")
    ap.add_argument("--wrist", action="store_true", help="log grip-check features (joints + wrist cam) per pick, "
                    "labelled by the scene scorer; one JSON line per pick in --wrist-out")
    ap.add_argument("--wrist-out", default="/tmp/grip_features.jsonl")
    ap.add_argument("--true-xy", action="store_true", help="grasp at the TRUE object xy (sim state) instead of the "
                    "perceived one; shape, axes and colour stay perceived. Answers: are misses position errors?")
    a = ap.parse_args(argv)
    ov = {}
    if a.closed:
        ov["hand.closed"] = [float(x) for x in a.closed.split(",")]
    if a.acc:
        ov["motion.arm_max_acc"] = a.acc
    if a.close_hold:
        ov["motion.close_hold_s"] = a.close_hold
    pol = FastSortPolicy(log=None, overrides=ov)
    pl, per, spec = pol.planner, pol.perception, pol.spec
    pl.mapping = dict(FIXED_MAPPING)
    sc = SceneClient()
    dr = bool(sc.info().get("dr"))
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=a.seeds[0], realtime=True, img_server_ip="127.0.0.1", time_limit=900.0,
                   head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")), cameras=a.wrist)
    by_shape, n_tot, n_ok = {}, 0, 0
    wout = open(a.wrist_out, "a") if a.wrist else None
    for s in a.seeds:
        home_arms(env, sc)
        env.rng = np.random.default_rng(s)
        if dr:
            sc.call(cmd="randomize", seed=s)
        obs = env.reset()
        arm = np.clip(np.asarray(obs["arm_q"], float), spec.arm_lower, spec.arm_upper)
        hand = np.clip(np.asarray(obs["hand_q"], float), spec.hand_lower, spec.hand_upper)
        A, Hh = pl.go_to(arm, hand, pol.home, pol.post["open"])
        n = 10
        A = np.vstack([A, np.repeat(pol.home[None], n, 0)]) if len(A) else np.repeat(arm[None], n, 0)
        Hh = np.vstack([Hh, np.repeat(pol.post["open"][None], n, 0)]) if len(Hh) else np.repeat(hand[None], n, 0)
        obs, why = run(env, A, Hh)
        cur_a, cur_h = A[-1].copy(), Hh[-1].copy()
        dets = per.detect(obs["head_rgb"], obs["head_depth"])
        if a.true_xy:
            truth = [np.asarray(o["pos"], float) for o in sc.state()["objects"] if o["active"]]
            errs = []
            for d in dets:
                if truth:
                    t = min(truth, key=lambda p: np.linalg.norm(p[:2] - d.top[:2]))
                    if np.linalg.norm(t[:2] - d.top[:2]) < 0.05:
                        errs.append(1000 * np.linalg.norm(t[:2] - d.top[:2]))
                        d.top = np.r_[t[:2], d.top[2]]
            print(f"  seed {s}: true xy used, perceived error {np.round(errs, 1).tolist()} mm")
        tried = 0
        for d in sorted(dets, key=lambda d: np.linalg.norm(d.top[:2] - pl.drop_xy(FIXED_MAPPING[d.color]))):
            b = FIXED_MAPPING[d.color]
            by = pl.bin_xy(b)[1]
            side = ("left" if by > 0 else "right") if abs(by) > 0.1 else ("left" if d.top[1] > 0 else "right")
            plan = pl._plan_side_hints(d, side, b, cur_a, cur_h)
            if not plan.ok:
                other = "right" if side == "left" else "left"
                plan = pl._plan_side_hints(d, other, b, cur_a, cur_h)
            if not plan.ok:
                print(f"  seed {s}: no plan for {d.color} {getattr(d, 'shape', '?')}: {plan.reason}")
                continue
            ph = {name: (s, e) for name, s, e in plan.phases}
            snap = {"lift0": ph["lift"][0] + 2, "lift1": ph["lift"][1] - 1} if (a.wrist and "lift" in ph) else None
            before = _n_correct(sc) if a.wrist else 0
            _, why = run(env, plan.arm, plan.hand, snap=snap)
            cur_a, cur_h = plan.arm[-1].copy(), plan.hand[-1].copy()
            tried += 1
            if a.wrist and snap and "lift1" in snap and not isinstance(snap["lift1"], int):
                import json
                from .wristcheck import grip_features, fmt, head_features
                time.sleep(0.5)
                ok = _n_correct(sc) > before
                ps = getattr(plan, "side", None) or side
                sl = pl.kin.sl(ps)
                o1, o0 = snap["lift1"], snap.get("lift0")
                f = grip_features(per, d.color, np.asarray(o1["hand_q"])[sl], plan.hand[ph["close"][1] - 1][sl],
                                  o1.get(f"{ps}_wrist_rgb"),
                                  None if isinstance(o0, int) or o0 is None else o0.get(f"{ps}_wrist_rgb"))
                f.update(head_features(per, d.color, o1.get("head_rgb"), o1.get("head_depth"), d.top[:2]))
                print(f"  seed {s} {ps} {d.color} {getattr(d, 'shape', '?')}: {'HELD' if ok else 'MISS'}  {fmt(f)}")
                wout.write(json.dumps({"seed": s, "side": ps, "color": d.color, "shape": getattr(d, "shape", None),
                                       "ok": bool(ok), **{k: v for k, v in f.items()}}) + "\n")
                wout.flush()
            if why:
                print(f"  seed {s}: episode ended during {d.color} pick: {why}")
                break
        time.sleep(1.0)
        ev = sc.evaluate("sort_fixed", dict(FIXED_MAPPING))
        objs = {o["name"]: o for o in sc.state()["objects"] if o["active"]}
        line = []
        for o in ev["outcomes"]:
            shape = objs.get(o["name"], {}).get("shape", o["name"].split("_")[2])
            ok = o["outcome"] == "correct"
            by_shape.setdefault(shape, [0, 0])
            by_shape[shape][0] += ok
            by_shape[shape][1] += 1
            n_ok += ok
            n_tot += 1
            line.append(f"{o['name'].replace('obj_', '')}:{o['outcome']}")
        print(f"seed {s}: {tried} picks, " + ", ".join(line))
    env.close()
    print("BY SHAPE: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(by_shape.items())))
    print(f"TOTAL {n_ok}/{n_tot} one-shot  (overrides {ov or 'none'}, DR {'on' if dr else 'off'})")


if __name__ == "__main__":
    sys.exit(main())
