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


def run(env, A, H, chunk=15):
    """Execute open-loop. Returns (obs, end_reason or None)."""
    obs = None
    for k in range(0, len(A), chunk):
        obs, info = env.step({"arm_q_target": A[k:k + chunk], "hand_q_target": H[k:k + chunk]})
        if info.get("episode_over"):
            return obs, f"{info.get('end_reason')} (clipped {info.get('clipped_steps')}, {info.get('clip_reasons')})"
    return obs, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--closed", default=None, help="hand.closed posture, 7 values, LEFT order")
    ap.add_argument("--acc", type=float, default=None, help="motion.arm_max_acc")
    ap.add_argument("--close-hold", type=float, default=None, help="motion.close_hold_s")
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
                   head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")), cameras=False)
    by_shape, n_tot, n_ok = {}, 0, 0
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
            _, why = run(env, plan.arm, plan.hand)
            cur_a, cur_h = plan.arm[-1].copy(), plan.hand[-1].copy()
            tried += 1
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
