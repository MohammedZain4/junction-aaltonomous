"""Perception benchmark against sim ground truth (no robot motion). Run inside the sim container:

    python -m aaltonomous.fastsort.perception_bench --seeds 1 2 3 ... [--save /tmp/pbench]

For each seed: scene service reset_random(seed) + randomize(seed) (domain randomization, if the sim runs with AALTO_DR),
one head RGB-D frame, fastsort's ShapePerception.detect(). Detections are matched to the true active objects
(same colour, within --match m in xy). Reports hits / misses / false positives (with position, pixel count, height,
method) and shape accuracy. Ground truth is privileged: for diagnosis only, never used by the policy.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from aaltonomous.baselines.classical.config import load_config
from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene
from aaltonomous.sim.scene_client import SceneClient

from .grasp import ShapePerception
from .policy import FAST_OVERRIDES


def frame(cli, after, timeout=6.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = cli.get()
        if f is not None and f[2] > after:
            return f
        time.sleep(0.05)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--match", type=float, default=0.025)
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-dr", action="store_true", help="skip randomize() even if the sim has DR")
    a = ap.parse_args(argv)

    cfg = load_config(None, dict(FAST_OVERRIDES))
    scene, cal = Scene.load(), Calibration.load()
    per = ShapePerception(cfg, scene, cal)
    sc = SceneClient()
    dr = bool(sc.info().get("dr")) and not a.no_dr
    # same sequence as sim/evaluate.py, then fastsort's own start: arms to the park pose, settle, look
    from aaltonomous.baselines.classical.config import hand_postures, home_arm
    from aaltonomous.baselines.classical.planner import Planner
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.interface.safety import JointSpec
    from aaltonomous.sim.evaluate import home_arms
    spec = JointSpec()
    post, park = hand_postures(cfg, spec), home_arm(cfg, spec)
    planner = Planner(cfg, scene, post, park, spec=spec)
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=a.seeds[0], realtime=True, img_server_ip="127.0.0.1",
                   head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")), cameras=False)
    tot = dict(hit=0, miss=0, fp=0, shape_ok=0, shape_n=0)
    conf = []   # (true shape, detected shape, hull/rect ratio)
    for s in a.seeds:
        home_arms(env, sc)
        env.rng = np.random.default_rng(s)
        if dr:
            sc.call(cmd="randomize", seed=s)
        obs = env.reset()
        arm = np.clip(np.asarray(obs["arm_q"], float), spec.arm_lower, spec.arm_upper)
        hand = np.clip(np.asarray(obs["hand_q"], float), spec.hand_lower, spec.hand_upper)
        A, H = planner.go_to(arm, hand, park, post["open"])
        n = int(0.5 * 30)
        A = np.vstack([A, np.repeat(park[None], n, 0)]) if len(A) else np.repeat(arm[None], n, 0)
        H = np.vstack([H, np.repeat(post["open"][None], n, 0)]) if len(H) else np.repeat(hand[None], n, 0)
        for k in range(0, len(A), 15):
            obs, _ = env.step({"arm_q_target": A[k:k + 15], "hand_q_target": H[k:k + 15]})
        rgb, depth = obs["head_rgb"], obs["head_depth"]
        if rgb is None or not rgb.any():
            print(f"seed {s}: no frame")
            continue
        truth = [o for o in sc.state()["objects"] if o["active"]]
        dets = per.detect(rgb, depth)
        used, fps = set(), []
        for d in dets:
            best, bj = None, None
            for j, o in enumerate(truth):
                if j in used or o["color"] != d.color:
                    continue
                e = float(np.linalg.norm(np.asarray(o["pos"][:2]) - d.top[:2]))
                if e <= a.match and (best is None or e < best):
                    best, bj = e, j
            if bj is None:
                fps.append(d)
            else:
                used.add(bj)
                tot["shape_n"] += 1
                tot["shape_ok"] += int(getattr(d, "shape", None) == truth[bj]["shape"])
                conf.append((truth[bj]["shape"], getattr(d, "shape", None), getattr(d, "shape_ratio", None)))
        miss = [o for j, o in enumerate(truth) if j not in used]
        tot["hit"] += len(used); tot["miss"] += len(miss); tot["fp"] += len(fps)
        line = f"seed {s:3d}: {len(truth)} objects, {len(used)} hit, {len(miss)} miss, {len(fps)} false"
        for d in fps:
            h = (d.top[2] - scene.counter_z) * 100
            line += (f"\n     FP {d.color:<6} xy=({d.top[0]:+.3f},{d.top[1]:+.3f}) h={h:4.1f}cm px={d.n_px} "
                     f"depth_px={d.n_depth} {d.method} shape={getattr(d, 'shape', None)} uv={tuple(round(x) for x in d.uv)}")
        for o in miss:
            line += f"\n     MISS {o['color']:<6} {o['shape']:<8} xy=({o['pos'][0]:+.3f},{o['pos'][1]:+.3f})"
        print(line)
        if a.save and (fps or miss):
            import cv2
            os.makedirs(a.save, exist_ok=True)
            img = rgb[..., ::-1].copy()
            for d in dets:
                c = (0, 0, 255) if any(d is x for x in fps) else (0, 255, 0)
                cv2.circle(img, (int(d.uv[0]), int(d.uv[1])), 10, c, 2)
                cv2.putText(img, d.color[0], (int(d.uv[0]) + 12, int(d.uv[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
            cv2.imwrite(os.path.join(a.save, f"seed{s}.png"), img)
    env.close()
    for ts in ("cube", "cylinder", "triangle"):
        rows = [(d, r) for t_, d, r in conf if t_ == ts]
        if rows:
            rs = np.array([r for _, r in rows if r is not None])
            got = {k: sum(1 for d, _ in rows if d == k) for k in ("cube", "cylinder", "triangle", None)}
            print(f"  true {ts:<8} n={len(rows):2d} -> {got}  ratio min/med/max "
                  f"{rs.min():.2f}/{np.median(rs):.2f}/{rs.max():.2f}" if len(rs) else f"  true {ts}: no ratios")
    n = tot["hit"] + tot["miss"]
    print(f"TOTAL: {tot['hit']}/{n} hit ({100 * tot['hit'] / max(n, 1):.0f}%), {tot['miss']} miss, {tot['fp']} false, "
          f"shape {tot['shape_ok']}/{tot['shape_n']}  (DR {'on' if dr else 'off'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
