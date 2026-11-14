"""FastSortPolicy: both arms at once, no wasted trips, instruction-aware.

Built on the classical baseline's perception (HSV + depth), IK and single-arm pick planner, but scheduled
differently:

  * Instruction -> colour/shape -> bin rules with fastsort.instruction (offline, paraphrase-robust).
  * Two LANES, one per arm. Each lane has its own queue of joint targets; act() pops both every tick, so the arms
    work in parallel. The left arm serves the left bin, the right arm the right bin, the centre bin goes to the arm
    on the object's side (or whichever lane is free).
  * Chaining: every single-arm plan ends with a move home. When the next plan for that arm is ready before the home
    move has started, the home move is cut and the next pick starts straight from the retreat pose above the bin.
  * Look once, then keep looking only when an arm runs out of work (no home + settle + look between every object).
  * Arm-arm collisions: every queued tick stores the arm's collision points (FK: palm, fingers, wrist, elbow).
    A new plan is only committed with the smallest start delay for which its points stay >= `min_arm_dist` from
    the other arm's committed timeline (holding still meanwhile). First come, first served; never a deadlock
    because every lane ends at its home pose, which is far from the other arm.
  * Planning runs in one background thread (kinematics is not thread-safe; only that thread touches it); act()
    only pops queues, so it answers in microseconds and the command rate stays at 30 Hz.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from aaltonomous.baselines.classical.config import hand_postures, home_arm, load_config
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.baselines.classical.perception import Perception
from aaltonomous.baselines.classical.planner import Planner
from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

from .instruction import parse_instruction

HZ = 30.0
SIDES = ("left", "right")

# Speed-oriented overrides on top of the baseline's `sim` profile. Env AALTO_FAST=0 disables them.
FAST_OVERRIDES = {
    "motion.arm_max_vel": 0.97,     # safety filter: 1.0
    "motion.arm_max_acc": 10.0,
    "motion.hand_max_vel": 2.8,     # safety filter: 3.0
    "motion.close_hold_s": 0.2,
    "motion.open_hold_s": 0.25,
    "motion.settle_s": 0.2,
    "motion.chunk": 15,             # ticks per reply: every reply carries ~3.5 MB of images to decode (27 Hz gate)
    "motion.hold_ticks": 10,
    # Park/home pose: arms raised out to the sides (shoulder roll 1.8). FK: hand points >= 31.8 cm from the centre
    # line, >= 27 cm above the pelvis, 64 cm apart, none in the head camera view. The sim's ARM_HOME parks both hands
    # above the middle of the object band (blocks the other arm, hides objects); the robot-side home (roll 0.7,
    # 21.6 cm out) still blocked cross-body reaches to the far edge (6.6 cm finger-to-finger).
    "home_arm_q": [0.0, 1.8, 0.0, 0.2, 0.0, 0.0, 0.0,
                   0.0, -1.8, 0.0, 0.2, 0.0, 0.0, 0.0],
}


def _lower_thread_priority(nice=10):
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice)
    except (AttributeError, OSError):
        pass


@dataclass(eq=False)  # identity comparison: field-wise __eq__ compares numpy arrays and raises
class Obj:
    det: object
    bin: str
    pref: str                       # lane that should do it
    tried: set = field(default_factory=set)
    reloc: bool = False
    not_before: int = 0             # absolute tick: don't retry before (waiting for the other arm)

    @property
    def xy(self):
        return np.asarray(self.det.top[:2], float)


@dataclass
class LanePlan:
    obj: Obj
    start: int                      # absolute tick
    end: int
    grasp_xy: np.ndarray


class Lane:
    def __init__(self, side):
        self.side = side
        self.sl = ArmKinematics.sl(side)
        self.qa = np.empty((0, 7))
        self.qh = np.empty((0, 7))
        self.pts = None             # (N, P, 3) collision points per queued tick
        self.cmd_a = self.cmd_h = None
        self.cmd_pts = None         # collision points of the last popped tick (= hold pose)
        self.chain_abs = None       # absolute tick where the final home move starts (cut point), or None
        self.chain_a = self.chain_h = self.chain_pts = None
        self.plans: list[LanePlan] = []
        self.idle_since = 0         # absolute tick at which the queue last ran empty
        self.version = 0            # +1 on every commit (the other lane re-validates its delay search against it)

    def tail(self):
        if len(self.qa):
            return self.qa[-1], self.qh[-1], self.pts[-1]
        return self.cmd_a, self.cmd_h, self.cmd_pts

    def pending(self, now):
        return [p for p in self.plans if p.end > now]


class FastSortPolicy:
    def __init__(self, config=None, scene=None, calibration=None, time_limit=120.0, log=print, overrides=None,
                 min_arm_dist=0.075):  # between link origins (palm, finger links, wrist, elbow), not surfaces
        ov = dict(FAST_OVERRIDES) if os.environ.get("AALTO_FAST", "1") != "0" else {}
        ov.update(overrides or {})
        self.cfg = config if isinstance(config, dict) else load_config(config, ov)
        self.scene = scene if isinstance(scene, Scene) else Scene.load(scene)
        self.cal = calibration if isinstance(calibration, Calibration) else Calibration.load(calibration)
        self.spec = JointSpec()
        self.post = hand_postures(self.cfg, self.spec)
        self.home = home_arm(self.cfg, self.spec)
        self.kin = ArmKinematics(self.cfg, self.spec)
        self.safety = SafetyConfig(dt=1.0 / HZ)
        self.perception = Perception(self.cfg, self.scene, self.cal)
        self.planner = Planner(self.cfg, self.scene, self.post, self.home, self.kin, self.safety, self.spec)
        self.time_limit = float(time_limit)
        self.min_arm_dist = float(min_arm_dist)
        self.log = log or (lambda *a: None)
        self._extra_frames = {}
        for side in SIDES:
            ids = []
            for name in (f"{side}_elbow_link", f"{side}_wrist_roll_link", f"{side}_wrist_yaw_link"):
                try:
                    fid = self.kin.model.getFrameId(name)
                    if fid < self.kin.model.nframes:
                        ids.append(fid)
                except Exception:
                    pass
            self._extra_frames[side] = ids
        sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))
        self._brain_thread = None
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, info=None):
        self.stop_brain()
        if isinstance(info, str):
            info = {"instruction": info}
        info = info or {}
        self.rules = parse_instruction(info.get("instruction", ""))
        self.mapping = self.rules.mapping
        self.planner.mapping = {c: b for c, b in self.mapping.items() if b is not None}
        self.time_limit = float(info.get("time_limit", self.time_limit))
        self.lock = threading.Lock()
        self.lanes = {s: Lane(s) for s in SIDES}
        self.tick = 0
        self.started = False
        self.finished = False
        self.latest_obs = None
        self.latest_obs_tick = -1
        self.pool: list[Obj] | None = None
        self.last_look_tick = -10_000
        self.last_look_obs_tick = -1
        self.attempts = []          # grasp xy of every committed pick
        self.failed = []            # xy that could not be planned
        self.events = []
        self._halt = False
        self._note(0.0, f"instruction -> {self.rules.summary()}  [{self.rules.instruction!r}]")

    def stop_brain(self):
        self._halt = True
        th = self._brain_thread
        if th is not None and th.is_alive():
            th.join(timeout=30.0)
        self._brain_thread = None

    def act(self, obs):
        if not self.started:
            self._start(obs)
        with self.lock:
            self.latest_obs, self.latest_obs_tick = obs, self.tick
            if self.finished:
                return self._pack(1, done=True)
            busy = any(len(l.qa) for l in self.lanes.values())
            k = int(self.cfg["motion"]["chunk"]) if busy else max(1, int(self.cfg["motion"].get("hold_ticks", 1)))
            return self._pack(k)

    # ------------------------------------------------------------------ act-thread helpers (hold self.lock)
    def _pack(self, k, done=False):
        A, H = np.empty((k, 14)), np.empty((k, 14))
        for lane in self.lanes.values():
            n = min(k, len(lane.qa))
            if n:
                A[:n, lane.sl], H[:n, lane.sl] = lane.qa[:n], lane.qh[:n]
                lane.cmd_a, lane.cmd_h, lane.cmd_pts = lane.qa[n - 1].copy(), lane.qh[n - 1].copy(), lane.pts[n - 1]
                lane.qa, lane.qh, lane.pts = lane.qa[n:], lane.qh[n:], lane.pts[n:]
                if not len(lane.qa):
                    lane.idle_since = self.tick + n
            A[n:, lane.sl], H[n:, lane.sl] = lane.cmd_a, lane.cmd_h
        if not done:
            self.tick += k
        return {"arm_q_target": A, "hand_q_target": H, "done": bool(done)}

    def _start(self, obs):
        sp = self.spec
        arm = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
        hand = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
        a, h = self.planner.go_to(arm, hand, self.home, self.post["open"])
        for lane in self.lanes.values():
            lane.cmd_a, lane.cmd_h = arm[lane.sl].copy(), hand[lane.sl].copy()
            lane.cmd_pts = self._points(lane.side, arm, hand)
            n_settle = int(round(self.cfg["motion"]["settle_s"] * HZ))
            qa = np.vstack([a[:, lane.sl], np.repeat(self.home[lane.sl][None], n_settle, 0)]) if len(a) else \
                np.repeat(lane.cmd_a[None], n_settle, 0)
            qh = np.vstack([h[:, lane.sl], np.repeat(self.post["open"][lane.sl][None], n_settle, 0)]) if len(h) else \
                np.repeat(lane.cmd_h[None], n_settle, 0)
            lane.qa, lane.qh = qa, qh
            # The first act() must answer fast (the watchdog stops the episode after 1 s without a target): FK on a
            # handful of samples of the initial move, repeated, instead of every tick.
            ks = np.unique(np.linspace(0, len(qa) - 1, min(len(qa), 6)).round().astype(int))
            sp_pts = self._lane_points(lane.side, qa[ks], qh[ks])
            nearest = np.abs(np.arange(len(qa))[:, None] - ks[None]).argmin(axis=1)
            lane.pts = sp_pts[nearest]
        self.started = True
        self._halt = False
        self._brain_thread = threading.Thread(target=self._brain, daemon=True)
        self._brain_thread.start()

    # ------------------------------------------------------------------ kinematics helpers (brain thread / init only)
    def _points(self, side, arm14, hand14):
        p = np.delete(self.kin.hand_points(side, arm14, hand14), 1, axis=0)  # drop the virtual grasp point
        extra = [self.kin.data.oMf[f].translation.copy() for f in self._extra_frames[side]]
        return np.vstack([p] + ([np.array(extra)] if extra else []))

    def _lane_points(self, side, qa7, qh7):
        sl = ArmKinematics.sl(side)
        arm, hand = self.home.copy(), self.post["open"].copy()
        out = []
        for a, h in zip(qa7, qh7):
            arm[sl], hand[sl] = a, h
            out.append(self._points(side, arm, hand))
        return np.array(out) if out else np.empty((0, 0, 3))

    # ------------------------------------------------------------------ brain
    def _brain(self):
        _lower_thread_priority()
        errors = 0
        while not self._halt:
            try:
                worked = self._brain_step()
                errors = 0
            except Exception:
                # Never end the episode over a bug: log (once per streak), back off, keep trying. The arms just hold.
                errors += 1
                if errors == 1 or errors % 100 == 0:
                    self.log(f"[fastsort] brain error (x{errors}): " + traceback.format_exc())
                time.sleep(0.05)
                worked = True
            if not worked:
                time.sleep(0.005)

    def _t(self):
        return self.tick / HZ

    def _brain_step(self):
        with self.lock:
            now = self.tick
            obs, obs_tick = self.latest_obs, self.latest_obs_tick
            idle = {s: (not len(l.qa)) for s, l in self.lanes.items()}
            settled = {s: idle[s] and obs_tick >= l.idle_since + int(round(self.cfg["motion"]["settle_s"] * HZ))
                       for s, l in self.lanes.items()}
        if self.finished or obs is None:
            return False

        # ---------------- look
        want_look = False
        if self.pool is None:
            want_look = all(settled.values())
        else:
            for s in SIDES:
                if settled[s] and not self._candidates(s) and not self.lanes[s].pending(now):
                    want_look = True
        if want_look and obs_tick > self.last_look_obs_tick and now - self.last_look_tick >= 15:
            self._look(obs, now)
            self.last_look_tick, self.last_look_obs_tick = now, obs_tick
            if all(settled.values()) and not self.pool and not any(l.pending(now) for l in self.lanes.values()):
                self._note(self._t(), "nothing left to sort: done")
                with self.lock:
                    self.finished = True
            return True
        if self.pool is None:
            return False

        # ---------------- plan (the lane with less queued work first)
        order = sorted(SIDES, key=lambda s: len(self.lanes[s].qa))
        for s in order:
            lane = self.lanes[s]
            if len(lane.pending(now)) >= 2:  # one executing + one look-ahead is enough
                continue
            obj = self._choose(s)
            if obj is None:
                continue
            self.pool.remove(obj)
            ok = self._plan_and_commit(lane, obj)
            if not ok and obj not in self.pool and not self._is_failed(obj.xy):
                self.pool.append(obj)
            return True
        return False

    def _is_failed(self, xy):
        return any(np.linalg.norm(xy - f) < self.cfg["policy"]["same_object_radius"] for f in self.failed)

    def _look(self, obs, now):
        dets = self.perception.detect(obs["head_rgb"], obs["head_depth"])
        r = self.cfg["policy"]["same_object_radius"]
        with self.lock:
            inflight = [(p.grasp_xy, l.side) for l in self.lanes.values() for p in l.pending(now)]
            busy_hands = []
            for l in self.lanes.values():
                if len(l.qa) and l.cmd_pts is not None:
                    busy_hands.append(l.cmd_pts[0, :2])  # palm xy of a moving arm
        pool = []
        for d in dets:
            xy = np.asarray(d.top[:2], float)
            b = self.rules.bin_for(d.color, getattr(d, "shape", None))
            if b is None:
                continue
            if self._is_failed(xy):
                continue
            if sum(1 for a in self.attempts if np.linalg.norm(xy - a) < r) >= self.cfg["policy"]["max_attempts"]:
                continue
            if any(np.linalg.norm(xy - g) < 0.06 for g, _ in inflight):
                continue
            if any(np.linalg.norm(xy - h) < 0.08 for h in busy_hands):
                continue
            by = self.planner.bin_xy(b)[1]
            pref = ("left" if by > 0 else "right") if abs(by) > 0.1 else ("left" if xy[1] > 0 else "right")
            o = Obj(det=d, bin=b, pref=pref)
            # same object as in the previous look: keep what we learned (which arm failed, relocation, back-off)
            old = min((p for p in (self.pool or []) if np.linalg.norm(p.xy - xy) < r and p.bin == b),
                      key=lambda p: np.linalg.norm(p.xy - xy), default=None)
            if old is not None:
                o.pref, o.tried, o.reloc, o.not_before = old.pref, set(old.tried), old.reloc, old.not_before
            pool.append(o)
        self.pool = pool
        self._note(self._t(), f"look: {len(dets)} detections, {len(pool)} to sort: "
                              + ", ".join(f"{o.det.color}->{o.bin}({o.pref[0]})" for o in pool))

    def _candidates(self, side):
        if not self.pool:
            return []
        ready = [o for o in self.pool if o.not_before <= self.tick]
        mine = [o for o in ready if o.pref == side and side not in o.tried]
        if mine:
            return mine
        # steal: centre-bin objects, or objects the other arm could not do
        return [o for o in ready if side not in o.tried and (o.bin == "center" or o.tried)]

    def _choose(self, side):
        c = self._candidates(side)
        if not c:
            return None
        # objects far from the centre line first (keeps the arms apart while both work)
        return max(c, key=lambda o: abs(o.xy[1]) - 0.5 * np.linalg.norm(o.xy - self.planner.drop_xy(o.bin)))

    def _plan_and_commit(self, lane, obj):
        side, sl = lane.side, lane.sl
        with self.lock:
            now = self.tick
            if lane.chain_abs is not None and lane.chain_abs - now > 3:
                use_chain = True
                sa, sh = lane.chain_a.copy(), lane.chain_h.copy()
            else:
                use_chain = False
                ta, th, _ = lane.tail()
                sa, sh = ta.copy(), th.copy()
        arm_q, hand_q = self.home.copy(), self.post["open"].copy()
        arm_q[sl], hand_q[sl] = sa, sh

        t0 = time.monotonic()
        if not obj.reloc:
            plan = self.planner._plan_side_hints(obj.det, side, obj.bin, arm_q, hand_q)
            if not plan.ok:
                self._side_failed(obj, side, f"no plan: {plan.reason}")
                return False
        else:
            obstacles = [o.xy for o in self.pool if o is not obj]
            plan = self.planner.plan_relocation(obj.det, obj.bin, arm_q, hand_q, obstacles)
            if plan.ok and plan.side != side:  # the relocation needs the other arm: hand it over
                obj.pref = plan.side
                return False
            if not plan.ok:
                self._side_failed(obj, side, f"no relocation: {plan.reason}")
                return False
        dt_plan = time.monotonic() - t0

        A, H = plan.arm[:, sl], plan.hand[:, sl]
        home_start = next((s for name, s, e in plan.phases if name == "home"), None)
        P = self._lane_points(side, A, H)
        hold_pts = self._points(side, arm_q, hand_q)

        other = self.lanes["right" if side == "left" else "left"]
        for _attempt in range(5):
            # snapshot under the lock, search the delay WITHOUT it (act() must never wait on this), re-validate
            with self.lock:
                now0 = self.tick
                if use_chain:
                    if lane.chain_abs is None or lane.chain_abs - now0 < 1:
                        return False  # the home move already started: replan from the new tail
                    cut0 = lane.chain_abs - now0
                else:
                    ta, th, _ = lane.tail()
                    if not (np.allclose(ta, sa) and np.allclose(th, sh)):
                        return False
                    cut0 = len(lane.qa)
                O = other.pts if len(other.qa) else np.empty((0,) + other.cmd_pts.shape)
                O_last = other.pts[-1] if len(other.qa) else other.cmd_pts
                other_ver, other_parked = other.version, not len(other.qa)
            d, closest = self._find_delay(cut0, hold_pts, P, O, O_last)
            self.lock.acquire()
            now = self.tick
            cut = cut0 - (now - now0)  # same absolute start tick as searched
            if other.version == other_ver and (cut >= 1 or (cut == 0 and now == now0)) and \
                    (use_chain or cut == len(lane.qa)):
                break
            self.lock.release()  # the other arm committed meanwhile or our start slipped into the past: redo
        else:
            return False
        try:
            if d is None:
                why = (f"{side} {self.PT_NAMES[closest[1]]} vs {other.side} {self.PT_NAMES[closest[2]]} "
                       f"{100 * closest[0]:.1f} cm")
                if other_parked:
                    # the other arm is not moving: waiting cannot help -> this arm can't do it, like a planning failure
                    self._side_failed(obj, side, f"collides with the parked {other.side} arm ({why})")
                else:
                    obj.not_before = now + 15
                    self._note(self._t(), f"{side}: {obj.det.color} would collide with the other arm ({why}): later")
                return False
            t_end = (now + cut + d + len(A)) / HZ
            if t_end > self.time_limit - 0.5:
                self.failed.append(obj.xy)
                self._note(self._t(), f"{side}: no time for {obj.det.color} (ends {t_end:.1f} s)")
                return False
            hold_a, hold_h = np.repeat(sa[None], d, 0), np.repeat(sh[None], d, 0)
            lane.qa = np.vstack([lane.qa[:cut], hold_a, A])
            lane.qh = np.vstack([lane.qh[:cut], hold_h, H])
            lane.pts = np.concatenate([lane.pts[:cut], np.repeat(hold_pts[None], d, 0), P]) if cut or d else P
            start_abs = now + cut + d
            if home_start is not None and home_start > 0:
                lane.chain_abs = start_abs + home_start
                lane.chain_a, lane.chain_h = A[home_start - 1].copy(), H[home_start - 1].copy()
            else:
                lane.chain_abs = None
            # the plan whose home move was cut now ends at the cut point
            for p in lane.plans:
                if p.start < now + cut < p.end:
                    p.end = now + cut
            gxy = np.asarray(plan.waypoints.get("grasp", np.r_[obj.xy, 0.0])[:2], float)
            lane.plans.append(LanePlan(obj, start_abs, now + cut + d + len(A), gxy))
            lane.version += 1
            self.attempts.append(obj.xy)
        finally:
            self.lock.release()
        what = "relocate" if plan.target_bin == "counter" else f"-> {obj.bin}"
        self._note(self._t(), f"{side}: {obj.det.color} {what}  start +{(cut + d) / HZ:.1f}s (delay {d / HZ:.1f}s, "
                              f"{'chained' if use_chain else 'from tail'}), {len(A) / HZ:.1f}s, planned in {dt_plan:.2f}s")
        return True

    def _find_delay(self, cut, hold_pts, P, O, O_last, max_delay=240, step=3):
        """Smallest d (ticks) such that [hold d ticks] + P, starting `cut` ticks from now, keeps every point >=
        min_arm_dist from the other arm's timeline O (then its final pose O_last). None if no d <= max_delay."""
        """Returns (d or None, closest) with closest = (distance, my point index, their point index) at d = 0."""
        dmin = self.min_arm_dist
        n_o = len(O)
        closest = None
        for d in range(0, max_delay + 1, step):
            L = d + len(P)
            idx = cut + np.arange(0, L, 2)
            mine = np.where((np.arange(0, L, 2) < d)[:, None, None], hold_pts[None],
                            P[np.clip(np.arange(0, L, 2) - d, 0, len(P) - 1)])
            theirs = np.where((idx < n_o)[:, None, None], O[np.clip(idx, 0, max(n_o - 1, 0))] if n_o else O_last[None],
                              O_last[None])
            dist = np.linalg.norm(mine[:, :, None, :] - theirs[:, None, :, :], axis=-1)
            if closest is None:
                _, i, j = np.unravel_index(dist.argmin(), dist.shape)
                closest = (float(dist.min()), int(i), int(j))
            if dist.min() >= dmin:
                return d, closest
            if idx[0] >= n_o:
                # the other arm is parked for the whole window: waiting longer cannot help
                return None, closest
        return None, closest

    # collision point names (order of _points): hand_points minus the virtual grasp point, then extra frames
    PT_NAMES = ("palm", "finger1", "finger2", "finger3", "tip1", "tip2", "tip3", "elbow", "wrist_roll", "wrist_yaw")

    def _side_failed(self, obj, side, reason):
        """This arm cannot do `obj`: let the other arm try, then fall back to a relocation, then give up."""
        obj.tried.add(side)
        other = "right" if side == "left" else "left"
        if not obj.reloc and other not in obj.tried:
            obj.pref = other
        elif not obj.reloc:
            obj.reloc, obj.tried = True, set()
            obj.pref = "left" if obj.xy[1] > 0 else "right"
        elif len(obj.tried) >= 2:
            self.failed.append(obj.xy)
            self._note(self._t(), f"give up on {obj.det.color} at {obj.xy.round(3)}: {reason}")
            return
        self._note(self._t(), f"{side}: can't do {obj.det.color}->{obj.bin}: {reason}")

    def _note(self, t, msg):
        self.events.append((round(t, 2), msg))
        self.log(f"[fastsort t={t:6.2f}] {msg}")
