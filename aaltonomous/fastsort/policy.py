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

import collections
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from aaltonomous.baselines.classical.config import hand_postures, home_arm, load_config
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from .calib import estimate_xy_offset, first_look_calibration, set_kin_world  # noqa: F401
from .grasp import ShapePerception, ShapePlanner

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

from .instruction import parse_instruction
from .wristcheck import fmt as grip_fmt, grip_features, head_features

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
    # Colour ranges valid in BOTH the sim and the real head camera (OpenCV HSV, H 0-180). Measured on the real
    # robot's Drop-0 video (2026-11-14): "red" objects read orange H 14 (S 150-180, V 250), yellow H 43-44
    # (S 72-88, V 190-250), blue H 111; bin tape H 4-5 (S 190, V 105-115), tape on the table H 4. The baseline's
    # sim ranges (red 0-8, yellow 18-35) would miss every real red and yellow object. Bins now fall in "red" in both
    # worlds and are removed geometrically (bin hull mask, footprint, bin-line rule); flat tape fails the height check.
    "grasp.relocate.max_spots": 3,   # each spot = a full finishing plan + a pick plan: 6 spots took 9-27 s under load
    "perception.hsv": {
        "red":    [[[0, 70, 70], [20, 255, 255]], [[170, 70, 70], [180, 255, 255]]],
        "yellow": [[[21, 45, 90], [55, 255, 255]]],
        "blue":   [[[104, 60, 90], [145, 255, 255]]],   # real: purple H 110-112; silver arm H 91-101, black fingers V < 60
    },
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


@dataclass(eq=False)  # identity: list.remove() on plans must not compare numpy fields
class LanePlan:
    obj: Obj
    start: int                      # absolute tick
    end: int
    grasp_xy: np.ndarray
    phases: dict = field(default_factory=dict)   # name -> (absolute start tick, absolute end tick)
    hand_closed: np.ndarray | None = None        # commanded 7 hand joints at the end of "close"
    checked: bool = False                        # grip check done
    plan_a: np.ndarray | None = None             # this lane's (N, 7) arm / hand trajectory, plan tick 0 = start
    plan_h: np.ndarray | None = None


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
    max_attempts_total = 3          # direct grasps of one object (any arm) before giving up on it
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
        self.perception = ShapePerception(self.cfg, self.scene, self.cal)
        self._xy_base = np.asarray(self.perception.xy_offset, float).copy()
        self._T_head0 = np.asarray(self.cal._T(self.perception.cam), float).copy()
        self.planner = ShapePlanner(self.cfg, self.scene, self.post, self.home, self.kin, self.safety, self.spec)
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
        self.obs_hist = collections.deque(maxlen=8)   # (tick, obs) of the last act() calls (~0.5 s apart)
        self.color_fix = []                            # (xy, colour) corrected by the wrist-camera colour check
        self.pushed = []                               # xy of standing prisms already pushed over
        self.shape_votes = []                          # per-object shape votes across looks (_vote_shape)
        # act (default): abort on the head-camera cue OR the finger/wrist verdict; head: head cue only;
        # log: never act; 0: off
        self.grip_mode = os.environ.get("AALTO_GRIP_CHECK", "act")
        self.pool: list[Obj] | None = None
        self.last_look_tick = -10_000
        self.last_look_obs_tick = -1
        self.attempts = []          # (xy, side, colour, is_relocation) of every committed pick
        self.bin_drops = {}         # bin -> releases committed so far (next release goes to the next slot)
        self.failed = []            # xy that could not be planned
        self.events = []
        self._halt = False
        # AALTO_RT_EMULATE=<rtf> (testing only): charge the brain's wall time (looks, planning incl. failed plans) to
        # the sim clock as if the sim ran at that real-time factor (1.0 = real robot). The scored 120 s run on the sim
        # clock, and a sim at RTF 0.15 otherwise hides ~85% of planning time. New plans never start before vtick.
        self.rt_emulate = float(os.environ.get("AALTO_RT_EMULATE", "0") or 0)
        self.vtick = 0.0
        self._note(0.0, f"instruction ->{self.rules.summary()}  [{self.rules.instruction!r}]")

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
            self.obs_hist.append((self.tick, obs))
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
        # an idle arm with nothing it may take re-looked every 0.5 s (40 looks of ~3 s CPU in one DR episode, all
        # charged in real time): with objects still in the pool (owned by the busy arm) look at most once per 1.5 s
        min_gap = 45 if (self.pool or not all(idle.values())) else 15
        if want_look and obs_tick > self.last_look_obs_tick and now - self.last_look_tick >= min_gap:
            self._look(obs, now)
            self.last_look_tick, self.last_look_obs_tick = now, obs_tick
            if all(settled.values()) and not self.pool and not any(l.pending(now) for l in self.lanes.values()):
                self._note(self._t(), "nothing left to sort: done")
                with self.lock:
                    self.finished = True
            return True
        if self.pool is None:
            return False

        # ---------------- grip check at the end of each lift (log only until the thresholds are fitted)
        if self.grip_mode != "0" and self._grip_check(obs, obs_tick):
            return True

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

    def _grip_check(self, obs, obs_tick):
        """Once per pick, on the first observation at/after the end of its lift: measure finger tracking error and the
        wrist-camera colour blob (fastsort.wristcheck), compared with an observation early in the lift."""
        did = False
        for side, lane in self.lanes.items():
            for p in list(lane.plans):
                if p.checked or "lift" not in p.phases or p.hand_closed is None:
                    continue
                ls, le = p.phases["lift"]
                if obs_tick < le + 4:    # a few ticks past the lift end: camera frames lag the joints
                    continue
                p.checked = True
                if obs_tick > le + 30:   # observation came too late (missed chunk): skip
                    continue
                with self.lock:
                    prev = next((o for t, o in reversed(self.obs_hist) if ls - 10 <= t <= le - 12), None)
                    at_ls = next((o for t, o in reversed(self.obs_hist) if t <= ls + 1), None)
                try:
                    f = grip_features(self.perception, p.obj.det.color, np.asarray(obs["hand_q"], float)[lane.sl],
                                      p.hand_closed, obs.get(f"{side}_wrist_rgb"),
                                      None if prev is None else prev.get(f"{side}_wrist_rgb"))
                    gxy = p.grasp_xy if getattr(p, "grasp_xy", None) is not None else p.obj.xy
                    f.update(head_features(self.perception, p.obj.det.color, obs.get("head_rgb"),
                                           obs.get("head_depth"), gxy))
                    stale = False
                    try:   # head frame unchanged since the lift started = stale image, no head verdict
                        hd0, hd1 = at_ls.get("head_depth"), obs.get("head_depth")
                        stale = hd0 is not None and hd1 is not None and np.array_equal(
                            np.asarray(hd0)[::8, ::8], np.asarray(hd1)[::8, ::8])
                    except Exception:
                        pass
                    if stale:
                        f["head_table_px"] = None
                        f["head_stale"] = True
                    head_miss = (f.get("head_table_px") or 0) >= self.grip_head_px
                    v = "miss" if head_miss else self._grip_verdict(f)
                    self._note(self._t(), f"{side}: grip check {p.obj.det.color} "
                                          f"{getattr(p.obj.det, 'shape', None) or '?'} at {p.obj.xy.round(3)}: "
                                          f"{v.upper()}  {grip_fmt(f)}")
                    if v == "miss" and (self.grip_mode == "act" or (self.grip_mode == "head" and head_miss)):
                        self._abort_pick(lane, p)
                    elif v == "held" and p.plan_a is not None:
                        oc = self._colour_mismatch(f)
                        if oc is not None:
                            self._note(self._t(), f"{side}: COLOUR CHECK: held object looks {oc}, not "
                                                  f"{p.obj.det.color}")
                            if self.grip_mode == "act":
                                self._put_back(lane, p, oc)
                except Exception as e:
                    self._note(self._t(), f"{side}: grip check failed: {e!r}")
                did = True
        return did

    # Grip verdict thresholds. Conservative by design: abort only when BOTH cues say empty (fingers closed fully AND
    # no object blob that moves with the wrist camera); a wrong abort costs a whole pick, a missed one the trip.
    grip_j_empty = 0.015    # rad: max finger tracking error below this = fingers reached the closed posture.
                            # grasp_bench --wrist, 12 labelled sim picks: every miss 0.00, every hold 0.02-0.05. On
                            # the real hand an empty grip likely reads > 0 -> "held" -> no action (fails safe)
    grip_frac_min = 0.03    # wrist blob of the object's colour, fraction of the image
    grip_iou_min = 0.5      # ... that stays put between early and late lift (moves with the camera)
    grip_head_px = 60       # head pixels of the object's colour still at table height within 3.5 cm of the grasp xy
                            # (a 4 cm object top is ~600-900 px from the head camera; the closing hand hides part)

    def _grip_verdict(self, f):
        j = f.get("jerr")
        if j is None:
            return "unsure"
        wrist_ok = bool(f.get("wrist_valid"))
        seen = f.get("frac", 0.0) >= self.grip_frac_min and (f.get("iou") is None or f["iou"] >= self.grip_iou_min)
        if j >= self.grip_j_empty or seen:
            return "held"
        if not wrist_ok:   # no usable wrist frame: joints alone, stricter
            return "miss" if j < 0.75 * self.grip_j_empty else "unsure"
        return "miss"

    def _colour_mismatch(self, f):
        """Another colour whose wrist blob is clearly the held object (big, stays put, dwarfs the expected colour's)."""
        best = None
        for oc, (fr, iou) in (f.get("others") or {}).items():
            if fr >= max(self.grip_frac_min * 2, 2.0 * f.get("frac", 0.0)) and iou is not None and iou >= 0.7:
                if best is None or fr > best[1]:
                    best = (oc, fr)
        return None if best is None else best[0]

    def _put_back(self, lane, p, new_color):
        """Held object is the wrong colour: lower it back along the (already validated) lift path, open, come back up
        the same way, then home (chainable). The next look sorts it with the corrected colour (color_fix)."""
        side, sl = lane.side, lane.sl
        if "close" not in p.phases or "lift" not in p.phases:
            return
        with self.lock:
            now = self.tick
            hold_pts, ca, ch = lane.cmd_pts, lane.cmd_a.copy(), lane.cmd_h.copy()
            later = [q for q in lane.plans if q.start > p.start]
        A0, H0 = p.plan_a, p.plan_h
        i_now = now - p.start                       # plan index of the next unsent tick
        cs, ce = (t - p.start for t in p.phases["close"])
        ls, le = (t - p.start for t in p.phases["lift"])
        if not (le <= i_now <= len(A0)) or i_now - ls > 3 * (le - ls) + 30:
            return                                  # too far into the carry: let it go (wrong bin, logged)
        down_a, down_h = A0[ls:i_now][::-1], H0[ls:i_now][::-1]
        open_h = H0[cs:ce][::-1]
        open_a = np.repeat(A0[ls][None], len(open_h), 0)
        h_open = H0[cs]
        up_a, up_h = A0[ls:le], np.repeat(h_open[None], le - ls, 0)
        arm, hand = self.home.copy(), self.post["open"].copy()
        arm[sl], hand[sl] = A0[le - 1], h_open
        tgt_a, tgt_h = arm.copy(), hand.copy()
        tgt_a[sl], tgt_h[sl] = self.home[sl], self.post["open"][sl]
        ha, hh = self.planner.go_to(arm, hand, tgt_a, tgt_h)
        mo = self.cfg["motion"]
        for k in range(0, len(ha), 3):
            if self.planner.clearance(side, ha[k], hh[k]) < mo["counter_clearance"] - 2e-3 or \
                    self.planner.bin_clearance(side, ha[k], hh[k]) < 0.003:
                self._note(self._t(), f"{side}: put-back not safe: finishing the pick")
                return
        A = np.vstack([down_a, open_a, up_a, ha[:, sl]])
        H = np.vstack([down_h, open_h, up_h, hh[:, sl]])
        full_a, full_h = np.repeat(self.home[None], len(A), 0), np.repeat(self.post["open"][None], len(A), 0)
        full_a[:, sl], full_h[:, sl] = A, H
        s_arm, s_hand = self.home.copy(), self.post["open"].copy()
        s_arm[sl], s_hand[sl] = ca, ch
        if self.planner.dry_run_safety(s_arm, s_hand, full_a, full_h):
            self._note(self._t(), f"{side}: put-back would clip: finishing the pick")
            return
        P = self._lane_points(side, A, H)
        other = self.lanes["right" if side == "left" else "left"]
        with self.lock:
            if self.tick != now or lane.cmd_pts is not hold_pts:
                return
            O = other.pts if len(other.qa) else np.empty((0,) + other.cmd_pts.shape)
            O_last = other.pts[-1] if len(other.qa) else other.cmd_pts
            d, _ = self._find_delay(0, hold_pts, P, O, O_last, max_delay=max(240, len(O) + 30))
            if d is None:
                self._note(self._t(), f"{side}: put-back would collide with the other arm: finishing the pick")
                return
            n_home = len(ha)
            lane.qa = np.vstack([np.repeat(ca[None], d, 0), A])
            lane.qh = np.vstack([np.repeat(ch[None], d, 0), H])
            lane.pts = np.concatenate([np.repeat(hold_pts[None], d, 0), P]) if d else P
            lane.chain_abs = now + d + len(A) - n_home
            lane.chain_a, lane.chain_h = A[len(A) - n_home - 1].copy(), H[len(A) - n_home - 1].copy()
            p.end = now + d + len(A)
            for q in later:
                lane.plans.remove(q)
            lane.version += 1
            self.color_fix.append((p.obj.xy.copy(), new_color))
        self._note(self._t(), f"{side}: put back {p.obj.det.color}->{new_color} object (delay {d / HZ:.1f}s)")

    def _abort_pick(self, lane, p):
        # planning the abort takes a few ms to ~0.1 s; a chunk going out meanwhile moves the arm, so replan from the
        # new commanded pose (was: give up silently, which happened to 3 of 9 MISS verdicts in one eval)
        for _ in range(4):
            if self._abort_pick_once(lane, p) != "retry":
                return
        self._note(self._t(), f"{lane.side}: abort raced the command stream 4x: finishing the pick")

    def _abort_pick_once(self, lane, p):
        """Empty hand after the lift: drop the rest of this pick (carry, open, home) and the look-ahead plan queued
        behind it, open the hand and head home from where it is. The new home move is chainable, so the next pick
        can start straight from here. Collision-checked against the other arm and the counter/bin rims; if it can't
        be done safely the original plan simply continues (it's only a wasted trip)."""
        side, sl = lane.side, lane.sl
        with self.lock:
            now = self.tick
            ca, ch = lane.cmd_a.copy(), lane.cmd_h.copy()
            hold_pts = lane.cmd_pts
            later = [q for q in lane.plans if q.start > p.start]
            if p.end <= now + 2:
                return
        arm, hand = self.home.copy(), self.post["open"].copy()
        arm[sl], hand[sl] = ca, ch
        tgt_a, tgt_h = arm.copy(), hand.copy()
        tgt_a[sl], tgt_h[sl] = self.home[sl], self.post["open"][sl]
        A, H = self.planner.go_to(arm, hand, tgt_a, tgt_h)
        mo = self.cfg["motion"]
        for k in range(0, len(A), 3):
            if self.planner.clearance(side, A[k], H[k]) < mo["counter_clearance"] - 2e-3 or \
                    self.planner.bin_clearance(side, A[k], H[k]) < 0.003:
                self._note(self._t(), f"{side}: abort not safe (counter/bin clearance): finishing the pick")
                return
        if self.planner.dry_run_safety(arm, hand, A, H):
            self._note(self._t(), f"{side}: abort would clip: finishing the pick")
            return
        A, H = A[:, sl], H[:, sl]
        P = self._lane_points(side, A, H)
        other = self.lanes["right" if side == "left" else "left"]
        with self.lock:
            if self.tick != now or lane.cmd_pts is not hold_pts:
                return "retry"   # a chunk went out meanwhile: plan again from the new commanded pose
            O = other.pts if len(other.qa) else np.empty((0,) + other.cmd_pts.shape)
            O_last = other.pts[-1] if len(other.qa) else other.cmd_pts
            d, _ = self._find_delay(0, hold_pts, P, O, O_last, max_delay=max(240, len(O) + 30))
            if d is None:
                self._note(self._t(), f"{side}: abort would collide with the other arm: finishing the pick")
                return
            lane.qa = np.vstack([np.repeat(ca[None], d, 0), A])
            lane.qh = np.vstack([np.repeat(ch[None], d, 0), H])
            lane.pts = np.concatenate([np.repeat(hold_pts[None], d, 0), P]) if d else P
            lane.chain_abs, lane.chain_a, lane.chain_h = now + d, ca.copy(), ch.copy()
            p.end = now + d + len(A)
            for q in later:
                lane.plans.remove(q)
                # by index: list.remove() compares tuples holding numpy arrays -> "truth value ... ambiguous"
                idx = next((i for i, a in enumerate(self.attempts) if a[0] is q.obj.xy or
                            (np.allclose(a[0], q.obj.xy) and a[1] == side)), None)
                if idx is not None:
                    del self.attempts[idx]
            lane.version += 1
        self._note(self._t(), f"{side}: EMPTY HAND after lift -> abort {p.obj.det.color}, home now (delay {d / HZ:.1f}s)"
                              + (f", dropped {len(later)} queued pick(s)" if later else ""))

    def _is_failed(self, xy):
        return any(np.linalg.norm(xy - f) < self.cfg["policy"]["same_object_radius"] for f in self.failed)

    # Wall-fit method bias (m, added to the calibrated offset). perception_bench, 20 DR seeds (2026-11-14): after the
    # bin fit every object read 9.9 mm too near (x) and 1.3 mm right (y), per-seed spread only 2 mm, while raw
    # detections were unbiased: the fit sees the walls ~9 mm too far (edge pixels mixed with the floor behind).
    CALIB_BIAS = (0.0099, 0.0013)

    def calib_bias(self):
        env = os.environ.get("AALTO_CALIB_BIAS")
        if env is not None:
            return np.asarray([float(v) for v in env.split(",")], float) / 1000.0 if env.strip() else np.zeros(2)
        return np.asarray(self.CALIB_BIAS, float)

    def _self_calibrate(self, depth):
        """Once per episode at the first look (arms parked): level the counter plane (head tilt / height, fastsort.calib)
        then fit the bins in xy. AALTO_SELF_CALIB=0 disables both, AALTO_LEVEL=0 only the levelling."""
        self.cal.cams[self.perception.cam].T_pelvis_optical = self._T_head0.copy()
        self.perception.xy_offset = self._xy_base.copy()
        set_kin_world(self.kin, np.eye(4))
        if os.environ.get("AALTO_SELF_CALIB", "1") == "0":
            return
        try:
            notes, W = first_look_calibration(self.perception, depth, self._T_head0, self._xy_base, self.calib_bias(),
                                              level=os.environ.get("AALTO_LEVEL", "1") != "0")
            set_kin_world(self.kin, W)   # body lean: arms planned in the levelled world frame too (identity otherwise)
        except Exception as e:   # never let calibration break an episode: fall back to the configured extrinsics
            self.cal.cams[self.perception.cam].T_pelvis_optical = self._T_head0.copy()
            self.perception.xy_offset = self._xy_base.copy()
            set_kin_world(self.kin, np.eye(4))
            notes = [f"self-calibration failed: {e!r}"]
        for n in notes:
            self._note(self._t(), n)

    def _charge(self, tick0, wall0):
        """Real-time emulation: the brain was busy from tick0 for (now - wall0) wall seconds."""
        if self.rt_emulate > 0:
            self.vtick = max(self.vtick, tick0) + (time.monotonic() - wall0) * HZ * self.rt_emulate

    def _look(self, obs, now):
        tick0, wall0 = self.tick, time.monotonic()
        try:
            self._look_inner(obs, now)
        finally:
            self._charge(tick0, wall0)

    def _look_inner(self, obs, now):
        if self.pool is None:
            self._self_calibrate(obs["head_depth"])
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
            fix = min(((np.linalg.norm(xy - fxy), c) for fxy, c in self.color_fix), default=None, key=lambda t: t[0])
            if fix is not None and fix[0] < 0.05:
                d.color = fix[1]   # the wrist camera saw this one up close
            self._vote_shape(d, xy)
            b = self.rules.bin_for(d.color, getattr(d, "shape", None))
            if b is None:
                continue
            if self._is_failed(xy):
                continue
            # earlier direct picks of a same-colour object near here that left it on the counter = failed grasps.
            # 6 cm, not same_object_radius: a failed grasp usually nudges the object (sim: one cube was retried 6x).
            prior = [a for a in self.attempts if a[2] == d.color and not a[3] and np.linalg.norm(xy - a[0]) < 0.06]
            if len(prior) >= self.max_attempts_total:
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
                o.n_blocked = getattr(old, "n_blocked", 0)
            # an arm whose grasp already failed here doesn't try the same thing again: other arm, then relocation
            failed_sides = {a[1] for a in prior}
            d.retry = len(prior)   # ShapePlanner starts from the next grasp axis on a retry
            if failed_sides and not o.reloc and b != "center":
                # side bins: the other arm can't carry there (cross-body 0/26), so "let the other arm try" only meant a
                # hand-off, and after a failed grasp at the hand-off spot it handed straight back: a relocation loop that
                # ate a whole DR episode (seed 1002). The bin-side arm retries with another grasp axis; one hand-off after
                # two misses.
                if len(prior) >= 2 and not any(a[3] for a in self.attempts if np.linalg.norm(xy - a[0]) < 0.10):
                    o.reloc, o.tried = True, set()
                    o.pref = "left" if xy[1] > 0 else "right"
                else:
                    o.tried = set()
            elif failed_sides and not o.reloc:
                o.tried |= failed_sides
                if len(o.tried) >= 2:
                    o.reloc, o.tried = True, set()
                    o.pref = "left" if xy[1] > 0 else "right"
                elif o.pref in o.tried:
                    o.pref = "right" if o.pref == "left" else "left"
            pool.append(o)
        self.pool = pool
        self._note(self._t(), f"look: {len(dets)} detections, {len(pool)} to sort: "
                              + ", ".join(f"{o.det.color} {getattr(o.det, 'shape', None) or '?'}->{o.bin}({o.pref[0]})" for o in pool))

    def _vote_shape(self, d, xy):
        """Shape double-check (#4): every look re-measures each object's top face; keep a per-object vote (same colour,
        within 3 cm) and use the majority, weighting confident measurements (hull/rect ratio far from the thresholds)
        higher. A single partly occluded view (an arm in front) can't flip the shape any more."""
        p = getattr(d, "shape_p", None)
        if not p:
            return
        mem = self._shape_mem(d.color, xy, create=True)
        mem["xy"] = xy.copy()
        self._add_shape_evidence(mem, p)
        self._apply_shape_mem(d, mem)

    def _shape_mem(self, color, xy, create=False):
        mem = next((m for m in self.shape_votes if m["color"] == color and np.linalg.norm(m["xy"] - xy) < 0.03), None)
        if mem is None and create:
            mem = {"color": color, "xy": np.asarray(xy, float).copy(), "logp": {}, "n": 0}
            self.shape_votes.append(mem)
        return mem

    @staticmethod
    def _add_shape_evidence(mem, p, source="head"):
        """Bayes: add log-likelihoods, each floored so one view can't veto. Head looks repeat nearly the same view,
        so they count as their AVERAGE times at most 2 views; other cameras (wrist) add as independent evidence."""
        key = "logp" if source == "head" else "logp_" + source
        mem.setdefault(key, {})
        for s in ("cube", "cylinder", "triangle"):
            mem[key][s] = mem[key].get(s, 0.0) + float(np.log(max(p.get(s, 0.0), 0.02)))
        if source == "head":
            mem["n"] += 1

    @staticmethod
    def shape_posterior(mem):
        n = max(mem.get("n", 0), 1)
        lp = {s: v / n * min(n, 2) for s, v in mem.get("logp", {}).items()}
        for k, extra in mem.items():
            if k.startswith("logp_"):
                for s, v in extra.items():
                    lp[s] = lp.get(s, 0.0) + v
        if not lp:
            return {"cube": 1 / 3, "cylinder": 1 / 3, "triangle": 1 / 3}
        m = max(lp.values())
        e = {s: float(np.exp(v - m)) for s, v in lp.items()}
        t = sum(e.values())
        return {s: v / t for s, v in e.items()}

    def _apply_shape_mem(self, d, mem):
        from .grasp import set_shape
        post = self.shape_posterior(mem)
        best = max(post, key=post.get)
        if best != d.shape:
            set_shape(d, best)   # this view's own axes for that shape (cube legs / ridge / tripod), maybe none
        d.shape_p = post
        d.shape_votes = {s: round(v, 2) for s, v in post.items()}

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
        try:
            plan = self._make_plan(lane, obj, side, arm_q, hand_q)
        finally:
            self._charge(now, t0)
        if plan is None:
            return False
        dt_plan = time.monotonic() - t0
        return self._commit(lane, obj, plan, side, sl, sa, sh, use_chain, arm_q, hand_q, dt_plan)

    def _make_plan(self, lane, obj, side, arm_q, hand_q):
        """The plan for obj with this arm, or None (failure already recorded / handed to the other arm)."""
        if self._wants_push(obj):
            obstacles = [o.xy for o in self.pool if o is not obj]
            plan = self.planner.plan_push(obj.det, side, arm_q, hand_q, obstacles)
            if plan.ok:
                return plan
            self._note(self._t(), f"{side}: no push for {obj.det.color} prism ({plan.reason}): grasping it standing")
        if not obj.reloc and obj.bin not in ("center", side):
            # cross-body carry (left arm -> right bin or vice versa): 0 successes in 26 tries over 7 eval runs, each
            # failure cost 2.5-4 s of planning. Go straight to the hand-off.
            obj.reloc, obj.tried = True, set()
        if not obj.reloc:
            k = self.bin_drops.get(obj.bin, 0)
            self.planner.drop_slot = {obj.bin: k}
            try:
                plan = self.planner._plan_side_hints(obj.det, side, obj.bin, arm_q, hand_q)
                if not plan.ok and k:   # slot out of reach: the usual release point (stacking beats no plan)
                    self.planner.drop_slot = {}
                    plan = self.planner._plan_side_hints(obj.det, side, obj.bin, arm_q, hand_q)
            finally:
                self.planner.drop_slot = {}
            if not plan.ok:
                self._side_failed(obj, side, f"no plan: {plan.reason}")
                return None
        else:
            obstacles = [o.xy for o in self.pool if o is not obj]
            plan = self.planner.plan_relocation(obj.det, obj.bin, arm_q, hand_q, obstacles)
            if plan.ok and plan.side != side:  # the relocation needs the other arm: hand it over
                obj.pref = plan.side
                return None
            if not plan.ok:
                self._side_failed(obj, side, f"no relocation: {plan.reason}")
                return None
        return plan

    push_min_p = 0.75   # P(triangle) needed before pushing (a pushed cube just slides: a wasted trip)

    def _wants_push(self, obj):
        """Standing prism (triangle top face, directed tripod axes), confident, not pushed here before."""
        d = obj.det
        if os.environ.get("AALTO_PUSH", "1") == "0" or obj.reloc:
            return False
        if getattr(d, "shape", None) != "triangle" or not getattr(d, "axes_directed", False):
            return False
        if len(getattr(d, "grasp_axes", None) or []) != 3:
            return False
        if float((getattr(d, "shape_p", None) or {}).get("triangle", 1.0)) < self.push_min_p:
            return False
        return not any(np.linalg.norm(obj.xy - p) < 0.07 for p in self.pushed)

    def _commit(self, lane, obj, plan, side, sl, sa, sh, use_chain, arm_q, hand_q, dt_plan):
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
            # search past the other arm's whole queued timeline: with a 8 s cap a chained other arm (pick + next pick
            # > 8 s) made this arm replan every 0.5 s for up to 8.5 s of sim time (17 "later"s in one DR episode)
            d0 = max(0, int(np.ceil(self.vtick - (now0 + cut0)))) if self.rt_emulate > 0 else 0
            d, closest = self._find_delay(cut0, hold_pts, P, O, O_last, max_delay=max(240, len(O) - cut0 + 30) + d0,
                                          d_min=d0)
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
                    # exponential back-off: every retry is a full plan (2-3 s of brain time); seed 2005 retried every
                    # 0.5 s for 7 s and the planning debt delayed every later move by 25-40 s
                    k = getattr(obj, "n_blocked", 0)
                    obj.n_blocked = k + 1
                    obj.not_before = now + min(120, 15 * 2 ** k)
                    self._note(self._t(), f"{side}: {obj.det.color} would collide with the other arm ({why}): "
                                          f"later (+{min(120, 15 * 2 ** k) / HZ:.1f}s)")
                return False
            wait = d - d0
            if use_chain and wait > self.max_wait_chain and not other_parked:
                # a chained arm waits at its chain point (over a bin, mid-workspace) where it blocks the other arm
                # (seed 2005: 40 s of "would collide ... later"): go home first, plan from there
                obj.not_before = (lane.chain_abs or now) + 2
                self._note(self._t(), f"{side}: {obj.det.color} would wait {wait / HZ:.1f}s at the chain point: "
                                      f"after the home move")
                return False
            if wait > self.max_wait and not other_parked:
                # a reservation far in the future pushes the other arm's next plans back too (28 s + 38 s snowball,
                # seed 2005): try again when the timeline has moved on
                k = getattr(obj, "n_blocked", 0)
                obj.n_blocked = k + 1
                obj.not_before = now + max(min(120, 15 * 2 ** k), (wait - self.max_wait) // 2)
                self._note(self._t(), f"{side}: {obj.det.color} would wait {wait / HZ:.1f}s: later")
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
            ph = {name: (start_abs + s, start_abs + e) for name, s, e in plan.phases}
            ce = next((e for name, s, e in plan.phases if name == "close"), None)
            lane.plans.append(LanePlan(obj, start_abs, now + cut + d + len(A), gxy, ph,
                                       None if ce is None else H[ce - 1].copy(), plan_a=A, plan_h=H))
            lane.version += 1
            self.attempts.append((obj.xy, side, obj.det.color, plan.target_bin == "counter"))
            if (plan.reason or "").startswith("push over"):
                self.pushed.append(np.asarray(obj.xy, float).copy())
            if plan.target_bin != "counter":
                self.bin_drops[obj.bin] = self.bin_drops.get(obj.bin, 0) + 1
        finally:
            self.lock.release()
        what = ("push" if (plan.reason or "").startswith("push over") else "relocate" if plan.target_bin == "counter" else f"-> {obj.bin}") + (f" [{plan.reason}]" if plan.reason else "")
        self._note(self._t(), f"{side}: {obj.det.color} {what}  start +{(cut + d) / HZ:.1f}s (delay {d / HZ:.1f}s, "
                              f"{'chained' if use_chain else 'from tail'}), {len(A) / HZ:.1f}s, planned in {dt_plan:.2f}s")
        return True

    max_wait_chain = 60    # ticks (2 s): longest wait accepted at a chain point
    max_wait = 180         # ticks (6 s): longest wait accepted for any new plan

    def _find_delay(self, cut, hold_pts, P, O, O_last, max_delay=240, step=3, d_min=0):
        """Smallest d (ticks) such that [hold d ticks] + P, starting `cut` ticks from now, keeps every point >=
        min_arm_dist from the other arm's timeline O (then its final pose O_last). None if no d <= max_delay."""
        """Returns (d or None, closest) with closest = (distance, my point index, their point index) at d = 0."""
        dmin = self.min_arm_dist
        n_o = len(O)
        closest = None
        for d in range(d_min, max_delay + 1, step):
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
