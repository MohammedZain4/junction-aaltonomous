"""Shape- and orientation-aware grasping on top of the baseline perception + planner.

ShapePerception: the baseline detection plus, from the 3D top-face points (pelvis frame, so no perspective):
  det.shape      cube | cylinder | triangle | None, by hull area / min-area-rectangle area
                 (square 1.0, circle pi/4 = 0.785, triangle 0.5)
  det.grasp_axes unit xy pinch axes (thumb <-> finger) that hold that shape best:
                 cube     -> the two face normals (pinch across parallel faces, not across corners)
                 triangle -> the three altitudes (flat side <-> opposite vertex, through the centroid)
                 cylinder -> none (any direction works)
ShapePlanner: tries those axes first as the finger direction (stronger IK weight), and accepts a plan only if the
solved hand really lines up (<= max_axis_err_deg at the grasp); then falls back to the baseline's finger hints.
The claw grasp pinches along the finger direction (fingers curl down in front, thumb comes from behind).
"""
from __future__ import annotations

import os

import numpy as np

from aaltonomous.baselines.classical.perception import Perception
from aaltonomous.baselines.classical.planner import Planner

RIDGE_MIN_LEN, RIDGE_MAX_WIDTH, RIDGE_ASPECT = 0.030, 0.022, 2.2   # top band of a lying prism: long thin strip
SQUARE_MIN, TRIANGLE_MAX = 0.88, 0.68     # hull/rect area ratio thresholds (DR bench medians: triangle 0.60, cylinder 0.80, cube 0.95)


def hull_rect_ratio(xy):
    """Convex hull area / minimum-area rectangle area of the top-face points (None if too few)."""
    import cv2
    if xy is None or len(xy) < 15:
        return None
    pts = np.asarray(xy, np.float32).reshape(-1, 1, 2)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return float(cv2.contourArea(cv2.convexHull(pts))) / max(float(w * h), 1e-9)


SHAPES = ("cube", "cylinder", "triangle")
# hull/rect ratio per shape (DR perception bench, 58 matched objects, seeds 31-46): median / spread
RATIO_MODEL = {"cube": (0.955, 0.03), "cylinder": (0.795, 0.025), "triangle": (0.60, 0.04)}
RIDGE_MAX_WIDTH_STRICT = 0.010   # a lying prism's top band: 6-9 mm wide and RECTANGULAR (hull/rect >= 0.82);
                                 # tilted-cylinder top slivers were 13-16 mm wide and rounded (0.73-0.77)
HALF_CUBE_ASPECT = 1.6           # triangle-shaped top band twice as long as wide = half of a (tilted) cube top
                                 # (5 cubes read as triangles: 28 x 55 mm; real standing prisms 39 x 45 mm)


def _tri_axes(tri):
    mode = os.environ.get("AALTO_TRI_AXIS", "tripod")
    axes = []
    for i in range(3):
        if mode == "altitude":   # flat side <-> opposite vertex: tested in sim, the vertex slips (1/4)
            v = 0.5 * (tri[(i + 1) % 3] + tri[(i + 2) % 3]) - tri[i]
        elif mode == "tripod":   # finger direction TOWARDS a vertex: the two fingers straddle that apex edge, the
            v = tri[i] - tri.mean(0)  # thumb pushes the opposite flat face (3 contacts; ShapePlanner centres it)
        else:                    # parallel to a side: both contacts on the two sloped faces, symmetric
            v = tri[(i + 2) % 3] - tri[(i + 1) % 3]
        n = np.linalg.norm(v)
        if n > 1e-6:
            axes.append(v / n)
    return axes


def classify_top_p(xy):
    """Soft shape classification of the top-face points xy (N, 2), m. Returns (probs {shape: p} | None,
    axes {shape: [unit xy pinch axes]}). Rules first (lying-prism ridge, half-visible cube top), else a Gaussian
    likelihood of the hull/rect area ratio; small / partial top faces get flatter probabilities."""
    import cv2
    if xy is None or len(xy) < 15:
        return None, {}
    pts = np.ascontiguousarray(xy, np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    a_hull = float(cv2.contourArea(hull))
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    a_rect = float(w * h)
    if a_rect <= 1e-8 or a_hull <= 1e-8:
        return None, {}
    r = a_hull / a_rect
    lo, hi = sorted((float(w), float(h)))
    t = np.radians(ang)
    rect_axes = [np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])]
    long_ax = rect_axes[0] if float(w) >= float(h) else rect_axes[1]
    tri, tri_fill = None, 0.0
    try:
        a_tri, tri = cv2.minEnclosingTriangle(hull)
        if tri is not None and float(a_tri) > 1e-12:
            tri = tri.reshape(3, 2).astype(float)
            tri_fill = a_hull / float(a_tri)
        else:
            tri = None
    except cv2.error:
        tri = None
    axes = {"cube": rect_axes, "cylinder": [], "triangle": _tri_axes(tri) if tri is not None else []}

    ridge = hi >= RIDGE_MIN_LEN and lo <= RIDGE_MAX_WIDTH and hi >= RIDGE_ASPECT * max(lo, 1e-6)
    if ridge and (r >= 0.82 or lo <= RIDGE_MAX_WIDTH_STRICT):
        # a triangular prism LYING on a rectangular side: its top is a ridge, so the top band is a thin straight
        # (rectangular) strip. Pinch along the ridge: the two triangular end faces are parallel (45 mm apart).
        axes["triangle"] = [long_ax]
        return {"cube": 0.07, "cylinder": 0.03, "triangle": 0.90}, axes
    if ridge:
        # rounded sliver (13-22 mm): the visible part of a tilted / occluded cylinder or cube top, rarely a ridge
        axes["triangle"] = [long_ax]
        return {"cube": 0.30, "cylinder": 0.45, "triangle": 0.25}, axes
    if tri is not None and tri_fill >= 0.85 and r <= 0.72 and hi >= HALF_CUBE_ASPECT * max(lo, 1e-6):
        # right-angled triangle twice as long as wide = half of a cube top (tilted cube: only the high half is in the
        # top band). Its two short sides are cube faces: pinch across those, i.e. along the legs at the right angle.
        def cos_at(i):
            u, v = tri[(i + 1) % 3] - tri[i], tri[(i + 2) % 3] - tri[i]
            return abs(float(u @ v)) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-12)
        k = int(np.argmin([cos_at(i) for i in range(3)]))   # the vertex closest to 90 deg
        legs = [tri[(k + 1) % 3] - tri[k], tri[(k + 2) % 3] - tri[k]]
        axes["cube"] = [v / np.linalg.norm(v) for v in legs]
        return {"cube": 0.75, "cylinder": 0.05, "triangle": 0.20}, axes
    like = {s: float(np.exp(-0.5 * ((r - m) / sd) ** 2)) + 1e-3 for s, (m, sd) in RATIO_MODEL.items()}
    tot = sum(like.values())
    p = {s: v / tot for s, v in like.items()}
    # partial view (arm in front, edge of the image): flatten towards uniform
    full = {"cube": 1.9e-3, "cylinder": 1.45e-3, "triangle": 1.0e-3}   # m^2, typical full top-band areas
    best = max(p, key=p.get)
    wgt = float(np.clip(a_hull / (0.6 * full[best]), 0.3, 1.0))
    p = {s: v ** wgt for s, v in p.items()}
    tot = sum(p.values())
    return {s: v / tot for s, v in p.items()}, axes


def set_shape(det, shape):
    """Set det.shape and the matching grasp axes (from det.axes_by_shape) + whether they are directed (tripod)."""
    det.shape = shape
    det.grasp_axes = list((getattr(det, "axes_by_shape", None) or {}).get(shape, [])) if shape else []
    # tripod prism grasp: the axes are DIRECTED (fingers towards the apex), not reversible pinch axes
    det.axes_directed = (shape == "triangle" and len(det.grasp_axes) == 3
                         and os.environ.get("AALTO_TRI_AXIS", "tripod") == "tripod")


def classify_top(xy):
    """xy: (N, 2) top-face points (m). Returns (shape | None, [unit xy pinch axes]) = the most likely shape."""
    p, axes = classify_top_p(xy)
    if p is None:
        return None, []
    s = max(p, key=p.get)
    return s, axes.get(s, [])


class ShapePerception(Perception):
    # Under domain randomization (bin tint + lighting + head-camera pose error) the orange bin walls drift into the
    # yellow HSV range, and the walls just outside the masked footprints (back walls, walls in the gaps between bins)
    # became phantom "yellow" objects: 13 in 8 DR scenes, all at or behind the bin row and 5.5-8.5 cm tall.
    bin_line_margin = 0.015   # m in front of the bins' near edge

    # Reference hues (OpenCV H, 0-180) of each colour in the sim (scene.yaml RGB) and on the real head camera (Drop 0).
    # Real blue class = purple/lavender: H 110-112 on the Drop 0/2 head camera, H ~127 in the organisers' lab notes.
    hue_prototypes = {"red": (0.0, 14.0), "yellow": (25.0, 43.0), "blue": (113.0, 111.0, 127.0)}
    dominant_frac = 0.7

    def masks(self, rgb):
        """Colour masks by BLOB, not by pixel: every pixel inside any configured colour range joins one mask, connected
        blobs are labelled by their pixels' nearest reference hue (circular distance); a blob whose dominant colour has
        >= dominant_frac of its pixels is assigned whole (a shaded yellow edge at H~20 stays yellow), otherwise it is
        split per pixel (two different objects touching). Fixed per-pixel ranges cannot separate sim yellow (H 21-29
        under DR) from real red (H 14) and the shaded edges of real yellow at the same time."""
        import cv2
        per_px = super().masks(rgb)
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        H = hsv[..., 0].astype(float)
        names = list(self.hue_prototypes)
        any_m = np.zeros(H.shape, np.uint8)
        for m in per_px.values():
            any_m |= m
        protos = [np.asarray(self.hue_prototypes[c], float) for c in names]
        sel_px = any_m > 0                                      # only the candidate pixels (was the whole image)
        h_sel = H[sel_px][:, None]
        dist = np.stack([np.min(np.minimum(np.abs(h_sel - p), 180 - np.abs(h_sel - p)), axis=-1) for p in protos])
        near = np.zeros(H.shape, np.int64)
        near[sel_px] = np.argmin(dist, axis=0)
        out = {c: np.zeros(H.shape, np.uint8) for c in names}
        n, lab = cv2.connectedComponents(any_m, connectivity=8)
        k5 = np.ones((5, 5), np.uint8)
        min_area = int(self.cfg.get("min_area_px", 120))
        for i in range(1, n):
            sel = lab == i
            counts = np.bincount(near[sel], minlength=len(names))
            k = int(counts.argmax())
            whole = counts[k] >= self.dominant_frac * counts.sum()
            if whole:
                # Real frames (Drop 2): a purple / yellow object touching an orange bin joins the bin's blob and was
                # out-voted -> painted "red". Absorb only thin shaded fringes: a minority part that survives a 5x5
                # opening with object-sized area is a separate object, so split per pixel.
                for j in range(len(names)):
                    if j == k or counts[j] < min_area:
                        continue
                    part = cv2.morphologyEx((sel & (near == j)).astype(np.uint8), cv2.MORPH_OPEN, k5)
                    if int(part.sum()) >= min_area:
                        whole = False
                        break
            if whole:
                out[names[k]][sel] = 255
            else:
                for j, c in enumerate(names):
                    out[c][sel & (near == j)] = 255
        return out

    def _reject(self, det):
        why = super()._reject(det)
        if why is not None:
            return why
        b = self.scene.cfg["bins"]
        near_x = min(self.scene.mat_to_pelvis(np.r_[np.asarray(c, float), 0.0])[0] for c in b["centers"].values()) \
            - float(np.asarray(b["outer_size"], float)[1]) / 2
        if det.top[0] >= near_x - self.bin_line_margin:
            return f"behind bin line (x {det.top[0]:.3f} >= {near_x - self.bin_line_margin:.3f})"
        # (A tighter height limit, 1.35 sizes, also removed them but lost 2 real objects in a DR scene whose head
        #  camera tilt made everything read ~1 cm taller: the bin-line rule alone removes all 13.)
        return None

    def _from_depth(self, color, u, v, d, area):
        det = super()._from_depth(color, u, v, d, area)
        if det is None:
            return det
        try:
            P = self.cal.pixel_to_pelvis(self.cam, u, v, d)
            z_top = np.percentile(P[:, 2], self.cfg["top_percentile"])
            top = P[P[:, 2] >= z_top - self.cfg["top_band"]]
            det.top_pts, det.all_pts = top[:, :3], P   # (diagnosis: perception_bench --out)
            det.shape_p, det.axes_by_shape = classify_top_p(top[:, :2])
            set_shape(det, None if det.shape_p is None else max(det.shape_p, key=det.shape_p.get))
            det.shape_ratio = hull_rect_ratio(top[:, :2])
        except Exception:
            det.shape, det.grasp_axes, det.shape_ratio, det.shape_p, det.axes_by_shape = None, [], None, None, {}
            det.axes_directed = False
        b = self.xy_bias(det.shape)
        if b is not None:
            det.top = np.r_[det.top[:2] - b, det.top[2:]]
        return det

    def xy_bias(self, shape):
        """Systematic top-centroid error (detected - true, m) to subtract, per perceived shape ("all" = fallback).
        From config `shape_xy_bias` or env AALTO_XY_BIAS="all:dx,dy;triangle:dx,dy" (mm, for benchmarking)."""
        if not hasattr(self, "_xy_bias"):
            tbl = {k: np.asarray(v, float) for k, v in (self.cfg.get("shape_xy_bias") or {}).items()}
            env = os.environ.get("AALTO_XY_BIAS", "").strip()
            for part in filter(None, env.split(";")):
                k, v = part.split(":")
                tbl[k.strip()] = np.asarray([float(x) for x in v.split(",")], float) / 1000.0
            self._xy_bias = tbl
        return self._xy_bias.get(shape, self._xy_bias.get("all"))


class ShapePlanner(Planner):
    drop_slot: dict = {}       # bin -> slot index for the next release (set by the policy per plan)
    slot_step = 0.05           # m between release slots (object 45 mm: neighbours never stack)

    def drop_xy(self, name):
        """Release point: the baseline point (inner corner nearest the pelvis, drop_inset inside the walls), moved to
        slot k inside the inset box. A second object released on the first one rests above the rim, and an object
        whose centre is above the rim does NOT count as in the bin (scene.classify_cube): 2 of 5 lost that way."""
        base = super().drop_xy(name)
        k = int(self.drop_slot.get(name, 0) or 0)
        inset = self.cfg["grasp"].get("drop_inset")
        if k <= 0 or inset is None:
            return base
        c = self.bin_xy(name)
        lo, hi = self.scene.bin_inner_box(name)
        a, b = self.scene.mat_to_pelvis(np.r_[lo[:2], 0.0])[:2], self.scene.mat_to_pelvis(np.r_[hi[:2], 0.0])[:2]
        lo2 = np.minimum(np.minimum(a, b) + inset, c)
        hi2 = np.maximum(np.maximum(a, b) - inset, c)
        mid = 0.5 * (lo2 + hi2)
        st = self.slot_step
        sx = 1.0 if base[0] <= mid[0] else -1.0
        if abs(base[1] - mid[1]) < 1e-3:      # centre bin: release point mid-width, spread sideways first
            slots = [(0.0, 0.0), (0.0, 0.8 * st), (0.0, -0.8 * st), (sx * st, 0.0), (sx * st, 0.8 * st),
                     (sx * st, -0.8 * st)]
        else:
            sy = 1.0 if base[1] <= mid[1] else -1.0
            slots = [(0.0, 0.0), (0.0, sy * st), (sx * st, 0.0), (sx * st, sy * st)]
        return np.clip(base + np.asarray(slots[k % len(slots)], float), lo2, hi2)

    max_axis_err_deg = 20.0
    axis_hint_weight = 1.0
    reloc_max_x = 0.31   # m (pelvis): hand-off spots in the near half of the object band

    def relocation_spots(self, det, bin_name, obstacles=()):
        """Like the baseline, but only spots in the near half of the band: spots next to the bins (x ~0.35) failed
        the bin-rim clearance for the finishing arm in DR runs (fingers reach forward over the near bin wall)."""
        rc = self.cfg["grasp"].get("relocate", {})
        L = self.scene.cfg["layout"]
        m = float(rc.get("region_margin", 0.02))
        xs = np.linspace(L["region_x"][0] - m, L["region_x"][1] + m, 9)
        ys = np.linspace(L["region_y"][0] - m, L["region_y"][1] + m, 6)
        pts = self.scene.mat_to_pelvis(np.array([[x, y, 0.0] for x in xs for y in ys]))[:, :2]
        clear = float(rc.get("clearance", L["min_spacing"]))
        obs = np.asarray(list(obstacles), float).reshape(-1, 2)
        keep = [p for p in pts if p[0] <= self.reloc_max_x and np.linalg.norm(p - det.top[:2]) > 0.06
                and (len(obs) == 0 or np.linalg.norm(obs - p, axis=1).min() > clear)]
        b = self.bin_xy(bin_name)
        keep.sort(key=lambda p: np.linalg.norm(p - b))
        return keep[: int(rc.get("max_spots", 6))]

    def _axis_hints(self, det, side):
        """Object pinch axes as finger directions: sign chosen to point forward (fingers can't point back at the
        robot); ordered by closeness to the arm's natural direction (forward, a little inward)."""
        axes = getattr(det, "grasp_axes", None) or []
        directed = bool(getattr(det, "axes_directed", False))
        natural = np.array([1.0, -0.4 if side == "left" else 0.4])
        natural /= np.linalg.norm(natural)
        out = []
        for a in axes:
            a = np.asarray(a, float)
            if directed:
                if a[0] < -0.2:   # this apex points back at the robot: the fingers can't come from there
                    continue
                out.append(a)
                continue
            if a[0] < 0 or (abs(a[0]) < 1e-6 and (a[1] > 0) == (side == "left")):
                a = -a
            if a[0] < -0.2:  # pointing back at the robot even after the flip: unreachable
                continue
            out.append(a)
        out.sort(key=lambda a: -float(a @ natural))
        return out

    def _axis_err_deg(self, side, plan, axis):
        """Angle between the solved finger direction at the grasp and the pinch axis (mod 180 deg)."""
        end = next((e for name, s, e in plan.phases if name == "descend"), None)
        if end is None or plan.arm is None:
            return 180.0
        self.kin._fk(plan.arm[end - 1])
        f = self.kin.data.oMf[self.kin.palm[side]].rotation[:2, 0]
        n = np.linalg.norm(f)
        if n < 1e-6:
            return 180.0
        c = abs(float(f @ axis) / n)
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    def _plan_side_hints(self, det, side, bin_name, arm_q, hand_q, **kw):
        axes = self._axis_hints(det, side) if os.environ.get("AALTO_SHAPE_GRASP", "1") != "0" else []
        k = int(getattr(det, "retry", 0) or 0)
        if axes and k:   # a retry after a failed grasp: start from the next axis (same grasp would likely fail again)
            axes = axes[k % len(axes):] + axes[:k % len(axes)]
        if axes:
            saved_h, saved_w = self.kin.hint, self.kin.w_hint
            saved_off = self.kin.offset[side].copy()
            try:
                self.kin.w_hint = self.axis_hint_weight
                if getattr(det, "axes_directed", False):
                    self.kin.offset[side] = saved_off * np.array([1.0, 1.0, 0.0])   # centred between the fingers
                for a in axes[:2]:  # best two: a failing plan per axis costs ~1-4 s of IK
                    self.kin.hint = np.r_[a, 0.0]
                    p = self._plan_side(det, side, bin_name, arm_q, hand_q, **kw)
                    if p.ok:
                        err = self._axis_err_deg(side, p, a)
                        if err <= self.max_axis_err_deg:
                            p.reason = f"{getattr(det, 'shape', '?')} axis grasp (err {err:.0f} deg)"
                            return p
            finally:
                self.kin.hint, self.kin.w_hint = saved_h, saved_w
                self.kin.offset[side] = saved_off
        return super()._plan_side_hints(det, side, bin_name, arm_q, hand_q, **kw)

    # ------------------------------------------------------------------ push-over (standing prisms)
    # A prism standing on a triangular end has no parallel pair of faces a top-down claw can pinch (tripod grasps:
    # ~1/10 in grasp_bench). Lying on a rectangular side, its two triangular ends ARE parallel, 45 mm apart: the
    # cube-like grasp. So: push it over first. Push from a vertex towards the opposite flat face, near the top: it
    # pivots on the far base edge (centroid 15 mm from it, centre of mass ~25 mm up: tips at ~31 deg) and lands on
    # that rectangular face. The next look sees a ridge -> pinch the two triangular ends.
    push_height = 0.036       # m above the counter: grasp point of the closed hand (fist) while pushing
    push_start = 0.085        # m from the object centre (behind the vertex) where the push starts
    push_end = 0.0            # m past the centre where the grasp point stops (the fist front is ~3 cm ahead)
    push_speed = 0.35         # fraction of the normal arm speed / acceleration for the push stroke

    def push_dirs(self, det, obstacles=()):
        """Candidate push directions (unit xy, vertex -> opposite face), best first: towards the robot / sideways
        rather than into the bins, and away from other objects."""
        axes = [np.asarray(a, float) for a in (getattr(det, "grasp_axes", None) or [])]
        if not getattr(det, "axes_directed", False) or len(axes) != 3:
            return []
        c = det.top[:2]
        obs = np.asarray(list(obstacles), float).reshape(-1, 2)
        out = []
        for a in axes:          # a: centroid -> vertex; the push goes the other way
            d = -a / np.linalg.norm(a)
            land = c + 0.04 * d
            pen = 0.0
            if len(obs):
                pen = float(np.clip(0.09 - np.linalg.norm(obs - land, axis=1), 0, None).sum()) * 20
            out.append((float(d[0]) + pen, d))
        out.sort(key=lambda t: t[0])
        return [d for _, d in out]

    def plan_push(self, det, side, arm_q, hand_q, obstacles=()):
        """Push a standing prism over with the closed hand. Plan with target_bin 'counter' (like a relocation):
        no grip check, not counted as a failed grasp; the object is re-detected at the next look."""
        from aaltonomous.baselines.classical.planner import Plan, ramp, time_scale
        from aaltonomous.baselines.classical.kinematics import ArmKinematics
        last = Plan(False, side=side, color=det.color, reason="no push direction")
        cfg, g, mo = self.cfg, self.cfg["grasp"], self.cfg["motion"]
        zc, H = self.scene.counter_z, self.height
        sl = ArmKinematics.sl(side)
        fist = hand_q.copy(); fist[sl] = self.post["closed"][sl]
        hand_open = hand_q.copy(); hand_open[sl] = self.post["open"][sl]
        c = np.asarray(det.top[:2], float)
        wall = float(self.scene.cfg["bins"]["wall_height"])
        z_safe = zc + max(g["carry_height"], wall + H + 0.03)
        saved_h, saved_w = self.kin.hint, self.kin.w_hint
        try:
            for d in self.push_dirs(det, obstacles)[:2]:
                plan = Plan(False, side=side, color=det.color, target_bin="counter")
                self.kin.hint, self.kin.w_hint = np.r_[d, 0.0], self.axis_hint_weight   # fingers along the push
                z = zc + self.push_height
                p0, p1 = np.r_[c - d * self.push_start, z], np.r_[c + d * self.push_end, z]
                q0, ok = self._ik(side, p0, arm_q, "push_start", plan)
                if not ok:
                    last = plan; last.reason = f"IK push start ({side})"; continue
                cl = self.clearance(side, q0, fist, fingers_only=True)
                if cl < 0.006:   # fist would scrape the counter: raise the stroke (still above the centre of mass)
                    z += 0.006 - cl
                    if z > zc + H - 0.008:
                        last = plan; last.reason = "fist too low for a push"; continue
                    p0, p1 = np.r_[p0[:2], z], np.r_[p1[:2], z]
                    q0, ok = self._ik(side, p0, arm_q, "push_start", plan)
                    if not ok:
                        last = plan; last.reason = f"IK push start ({side})"; continue
                pre = np.r_[p0[:2], max(z + g["approach_height"], z_safe)]
                q_pre, ok = self._ik(side, pre, q0, "pregrasp", plan)
                if not ok:
                    last = plan; last.reason = f"IK push pre ({side})"; continue
                down = self._cartesian(side, pre, p0, q_pre, "descend", plan)
                stroke = down and self._cartesian(side, p0, p1, down[-1], "push", plan)
                up = stroke and self._cartesian(side, p1, np.r_[p1[:2], z_safe], stroke[-1], "retreat", plan)
                if not up:
                    last = plan; last.reason = plan.reason or f"IK push path ({side})"; continue
                vmax, amax, hv = mo["arm_max_vel"], mo["arm_max_acc"], mo["hand_max_vel"]
                home = arm_q.copy(); home[sl] = self.home[sl]
                parts, phases, cur = [], [], [arm_q.copy(), hand_q.copy()]

                def add(name, Qs, hand_to=None, speed=1.0):
                    a = time_scale([cur[0]] + list(Qs), vmax * speed, amax * speed, self.dt)
                    h = np.repeat(cur[1][None], len(a), 0)
                    if hand_to is not None:
                        hr = ramp(cur[1], hand_to, hv, self.dt)
                        if len(hr) > len(a):
                            a = np.vstack([a, np.repeat((a[-1] if len(a) else cur[0])[None], len(hr) - len(a), 0)])
                            h = np.repeat(cur[1][None], len(a), 0)
                        h[:len(hr)] = hr
                        h[len(hr):] = hand_to
                        cur[1] = np.asarray(hand_to, float).copy()
                    if len(a):
                        s0 = sum(len(x[0]) for x in parts)
                        parts.append((a, h))
                        phases.append((name, s0, s0 + len(a)))
                        cur[0] = a[-1].copy()

                add("to_pregrasp", [q_pre], hand_to=fist)
                add("descend", down)
                add("push", stroke, speed=self.push_speed)
                add("retreat", up)
                add("home", [home], hand_to=hand_open)
                plan.arm = np.vstack([a for a, _ in parts])
                plan.hand = np.vstack([h for _, h in parts])
                plan.phases = phases
                plan.waypoints["grasp"] = p0          # policy bookkeeping: where this arm works
                bad = None
                for k in range(0, len(plan.arm), 3):
                    name = next(n for n, s, e in phases if s <= k < e)
                    need = 0.004 if name in ("descend", "push") else mo["counter_clearance"]
                    if self.clearance(side, plan.arm[k], plan.hand[k]) < need - 2e-3:
                        bad = f"hand too low during {name}"
                        break
                    if self.bin_clearance(side, plan.arm[k], plan.hand[k]) < 0.003:
                        bad = f"hand over a bin rim during {name}"
                        break
                if bad:
                    last = plan; last.reason = bad; continue
                plan.safety_clips = self.dry_run_safety(arm_q, hand_q, plan.arm, plan.hand)
                if plan.safety_clips:
                    last = plan; last.reason = f"{plan.safety_clips} safety clips in dry run"; continue
                plan.ok = True
                plan.reason = f"push over towards ({d[0]:+.2f},{d[1]:+.2f})"
                return plan
        finally:
            self.kin.hint, self.kin.w_hint = saved_h, saved_w
        return last
