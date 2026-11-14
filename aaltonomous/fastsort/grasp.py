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


def classify_top(xy):
    """xy: (N, 2) top-face points (m). Returns (shape | None, [unit xy pinch axes])."""
    import cv2
    if xy is None or len(xy) < 15:
        return None, []
    pts = np.asarray(xy, np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    a_hull = float(cv2.contourArea(hull))
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    a_rect = float(w * h)
    if a_rect <= 1e-8 or a_hull <= 1e-8:
        return None, []
    r = a_hull / a_rect
    lo, hi = sorted((float(w), float(h)))
    if hi >= RIDGE_MIN_LEN and lo <= RIDGE_MAX_WIDTH and hi >= RIDGE_ASPECT * max(lo, 1e-6):
        # a triangular prism LYING on a rectangular side (organisers: prisms may lie on a face or a side): from above
        # it is a ~52 x 45 mm rectangle as tall as a cube, but its top is a ridge, so the top band is a thin strip.
        # Pinch along the ridge: the two triangular end faces are parallel (45 mm apart).
        t = np.radians(ang)
        ax = np.array([np.cos(t), np.sin(t)]) if float(w) >= float(h) else np.array([-np.sin(t), np.cos(t)])
        return "triangle", [ax]
    if r >= SQUARE_MIN:
        t = np.radians(ang)
        axes = [np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])]
        return "cube", axes
    if r <= TRIANGLE_MAX:
        try:
            _, tri = cv2.minEnclosingTriangle(hull)
        except cv2.error:
            return "triangle", []
        tri = tri.reshape(3, 2).astype(float)
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
        return "triangle", axes
    return "cylinder", []


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
            det.shape, det.grasp_axes = classify_top(top[:, :2])
            # tripod prism grasp: the axes are DIRECTED (fingers towards the apex), not reversible pinch axes
            det.axes_directed = (det.shape == "triangle" and len(det.grasp_axes) == 3
                                 and os.environ.get("AALTO_TRI_AXIS", "tripod") == "tripod")
            det.shape_ratio = hull_rect_ratio(top[:, :2])
        except Exception:
            det.shape, det.grasp_axes, det.shape_ratio = None, [], None
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
