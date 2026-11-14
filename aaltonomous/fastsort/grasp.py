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

SQUARE_MIN, TRIANGLE_MAX = 0.88, 0.64     # hull/rect area ratio thresholds


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
        mode = os.environ.get("AALTO_TRI_AXIS", "side")
        axes = []
        for i in range(3):
            if mode == "altitude":   # flat side <-> opposite vertex: tested in sim, the vertex slips (1/4)
                v = 0.5 * (tri[(i + 1) % 3] + tri[(i + 2) % 3]) - tri[i]
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
    hue_prototypes = {"red": (0.0, 14.0), "yellow": (25.0, 43.0), "blue": (113.0, 111.0)}
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
        dist = np.stack([np.min(np.minimum(np.abs(H[..., None] - p), 180 - np.abs(H[..., None] - p)), axis=-1)
                         for p in protos])                      # (colours, h, w)
        near = np.argmin(dist, axis=0)
        out = {c: np.zeros(H.shape, np.uint8) for c in names}
        n, lab = cv2.connectedComponents(any_m, connectivity=8)
        for i in range(1, n):
            sel = lab == i
            counts = np.bincount(near[sel], minlength=len(names))
            k = int(counts.argmax())
            if counts[k] >= self.dominant_frac * counts.sum():
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
        except Exception:
            det.shape, det.grasp_axes = None, []
        return det


class ShapePlanner(Planner):
    max_axis_err_deg = 20.0
    axis_hint_weight = 1.0

    def _axis_hints(self, det, side):
        """Object pinch axes as finger directions: sign chosen to point forward (fingers can't point back at the
        robot); ordered by closeness to the arm's natural direction (forward, a little inward)."""
        axes = getattr(det, "grasp_axes", None) or []
        natural = np.array([1.0, -0.4 if side == "left" else 0.4])
        natural /= np.linalg.norm(natural)
        out = []
        for a in axes:
            a = np.asarray(a, float)
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
        if axes:
            saved_h, saved_w = self.kin.hint, self.kin.w_hint
            try:
                self.kin.w_hint = self.axis_hint_weight
                for a in axes:
                    self.kin.hint = np.r_[a, 0.0]
                    p = self._plan_side(det, side, bin_name, arm_q, hand_q, **kw)
                    if p.ok:
                        err = self._axis_err_deg(side, p, a)
                        if err <= self.max_axis_err_deg:
                            p.reason = f"{getattr(det, 'shape', '?')} axis grasp (err {err:.0f} deg)"
                            return p
            finally:
                self.kin.hint, self.kin.w_hint = saved_h, saved_w
        return super()._plan_side_hints(det, side, bin_name, arm_q, hand_q, **kw)
