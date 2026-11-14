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
