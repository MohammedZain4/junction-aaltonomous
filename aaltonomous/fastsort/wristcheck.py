"""Grip check features from the hand joints and the wrist camera (mount-agnostic).

The sim wrist cameras sit on the forearm and look at the hand from behind; the real ones (Drop 2) sit in the palm and
look forward between the fingers. So nothing here assumes where the object appears in the image:

  finger tracking error  a held object blocks the fingers short of the commanded closed posture; an empty hand
                         reaches it (max |actual - commanded| over the 7 hand joints)
  wrist colour blob      largest blob of the object's colour (same blob-level colour masks as the head camera),
                         as a fraction of the image
  blob stays put         a held object moves WITH the camera: between a frame early in the lift and one at its end
                         the world slides through the image, the held blob keeps its place and size

grip_features() only measures; thresholds are fitted on labelled sim picks (grasp_bench --wrist) before the policy
acts on them.
"""
from __future__ import annotations

import numpy as np


def _blob(perception, rgb, color):
    """(area fraction, centroid xy px, mask) of the largest blob of `color`, or (0, None, None)."""
    import cv2
    if rgb is None:
        return 0.0, None, None
    img = np.asarray(rgb)
    if img.ndim != 3 or img.size == 0 or not img.any():
        return 0.0, None, None
    m = perception.masks(img).get(color)
    if m is None or not m.any():
        return 0.0, None, None
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return 0.0, None, None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return float(st[k, cv2.CC_STAT_AREA]) / float(m.size), cen[k].astype(float), lab == k


def grip_features(perception, color, hand_q7, hand_cmd7, wrist_rgb, wrist_rgb_prev=None):
    f = {}
    try:
        e = np.asarray(hand_q7, float) - np.asarray(hand_cmd7, float)
        f["jerr"] = float(np.max(np.abs(e)))
        f["jerr_vec"] = np.round(e, 3).tolist()
    except Exception:
        f["jerr"], f["jerr_vec"] = None, None
    frac, c, m = _blob(perception, wrist_rgb, color)
    f["frac"], f["cxy"] = frac, (None if c is None else np.round(c).astype(int).tolist())
    if wrist_rgb_prev is not None:
        frac0, c0, m0 = _blob(perception, wrist_rgb_prev, color)
        f["frac_prev"] = frac0
        f["shift_px"] = None if (c is None or c0 is None) else float(np.linalg.norm(c - c0))
        f["iou"] = None if (m is None or m0 is None) else float((m & m0).sum()) / float((m | m0).sum())
    return f


def fmt(f):
    s = f"finger err {f['jerr']:.2f}" if f.get("jerr") is not None else "finger err ?"
    s += f", wrist blob {100 * f['frac']:.1f}%"
    if "frac_prev" in f:
        s += f" (earlier {100 * f['frac_prev']:.1f}%"
        s += f", shift {f['shift_px']:.0f}px" if f.get("shift_px") is not None else ""
        s += f", IoU {f['iou']:.2f})" if f.get("iou") is not None else ")"
    return s
