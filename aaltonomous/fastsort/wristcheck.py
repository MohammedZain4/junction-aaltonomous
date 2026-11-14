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
    f["wrist_valid"] = wrist_rgb is not None and np.asarray(wrist_rgb).size > 0 and bool(np.asarray(wrist_rgb).any())
    frac, c, m = _blob(perception, wrist_rgb, color)
    f["frac"], f["cxy"] = frac, (None if c is None else np.round(c).astype(int).tolist())
    if wrist_rgb_prev is not None:
        frac0, c0, m0 = _blob(perception, wrist_rgb_prev, color)
        f["frac_prev"] = frac0
        f["shift_px"] = None if (c is None or c0 is None) else float(np.linalg.norm(c - c0))
        f["iou"] = None if (m is None or m0 is None) else float((m & m0).sum()) / float((m | m0).sum())
    # the other colours too (colour double-check): largest blob + whether it stays put
    f["others"] = {}
    for oc in getattr(perception, "hue_prototypes", {}):
        if oc == color:
            continue
        fr, _, mo = _blob(perception, wrist_rgb, oc)
        iou = None
        if wrist_rgb_prev is not None and mo is not None:
            _, _, mo0 = _blob(perception, wrist_rgb_prev, oc)
            iou = None if mo0 is None else float((mo & mo0).sum()) / float((mo | mo0).sum())
        f["others"][oc] = (fr, iou)
    return f


def head_features(perception, color, head_rgb, head_depth, xy, r_table=0.035, r_up=0.09):
    """Head-camera grip cue, independent of the wrist mount: at the end of the lift, count head pixels of the object's
    colour whose 3D point is (a) still at table height within r_table of the grasp xy (object left behind) and (b)
    lifted well above the table within r_up (object in the hand). The hand may hide the object from the head camera,
    so only (a) is used as evidence of a miss; (b) is logged."""
    out = {"head_table_px": None, "head_up_px": None}
    try:
        if head_rgb is None or head_depth is None:
            return out
        m = perception.masks(np.asarray(head_rgb)).get(color)
        if m is None:
            return out
        d = np.asarray(head_depth, float)
        dlo, dhi = perception.cfg["depth_range_mm"]
        vv, uu = np.nonzero((m > 0) & (d >= dlo) & (d <= dhi))
        if len(uu) == 0:
            out["head_table_px"] = out["head_up_px"] = 0
            return out
        P = perception.cal.pixel_to_pelvis(perception.cam, uu, vv, d[vv, uu])
        P[:, :2] += getattr(perception, "xy_offset", 0.0)
        h = (P[:, 2] - perception.scene.counter_z) / perception.height
        r = np.linalg.norm(P[:, :2] - np.asarray(xy, float)[:2], axis=1)
        out["head_table_px"] = int(((r < r_table) & (h >= 0.25) & (h <= 1.4)).sum())
        out["head_up_px"] = int(((r < r_up) & (h > 1.8)).sum())
    except Exception as e:
        out["head_err"] = repr(e)
    return out


def fmt(f):
    s = f"finger err {f['jerr']:.2f}" if f.get("jerr") is not None else "finger err ?"
    s += f", wrist blob {100 * f['frac']:.1f}%"
    if "frac_prev" in f:
        s += f" (earlier {100 * f['frac_prev']:.1f}%"
        s += f", shift {f['shift_px']:.0f}px" if f.get("shift_px") is not None else ""
        s += f", IoU {f['iou']:.2f})" if f.get("iou") is not None else ")"
    for oc, (fr, iou) in (f.get("others") or {}).items():
        if fr >= 0.01:
            s += f"; {oc} {100 * fr:.1f}%" + (f" IoU {iou:.2f}" if iou is not None else "")
    if f.get("head_table_px") is not None:
        s += f"; head: {f['head_table_px']} px left on table, {f['head_up_px']} px lifted"
    return s
