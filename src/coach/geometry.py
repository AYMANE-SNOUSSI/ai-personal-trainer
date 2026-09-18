"""2D geometry on image coordinates (x to the right, y downwards)."""
from __future__ import annotations

import numpy as np

# COCO keypoint indices used by YOLO pose models
COCO = {
    "left":  {"shoulder": 5, "elbow": 7, "wrist": 9,  "hip": 11, "knee": 13, "ankle": 15},
    "right": {"shoulder": 6, "elbow": 8, "wrist": 10, "hip": 12, "knee": 14, "ankle": 16},
}


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b, in degrees, between segments b->a and b->c. Always in [0, 180]."""
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    cos = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def vertical_offset_from_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """How far p sits below the straight line a-b, as a fraction of |ab|.

    Positive: p is lower in the image than the line (e.g. hips sagging in a plank).
    Negative: p is higher (hips piked up). Works whichever way the person faces.
    """
    d = b - a
    length = np.linalg.norm(d)
    if length == 0 or abs(d[0]) < 1e-6:
        return float("nan")
    y_on_line = a[1] + (p[0] - a[0]) * d[1] / d[0]
    return float((p[1] - y_on_line) / length)


def angle_from_vertical(top: np.ndarray, bottom: np.ndarray) -> float:
    """Angle in degrees between the segment bottom->top and the upward vertical."""
    v = top - bottom
    n = np.linalg.norm(v)
    if n == 0:
        return float("nan")
    cos = np.clip(np.dot(v, np.array([0.0, -1.0])) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def tilt_above_line(a: np.ndarray, b: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """Signed angle in degrees of segment a->b relative to a reference line (e.g. the floor).

    Positive when b is further from the line than a on the "up" side of the image, 0 when a-b is
    parallel to the line. "Up" is the side of the line facing the top of the image, so the camera
    only needs to be roughly upright; a tilted floor is handled by the reference line itself.
    """
    d = line_end - line_start
    n = np.linalg.norm(d)
    seg = b - a
    ns = np.linalg.norm(seg)
    if n == 0 or ns == 0:
        return float("nan")
    normal = np.array([d[1], -d[0]]) / n
    if normal[1] > 0:                      # make the normal point towards the top of the image
        normal = -normal
    return float(np.degrees(np.arcsin(np.clip(np.dot(seg, normal) / ns, -1.0, 1.0))))
