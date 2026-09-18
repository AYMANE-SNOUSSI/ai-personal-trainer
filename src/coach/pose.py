"""Access to one side of a detected body, with confidence gating."""
from __future__ import annotations

import numpy as np

from .geometry import COCO


class SidePose:
    """Keypoints of one body side. `get` returns None for joints below the confidence threshold."""

    def __init__(self, keypoints: np.ndarray, side: str, min_conf: float):
        self.kp, self.side, self.min_conf = keypoints, side, min_conf

    def get(self, joint: str) -> np.ndarray | None:
        """Pixel position of the joint, or None when the detector is not confident enough."""
        row = self.kp[COCO[self.side][joint]]
        return np.array(row[:2], dtype=float) if row[2] >= self.min_conf else None

    def get_3d(self, joint: str) -> np.ndarray | None:
        """3D position in metres when the detector provides one, else None."""
        if self.kp.shape[1] < 6:
            return None
        row = self.kp[COCO[self.side][joint]]
        return np.array(row[3:6], dtype=float) if row[2] >= self.min_conf else None

    def points(self, *joints: str) -> list[np.ndarray] | None:
        """The joints in 3D if available, otherwise in 2D; None if any of them is missing.

        Angles between segments are meaningful in both cases, so a metric written against this
        works with a detector that estimates depth and with one that does not.
        """
        pts = [self.get_3d(j) for j in joints]
        missing = any(p is None for p in pts)
        # a detector that returns zeros for depth would collapse every joint onto one point
        degenerate = not missing and all(np.allclose(p, pts[0]) for p in pts[1:])
        if missing or degenerate:
            pts = [self.get(j) for j in joints]
        return None if any(p is None for p in pts) else pts

    def mean_conf(self, joints: list[str]) -> float:
        return float(np.mean([self.kp[COCO[self.side][j]][2] for j in joints]))


def pick_side(keypoints: np.ndarray, joints: list[str], current: str | None, margin: float = 0.1) -> str:
    """The side facing the camera, judged on the joints the exercise needs.

    Sticky: switching requires the other side to be better by `margin`, so the analysed side
    does not flicker from frame to frame.
    """
    conf = {s: SidePose(keypoints, s, 0).mean_conf(joints) for s in ("left", "right")}
    best = max(conf, key=conf.get)
    if current is None or conf[best] > conf[current] + margin:
        return best
    return current
