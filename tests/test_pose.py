"""Point selection: 3D when usable, 2D otherwise, never a crash."""
import numpy as np

from coach.geometry import COCO
from coach.pose import SidePose

R = COCO["right"]


def keypoints(columns: int, conf=0.9) -> np.ndarray:
    kp = np.zeros((17, columns))
    for i, joint in enumerate(("shoulder", "elbow", "wrist")):
        kp[R[joint], :3] = [100 + 10 * i, 200 + 20 * i, conf]
        if columns == 6:
            kp[R[joint], 3:] = [0.1 * i, 0.2 * i, 0.3 * i]
    return kp


def test_3d_is_used_when_present():
    pts = SidePose(keypoints(6), "right", 0.5).points("shoulder", "elbow", "wrist")
    assert all(p.shape == (3,) for p in pts)


def test_falls_back_to_2d_without_depth_columns():
    pts = SidePose(keypoints(3), "right", 0.5).points("shoulder", "elbow", "wrist")
    assert all(p.shape == (2,) for p in pts)


def test_falls_back_to_2d_when_depth_is_all_zeros():
    kp = keypoints(6)
    kp[:, 3:] = 0.0
    pts = SidePose(kp, "right", 0.5).points("shoulder", "elbow", "wrist")
    assert all(p.shape == (2,) for p in pts)


def test_missing_joint_in_3d_falls_back_and_then_gives_up_cleanly():
    kp = keypoints(6)
    kp[R["wrist"], 2] = 0.1                      # wrist below the confidence threshold
    assert SidePose(kp, "right", 0.5).points("shoulder", "elbow", "wrist") is None
