"""The point of using the model's 3D estimate: angles must not depend on the camera angle."""
import numpy as np
import pytest

from coach import EXERCISES, RepCounter
from coach.geometry import COCO
from coach.pose import SidePose

SIDE = COCO["right"]


def arm_keypoints(bend_deg: float, turn_deg: float) -> np.ndarray:
    """A right arm bent by `bend_deg` in the sagittal plane, the body turned `turn_deg` from side-on.

    Columns 0 to 2 are what a camera sees (the 3D points projected, losing depth), columns 3 to 5
    are the metric 3D positions a model like MediaPipe estimates.
    """
    a = np.radians(bend_deg)
    shoulder = np.array([0.0, 0.0, 0.0])
    elbow = shoulder + np.array([0.0, 0.30, 0.0])                      # straight down
    forearm = np.array([np.sin(a), np.cos(a), 0.0]) * 0.28             # rotated in the sagittal plane
    wrist = elbow + forearm
    hip = shoulder + np.array([0.0, 0.50, 0.0])

    turn = np.radians(turn_deg)
    rot = np.array([[np.cos(turn), 0, -np.sin(turn)], [0, 1, 0], [np.sin(turn), 0, np.cos(turn)]])
    kp = np.zeros((17, 6))
    for joint, point in dict(shoulder=shoulder, elbow=elbow, wrist=wrist, hip=hip).items():
        p = rot @ point
        kp[SIDE[joint], :3] = [300 + 400 * p[0], 200 + 400 * p[1], 0.95]   # camera projection
        kp[SIDE[joint], 3:] = p
    return kp


def primary(kp: np.ndarray) -> float:
    return RepCounter(EXERCISES["biceps_curl"])._primary_angle(SidePose(kp, "right", 0.5))


def test_3d_angle_is_the_same_from_any_viewpoint():
    values = [primary(arm_keypoints(120, turn)) for turn in (0, 30, 60, 85)]
    assert all(v == pytest.approx(60, abs=1.0) for v in values)


def test_the_flat_image_angle_would_have_been_wrong():
    """Same skeletons without the 3D columns: the measured angle collapses as the person turns."""
    flat = [primary(arm_keypoints(120, turn)[:, :3]) for turn in (0, 60, 85)]
    assert flat[0] == pytest.approx(60, abs=1.0)
    assert flat[1] == pytest.approx(41, abs=2.0)      # 19 degrees off at a three-quarter view
    assert flat[2] < 15                               # nearly facing the camera: meaningless
