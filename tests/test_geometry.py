import numpy as np
import pytest

from coach.geometry import joint_angle, tilt_above_line

A = np.array


def test_segment_parallel_to_floor_has_zero_tilt():
    assert tilt_above_line(A([100, 300]), A([160, 300]), A([0, 400]), A([500, 400])) == pytest.approx(0, abs=1e-6)


def test_shoulder_above_elbow_is_positive_whichever_way_the_floor_is_drawn():
    elbow, shoulder = A([100, 300]), A([140, 260])            # shoulder 45 degrees above (y is down)
    assert tilt_above_line(elbow, shoulder, A([0, 400]), A([500, 400])) == pytest.approx(45, abs=1e-6)
    assert tilt_above_line(elbow, shoulder, A([500, 400]), A([0, 400])) == pytest.approx(45, abs=1e-6)


def test_tilt_is_measured_against_a_tilted_floor():
    floor_a, floor_b = A([0, 400]), A([400, 300])            # floor rising to the right (camera tilt)
    d = (floor_b - floor_a) / np.linalg.norm(floor_b - floor_a)
    elbow = A([100, 200])
    assert tilt_above_line(elbow, elbow + 50 * d, floor_a, floor_b) == pytest.approx(0, abs=1e-6)


def test_straight_body_line_is_180():
    assert joint_angle(A([0, 0]), A([100, 20]), A([200, 40])) == pytest.approx(180, abs=1e-6)
