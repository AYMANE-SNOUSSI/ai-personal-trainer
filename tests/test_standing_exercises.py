"""Counting and form rules for the standing exercises, on synthetic skeletons."""
import pytest

from coach import EXERCISES, RepCounter
from coach.geometry import COCO, joint_angle
import numpy as np

from synthetic import angle_cycle, arm_exercise_frames, standing_arm_frame

FPS = 30


def run(counter, frames):
    return [counter.update(kp, i / FPS) for i, kp in enumerate(frames)]


def test_synthetic_standing_skeleton_has_the_requested_elbow_angle():
    kp = standing_arm_frame(75)
    R = COCO["right"]
    assert joint_angle(kp[R["shoulder"], :2], kp[R["elbow"], :2], kp[R["wrist"], :2]) == pytest.approx(75, abs=0.5)


def test_biceps_curls_are_counted():
    c = RepCounter(EXERCISES["biceps_curl"])
    run(c, [standing_arm_frame(a) for a in angle_cycle(6, top=160, bottom=45)])
    assert c.summary()["reps"] == 6 and c.summary()["clean_reps"] == 6


def test_curl_on_an_inclined_bench_is_not_flagged():
    """Upper arm resting forward on a pad and a leaning torso are correct, as long as both stay put."""
    angles = angle_cycle(3, top=160, bottom=45)
    c = RepCounter(EXERCISES["biceps_curl"])
    run(c, arm_exercise_frames(angles, upper_arm=[45.0] * len(angles), torso=[35.0] * len(angles)))
    assert c.summary()["clean_reps"] == 3


def test_curl_with_a_swinging_upper_arm_is_flagged():
    angles = angle_cycle(3, top=160, bottom=45)
    swing = 40 * (1 - np.cos(np.linspace(0, 6 * np.pi, len(angles)))) / 2      # arm lifts on each rep
    c = RepCounter(EXERCISES["biceps_curl"])
    run(c, arm_exercise_frames(angles, upper_arm=swing))
    assert all("elbow_drifting" in r.faults for r in c.reps)


def test_curl_with_body_swing_is_flagged():
    angles = angle_cycle(3, top=160, bottom=45)
    swing = 30 * (1 - np.cos(np.linspace(0, 6 * np.pi, len(angles)))) / 2
    c = RepCounter(EXERCISES["biceps_curl"])
    run(c, arm_exercise_frames(angles, torso=swing))
    # not every repetition: a rep whose window happens to sit on the peak of the swing sees little
    # movement within itself, which is a real limit of judging a rep in isolation
    assert sum("torso_swinging" in r.faults for r in c.reps) >= 2


def test_shoulder_presses_are_counted():
    c = RepCounter(EXERCISES["shoulder_press"])
    run(c, [standing_arm_frame(a, overhead=True) for a in angle_cycle(5, top=170, bottom=80)])
    assert c.summary()["reps"] == 5


def test_press_while_leaning_back_is_flagged():
    angles = angle_cycle(3, top=170, bottom=80)
    lean = 25 * (1 - np.cos(np.linspace(0, 6 * np.pi, len(angles)))) / 2
    c = RepCounter(EXERCISES["shoulder_press"])
    run(c, arm_exercise_frames(angles, torso=lean, overhead=True))
    assert all("leaning_back" in r.faults for r in c.reps)


def test_seated_press_with_a_steady_torso_is_clean():
    angles = angle_cycle(3, top=170, bottom=80)
    c = RepCounter(EXERCISES["shoulder_press"])
    run(c, arm_exercise_frames(angles, torso=[20.0] * len(angles), overhead=True))
    assert c.summary()["clean_reps"] == 3


def test_tiny_movements_are_not_counted_as_repetitions():
    """Under the exercise's minimum range (40 degrees), nothing counts, however many times it repeats."""
    c = RepCounter(EXERCISES["biceps_curl"])
    run(c, [standing_arm_frame(a) for a in angle_cycle(4, top=160, bottom=130)])
    assert c.summary()["reps"] == 0


def test_a_short_but_consistent_range_is_counted():
    """Measured ranges differ between people, cameras and detectors: on one clip a fully extended
    arm read 141 degrees and a curl bottomed out at 91, a 50-degree range that fixed thresholds
    missed entirely. Thresholds sit inside the range the person actually produces."""
    c = RepCounter(EXERCISES["biceps_curl"])
    run(c, [standing_arm_frame(a) for a in angle_cycle(5, top=141, bottom=91)])
    assert c.summary()["reps"] == 5
