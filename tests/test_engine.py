import numpy as np
import pytest

from coach import EXERCISES, RepCounter
from coach.geometry import joint_angle
from synthetic import angle_cycle, knee_pushup_frame, pushup_frame, squat_frame

FPS = 30


def run(counter, frames):
    return [counter.update(kp, i / FPS) for i, kp in enumerate(frames)]


def test_synthetic_skeletons_have_the_requested_angles():
    kp = pushup_frame(90)
    from coach.geometry import COCO
    L = COCO["left"]
    assert joint_angle(kp[L["shoulder"], :2], kp[L["elbow"], :2], kp[L["wrist"], :2]) == pytest.approx(90, abs=0.5)
    kp = squat_frame(95)
    R = COCO["right"]
    assert joint_angle(kp[R["hip"], :2], kp[R["knee"], :2], kp[R["ankle"], :2]) == pytest.approx(95, abs=0.5)


def test_joint_angle_is_never_above_180():
    assert joint_angle(np.array([0, 0]), np.array([1, 0]), np.array([2, 0.001])) <= 180


def test_counts_clean_pushups():
    c = RepCounter(EXERCISES["pushup"])
    run(c, [pushup_frame(a) for a in angle_cycle(5, top=150, bottom=80)])
    s = c.summary()
    assert s["reps"] == 5 and s["clean_reps"] == 5 and s["partial_reps"] == 0


def test_pushup_depth_is_not_judged_until_keypoints_are_validated():
    c = RepCounter(EXERCISES["pushup"])
    run(c, [pushup_frame(a) for a in angle_cycle(3, top=150, bottom=105)])
    assert c.summary()["reps"] == 3
    assert all("not_deep_enough" not in r.faults for r in c.reps)


def test_half_movements_are_partial_not_reps():
    """A movement of 32 degrees stays under the exercise's minimum range, so it is not a repetition."""
    c = RepCounter(EXERCISES["pushup"])
    run(c, [pushup_frame(a) for a in angle_cycle(4, top=150, bottom=118)])
    s = c.summary()
    assert s["reps"] == 0 and s["partial_reps"] == 4


def test_sagging_hips_are_detected():
    c = RepCounter(EXERCISES["pushup"])
    run(c, [pushup_frame(a, hip_drop=60) for a in angle_cycle(3, top=150, bottom=80)])
    assert c.summary()["reps"] == 3
    assert all("hips_sagging" in r.faults for r in c.reps)


def test_piked_hips_are_detected():
    c = RepCounter(EXERCISES["pushup"])
    run(c, [pushup_frame(a, hip_drop=-60) for a in angle_cycle(2, top=150, bottom=80)])
    assert all("hips_piked" in r.faults for r in c.reps)


def test_jitter_around_threshold_does_not_double_count():
    rng = np.random.default_rng(0)
    angles = np.array(angle_cycle(4, top=150, bottom=80)) + rng.normal(0, 6, size=len(angle_cycle(4, 170, 80)))
    c = RepCounter(EXERCISES["pushup"])
    run(c, [pushup_frame(a) for a in angles])
    assert c.summary()["reps"] == 4


def test_short_detection_gaps_are_tolerated():
    frames = [pushup_frame(a) for a in angle_cycle(3, top=150, bottom=80)]
    for i in range(0, len(frames), 7):          # lose the body on 1 frame out of 7
        frames[i] = None
    c = RepCounter(EXERCISES["pushup"])
    run(c, frames)
    assert c.summary()["reps"] == 3


def test_long_gap_abandons_the_current_rep():
    angles = angle_cycle(1, top=150, bottom=80)
    mid = len(angles) // 2                       # at the bottom
    frames = [pushup_frame(a) for a in angles[:mid]] + [None] * 60 + [pushup_frame(a) for a in angles[mid:]]
    c = RepCounter(EXERCISES["pushup"])
    run(c, frames)
    assert c.summary()["reps"] == 0


def test_low_confidence_keypoints_are_ignored():
    frames = [pushup_frame(a) for a in angle_cycle(2, top=150, bottom=80)]
    for kp in frames:
        kp[:, 2] = 0.2
    c = RepCounter(EXERCISES["pushup"], min_conf=0.5)
    res = run(c, frames)
    assert c.summary()["reps"] == 0 and not any(r.visible for r in res)


def test_works_on_either_side():
    c = RepCounter(EXERCISES["pushup"])
    run(c, [pushup_frame(a, side="right") for a in angle_cycle(3, top=150, bottom=80)])
    assert c.summary()["reps"] == 3


def test_squats_counted_and_depth_checked():
    c = RepCounter(EXERCISES["squat"])
    run(c, [squat_frame(a) for a in angle_cycle(4, top=175, bottom=85)])
    s = c.summary()
    assert (s["exercise"], s["reps"], s["clean_reps"], s["partial_reps"], s["faults"]) == ("squat", 4, 4, 0, {})
    c = RepCounter(EXERCISES["squat"])
    run(c, [squat_frame(a) for a in angle_cycle(2, top=175, bottom=112)])
    assert c.summary()["faults"] == {"not_deep_enough": 2}


def test_squat_forward_lean_is_detected():
    c = RepCounter(EXERCISES["squat"])
    run(c, [squat_frame(a, lean_deg=70) for a in angle_cycle(2, top=175, bottom=85)])
    assert c.summary()["faults"].get("torso_too_forward") == 2


def test_starting_at_the_bottom_does_not_count_a_half_rep():
    angles = angle_cycle(2, top=150, bottom=80)
    start = 15 + 22                                   # first frame at the bottom of rep 1
    c = RepCounter(EXERCISES["pushup"])
    run(c, [pushup_frame(a) for a in angles[start:]])
    assert c.summary()["reps"] == 1


def test_knee_pushup_with_straight_body_is_clean():
    c = RepCounter(EXERCISES["pushup"])
    run(c, [knee_pushup_frame(a) for a in angle_cycle(3, top=150, bottom=80)])
    assert c.summary()["clean_reps"] == 3


def test_knee_pushup_with_high_hips_is_flagged():
    c = RepCounter(EXERCISES["pushup"])
    run(c, [knee_pushup_frame(a, hip_rise=60) for a in angle_cycle(2, top=150, bottom=80)])
    assert all("hips_piked" in r.faults for r in c.reps)


def test_small_wobble_at_the_top_is_not_a_partial_rep():
    wobble = [150] * 10 + [138, 134, 136, 141, 150] + [150] * 10
    c = RepCounter(EXERCISES["pushup"])
    run(c, [pushup_frame(a) for a in wobble])
    assert c.summary()["partial_reps"] == 0


def test_no_form_feedback_while_setting_up_before_the_first_rep():
    frames = [pushup_frame(150, hip_drop=-60)] * 40          # hips high while getting into position
    c = RepCounter(EXERCISES["pushup"])
    res = run(c, frames)
    assert all(r.live_faults == [] for r in res)


def test_slight_hip_sag_within_tolerance_is_not_flagged():
    c = RepCounter(EXERCISES["pushup"])
    run(c, [pushup_frame(a, hip_drop=15) for a in angle_cycle(3, top=150, bottom=80)])
    assert c.summary()["clean_reps"] == 3
