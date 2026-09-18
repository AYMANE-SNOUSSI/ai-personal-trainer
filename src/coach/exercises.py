"""Exercise definitions. Adding an exercise means adding one ExerciseSpec, not new code.

Push-up criteria follow the standard test protocols (FITNESSGRAM 90-degree push-up, President's
Challenge, US Army APFT): body in a straight line from head to heels, lowered until the elbows
reach about 90 degrees with the upper arms parallel to the floor.
Squat criteria are NOT yet checked against a reference and must not be presented as validated.

Thresholds are expressed in what the pose model measures, which differs from true joint angles
(a straight arm reads 150 to 175 degrees depending on camera angle). They still need calibration
on labelled recordings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .geometry import angle_from_vertical, joint_angle, tilt_above_line, vertical_offset_from_line
from .pose import SidePose

Metric = Callable[[SidePose], float | None]


def angle_metric(a: str, b: str, c: str) -> Metric:
    """Angle at joint b, measured in 3D when the detector estimates depth."""
    def f(p: SidePose) -> float | None:
        pts = p.points(a, b, c)
        return None if pts is None else joint_angle(*pts)
    return f


def _body_end(p: SidePose) -> np.ndarray | None:
    """Lower contact point of the body line: ankle, or knee for knee push-ups (feet lifted)."""
    s, k, a = p.get("shoulder"), p.get("knee"), p.get("ankle")
    if k is not None and a is not None and s is not None:
        return k if a[1] < k[1] - 0.15 * np.linalg.norm(s - k) else a
    return a if a is not None else k


def upper_arm_vs_floor(p: SidePose) -> float | None:
    """Standard's depth criterion measured literally: shoulder height above the elbow, as an angle
    to the floor line (wrist to feet or knees). 0 = upper arm parallel to the floor; positive =
    shoulder still above elbow level; negative = lower. Independent of how a detector places the
    wrist along the forearm, unlike the elbow angle."""
    s, e, w, end = p.get("shoulder"), p.get("elbow"), p.get("wrist"), _body_end(p)
    if s is None or e is None or w is None or end is None:
        return None
    v = tilt_above_line(e, s, w, end)
    return None if np.isnan(v) else v


def body_line_angle(p: SidePose) -> float | None:
    """Shoulder-hip-(ankle or knee) angle: 180 for a perfectly straight body. 3D when available."""
    end2d = _body_end(p)
    if end2d is None:
        return None
    end_joint = "knee" if p.get("knee") is not None and np.allclose(end2d, p.get("knee")) else "ankle"
    pts = p.points("shoulder", "hip", end_joint)
    return None if pts is None else joint_angle(*pts)


def body_bend(p: SidePose) -> float | None:
    """How far the body is from a straight line, in degrees, signed by direction.

    180 - (shoulder-hip-ankle angle), positive when the hips drop below the line, negative when
    they rise above it. Chosen because YOLO and MediaPipe agreed on it within 2 to 5 degrees on
    the same clips, unlike depth measurements that differed by up to 48 degrees.
    """
    angle, offset = body_line_angle(p), hip_offset(p)
    if angle is None or offset is None:
        return None
    return (180.0 - angle) * (1.0 if offset >= 0 else -1.0)


def hip_offset(p: SidePose) -> float | None:
    """Hip position relative to the straight line the body should form.

    Full push-up: shoulder to ankle. Knee push-up: shoulder to knee, because the feet are lifted
    and the ankle is no longer on that line. A knee push-up is recognised when the ankle sits
    clearly higher in the image than the knee.
    """
    s, h, k, a = p.get("shoulder"), p.get("hip"), p.get("knee"), p.get("ankle")
    if s is None or h is None:
        return None
    if k is not None and a is not None:
        on_knees = a[1] < k[1] - 0.15 * np.linalg.norm(s - k)
        end = k if on_knees else a
    else:
        end = k if a is None else a
    if end is None:
        return None
    v = vertical_offset_from_line(h, s, end)
    return None if np.isnan(v) else v


def torso_lean(p: SidePose) -> float | None:
    """Angle of the torso (hip to shoulder) away from vertical; 0 = upright."""
    s, h = p.get("shoulder"), p.get("hip")
    return None if s is None or h is None else angle_from_vertical(s, h)


def upper_arm_from_vertical(p: SidePose) -> float | None:
    """Angle of the upper arm (elbow to shoulder) away from vertical; 0 = elbow under the shoulder."""
    s, e = p.get("shoulder"), p.get("elbow")
    return None if s is None or e is None else angle_from_vertical(s, e)


def upper_arm_vs_torso(p: SidePose) -> float | None:
    """Angle between the upper arm and the torso: 0 = arm along the body, 90 = arm out in front.

    Measured against the torso rather than against vertical, so it means the same thing standing,
    seated, or leaning on a preacher bench, where the whole body is inclined.
    """
    pts = p.points("elbow", "shoulder", "hip")
    if pts is None:
        return None
    v = joint_angle(*pts)
    return None if np.isnan(v) else v


@dataclass(frozen=True)
class FormRule:
    name: str
    message: str
    metric: Metric
    is_fault: Callable[[float], bool]
    joints: tuple[str, ...]
    min_fraction: float = 0.3      # fault if present on at least this share of the rep's frames


@dataclass(frozen=True)
class RepRule:
    """A check on the whole repetition: it looks at how a measurement moved, not at one frame."""
    name: str
    message: str
    observation: str
    is_fault: Callable[[dict], bool]


@dataclass(frozen=True)
class ExerciseSpec:
    name: str
    primary: tuple[str, str, str]  # joint triplet whose angle drives the repetition
    down_below: float              # fallback angle for the bottom, before the person's range is known
    up_above: float                # fallback angle for the top (hysteresis with down_below)
    depth_target: float | None     # min angle a rep should reach; None = depth not judged
    depth_message: str
    min_range: float = 40.0        # a movement smaller than this is not treated as a repetition
    rules: tuple[FormRule, ...] = field(default_factory=tuple)
    rep_rules: tuple[RepRule, ...] = field(default_factory=tuple)
    # measurements recorded for analysis and calibration; they do not affect any verdict
    observations: dict = field(default_factory=dict)

    @property
    def joints(self) -> list[str]:
        names = list(self.primary)
        for r in self.rules:
            names += [j for j in r.joints if j not in names]
        return names


PUSHUP = ExerciseSpec(
    name="pushup",
    primary=("shoulder", "elbow", "wrist"),
    # up_above measured on a real clip: a fully extended arm read 147-150 degrees from YOLO
    # keypoints, so 150 was never reached. 140 keeps a 30-degree hysteresis with down_below.
    down_below=110, up_above=140,
    # Depth is NOT judged: on the same clips, YOLO and MediaPipe disagreed by up to 48 degrees on
    # where the upper arm is at the bottom. A correction that depends on the detector must not
    # be given to a user until keypoint accuracy has been checked against annotated frames.
    depth_target=None, depth_message="Go a little lower. Bend your elbows to 90 degrees.",
    rules=(
        # Provisional tolerance (25 degrees of bend): correct clips measured 167-179 degrees,
        # the knee push-up clip with visibly raised hips 117-133, with both detectors.
        # Three clips are not enough to fix this value; it must be re-checked on more videos.
        FormRule("hips_sagging", "Your hips are dropping. Tighten your stomach and keep your body straight.",
                 body_bend, lambda v: v > 25, ("shoulder", "hip", "knee")),
        FormRule("hips_piked", "Your hips are too high. Lower them so your body makes a straight line.",
                 body_bend, lambda v: v < -25, ("shoulder", "hip", "knee")),
    ),
    observations={"upper_arm_vs_floor": upper_arm_vs_floor, "body_line_angle": body_line_angle,
                  "body_bend": body_bend},
)

SQUAT = ExerciseSpec(
    name="squat",
    primary=("hip", "knee", "ankle"),
    down_below=120, up_above=160,
    depth_target=100, depth_message="Go a bit lower. Try to bring your thighs level with the floor.",
    rules=(
        FormRule("torso_too_forward", "Keep your chest up.", torso_lean,
                 lambda v: v > 55, ("shoulder", "hip")),
    ),
    observations={"torso_lean": torso_lean, "knee_angle_check": angle_metric("hip", "knee", "ankle")},
)

# Standing exercises. The movement criteria below are the usual coaching cues (keep the elbows at
# your sides, do not swing the torso, control the range). They are NOT taken from a measurement
# protocol like the push-up test, and the tolerances are first guesses: counting is the part to
# trust today, and each rule has to be checked on annotated clips before it is presented as advice.
BICEPS_CURL = ExerciseSpec(
    name="biceps_curl",
    primary=("shoulder", "elbow", "wrist"),
    down_below=70, up_above=140,
    depth_target=None, depth_message="Curl a little higher.",
    rules=(),
    rep_rules=(
        # The arm is not always vertical: on a preacher or incline bench it rests forward on a pad,
        # which is correct. The fault is the upper arm MOVING during the curl, so both rules below
        # look at how much a measurement changed within the repetition, not at its absolute value.
        RepRule("elbow_drifting", "Keep your elbow still instead of swinging your upper arm.",
                "upper_arm_vs_torso", lambda st: st["max"] - st["min"] > 25),
        # A curl on an incline or preacher bench keeps the torso tilted the whole time, which is
        # correct. What is not correct is the torso MOVING to help the arm, so the fault is the
        # swing during the rep, not the lean itself.
        RepRule("torso_swinging", "Keep your body still and let your arm do the work.",
                "torso_lean", lambda st: st["max"] - st["min"] > 20),
    ),
    observations={"upper_arm_vs_torso": upper_arm_vs_torso, "torso_lean": torso_lean},
)

SHOULDER_PRESS = ExerciseSpec(
    name="shoulder_press",
    primary=("shoulder", "elbow", "wrist"),
    down_below=100, up_above=150,
    depth_target=None, depth_message="Press all the way up.",
    rules=(),
    rep_rules=(
        RepRule("leaning_back", "Keep your torso steady instead of leaning back to push the weight.",
                "torso_lean", lambda st: st["max"] - st["min"] > 15),
    ),
    observations={"torso_lean": torso_lean, "upper_arm_vs_torso": upper_arm_vs_torso},
)

EXERCISES: dict[str, ExerciseSpec] = {s.name: s for s in (PUSHUP, SQUAT, BICEPS_CURL, SHOULDER_PRESS)}
