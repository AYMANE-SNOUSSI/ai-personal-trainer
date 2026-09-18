"""Synthetic side-view skeletons with known joint angles, to test the engine without video."""
from __future__ import annotations

import numpy as np

from coach.geometry import COCO


def _blank() -> np.ndarray:
    kp = np.zeros((17, 3))
    return kp


def _set(kp, side, joint, xy, conf=0.9):
    kp[COCO[side][joint]] = [xy[0], xy[1], conf]


def pushup_frame(elbow_deg: float, hip_drop: float = 0.0, side: str = "left") -> np.ndarray:
    """Body horizontal, facing right. hip_drop > 0 moves the hip down (sagging), in pixels."""
    upper, fore = 60.0, 60.0
    wrist = np.array([300.0, 400.0])
    c = np.sqrt(upper**2 + fore**2 - 2 * upper * fore * np.cos(np.radians(elbow_deg)))
    shoulder = wrist + np.array([0.0, -c])                    # shoulder straight above the hand
    # elbow: triangle apex, placed behind the arm line
    along = (fore**2 - upper**2 + c**2) / (2 * c)
    h = np.sqrt(max(fore**2 - along**2, 0.0))
    elbow = wrist + np.array([-h, -along])
    ankle = shoulder + np.array([-400.0, 400.0 - shoulder[1] + 0.0])  # feet on the floor
    ankle[1] = 400.0
    hip = shoulder + (ankle - shoulder) * 0.45 + np.array([0.0, hip_drop])
    knee = shoulder + (ankle - shoulder) * 0.72
    kp = _blank()
    for j, xy in dict(shoulder=shoulder, elbow=elbow, wrist=wrist, hip=hip, knee=knee, ankle=ankle).items():
        _set(kp, side, j, xy)
    return kp


def squat_frame(knee_deg: float, lean_deg: float = 20.0, side: str = "right") -> np.ndarray:
    """Shin vertical; thigh set so the hip-knee-ankle angle equals knee_deg."""
    thigh, shin, torso = 90.0, 90.0, 110.0
    ankle = np.array([300.0, 500.0])
    knee = ankle + np.array([0.0, -shin])
    # direction knee->ankle is straight down; rotate it by knee_deg to get knee->hip
    a = np.radians(knee_deg)
    down = np.array([0.0, 1.0])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    hip = knee + thigh * (rot @ down)
    l = np.radians(lean_deg)
    shoulder = hip + torso * np.array([np.sin(l), -np.cos(l)])
    kp = _blank()
    for j, xy in dict(shoulder=shoulder, hip=hip, knee=knee, ankle=ankle).items():
        _set(kp, side, j, xy)
    return kp


def angle_cycle(n_reps: int, top: float, bottom: float, fps: int = 30, rep_s: float = 1.5,
                hold_s: float = 0.5) -> list[float]:
    """Top hold, then n cosine-shaped reps between top and bottom, then top hold."""
    hold = [top] * int(hold_s * fps)
    k = np.arange(int(rep_s * fps))
    one = top - (top - bottom) * (1 - np.cos(2 * np.pi * k / len(k))) / 2
    return hold + list(one) * n_reps + hold


def knee_pushup_frame(elbow_deg: float, hip_rise: float = 0.0, side: str = "left") -> np.ndarray:
    """Knees on the floor, shins and feet raised behind. hip_rise > 0 lifts the hips (piking)."""
    kp = pushup_frame(elbow_deg, side=side)
    L = COCO[side]
    shoulder = kp[L["shoulder"], :2]
    knee = np.array([shoulder[0] - 250.0, 400.0])
    hip = shoulder + (knee - shoulder) * 0.55 + np.array([0.0, -hip_rise])
    ankle = knee + np.array([-90.0, -90.0])           # feet up in the air
    for j, xy in dict(hip=hip, knee=knee, ankle=ankle).items():
        kp[L[j]] = [xy[0], xy[1], 0.9]
    return kp


def arm_exercise_frames(angles, upper_arm=None, torso=None, overhead=False, side="right"):
    """One frame per elbow angle; `upper_arm` and `torso` may be per-frame sequences."""
    n = len(angles)
    ua = [0.0] * n if upper_arm is None else list(upper_arm)
    to = [0.0] * n if torso is None else list(torso)
    return [standing_arm_frame(a, ua[i], to[i], overhead=overhead, side=side) for i, a in enumerate(angles)]


def standing_arm_frame(elbow_deg: float, upper_arm_deg: float = 0.0, torso_deg: float = 0.0,
                       overhead: bool = False, side: str = "right") -> np.ndarray:
    """Standing body seen from the side, with a controlled elbow angle.

    upper_arm_deg tilts the upper arm away from vertical (elbow drifting forward),
    torso_deg leans the whole torso. overhead places the upper arm above the shoulder (press).
    """
    upper, fore, torso, thigh, shin = 70.0, 70.0, 150.0, 120.0, 120.0
    ankle = np.array([300.0, 700.0])
    knee = ankle + np.array([0.0, -shin])
    hip = knee + np.array([0.0, -thigh])
    l = np.radians(torso_deg)
    shoulder = hip + torso * np.array([np.sin(l), -np.cos(l)])
    a = np.radians(upper_arm_deg)
    direction = np.array([np.sin(a), -np.cos(a)]) if overhead else np.array([np.sin(a), np.cos(a)])
    elbow = shoulder + upper * direction
    # rotate the forearm away from the upper arm by (180 - elbow_deg)
    turn = np.radians(180.0 - elbow_deg)
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    wrist = elbow + fore * (rot @ direction)
    kp = _blank()
    for j, xy in dict(shoulder=shoulder, elbow=elbow, wrist=wrist, hip=hip, knee=knee, ankle=ankle).items():
        _set(kp, side, j, xy)
    return kp
