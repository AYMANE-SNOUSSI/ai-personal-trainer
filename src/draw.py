"""OpenCV overlay for the coach engine."""
from __future__ import annotations

import cv2
import numpy as np

from coach import FrameResult, RepCounter
from coach.feedback import Message
from coach.geometry import COCO

GREEN, RED, BLUE, ORANGE, WHITE, GREY = (80, 200, 80), (60, 60, 230), (230, 140, 40), (0, 165, 255), (255, 255, 255), (170, 170, 170)
BODY_CHAIN = ["wrist", "elbow", "shoulder", "hip", "knee", "ankle"]


def draw(frame: np.ndarray, kp: np.ndarray | None, res: FrameResult, counter: RepCounter,
         message: Message | None) -> None:
    h, w = frame.shape[:2]
    scale = w / 960
    if kp is not None and res.side is not None:
        idx = COCO[res.side]
        colour = RED if res.live_faults else GREEN
        pts = [(int(kp[idx[j], 0]), int(kp[idx[j], 1])) if kp[idx[j], 2] >= counter.min_conf else None
               for j in BODY_CHAIN]
        for a, b in zip(pts, pts[1:]):
            if a is not None and b is not None:
                cv2.line(frame, a, b, colour, max(2, int(4 * scale)))
        primary = [pts[BODY_CHAIN.index(j)] for j in counter.spec.primary]
        if all(p is not None for p in primary):
            arm_colour = ORANGE if res.phase == "bottom" else BLUE
            cv2.line(frame, primary[0], primary[1], arm_colour, max(2, int(5 * scale)))
            cv2.line(frame, primary[1], primary[2], arm_colour, max(2, int(5 * scale)))
            if res.angle is not None:
                cv2.putText(frame, f"{res.angle:.0f}", (primary[1][0] + 8, primary[1][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, WHITE, max(1, int(2 * scale)))

    cv2.rectangle(frame, (0, 0), (int(330 * scale), int(95 * scale)), (0, 0, 0), -1)
    cv2.putText(frame, str(res.reps), (int(15 * scale), int(75 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                2.2 * scale, WHITE, max(2, int(4 * scale)))
    cv2.putText(frame, f"{counter.spec.name.upper()}  {res.phase}", (int(110 * scale), int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, GREY, max(1, int(2 * scale)))
    if not res.visible:
        cv2.putText(frame, "body not visible", (int(110 * scale), int(75 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, ORANGE, max(1, int(2 * scale)))
    if message is not None:
        col = GREEN if message.kind == "good" else (ORANGE if message.kind == "info" else RED)
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, max(2, int(2 * scale))
        margin, line_h = int(15 * scale), int(38 * scale)
        lines, line = [], ""
        for word in message.text.split():            # wrap to the frame width
            trial = f"{line} {word}".strip()
            if cv2.getTextSize(trial, font, fs, th)[0][0] > w - 2 * margin and line:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        top = h - line_h * len(lines) - int(15 * scale)
        cv2.rectangle(frame, (0, top - int(10 * scale)), (w, h), (0, 0, 0), -1)
        for k, text in enumerate(lines):
            cv2.putText(frame, text, (margin, top + line_h * k + int(28 * scale)), font, fs, col, th)
