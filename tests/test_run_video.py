"""End-to-end test of the video pipeline with a fake detector: no model, no real video."""
import json

import cv2
import numpy as np

import run_video
from synthetic import angle_cycle, pushup_frame


def test_pipeline_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    angles = angle_cycle(3, top=150, bottom=80)
    video = tmp_path / "blank.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for _ in angles:
        w.write(np.zeros((480, 640, 3), np.uint8))
    w.release()

    frames = iter([pushup_frame(a) for a in angles])
    summary = run_video.run(str(video), "pushup", "t", show=False, detector=lambda f: next(frames))

    assert summary["reps"] == 3
    out = tmp_path / "runs" / "t"
    assert (out / "annotated.mp4").stat().st_size > 0
    assert json.loads((out / "summary.json").read_text())["reps"] == 3
    assert len((out / "frames.csv").read_text().splitlines()) == len(angles) + 1


def test_pipeline_handles_keypoints_that_carry_3d_columns(tmp_path, monkeypatch):
    """MediaPipe rows are (x, y, confidence, X, Y, Z); nothing downstream may assume three columns."""
    monkeypatch.chdir(tmp_path)
    angles = angle_cycle(2, top=150, bottom=80)
    video = tmp_path / "blank.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for _ in angles:
        w.write(np.zeros((480, 640, 3), np.uint8))
    w.release()

    def detector_with_depth(frame, frames=iter([pushup_frame(a) for a in angles])):
        kp = next(frames)
        world = np.hstack([kp[:, :2] / 400.0, np.zeros((17, 1))])    # same pose, in metres
        return np.hstack([kp, world])

    summary = run_video.run(str(video), "pushup", "3d", show=False, detector=detector_with_depth)
    assert summary["reps"] == 2


def test_pipeline_survives_frames_with_no_person(tmp_path, monkeypatch):
    """A frame where nothing is detected must not crash any of the per-frame checks."""
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "blank.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(10):
        w.write(np.zeros((240, 320, 3), np.uint8))
    w.release()

    summary = run_video.run(str(video), "pushup", "empty", show=False, detector=lambda f: None)
    assert summary["reps"] == 0 and summary["visible_frames"] == 0
