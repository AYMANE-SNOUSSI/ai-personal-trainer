"""Pose detectors. Each one turns a BGR frame into keypoints (17, 3) in COCO order, or None.

The coach engine only knows the COCO layout, so any pose model can drive it through an adapter.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

# MediaPipe BlazePose (33 landmarks) -> COCO (17 keypoints)
MEDIAPIPE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

MEDIAPIPE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                       "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task")


def body_size(kp: np.ndarray, min_conf: float = 0.3) -> float:
    """Diagonal of the box around the confident keypoints: how close the person is to the camera."""
    pts = kp[kp[:, 2] >= min_conf, :2]
    if len(pts) < 2:
        return 0.0
    return float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))


def facing_camera(kp: np.ndarray | None, min_conf: float = 0.5) -> bool | None:
    """True when the person faces the camera rather than standing side-on.

    Seen from the side, one shoulder hides the other, so the gap between left and right shoulder
    is small compared with the torso. Facing the camera, the shoulders are wide apart. Every angle
    here is measured in the image plane, so a front view distorts them and must be flagged.
    """
    if kp is None:
        return None
    ls, rs, lh = kp[5], kp[6], kp[11]
    if min(ls[2], rs[2], lh[2]) < min_conf:
        return None
    torso = np.hypot(*(ls[:2] - lh[:2]))
    if torso <= 0:
        return None
    return bool(abs(ls[0] - rs[0]) / torso > 0.45)


def joints_outside_frame(kp: np.ndarray | None, shape: tuple[int, int, int], min_conf: float = 0.5) -> int:
    """Number of confident keypoints that fall outside the image.

    MediaPipe extrapolates landmarks beyond the frame and still reports them as visible, so a hand
    raised above the top edge gets coordinates that were never seen. Counting them lets the caller
    warn the user instead of silently measuring an invented position.
    """
    if kp is None:
        return 0
    h, w = shape[:2]
    conf = kp[:, 2] >= min_conf
    outside = (kp[:, 0] < 0) | (kp[:, 0] > w) | (kp[:, 1] < 0) | (kp[:, 1] > h)
    return int((conf & outside).sum())


class YoloDetector:
    name = "yolo"

    def __init__(self, model: str = "yolov8n-pose.pt", imgsz: int = 640):
        from ultralytics import YOLO
        self.model, self.imgsz = YOLO(model), imgsz
        self.config = {"model": model, "imgsz": imgsz}

    def __call__(self, frame: np.ndarray) -> np.ndarray | None:
        result = self.model(frame, verbose=False, imgsz=self.imgsz)[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            return None
        xyxy = result.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        return result.keypoints.data[int(np.argmax(areas))].cpu().numpy()


def landmarks_to_coco(landmarks, width: int, height: int, world=None) -> np.ndarray:
    """MediaPipe landmarks -> keypoints in COCO order.

    Columns 0 to 2 are pixels and visibility, used for drawing and for anything defined in the
    image (the floor line, the frame edges). When `world` is given, columns 3 to 5 carry the
    model's 3D estimate in metres, centred on the hips. Angles computed from those do not change
    when the person turns away from a perfect side view, which a 2D angle cannot handle.
    """
    kp = np.zeros((17, 6 if world is not None else 3), dtype=float)
    for coco_i, mp_i in enumerate(MEDIAPIPE_TO_COCO):
        lm = landmarks[mp_i]
        kp[coco_i, :3] = [lm.x * width, lm.y * height, lm.visibility]
        if world is not None:
            w = world[mp_i]
            kp[coco_i, 3:] = [w.x, w.y, w.z]
    return kp


class MediaPipeDetector:
    """BlazePose via the MediaPipe Tasks API. Tracks a single person, which suits a home workout."""
    name = "mediapipe"

    def __init__(self, model_path: str = "pose_landmarker_full.task", max_poses: int = 1):
        if not Path(model_path).exists():
            variant = "full" if "full" in model_path else ("lite" if "lite" in model_path else "heavy")
            raise SystemExit(f"missing {model_path}. Download it with:\n"
                             f"  curl.exe -L -o {model_path} {MEDIAPIPE_MODEL_URL.format(variant=variant)}")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            # One pose is the fast path. Raise it when other people are in shot (a gym), then the
            # closest person is kept: each extra pose costs another run of the landmark model.
            num_poses=max_poses,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.config = {"model": model_path, "max_poses": max_poses}
        self._t0 = time.perf_counter()
        self._last_ms = -1

    def __call__(self, frame: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(1000 * (time.perf_counter() - self._t0)), self._last_ms + 1)  # must strictly increase
        self._last_ms = ts
        result = self.landmarker.detect_for_video(image, ts)
        if not result.pose_landmarks:
            return None
        h, w = frame.shape[:2]
        worlds = result.pose_world_landmarks or [None] * len(result.pose_landmarks)
        poses = [landmarks_to_coco(lms, w, h, world) for lms, world in zip(result.pose_landmarks, worlds)]
        return max(poses, key=body_size)
