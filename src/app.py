"""Streamlit front end: same coach engine, live from a webcam.

    streamlit run src/app.py
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

from coach import EXERCISES, RepCounter
from coach.feedback import LiveCoach
from detectors import MediaPipeDetector, YoloDetector, facing_camera, joints_outside_frame
from draw import draw

st.set_page_config(page_title="AI Personal Trainer", layout="wide")

PROCESS_WIDTH = 960          # pose models are run on a downscaled frame: a 4K frame is many times slower
DISPLAY_EVERY = 2            # every frame is analysed; only every other one is sent to the browser
DISPLAY_WIDTH = 720          # the browser preview is resized and JPEG-encoded: full-size PNG
JPEG_QUALITY = 75            # was costing more time than the pose model itself


@st.cache_resource
def load_detector(kind: str, imgsz: int, mp_model: str, max_poses: int):
    return MediaPipeDetector(mp_model, max_poses=max_poses) if kind == "mediapipe" else YoloDetector(imgsz=imgsz)


def show_summary(counter: RepCounter) -> None:
    summary = counter.summary()
    messages = counter.messages()
    st.subheader("Session summary")
    a, b, c = st.columns(3)
    a.metric("Repetitions", summary["reps"])
    b.metric("Clean repetitions", summary["clean_reps"])
    c.metric("Half repetitions", summary["partial_reps"])
    if summary["faults"]:
        st.write("What to work on:")
        for name, count in sorted(summary["faults"].items(), key=lambda kv: -kv[1]):
            st.write(f"- {messages[name]} ({count} of {summary['reps']} repetitions)")
    elif summary["reps"]:
        st.success("No form problem detected on any repetition.")
    with st.expander("Details of each repetition"):
        st.caption("Deepest point: the smallest joint angle reached in the repetition. "
                   "A smaller number means you went further down.")
        st.table([{"#": r.index, "when": f"{r.start:.1f}s to {r.end:.1f}s",
                   "deepest point": f"{r.min_angle:.0f}\u00b0",
                   "form": "OK" if r.clean else ", ".join(messages[f].rstrip(".") for f in r.faults)}
                  for r in counter.reps])


st.title("AI Personal Trainer")

with st.sidebar:
    exercise = st.selectbox("Exercise", sorted(EXERCISES))
    source_kind = st.radio("Video source", ["Webcam", "Video file"])
    upload = st.file_uploader("Your video", type=["mp4", "mov", "avi"]) if source_kind == "Video file" else None
    with st.expander("Advanced"):
        kind = st.radio("Pose detector", ["mediapipe", "yolo"],
                        help="MediaPipe is about six times faster on CPU and tracks one person.")
        imgsz = st.select_slider("YOLO input size", [320, 480, 640], value=640) if kind == "yolo" else 640
        camera = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1,
                                 help="0 is usually the built-in webcam; try 1 or 2 if it fails.")
        crowded = st.checkbox("Other people in shot (slower)",
                              help="Detects up to three people and keeps the one closest to the camera.")
    if st.button("Start", type="primary"):
        st.session_state.running = True
    if st.button("Stop"):
        st.session_state.running = False

col_video, col_stats = st.columns([3, 1])
frame_slot = col_video.empty()
with col_stats:
    reps_slot, state_slot, message_slot, speed_slot = st.empty(), st.empty(), st.empty(), st.empty()

if st.session_state.get("running"):
    detector = load_detector(kind, imgsz, "pose_landmarker_full.task", 3 if crowded else 1)
    counter = RepCounter(EXERCISES[exercise])
    coach = LiveCoach(counter.messages())
    if upload is not None:
        path = Path(tempfile.gettempdir()) / f"coach_upload{Path(upload.name).suffix}"
        path.write_bytes(upload.getbuffer())
        cap, live = cv2.VideoCapture(str(path)), False
    else:
        cap, live = cv2.VideoCapture(int(camera)), True
    if not cap.isOpened():
        st.error("Cannot open the video source. For a webcam, try another camera index under Advanced.")
    else:
        t0, frames, last_report, outside, front = time.perf_counter(), 0, 0.0, 0, 0
        try:
            # Pressing Stop reruns the script, which ends this loop with running set to False.
            while cap.isOpened() and st.session_state.get("running"):
                ok, frame = cap.read()
                if not ok:
                    if live:
                        st.error("Lost the camera feed.")
                    break
                if frame.shape[1] > PROCESS_WIDTH:
                    fh, fw = frame.shape[:2]
                    frame = cv2.resize(frame, (PROCESS_WIDTH, int(fh * PROCESS_WIDTH / fw)),
                                       interpolation=cv2.INTER_AREA)
                if live:
                    frame = cv2.flip(frame, 1)             # mirror, so left and right feel natural
                    t = time.perf_counter() - t0
                else:
                    t = frames / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
                t_det = time.perf_counter()
                kp = detector(frame)
                infer_ms = 1000 * (time.perf_counter() - t_det)
                res = counter.update(kp, t)
                coach.update(res)
                draw(frame, kp, res, counter, coach.current)
                outside += joints_outside_frame(kp, frame.shape) > 0
                front += facing_camera(kp) is True
                h, w = frame.shape[:2]
                preview = cv2.resize(frame, (DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w))) if w > DISPLAY_WIDTH else frame
                if frames % DISPLAY_EVERY == 0:
                    ok_jpg, buf = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    if ok_jpg:
                        frame_slot.image(buf.tobytes(), use_container_width=True)

                summary = counter.summary()
                reps_slot.metric("Repetitions", summary["reps"], f"{summary['clean_reps']} clean")
                state_slot.info(f"State: {res.phase}" + ("" if res.visible else " (body not visible)"))
                if coach.current is None:
                    message_slot.empty()
                elif coach.current.kind == "fault":
                    message_slot.warning(coach.current.text)
                else:
                    message_slot.success(coach.current.text)
                frames += 1
                if t - last_report > 1.0:                  # refresh the read-outs once a second
                    speed_slot.caption(f"{infer_ms:.0f} ms of pose detection per frame, "
                                       f"{frames / max(t, 1e-6):.0f} fps displayed")
                    if outside > 0.2 * frames:
                        state_slot.warning("Part of your body leaves the frame. Step back from the camera.")
                    elif front > 0.5 * frames:
                        state_slot.warning("Turn side-on to the camera: angles are measured from a side view.")
                    last_report = t
        finally:
            cap.release()
        st.session_state.running = False
        show_summary(counter)
else:
    st.info("Pick an exercise and press Start. Film yourself from the side, with your whole body in view.")
