"""Exercise analysis engine: pose keypoints in, repetitions and form feedback out.

No dependency on OpenCV, Ultralytics or Streamlit, so it can be tested alone and driven by
any frontend (video file, webcam, web app).
"""
from .engine import FrameResult, RepCounter, RepRecord
from .exercises import EXERCISES

__all__ = ["RepCounter", "FrameResult", "RepRecord", "EXERCISES"]
