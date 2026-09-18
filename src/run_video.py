"""Analyse an exercise from a video file or a camera.

    python src/run_video.py --exercise pushup --source data/test1_pushup.mp4
    python src/run_video.py --exercise squat --source 0            # webcam index 0
    python src/run_video.py --exercise pushup --source clip.mp4 --detector mediapipe

Writes an annotated video, a per-frame CSV and a JSON summary to runs/<tag>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from coach import EXERCISES, RepCounter
from coach.feedback import LiveCoach
from coach.geometry import COCO
from detectors import MediaPipeDetector, YoloDetector, facing_camera, joints_outside_frame
from draw import draw

PROCESS_WIDTH = 960   # frames are resized before detection and drawing; 4K input is otherwise very slow


def keypoint_columns(kp: np.ndarray | None, side: str | None) -> dict:
    """Raw keypoints of the analysed side, kept for calibrating thresholds later."""
    out = {}
    for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        if kp is None or side is None:
            out[f"{joint}_x"] = out[f"{joint}_y"] = out[f"{joint}_c"] = ""
        else:
            x, y, c = kp[COCO[side][joint], :3]      # rows may carry 3D columns as well
            out[f"{joint}_x"], out[f"{joint}_y"], out[f"{joint}_c"] = round(float(x), 1), round(float(y), 1), round(float(c), 2)
    return out


def open_source(source: str) -> tuple[cv2.VideoCapture, bool]:
    is_camera = source.isdigit()
    cap = cv2.VideoCapture(int(source)) if is_camera else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {source}")
    return cap, is_camera


def run(source: str, exercise: str, tag: str, show: bool, detector) -> dict:
    cap, is_camera = open_source(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_dir = Path("runs") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    counter = RepCounter(EXERCISES[exercise])
    coach = LiveCoach(counter.messages())
    writer, rows = None, []
    t0, i, stopped_by_user = time.perf_counter(), 0, False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > PROCESS_WIDTH:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (PROCESS_WIDTH, int(h * PROCESS_WIDTH / w)), interpolation=cv2.INTER_AREA)
        t = (time.perf_counter() - t0) if is_camera else i / fps
        t_det = time.perf_counter()
        kp = detector(frame)
        infer_ms = 1000 * (time.perf_counter() - t_det)
        res = counter.update(kp, t)
        outside = joints_outside_frame(kp, frame.shape)
        front = facing_camera(kp)

        coach.update(res)

        draw(frame, kp, res, counter, coach.current)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        rows.append({"frame": i, "t": round(t, 3), "visible": res.visible, "side": res.side,
                     "angle": None if res.angle is None else round(res.angle, 1), "phase": res.phase,
                     "reps": res.reps, "live_faults": "|".join(res.live_faults),
                     "rep_completed": res.completed.index if res.completed else "",
                     "infer_ms": round(infer_ms, 1), "joints_outside_frame": outside, "facing_camera": "" if front is None else int(front),
                     "message": coach.current.text if coach.current else "",
                     **{k: ("" if v is None else round(v, 2)) for k, v in res.observations.items()},
                     **keypoint_columns(kp, res.side)})
        i += 1
        if show:
            cv2.imshow("coach", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stopped_by_user = True
                break

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    if rows:
        with open(out_dir / "frames.csv", "w", newline="") as f:
            fields = list(dict.fromkeys(k for r in rows for k in r))     # union, first-seen order
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)
    summary = counter.summary() | {
        "source": source, "frames": i, "stopped_by_user": stopped_by_user,
        "visible_frames": sum(r["visible"] for r in rows),
        "frames_with_joints_outside": sum(r["joints_outside_frame"] > 0 for r in rows),
        "frames_facing_camera": sum(r["facing_camera"] == 1 for r in rows),
        "video_fps": round(fps, 1),
        "infer_ms_p50": round(float(np.median([r["infer_ms"] for r in rows[5:]])), 1) if len(rows) > 5 else None,
        "detector": getattr(detector, "name", "custom"),
        "detector_config": getattr(detector, "config", {}),
        "messages": [{"t": round(m.since, 2), "text": m.text} for m in coach.log],
        "infer_ms_p95": round(float(np.percentile([r["infer_ms"] for r in rows[5:]], 95)), 1) if len(rows) > 5 else None,
        "reps_detail": [{"index": r.index, "start_s": round(r.start, 2), "end_s": round(r.end, 2),
                         "min_angle": round(r.min_angle, 1), "faults": r.faults,
                         "observations": r.observations} for r in counter.reps],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exercise", choices=sorted(EXERCISES), required=True)
    ap.add_argument("--source", required=True, help="video path, or camera index (0, 1, 2...)")
    ap.add_argument("--tag", default=None, help="output folder name under runs/")
    ap.add_argument("--detector", choices=["yolo", "mediapipe"], default="mediapipe",
                    help="mediapipe: ~6x faster on CPU, single person; yolo: multi-person")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO input size; 320 is faster, less precise")
    ap.add_argument("--mp-model", default="pose_landmarker_full.task", help="MediaPipe model file")
    ap.add_argument("--poses", type=int, default=1,
                    help="people to detect per frame; use 2 or 3 when others are in shot (slower)")
    ap.add_argument("--no-show", action="store_true", help="do not open a preview window")
    args = ap.parse_args()
    detector = (MediaPipeDetector(args.mp_model, max_poses=args.poses) if args.detector == "mediapipe"
                else YoloDetector(imgsz=args.imgsz))
    tag = args.tag or f"{args.exercise}_{Path(args.source).stem}_{args.detector}"
    summary = run(args.source, args.exercise, tag, show=not args.no_show, detector=detector)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("reps_detail", "messages")}, indent=2))
    if summary["stopped_by_user"]:
        print("WARNING: stopped before the end of the video, counts are incomplete")
    print(f"outputs in runs/{tag}/")


if __name__ == "__main__":
    main()
