"""Compare the per-rep measurements of several runs of the same video (e.g. YOLO vs MediaPipe).

    python src/compare_runs.py runs/pushup_test1_pushup_yolo runs/pushup_test1_pushup_mediapipe

A measurement that describes the movement, rather than the detector, should give similar values
whichever detector produced the keypoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROWS = [("elbow angle, min", lambda r: r["min_angle"]),
        ("upper arm vs floor, min", lambda r: r.get("observations", {}).get("upper_arm_vs_floor", {}).get("min")),
        ("body line angle, median", lambda r: r.get("observations", {}).get("body_line_angle", {}).get("median"))]


def main(paths: list[str]) -> None:
    runs = []
    for p in paths:
        s = json.loads((Path(p) / "summary.json").read_text())
        runs.append((s.get("detector", Path(p).name), s))
    n = max(len(s["reps_detail"]) for _, s in runs)
    print("reps per run: " + ", ".join(f"{d}={len(s['reps_detail'])}" for d, s in runs))
    header = f"{'rep':>4} {'measurement':26s}" + "".join(f"{d:>12s}" for d, _ in runs)
    print(header)
    print("-" * len(header))
    for i in range(n):
        for label, get in ROWS:
            vals = []
            for _, s in runs:
                reps = s["reps_detail"]
                v = get(reps[i]) if i < len(reps) else None
                vals.append("-" if v is None else f"{v:.1f}")
            print(f"{i + 1:>4} {label:26s}" + "".join(f"{v:>12s}" for v in vals))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
