"""Turn an annotated run into a small GIF for the README.

    python src/make_gif.py runs/pushup_test1_pushup_mediapipe/annotated.mp4 assets/pushup.gif
    python src/make_gif.py runs/.../annotated.mp4 out.gif --seconds 8 --max-mb 3

GitHub serves the README GIFs on every page view, so size matters more than smoothness: the
script drops frames, shrinks the image and reduces the palette until the file fits the budget.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SETTINGS = [  # tried in order, from best looking to smallest
    (480, 12, 128), (480, 10, 96), (400, 10, 64), (360, 8, 64), (320, 8, 48), (280, 6, 32),
]


def read_frames(path: str, start: float, seconds: float | None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    limit = None if seconds is None else int(seconds * fps)
    frames = []
    while limit is None or len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit("no frames read")
    return frames, fps


def write_gif(frames: list[np.ndarray], fps: float, out: Path, width: int, target_fps: int,
              colors: int) -> int:
    step = max(1, round(fps / target_fps))
    kept = frames[::step]
    images = []
    for frame in kept:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        rgb = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        images.append(rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT))
    images[0].save(out, save_all=True, append_images=images[1:], loop=0, optimize=True,
                   duration=int(1000 * step / fps))
    return out.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("output")
    ap.add_argument("--start", type=float, default=0.0, help="seconds to skip at the beginning")
    ap.add_argument("--seconds", type=float, default=None, help="length to keep (default: all)")
    ap.add_argument("--max-mb", type=float, default=4.0)
    args = ap.parse_args()

    frames, fps = read_frames(args.video, args.start, args.seconds)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    budget = args.max_mb * 1e6
    for width, target_fps, colors in SETTINGS:
        size = write_gif(frames, fps, out, width, target_fps, colors)
        print(f"{width}px, {target_fps} fps, {colors} colours -> {size / 1e6:.1f} MB")
        if size <= budget:
            print(f"kept {out} ({size / 1e6:.1f} MB)")
            return
    print(f"warning: still {out.stat().st_size / 1e6:.1f} MB, above the {args.max_mb} MB budget. "
          f"Use --seconds to keep a shorter clip.")


if __name__ == "__main__":
    main()
