"""Find a working camera: tries every index with every backend and reports what happens.

    python src/check_camera.py
"""
from __future__ import annotations

import platform

import cv2

BACKENDS = ([("MSMF", cv2.CAP_MSMF), ("DirectShow", cv2.CAP_DSHOW), ("default", cv2.CAP_ANY)]
            if platform.system() == "Windows" else [("default", cv2.CAP_ANY)])


def main() -> None:
    working = []
    for index in range(4):
        for name, api in BACKENDS:
            cap = cv2.VideoCapture(index, api)
            if not cap.isOpened():
                print(f"camera {index} with {name}: cannot open")
                cap.release()
                continue
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                print(f"camera {index} with {name}: OK, {w}x{h}")
                working.append((index, name))
            else:
                print(f"camera {index} with {name}: opens but returns no image")
    print()
    if working:
        index, name = working[0]
        print(f"Use camera index {index} ({name} backend).")
    else:
        print("No camera produced an image. Check that no other application is using it, "
              "and that Windows privacy settings allow camera access for desktop apps.")


if __name__ == "__main__":
    main()
