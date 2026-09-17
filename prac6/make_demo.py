"""Stitch a short captioned demo reel from Practical 6 outputs."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "report" / "figures"
DATA = HERE / "data"
OUT = HERE / "report" / "Prac6_Demo.mp4"


def _title_card(text: str, size=(960, 540), n=24):
    frames = []
    for _ in range(n):
        img = np.full((size[1], size[0], 3), 30, np.uint8)
        y = size[1] // 2 - 20
        for i, line in enumerate(text.split("\n")):
            cv2.putText(img, line, (40, y + i * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
        frames.append(img)
    return frames


def _fit(img, size=(960, 540)):
    h, w = img.shape[:2]
    canvas = np.full((size[1], size[0], 3), 20, np.uint8)
    scale = min(size[0] / w, size[1] / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    x0 = (size[0] - nw) // 2
    y0 = (size[1] - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _from_video(path: Path, max_frames=45, size=(960, 540)):
    if not path.exists():
        return []
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(_fit(fr, size))
    cap.release()
    return frames


def main():
    size = (960, 540)
    seq = []
    seq += _title_card("CSC 6302 Practical 6\nMunyeme Sinyani (SINYANI)\nMotion · Tracking · AR · Stereo", size, 30)
    seq += _title_card("Task 1 — Optical flow (LK + Farneback)", size, 18)
    for name in ["t1_lk_trails.png", "t1_farneback.png", "t1_flow_failure.png"]:
        p = FIG / name
        if p.exists():
            seq += [_fit(cv2.imread(str(p)), size)] * 18
    seq += _from_video(DATA / "flow_scene.mp4", 36, size)

    seq += _title_card("Task 2 — Kalman coasts through occlusion\n(green=meas, red=KF)", size, 20)
    seq += _from_video(FIG / "t2_kalman.mp4", 60, size)
    if (FIG / "t2_kf_plot.png").exists():
        seq += [_fit(cv2.imread(str(FIG / "t2_kf_plot.png")), size)] * 24

    seq += _title_card("Task 3 — Camera calibration / undistort", size, 16)
    if (FIG / "t3_undistort.png").exists():
        seq += [_fit(cv2.imread(str(FIG / "t3_undistort.png")), size)] * 30

    seq += _title_card("Task 4 — AR cube on chessboard", size, 16)
    seq += _from_video(FIG / "t4_ar_cube.mp4", 48, size)

    seq += _title_card("Task 5 — Stereo disparity (SGBM)", size, 16)
    if (FIG / "t5_stereo.png").exists():
        seq += [_fit(cv2.imread(str(FIG / "t5_stereo.png")), size)] * 30

    seq += _title_card("End — see README for exact commands", size, 24)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), 12, size)
    for fr in seq:
        vw.write(fr)
    vw.release()
    print("Wrote", OUT, "frames", len(seq))


if __name__ == "__main__":
    main()
