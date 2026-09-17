"""Task 4 — AR: solvePnP cube + axes locked to the chessboard across a pan video."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PATTERN = (9, 6)


def _draw_cube(img, imgpts):
    imgpts = imgpts.reshape(-1, 2).astype(int)
    # base
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(img, tuple(imgpts[a]), tuple(imgpts[b]), (0, 255, 0), 2)
    # top
    for a, b in [(4, 5), (5, 6), (6, 7), (7, 4)]:
        cv2.line(img, tuple(imgpts[a]), tuple(imgpts[b]), (255, 0, 0), 2)
    # pillars
    for a, b in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(img, tuple(imgpts[a]), tuple(imgpts[b]), (0, 0, 255), 2)
    return img


def run():
    cam = DATA / "camera.npz"
    if not cam.exists():
        cam = OUT / "camera.npz"
    if not cam.exists():
        from task3_calibrate import run as calib

        calib()
        cam = DATA / "camera.npz"
    npz = np.load(cam)
    K, dist = npz["K"], npz["dist"]

    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2)
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]])
    cube = np.float32(
        [
            [0, 0, 0],
            [0, 2, 0],
            [2, 2, 0],
            [2, 0, 0],
            [0, 0, -2],
            [0, 2, -2],
            [2, 2, -2],
            [2, 0, -2],
        ]
    )

    video = DATA / "ar_pan.mp4"
    if not video.exists():
        from prepare_assets import make_ar_pan_video

        make_ar_pan_video()

    cap = cv2.VideoCapture(str(video))
    ok0, frame0 = cap.read()
    if not ok0:
        raise RuntimeError("Could not read AR pan video")
    h, w = frame0.shape[:2]
    out_path = OUT / "t4_ar_cube.mp4"
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (w, h))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    still = None
    n_ok = 0
    n_total = 0
    # rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_total += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        vis = frame.copy()
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            ok_pnp, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
            if ok_pnp:
                n_ok += 1
                imgpts, _ = cv2.projectPoints(cube, rvec, tvec, K, dist)
                vis = _draw_cube(vis, imgpts)
                axpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
                origin = tuple(corners[0].ravel().astype(int))
                axpts = axpts.reshape(-1, 2).astype(int)
                cv2.line(vis, origin, tuple(axpts[0]), (255, 0, 0), 3)
                cv2.line(vis, origin, tuple(axpts[1]), (0, 255, 0), 3)
                cv2.line(vis, origin, tuple(axpts[2]), (0, 0, 255), 3)
                if still is None and n_ok == max(1, n_total // 2):
                    still = vis.copy()
        vw.write(vis)
    cap.release()
    vw.release()
    if still is None:
        # fallback last good — re-run one mid frame from calib
        sample = sorted((DATA / "calib_images").glob("calib_*.jpg"))
        if sample:
            mid = cv2.imread(str(sample[len(sample) // 2]))
            gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
            gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                ok_pnp, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
                imgpts, _ = cv2.projectPoints(cube, rvec, tvec, K, dist)
                still = _draw_cube(mid.copy(), imgpts)
    if still is not None:
        cv2.imwrite(str(OUT / "t4_ar_cube.png"), still)

    (OUT / "t4_note.txt").write_text(
        "solvePnP recovers the camera pose (rvec, tvec) that maps known 3-D chessboard "
        "points into the observed 2-D corners. It needs the intrinsics (K, dist) from "
        "calibration first — otherwise the projection model is wrong and the cube drifts "
        "or warps instead of locking to the board.\n"
        f"frames_with_pose={n_ok}/{n_total}\n",
        encoding="utf-8",
    )
    print(f"Task4 AR cube video → {out_path} ({n_ok}/{n_total} posed)")


if __name__ == "__main__":
    run()
