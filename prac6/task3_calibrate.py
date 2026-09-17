"""Task 3 — Calibrate camera from my chessboard captures."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PATTERN = (9, 6)  # inner corners


def run():
    calib_dir = DATA / "calib_images"
    imgs = sorted(calib_dir.glob("calib_*.jpg"))
    if len(imgs) < 10:
        from prepare_assets import make_calibration_views

        make_calibration_views()
        imgs = sorted(calib_dir.glob("calib_*.jpg"))

    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for p in imgs:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        size = gray.shape[::-1]
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners)

    if len(objpoints) < 8:
        raise RuntimeError(f"Only {len(objpoints)} valid chessboard views — need ≥10")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, size, None, None
    )
    sample = cv2.imread(str(imgs[0]))
    und = cv2.undistort(sample, K, dist)
    # draw corners on a mid view for the report
    mid = cv2.imread(str(imgs[len(imgs) // 2]))
    gray_m = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray_m, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if found:
        cv2.drawChessboardCorners(mid, PATTERN, corners, found)
    cv2.imwrite(str(OUT / "t3_corners.png"), mid)
    cv2.imwrite(str(OUT / "t3_undistort.png"), np.hstack([sample, und]))

    fx, fy = float(K[0, 0]), float(K[1, 1])
    # Typical phone sensor width ~5–6 mm; use 5.6 mm for a rough mm focal length
    sensor_width_mm = 5.6
    focal_mm = fx * sensor_width_mm / size[0]

    text = (
        f"rms_reprojection_px={rms:.4f}\n"
        f"views_used={len(objpoints)}\n"
        f"image_size_wxh={size[0]}x{size[1]}\n"
        f"camera_matrix=\n{K}\n"
        f"dist_coeffs={dist.ravel()}\n"
        f"fx_pixels={fx:.2f}\n"
        f"fy_pixels={fy:.2f}\n"
        f"sensor_width_mm_assumed={sensor_width_mm}\n"
        f"focal_length_mm_approx={focal_mm:.2f}\n"
    )
    (OUT / "t3_calib.txt").write_text(text, encoding="utf-8")
    np.savez(OUT / "camera.npz", K=K, dist=dist, rms=rms, image_size=np.array(size))
    # copy for Task 4 convenience
    np.savez(DATA / "camera.npz", K=K, dist=dist, rms=rms, image_size=np.array(size))
    print(text)
    if rms >= 1.0:
        print("WARNING: RMS >= 1 px — retake / regenerate views")
    print("Task3 calibration →", OUT)


if __name__ == "__main__":
    run()
