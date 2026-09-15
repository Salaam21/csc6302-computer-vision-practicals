"""Run Practical 6 tasks and save figures for my report."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
OUT = Path(__file__).resolve().parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


class KF:
    # My Kalman filter for a 2D point - state is x, y, vx, vy

    def __init__(self, dt=1.0, q=1e-2, r=5.0):
        self.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.Q = q * np.eye(4)
        self.R = r * np.eye(2)
        self.P = 100 * np.eye(4)
        self.x = np.zeros((4, 1))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        if z is None:
            return  # no measurement - I just coast on predict
        z = np.asarray(z, float).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.copy()


def detect_red_centroid(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    m2 = cv2.inRange(hsv, (160, 80, 60), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 30:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return np.array([[M["m10"] / M["m00"]], [M["m01"] / M["m00"]]])


def task1_flow():
    frames = [cv2.imread(str(p)) for p in sorted((IMG / "motion_frames").glob("frame_*.png"))]
    old = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(old, 80, 0.01, 8)
    trail = frames[0].copy()
    for fr in frames[1:15]:
        new = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(old, new, pts, None)
        if nxt is None:
            break
        good_new = nxt[st == 1]
        good_old = pts[st == 1]
        for p0, p1 in zip(good_old, good_new):
            cv2.line(trail, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 255, 0), 1)
            cv2.circle(trail, tuple(p1.astype(int)), 2, (0, 0, 255), -1)
        pts = good_new.reshape(-1, 1, 2)
        old = new
    cv2.imwrite(str(OUT / "t1_lk_trails.png"), trail)

    g0 = cv2.cvtColor(frames[5], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frames[8], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*g0.shape, 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    dense = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(OUT / "t1_farneback.png"), dense)

    # brightness change breaks brightness constancy
    bright = np.clip(frames[10].astype(np.float32) * 2.2, 0, 255).astype(np.uint8)
    g_a = cv2.cvtColor(frames[10], cv2.COLOR_BGR2GRAY)
    g_b = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
    flow_bad = cv2.calcOpticalFlowFarneback(g_a, g_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_b, ang_b = cv2.cartToPolar(flow_bad[..., 0], flow_bad[..., 1])
    hsv2 = np.zeros((*g_a.shape, 3), np.uint8)
    hsv2[..., 0] = ang_b * 180 / np.pi / 2
    hsv2[..., 1] = 255
    hsv2[..., 2] = cv2.normalize(mag_b, None, 0, 255, cv2.NORM_MINMAX)
    bad = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(OUT / "t1_flow_failure.png"), np.hstack([frames[10], bright, bad]))


def task2_kalman():
    frames = [cv2.imread(str(p)) for p in sorted((IMG / "motion_frames").glob("frame_*.png"))]
    kf = KF(dt=1.0, q=5e-2, r=8.0)
    xs_meas, xs_est, coast = [], [], []
    for i, fr in enumerate(frames):
        z = detect_red_centroid(fr)
        if z is not None:
            kf.x[:2] = z
            kf.x[2:] = 0
            break
    annotated = []
    for i, fr in enumerate(frames):
        kf.predict()
        z = detect_red_centroid(fr)
        kf.update(z)
        est = kf.x[:2].ravel()
        vis = fr.copy()
        if z is not None:
            cv2.circle(vis, (int(z[0, 0]), int(z[1, 0])), 6, (0, 255, 0), -1)
            xs_meas.append((i, float(z[0, 0])))
            coast.append(False)
        else:
            coast.append(True)
        cv2.circle(vis, (int(est[0]), int(est[1])), 6, (0, 0, 255), 2)
        xs_est.append((i, float(est[0])))
        annotated.append(vis)

    h, w = annotated[0].shape[:2]
    writer = cv2.VideoWriter(str(OUT / "t2_kalman.avi"), cv2.VideoWriter_fourcc(*"XVID"), 10, (w, h))
    for fr in annotated:
        writer.write(fr)
    writer.release()

    fig, ax = plt.subplots(figsize=(7, 3))
    if xs_meas:
        ax.plot([t for t, _ in xs_meas], [x for _, x in xs_meas], "go", label="measurement")
    ax.plot([t for t, _ in xs_est], [x for _, x in xs_est], "r-", label="KF estimate")
    for i, c in enumerate(coast):
        if c:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.2)
    ax.set_xlabel("frame")
    ax.set_ylabel("x position")
    ax.legend()
    ax.set_title("Kalman coasts through occlusion (grey band)")
    fig.tight_layout()
    fig.savefig(OUT / "t2_kf_plot.png", dpi=140)
    plt.close(fig)
    (OUT / "t2_kf_tune.txt").write_text("q=5e-2 r=8.0  (higher r = trust detector less)\n")

    try:
        tracker = cv2.TrackerCSRT_create()
        init_fr = frames[5]
        z = detect_red_centroid(init_fr)
        if z is not None:
            bbox = (int(z[0, 0] - 18), int(z[1, 0] - 18), 36, 36)
            tracker.init(init_fr, bbox)
            ok_count = 0
            for fr in frames[6:]:
                ok, box = tracker.update(fr)
                ok_count += int(ok)
            note = f"CSRT updates succeeded on {ok_count}/{len(frames)-6} frames after init"
        else:
            note = "CSRT skipped - no init detection"
    except Exception as e:
        note = f"CSRT unavailable: {e}"
    (OUT / "t2_csrt.txt").write_text(note + "\n")


def task3_calibration():
    board = cv2.imread(str(IMG / "checkerboard.png"), cv2.IMREAD_GRAYSCALE)
    pattern = (8, 5)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    calib_dir = OUT / "calib_views"
    calib_dir.mkdir(exist_ok=True)
    h, w = board.shape
    for i, ang in enumerate(range(-25, 30, 5)):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        M[0, 2] += i * 2
        view = cv2.warpAffine(board, M, (w, h), borderValue=180)
        cv2.imwrite(str(calib_dir / f"view_{i:02d}.png"), view)
        found, corners = cv2.findChessboardCorners(view, pattern)
        if found:
            corners = cv2.cornerSubPix(
                view, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            objpoints.append(objp)
            imgpoints.append(corners)
    if len(objpoints) < 5:
        (OUT / "t3_calib.txt").write_text(f"Only found {len(objpoints)} boards - need better views.\n")
        return
    rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    sample = cv2.imread(str(calib_dir / "view_00.png"))
    und = cv2.undistort(sample, K, dist)
    cv2.imwrite(str(OUT / "t3_undistort.png"), np.hstack([sample, und]))
    fx = K[0, 0]
    (OUT / "t3_calib.txt").write_text(
        f"rms_reprojection_px={rms:.4f}\n"
        f"camera_matrix=\n{K}\n"
        f"dist_coeffs={dist.ravel()}\n"
        f"fx_pixels={fx:.2f}\n"
        f"views_used={len(objpoints)}\n"
    )
    np.savez(OUT / "camera.npz", K=K, dist=dist, rms=rms)


def task4_ar():
    data = OUT / "camera.npz"
    if not data.exists():
        (OUT / "t4_ar_SKIPPED.txt").write_text("Run calibration first.\n")
        return
    npz = np.load(data)
    K, dist = npz["K"], npz["dist"]
    pattern = (8, 5)
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]])
    cube = np.float32([
        [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
        [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1],
    ])
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)

    views = sorted((OUT / "calib_views").glob("view_*.png"))
    if not views:
        return
    sample = cv2.imread(str(views[len(views) // 2]))
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern)
    if not found:
        (OUT / "t4_ar_SKIPPED.txt").write_text("No board in AR sample frame.\n")
        return
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    imgpts, _ = cv2.projectPoints(cube, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    out = sample.copy()
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(out, tuple(imgpts[a]), tuple(imgpts[b]), (0, 255, 0), 2)
    for a, b in [(4, 5), (5, 6), (6, 7), (7, 4)]:
        cv2.line(out, tuple(imgpts[a]), tuple(imgpts[b]), (255, 0, 0), 2)
    for a, b in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(out, tuple(imgpts[a]), tuple(imgpts[b]), (0, 0, 255), 2)
    axpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    origin = tuple(corners[0].ravel().astype(int))
    axpts = axpts.reshape(-1, 2).astype(int)
    cv2.line(out, origin, tuple(axpts[0]), (255, 0, 0), 2)
    cv2.line(out, origin, tuple(axpts[1]), (0, 255, 0), 2)
    cv2.line(out, origin, tuple(axpts[2]), (0, 0, 255), 2)
    cv2.imwrite(str(OUT / "t4_ar_cube.png"), out)


def task5_stereo():
    left = cv2.imread(str(IMG / "stereo_left.jpg"), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(IMG / "stereo_right.jpg"), cv2.IMREAD_GRAYSCALE)
    outs = []
    for nd, bs in ((64, 5), (96, 9)):
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=nd,
            blockSize=bs,
            P1=8 * 3 * bs * bs,
            P2=32 * 3 * bs * bs,
        )
        disp = stereo.compute(left, right).astype(np.float32) / 16.0
        vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        outs.append(vis)
        cv2.imwrite(str(OUT / f"t5_disp_nd{nd}_bs{bs}.png"), vis)
    cv2.imwrite(str(OUT / "t5_stereo.png"), np.hstack([left, right, outs[0], outs[1]]))
    (OUT / "t5_notes.txt").write_text(
        "Nearer objects should look brighter (larger disparity). "
        "My synthetic pair is roughly rectified; real phone pairs need proper rectification.\n"
    )


def main():
    task1_flow()
    task2_kalman()
    task3_calibration()
    task4_ar()
    task5_stereo()
    print("Practical 6 figures saved to", OUT)


if __name__ == "__main__":
    main()
