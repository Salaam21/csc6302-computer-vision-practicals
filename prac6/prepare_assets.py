"""
Build Practical 6 inputs: flow clip frames, red-ball occlusion video,
chessboard calibration views + AR pan, using my phone street clip where useful.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(parents=True, exist_ok=True)

PATTERN = (9, 6)  # inner corners — matches OpenCV 9x6 chessboard brief


def extract_flow_frames(n=40, max_side=640) -> Path:
    """Sample frames from my street_clip.mp4 for Task 1."""
    out = DATA / "flow_frames"
    out.mkdir(exist_ok=True)
    src = IMG / "street_clip.mp4"
    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, min(total - 1, 200), n, dtype=int)
    saved = 0
    for i, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            fr = cv2.resize(fr, (int(w * scale), int(h * scale)))
        cv2.imwrite(str(out / f"frame_{saved:03d}.png"), fr)
        saved += 1
    cap.release()
    # Also write a short mp4 for the demo
    frames = sorted(out.glob("frame_*.png"))
    if frames:
        im0 = cv2.imread(str(frames[0]))
        h, w = im0.shape[:2]
        vw = cv2.VideoWriter(
            str(DATA / "flow_scene.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            12,
            (w, h),
        )
        for p in frames:
            vw.write(cv2.imread(str(p)))
        vw.release()
    print(f"flow frames: {saved} → {out}")
    return out


def make_occlusion_video(n=60, w=640, h=360) -> Path:
    """Bright red ball passes behind a grey pillar (~1s occlusion)."""
    path = DATA / "occlusion_ball.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 15, (w, h))
    frames_dir = DATA / "occlusion_frames"
    frames_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(3)
    # Ball path: left → right with slight bounce
    xs = np.linspace(60, w - 60, n)
    ys = 180 + 40 * np.sin(np.linspace(0, 2 * np.pi, n))
    # Pillar covers roughly frames where x in [pillar]
    pillar_x0, pillar_x1 = 280, 360
    for i in range(n):
        fr = np.full((h, w, 3), 40, np.uint8)
        # textured floor strip
        fr[280:, :] = (55, 60, 50)
        noise = rng.integers(-8, 8, (h, w, 3), dtype=np.int16)
        fr = np.clip(fr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cx, cy = int(xs[i]), int(ys[i])
        # draw ball under pillar if occluded
        occluded = pillar_x0 - 25 < cx < pillar_x1 + 25
        if not occluded:
            cv2.circle(fr, (cx, cy), 22, (40, 40, 220), -1)  # red in BGR
            cv2.circle(fr, (cx - 6, cy - 6), 6, (180, 180, 255), -1)
        # pillar on top
        cv2.rectangle(fr, (pillar_x0, 40), (pillar_x1, h - 20), (150, 150, 150), -1)
        cv2.putText(fr, "obstacle", (pillar_x0 - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        vw.write(fr)
        cv2.imwrite(str(frames_dir / f"frame_{i:03d}.png"), fr)
    vw.release()
    print("occlusion video:", path)
    return path


def _chessboard_image(sq=40) -> np.ndarray:
    """9x6 inner corners → 10x7 squares."""
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1
    board = np.zeros((rows * sq, cols * sq), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                board[i * sq : (i + 1) * sq, j * sq : (j + 1) * sq] = 255
    # margin
    canvas = np.full((board.shape[0] + 80, board.shape[1] + 80), 180, np.uint8)
    canvas[40 : 40 + board.shape[0], 40 : 40 + board.shape[1]] = board
    return canvas


def _render_board_view(K, dist, rvec, tvec, w=640, h=480) -> np.ndarray:
    """Draw each chessboard square by projecting its 3-D corners (clean edges)."""
    img = np.full((h, w, 3), 45, np.uint8)
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1
    for i in range(rows):
        for j in range(cols):
            colour = (245, 245, 245) if (i + j) % 2 == 0 else (15, 15, 15)
            pts3 = np.float32([
                [j, i, 0],
                [j + 1, i, 0],
                [j + 1, i + 1, 0],
                [j, i + 1, 0],
            ])
            pts2, _ = cv2.projectPoints(pts3, rvec, tvec, K, dist)
            poly = pts2.reshape(-1, 2).astype(np.int32)
            cv2.fillConvexPoly(img, poly, colour)
    return img


def make_calibration_views(n=18) -> Path:
    """
    Simulate phone captures: project a flat chessboard with a realistic K +
    barrel distortion from many angles. Calibration data is unique to this run.
    """
    out = DATA / "calib_images"
    out.mkdir(exist_ok=True)
    for p in out.glob("*.jpg"):
        p.unlink()

    w, h = 640, 480
    fx = fy = 520.0
    K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float64)
    # Mild distortion — strong barrel + harsh warp kills findChessboardCorners
    dist = np.array([-0.18, 0.05, 0.0, 0.0, 0.0], dtype=np.float64)

    rng = np.random.default_rng(7)
    saved = 0
    # Deterministic mild grid of poses first (reliable detections), then random extras
    pose_grid = []
    for yaw_d in range(-28, 29, 8):
        for pitch_d in range(-22, 23, 10):
            pose_grid.append((yaw_d, pitch_d, 0, 0.0, 0.0, 12.0))
    for yaw_d, pitch_d, roll_d, tx, ty, tz in pose_grid:
        yaw, pitch, roll = map(np.deg2rad, (yaw_d, pitch_d, roll_d))
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        # Centre the board in front of the camera
        t = np.array(
            [[tx - (PATTERN[0] + 1) / 2], [ty - (PATTERN[1] + 1) / 2], [tz]],
            dtype=np.float64,
        )
        rvec, _ = cv2.Rodrigues(R)
        view = _render_board_view(K, dist, rvec, t, w, h)
        noise = rng.normal(0, 2, view.shape).astype(np.float32)
        view = np.clip(view.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, _ = cv2.findChessboardCorners(gray, PATTERN, flags)
        if found:
            cv2.imwrite(str(out / f"calib_{saved:02d}.jpg"), view)
            saved += 1
        if saved >= n:
            break

    (DATA / "calib_note.txt").write_text(
        f"pattern_inner_corners={PATTERN}\n"
        f"sim_fx={fx} sim_fy={fy} image_size={w}x{h}\n"
        f"sim_dist={dist.tolist()}\n"
        f"views_written={saved}\n"
        "Synthetic phone-like captures of a printed 9x6 board for this practical "
        "(physical print can replace these later; do not share identical intrinsics).\n",
        encoding="utf-8",
    )
    cv2.imwrite(str(IMG / "checkerboard.png"), _chessboard_image(40))
    print(f"calib views: {saved} → {out}")
    return out


def make_ar_pan_video() -> Path:
    """Slow pan over chessboard for Task 4 (same renderer as calibration)."""
    path = DATA / "ar_pan.mp4"
    w, h = 640, 480
    fx = fy = 520.0
    K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.array([-0.18, 0.05, 0, 0, 0], dtype=np.float64)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (w, h))
    n = 48
    for i in range(n):
        yaw = np.deg2rad(-22 + 44 * i / (n - 1))
        pitch = np.deg2rad(-6 + 3 * np.sin(i / 5))
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        R = Ry @ Rx
        t = np.array(
            [
                [-(PATTERN[0] + 1) / 2],
                [-(PATTERN[1] + 1) / 2],
                [12.0 - 0.4 * np.sin(i / 6)],
            ],
            dtype=np.float64,
        )
        rvec, _ = cv2.Rodrigues(R)
        view = _render_board_view(K, dist, rvec, t, w, h)
        vw.write(view)
    vw.release()
    print("AR pan:", path)
    return path


def main():
    extract_flow_frames()
    make_occlusion_video()
    make_calibration_views()
    make_ar_pan_video()
    print("Prac6 data ready under", DATA)


if __name__ == "__main__":
    main()
