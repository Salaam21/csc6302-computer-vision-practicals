"""
I generate demo images so I can run all practicals without my phone photos yet.
Swap files in shared/images/ with real captures before I submit.
"""

from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent / "images"
ROOT.mkdir(parents=True, exist_ok=True)


def colourful_scene(path: Path) -> None:
    img = np.full((480, 640, 3), 40, np.uint8)
    cv2.rectangle(img, (40, 40), (280, 220), (40, 80, 220), -1)
    cv2.circle(img, (480, 140), 70, (40, 200, 60), -1)
    cv2.rectangle(img, (80, 280), (560, 420), (210, 160, 40), -1)
    cv2.putText(img, "CSC6302 demo scene", (150, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


def low_light(path: Path) -> None:
    img = np.full((360, 480, 3), 18, np.uint8)
    cv2.rectangle(img, (60, 50), (200, 180), (90, 90, 90), -1)
    cv2.circle(img, (340, 120), 50, (120, 100, 80), -1)
    noise = np.random.normal(0, 25, img.shape)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


def coins(path: Path, n: int = 8) -> None:
    img = np.full((420, 560, 3), 230, np.uint8)
    rng = np.random.default_rng(7)
    placed = []
    for _ in range(n):
        for _try in range(40):
            r = int(rng.integers(28, 42))
            cx = int(rng.integers(50, 510))
            cy = int(rng.integers(50, 370))
            if all((cx - x) ** 2 + (cy - y) ** 2 > (r + rr + 4) ** 2 for x, y, rr in placed):
                placed.append((cx, cy, r))
                colour = (int(rng.integers(40, 90)),) * 3
                cv2.circle(img, (cx, cy), r, colour, -1)
                cv2.circle(img, (cx, cy), r, (20, 20, 20), 2)
                break
    cv2.circle(img, (200, 200), 34, (70, 70, 70), -1)
    cv2.circle(img, (265, 200), 34, (65, 65, 65), -1)
    cv2.imwrite(str(path), img)


def checkerboard(path: Path, squares=(9, 6), sq=40) -> None:
    rows, cols = squares[1] * sq, squares[0] * sq
    board = np.zeros((rows, cols), np.uint8)
    for i in range(squares[1]):
        for j in range(squares[0]):
            if (i + j) % 2 == 0:
                board[i * sq:(i + 1) * sq, j * sq:(j + 1) * sq] = 255
    canvas = np.full((rows + 80, cols + 80), 180, np.uint8)
    canvas[40:40 + rows, 40:40 + cols] = board
    cv2.imwrite(str(path), canvas)


def panorama_pair(path_a: Path, path_b: Path, path_c: Path) -> None:
    base = np.full((360, 720, 3), 90, np.uint8)
    cv2.rectangle(base, (40, 80), (180, 280), (30, 60, 180), -1)
    cv2.rectangle(base, (250, 60), (380, 300), (40, 160, 80), -1)
    cv2.rectangle(base, (450, 100), (620, 260), (180, 80, 40), -1)
    cv2.circle(base, (200, 200), 35, (0, 220, 220), -1)
    cv2.putText(base, "UNZA wall demo", (240, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for x in range(0, 720, 20):
        cv2.line(base, (x, 0), (x, 360), (70, 70, 70), 1)
    for y in range(0, 360, 20):
        cv2.line(base, (0, y), (720, y), (70, 70, 70), 1)
    noise = np.random.normal(0, 6, base.shape)
    base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    a = base[:, 0:480].copy()
    b = base[:, 120:600].copy()
    c = base[:, 240:720].copy()
    cv2.imwrite(str(path_a), a)
    cv2.imwrite(str(path_b), b)
    cv2.imwrite(str(path_c), c)


def document_uneven(path: Path) -> None:
    img = np.full((500, 400), 240, np.uint8)
    for i, line in enumerate([
        "Computer Vision Practical",
        "Thresholding under uneven light",
        "Line three of the document",
        "Another paragraph of text here",
        "Final line for testing Otsu",
    ]):
        cv2.putText(img, line, (30, 80 + i * 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 20, 2)
    ramp = np.linspace(0.55, 1.15, img.shape[1], dtype=np.float32)
    lit = np.clip(img.astype(np.float32) * ramp[None, :], 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), lit)


def portrait(path: Path) -> None:
    img = np.full((420, 320, 3), 210, np.uint8)
    cv2.ellipse(img, (160, 210), (90, 140), 0, 0, 360, (40, 90, 180), -1)
    cv2.circle(img, (160, 120), 55, (60, 140, 220), -1)
    cv2.circle(img, (145, 115), 6, (20, 20, 20), -1)
    cv2.circle(img, (175, 115), 6, (20, 20, 20), -1)
    cv2.ellipse(img, (160, 135), (18, 10), 0, 0, 180, (20, 20, 20), 2)
    cv2.imwrite(str(path), img)


def street_scene(path: Path) -> None:
    img = np.full((360, 640, 3), 170, np.uint8)
    cv2.rectangle(img, (0, 220), (640, 360), (80, 80, 80), -1)
    cv2.rectangle(img, (60, 200), (150, 250), (40, 40, 200), -1)
    cv2.rectangle(img, (280, 190), (380, 250), (40, 180, 40), -1)
    cv2.rectangle(img, (480, 205), (580, 255), (200, 80, 40), -1)
    for x in (200, 420, 520):
        cv2.circle(img, (x, 170), 12, (30, 30, 30), -1)
        cv2.rectangle(img, (x - 10, 182), (x + 10, 230), (30, 30, 30), -1)
    cv2.imwrite(str(path), img)


def stereo_pair(path_l: Path, path_r: Path) -> None:
    left = np.full((240, 320, 3), 160, np.uint8)
    cv2.rectangle(left, (40, 80), (100, 180), (30, 30, 200), -1)
    cv2.rectangle(left, (160, 100), (210, 160), (30, 160, 30), -1)
    cv2.rectangle(left, (250, 110), (290, 150), (180, 80, 30), -1)
    right = np.full_like(left, 160)
    shifts = [(40, 100, 80, 180, 18), (160, 210, 100, 160, 10), (250, 290, 110, 150, 5)]
    colours = [(30, 30, 200), (30, 160, 30), (180, 80, 30)]
    for (x0, x1, y0, y1, d), col in zip(shifts, colours):
        cv2.rectangle(right, (x0 - d, y0), (x1 - d, y1), col, -1)
    cv2.imwrite(str(path_l), left)
    cv2.imwrite(str(path_r), right)


def motion_frames(folder: Path, n: int = 40) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        frame = np.full((240, 320, 3), 40, np.uint8)
        x = 30 + i * 6
        hide = 18 <= i <= 26
        if not hide:
            cv2.circle(frame, (x, 120), 18, (40, 40, 220), -1)
        cv2.rectangle(frame, (150, 40), (190, 200), (90, 90, 90), -1)
        cv2.imwrite(str(folder / f"frame_{i:03d}.png"), frame)


def group_faces_proxy(path: Path) -> None:
    # cartoon faces - I will replace with a real group photo later
    img = np.full((300, 500, 3), 200, np.uint8)
    for cx in (100, 250, 400):
        cv2.circle(img, (cx, 140), 45, (90, 160, 220), -1)
        cv2.circle(img, (cx - 15, 130), 5, (20, 20, 20), -1)
        cv2.circle(img, (cx + 15, 130), 5, (20, 20, 20), -1)
        cv2.ellipse(img, (cx, 155), (14, 8), 0, 0, 180, (20, 20, 20), 2)
    cv2.imwrite(str(path), img)


def leaf_classes(root: Path) -> None:
    classes = {
        "healthy": (40, 160, 60),
        "rust": (40, 90, 180),
        "blight": (30, 30, 140),
    }
    rng = np.random.default_rng(42)
    for split, n in (("train", 42), ("val", 9), ("test", 9)):
        for name, colour in classes.items():
            d = root / split / name
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = np.full((128, 128, 3), 220, np.uint8)
                pts = np.array([
                    [64 + int(rng.integers(-8, 8)), 20],
                    [20, 100],
                    [108, 100],
                ], np.int32)
                cv2.fillConvexPoly(img, pts, colour)
                if name != "healthy":
                    for _ in range(25):
                        x, y = int(rng.integers(30, 98)), int(rng.integers(40, 100))
                        cv2.circle(img, (x, y), 2, (20, 20, 20), -1)
                noise = rng.normal(0, 8, img.shape)
                out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                cv2.imwrite(str(d / f"{name}_{i:02d}.png"), out)


def main() -> None:
    colourful_scene(ROOT / "colourful.jpg")
    low_light(ROOT / "lowlight.jpg")
    coins(ROOT / "coins.jpg")
    checkerboard(ROOT / "checkerboard.png")
    panorama_pair(ROOT / "pano_left.jpg", ROOT / "pano_mid.jpg", ROOT / "pano_right.jpg")
    document_uneven(ROOT / "document.jpg")
    portrait(ROOT / "portrait.jpg")
    street_scene(ROOT / "street.jpg")
    stereo_pair(ROOT / "stereo_left.jpg", ROOT / "stereo_right.jpg")
    motion_frames(ROOT / "motion_frames")
    group_faces_proxy(ROOT / "group.jpg")
    leaf_classes(ROOT / "leaf_dataset")
    gray = cv2.imread(str(ROOT / "colourful.jpg"), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(ROOT / "clean_gray.png"), gray)
    print(f"Wrote demo images under {ROOT}")


if __name__ == "__main__":
    main()
