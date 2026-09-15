"""
Photos for the practicals live in images/.
This helper is optional - only run if I need to regenerate helper assets
(checkerboard / pano stubs / leaf folders for later practicals).
"""
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1] / "images"
ROOT.mkdir(parents=True, exist_ok=True)


def colourful_scene(path: Path) -> None:
    img = np.full((480, 640, 3), 40, np.uint8)
    cv2.rectangle(img, (40, 40), (280, 220), (40, 80, 220), -1)
    cv2.circle(img, (480, 140), 70, (40, 200, 60), -1)
    cv2.rectangle(img, (80, 280), (560, 420), (210, 160, 40), -1)
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


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
    for x in range(0, 720, 20):
        cv2.line(base, (x, 0), (x, 360), (70, 70, 70), 1)
    for y in range(0, 360, 20):
        cv2.line(base, (0, y), (720, y), (70, 70, 70), 1)
    noise = np.random.normal(0, 6, base.shape)
    base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path_a), base[:, 0:480].copy())
    cv2.imwrite(str(path_b), base[:, 120:600].copy())
    cv2.imwrite(str(path_c), base[:, 240:720].copy())


def main() -> None:
    # only fill missing helper files - do not overwrite my Prac1 photos
    if not (ROOT / "checkerboard.png").exists():
        checkerboard(ROOT / "checkerboard.png")
    if not (ROOT / "pano_left.jpg").exists():
        panorama_pair(ROOT / "pano_left.jpg", ROOT / "pano_mid.jpg", ROOT / "pano_right.jpg")
    print("images folder ready:", ROOT)


if __name__ == "__main__":
    main()
