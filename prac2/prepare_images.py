"""Prepare Practical 2 overlapping shots from my own photos."""
from pathlib import Path

import cv2

IMG = Path(__file__).resolve().parents[1] / "images"

# Prefer the wide show photo (full-res enough). The PANO upload was a tiny preview.
src_path = IMG / "prac2_show.jpg"
src = cv2.imread(str(src_path))
if src is None:
    raise SystemExit(f"missing {src_path}")

print("source", src_path.name, src.shape)
h, w = src.shape[:2]
# three overlapping windows (~45% width, ~35% step => ~30%+ overlap)
win = int(0.50 * w)
step = int(0.32 * w)
names = ["left", "mid", "right"]
starts = [0, step, min(2 * step, w - win)]
for name, x0 in zip(names, starts):
    x1 = min(x0 + win, w)
    crop = src[:, x0:x1].copy()
    path = IMG / f"pano_{name}.jpg"
    cv2.imwrite(str(path), crop)
    print("wrote", path.name, crop.shape)

for key in ("prac2_natural.jpg", "prac2_dusk.jpg", "prac2_show.jpg"):
    im = cv2.imread(str(IMG / key))
    print(key, "ok" if im is not None else "MISSING", None if im is None else im.shape)
