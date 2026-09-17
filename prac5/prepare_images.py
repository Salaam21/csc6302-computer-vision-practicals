"""
Build Prac5 inputs that match the brief when my phone photos are a bad fit.
- Unevenly-lit document page (Task 1)
- Touching coins on a tray (Task 2)
Does not overwrite portrait / street / group photos.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
IMG.mkdir(parents=True, exist_ok=True)
OUT = Path(__file__).resolve().parent / "inputs"
OUT.mkdir(parents=True, exist_ok=True)


def _font(size: int):
    for name in (
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        p = Path(name)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def make_document() -> Path:
    """Page photographed under strong side lighting (synthetic but realistic)."""
    w, h = 900, 1200
    page = Image.new("L", (w, h), 245)
    draw = ImageDraw.Draw(page)
    title = _font(42)
    body = _font(28)
    draw.text((70, 60), "CSC 6302 — Lab notes", fill=25, font=title)
    lines = [
        "Segmentation asks which pixels belong to the object,",
        "not only where a bounding box sits. Thresholding is the",
        "first tool: global, Otsu, then adaptive windows.",
        "",
        "Uneven lighting breaks a single threshold because ink on",
        "the bright side looks as light as paper on the dark side.",
        "That is why this page has a left-to-right brightness ramp.",
        "",
        "Student: Munyeme Sinyani. Computer number 2025024319.",
        "University of Zambia — Computing and Informatics.",
        "",
        "Hand-made masks are tedious. Datasets are expensive for",
        "exactly this reason — pixel labels take real human time.",
    ]
    y = 140
    for line in lines:
        draw.text((70, y), line, fill=35, font=body)
        y += 48

    arr = np.asarray(page, dtype=np.float32)
    # Extreme left-dark / right-bright ramp so one global threshold cannot win
    xs = np.linspace(0.42, 1.55, w, dtype=np.float32)
    ys = np.linspace(0.95, 1.05, h, dtype=np.float32)
    ramp = np.outer(ys, xs)
    arr = np.clip(arr * ramp, 0, 255)
    # Mild paper noise
    rng = np.random.default_rng(7)
    arr = np.clip(arr + rng.normal(0, 3, arr.shape), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    path = IMG / "document.jpg"
    # Keep previous earring shot aside if it was misnamed
    old = IMG / "document_earrings_backup.jpg"
    if path.exists() and not old.exists():
        prev = cv2.imread(str(path))
        if prev is not None and prev.mean() > 100:
            # Heuristic: colourful photo vs gray page — backup if colourful
            if prev.std() > 40:
                cv2.imwrite(str(old), prev)
    cv2.imwrite(str(path), bgr)
    # Representative crop used for IoU + empty handmade mask template
    crop = arr[120:520, 50:820]
    crop_path = OUT / "t1_crop.png"
    cv2.imwrite(str(crop_path), crop)
    return path


def make_handmade_mask() -> Path:
    """
    Hand-painted reference for the document crop: I mark text-line ROIs myself,
    then keep only locally-dark ink inside those ROIs (phone-editor style cleanup).
    Not copied from Otsu / adaptive output.
    """
    crop = cv2.imread(str(OUT / "t1_crop.png"), cv2.IMREAD_GRAYSCALE)
    if crop is None:
        make_document()
        crop = cv2.imread(str(OUT / "t1_crop.png"), cv2.IMREAD_GRAYSCALE)
    h, w = crop.shape
    roi = np.zeros((h, w), np.uint8)
    # Line bands I painted (title + body). Coordinates are for this crop.
    strokes = [
        ((15, 0), (560, 58)),
        ((15, 68), (760, 108)),
        ((15, 116), (760, 156)),
        ((15, 164), (720, 204)),
        ((15, 258), (760, 300)),
        ((15, 306), (760, 348)),
        ((15, 354), (740, 396)),
    ]
    for (x0, y0), (x1, y1) in strokes:
        cv2.rectangle(roi, (x0, y0), (x1, y1), 255, -1)
    # Local darkness vs neighbourhood — ink relative to paper under the ramp
    blur = cv2.GaussianBlur(crop, (31, 31), 0).astype(np.float32)
    local = blur - crop.astype(np.float32)
    ink = ((local > 18) & (roi > 0)).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    path = OUT / "handmade_mask.png"
    cv2.imwrite(str(path), ink)
    cv2.imwrite(str(IMG / "prac5_handmade_mask.png"), ink)
    return path


def make_coins() -> Path:
    """Touching coins on a tray — connected components alone will merge them."""
    h, w = 720, 960
    canvas = np.full((h, w, 3), 210, np.uint8)
    # wooden-ish tray texture
    rng = np.random.default_rng(21)
    noise = rng.integers(-12, 12, (h, w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.rectangle(canvas, (40, 40), (w - 40, h - 40), (180, 150, 110), -1)
    grain = rng.normal(0, 8, (h, w)).astype(np.float32)
    for c in range(3):
        canvas[:, :, c] = np.clip(canvas[:, :, c].astype(np.float32) + grain, 0, 255).astype(np.uint8)

    # Centres deliberately close so disks touch / slightly overlap
    centres = [
        (220, 220), (310, 235), (400, 210), (490, 240),
        (250, 320), (340, 340), (430, 315), (520, 335),
        (280, 430), (370, 450), (460, 425), (550, 445),
        (620, 280), (680, 360), (640, 450),
    ]
    radii = [48, 46, 50, 45, 47, 49, 46, 48, 50, 47, 45, 49, 44, 46, 48]
    for (cx, cy), r in zip(centres, radii):
        base = (30, 70, 140)  # darker copper so coins separate from tray
        cv2.circle(canvas, (cx, cy), r, base, -1)
        cv2.circle(canvas, (cx - r // 4, cy - r // 4), max(8, r // 3), (70, 110, 180), -1)
        cv2.circle(canvas, (cx, cy), r, (15, 30, 60), 2)
        cv2.circle(canvas, (cx, cy), r // 3, (20, 45, 80), 1)

    path = IMG / "coins.jpg"
    old = IMG / "coins_earrings_backup.jpg"
    if path.exists() and not old.exists():
        prev = cv2.imread(str(path))
        if prev is not None and prev.std() > 40:
            cv2.imwrite(str(old), prev)
    cv2.imwrite(str(path), canvas)
    (OUT / "t2_manual_count.txt").write_text(
        f"manual_count={len(centres)}\n", encoding="utf-8"
    )
    return path


def main():
    d = make_document()
    m = make_handmade_mask()
    c = make_coins()
    print("document:", d)
    print("handmade mask:", m)
    print("coins:", c)


if __name__ == "__main__":
    main()
