"""
CSC 6302 Practical 1 — Seeing Like a Machine

CLI image toolbox. Two from-scratch pieces (brief rule):
  - psnr()     Task 2
  - my_sobel() Task 4 (uses conv2d)

Everything else may use OpenCV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def psnr(clean, restored):
    """PSNR from the definition. No cv2.PSNR."""
    clean = clean.astype(np.float64)
    restored = restored.astype(np.float64)
    mse = np.mean((clean - restored) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def conv2d(img, kernel):
    """
    2-D convolution with nested loops (seminar-explainable).
    Reflect padding matches OpenCV Sobel's default BORDER_REFLECT_101.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(
        img.astype(np.float64),
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="reflect",
    )
    out = np.zeros_like(img, dtype=np.float64)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            region = padded[y : y + kh, x : x + kw]
            out[y, x] = np.sum(region * kernel)
    return out


def my_sobel(gray):
    """Own Sobel: convolve with Gx/Gy, return gradient magnitude."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = conv2d(gray, kx)
    gy = conv2d(gray, ky)
    return np.sqrt(gx**2 + gy**2)


def op_pick(img_bgr):
    """HSV colour picker for the lime-green motorcycle in colourful.jpg."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Hue ~ green, high saturation so dull grass is less included
    mask = cv2.inRange(hsv, (40, 110, 90), (85, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)


def op_denoise(img_bgr, ksize=5):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img_bgr, ksize)


def op_sharpen(img_bgr, amount=1.0):
    """Unsharp mask in float, then clip to uint8."""
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
    orig = img_bgr.astype(np.float32)
    sharp = orig + amount * (orig - blur.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)


def op_edges(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 80, 160)


def _earring_mask(img_bgr):
    """
    Build a cleaned binary mask for earrings on the white card.
    Crop away header/footer text so letters are not counted as objects.
    """
    h, w = img_bgr.shape[:2]
    # Keep the jewellery band; drop "FASHION EARRINGS" and "K20..." text
    y0, y1 = int(0.18 * h), int(0.82 * h)
    x0, x1 = int(0.08 * w), int(0.92 * w)
    roi = img_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 5x5 ellipse: fill small holes in hoops, open removes speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask, kernel, iterations=1)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    full = {
        "roi_box": (x0, y0, x1, y1),
        "threshold": mask,
        "eroded": eroded,
        "dilated": dilated,
        "opened": opened,
        "closed": closed,
    }
    return full


def op_count(img_bgr):
    """
    Threshold + morphology + connectedComponents.
    Returns (visualisation, count).
    """
    steps = _earring_mask(img_bgr)
    x0, y0, x1, y1 = steps["roi_box"]
    closed = steps["closed"]
    rh, rw = closed.shape

    n, _labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    # Area gate: drop dust; keep studs and larger hoop clusters
    min_a = (rh * rw) * 0.0015
    max_a = (rh * rw) * 0.12
    count = 0
    vis = img_bgr.copy()
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_a <= area <= max_a:
            count += 1
            x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT : cv2.CC_STAT_LEFT + 4]
            cv2.rectangle(
                vis,
                (x0 + x, y0 + y),
                (x0 + x + bw, y0 + y + bh),
                (0, 220, 0),
                2,
            )
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 180, 255), 2)
    cv2.putText(
        vis,
        f"count={count}",
        (x0, max(20, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 180, 255),
        2,
    )
    return vis, count, steps


def main():
    ap = argparse.ArgumentParser(description="CSC 6302 Practical 1 image toolbox")
    ap.add_argument("--input", required=True)
    ap.add_argument(
        "--op",
        required=True,
        choices=["pick", "denoise", "sharpen", "edges", "count"],
    )
    ap.add_argument("--ksize", type=int, default=5)
    ap.add_argument("--amount", type=float, default=1.0)
    ap.add_argument("--output", default="out.png")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)

    if args.op == "pick":
        out = op_pick(img)
    elif args.op == "denoise":
        out = op_denoise(img, args.ksize)
    elif args.op == "sharpen":
        out = op_sharpen(img, args.amount)
    elif args.op == "edges":
        out = op_edges(img)
    else:
        out, count, _ = op_count(img)
        print("object_count =", count)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, out)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
