"""
CSC 6302 Practical 1 — Seeing Like a Machine

A computer does not see a picture. It sees a grid of numbers.
This CLI just loads an image, changes those numbers, and saves the result.

From-scratch bits (brief says I must write these myself):
- psnr
- my_sobel (+ my own conv2d)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def psnr(clean, restored):
    # PSNR = 10 * log10(MAX^2 / MSE). Higher = closer to the clean image.
    # No cv2.PSNR — I have to do this myself.
    clean = clean.astype(np.float64)
    restored = restored.astype(np.float64)
    mse = np.mean((clean - restored) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def conv2d(img, kernel):
    # basic 2-D convolution. nested loops so I can explain it in the seminar.
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            region = padded[y:y + kh, x:x + kw]
            out[y, x] = np.sum(region * kernel)
    return out


def my_sobel(gray):
    # Sobel kernels from the notes — one for x, one for y.
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = conv2d(gray, kx)
    gy = conv2d(gray, ky)
    # magnitude = how strong the edge is at that pixel
    return np.sqrt(gx ** 2 + gy ** 2)


def op_pick(img_bgr):
    # pick the lime-green bike (high saturation so plain grass is less included)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 110, 90), (85, 255, 255))
    # clean small speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)


def op_denoise(img_bgr, ksize=5):
    # median is good on salt-and-pepper (from the filter experiment)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img_bgr, ksize)


def op_sharpen(img_bgr, amount=1.0):
    # unsharp mask: original + amount * (original - blur)
    # work in float then clip — uint8 maths overflows quietly
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
    orig = img_bgr.astype(np.float32)
    sharp = orig + amount * (orig - blur.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)


def op_edges(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 80, 160)


def op_count(img_bgr):
    # earrings on a white card: gold bits are darker than the card
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    # drop background + tiny text speckles + huge card blobs
    h, w = gray.shape
    min_a = (h * w) * 0.0003
    max_a = (h * w) * 0.02
    count = 0
    vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_a <= area <= max_a:
            count += 1
            x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
    return vis, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--op", required=True, choices=["pick", "denoise", "sharpen", "edges", "count"])
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
        out, count = op_count(img)
        print("object_count =", count)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, out)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
