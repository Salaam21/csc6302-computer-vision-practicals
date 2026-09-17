"""
Generate all Practical 1 report figures from my own photos in images/.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from process import my_sobel, op_count, op_pick, op_sharpen, psnr

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
OUT = Path(__file__).resolve().parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save_row(images, titles, path, figsize_scale=3.2):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(figsize_scale * n, figsize_scale))
    if n == 1:
        axes = [axes]
    for ax, im, t in zip(axes, images, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def ensure_clean_gray():
    """Clean grayscale for Task 2/4 from my colourful scene (resized for speed)."""
    path = IMG / "clean_gray.png"
    src = cv2.imread(str(IMG / "colourful.jpg"))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(path), gray)
    return gray


def task1_colour():
    img = cv2.imread(str(IMG / "colourful.jpg"))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    save_row(
        [hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]],
        ["H (hue)", "S (saturation)", "V (value/brightness)"],
        OUT / "t1_hsv_channels.png",
    )
    save_row(
        [lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]],
        ["L (lightness)", "A (green–red)", "B (blue–yellow)"],
        OUT / "t1_lab_channels.png",
    )
    picked = op_pick(img)
    save_row([img, picked], ["Original (colourful)", "HSV colour pick (lime bike)"], OUT / "t1_picker.png")

    # Brief asks for a low-light photo among my inputs — show it here for evidence
    low = cv2.imread(str(IMG / "lowlight.jpg"))
    if low is not None:
        save_row([low], ["Low-light / noisy input photo"], OUT / "t1_lowlight.png")


def add_gaussian(gray, sigma=15):
    noise = np.random.normal(0, sigma, gray.shape)
    return np.clip(gray.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_sap(gray, amount=0.05):
    out = gray.copy()
    n = out.size
    n_flip = int(amount * n)
    flat_idx = np.random.choice(n, n_flip, replace=False)
    coords = np.unravel_index(flat_idx, out.shape)
    half = n_flip // 2
    out[coords[0][:half], coords[1][:half]] = 0
    out[coords[0][half:], coords[1][half:]] = 255
    return out


def task2_noise(clean):
    g_noisy = add_gaussian(clean, 15)
    s_noisy = add_sap(clean, 0.05)
    save_row(
        [clean, g_noisy, s_noisy],
        ["Clean grayscale", "Gaussian σ=15", "Salt-and-pepper 5%"],
        OUT / "t2_noise.png",
    )

    rows = []
    for noise_name, noisy in [("gaussian", g_noisy), ("sap", s_noisy)]:
        for k in (3, 5, 7):
            for fname, restored in [
                ("box", cv2.blur(noisy, (k, k))),
                ("gaussian", cv2.GaussianBlur(noisy, (k, k), 0)),
                ("median", cv2.medianBlur(noisy, k)),
            ]:
                rows.append(
                    {
                        "noise": noise_name,
                        "filter": fname,
                        "ksize": k,
                        "psnr": round(psnr(clean, restored), 3),
                    }
                )

    with open(OUT / "t2_psnr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["noise", "filter", "ksize", "psnr"])
        w.writeheader()
        w.writerows(rows)

    # Best picks from the table: Gaussian filter on Gaussian noise; median on S&P
    save_row(
        [cv2.GaussianBlur(g_noisy, (3, 3), 0), cv2.medianBlur(s_noisy, 3)],
        ["Gaussian noise → Gaussian k=3", "S&P → median k=3"],
        OUT / "t2_restored.png",
    )


def task3_sharpen():
    img = cv2.imread(str(IMG / "colourful.jpg"))
    save_row(
        [img, op_sharpen(img, 0.5), op_sharpen(img, 1.0), op_sharpen(img, 2.0)],
        ["Original", "amount=0.5", "amount=1.0", "amount=2.0 (halos)"],
        OUT / "t3_sharpen.png",
    )


def task4_edges(gray):
    mine = my_sobel(gray)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ref = np.sqrt(gx**2 + gy**2)

    max_diff = float(np.max(np.abs(mine - ref)))
    (OUT / "t4_sobel_maxdiff.txt").write_text(
        f"max_abs_diff={max_diff:.6e}\n"
        f"note=own Sobel vs cv2.Sobel magnitude; reflect pad matches OpenCV default\n",
        encoding="utf-8",
    )

    mine_u8 = cv2.convertScaleAbs(mine)
    ref_u8 = cv2.convertScaleAbs(ref)
    save_row([mine_u8, ref_u8], ["My Sobel magnitude", "OpenCV Sobel magnitude"], OUT / "t4_sobel.png")

    canny_imgs, titles = [], []
    for lo, hi in ((30, 90), (80, 160), (150, 250)):
        canny_imgs.append(cv2.Canny(gray, lo, hi))
        titles.append(f"Canny {lo}/{hi}")
    save_row(canny_imgs, titles, OUT / "t4_canny.png")

    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    log_u8 = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_64F))
    save_row([mine_u8, log_u8], ["Sobel magnitude", "Laplacian of Gaussian"], OUT / "t4_log.png")


def task5_morph():
    img = cv2.imread(str(IMG / "coins.jpg"))
    vis, count, steps = op_count(img)
    (OUT / "t5_count.txt").write_text(
        f"object_count={count}\n"
        "structuring_element=5x5 ellipse\n"
        "roi=cropped to jewellery band (excludes card text)\n"
        "pipeline=Otsu INV -> open -> close -> connectedComponents + area gate\n",
        encoding="utf-8",
    )
    save_row(
        [
            steps["threshold"],
            steps["eroded"],
            steps["dilated"],
            steps["opened"],
            steps["closed"],
        ],
        ["Threshold", "Erode", "Dilate", "Open", f"Close (count={count})"],
        OUT / "t5_morph.png",
    )
    cv2.imwrite(str(OUT / "t5_count_vis.png"), vis)
    save_row([img, vis], ["Input earrings photo", f"Counted objects = {count}"], OUT / "t5_before_after.png")


def main():
    np.random.seed(0)
    clean = ensure_clean_gray()
    task1_colour()
    task2_noise(clean)
    task3_sharpen()
    task4_edges(clean)
    task5_morph()
    print("Practical 1 figures saved to", OUT)
    print((OUT / "t4_sobel_maxdiff.txt").read_text(encoding="utf-8").strip())
    print((OUT / "t5_count.txt").read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
