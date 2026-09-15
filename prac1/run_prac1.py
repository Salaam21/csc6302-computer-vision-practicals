"""
Run Practical 1 tasks and save figures for my report.
"""

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


def save_row(images, titles, path):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
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


def task1_colour():
    # load colourful scene, look at HSV and LAB channels, then pick red
    img = cv2.imread(str(IMG / "colourful.jpg"))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    save_row(
        [hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]],
        ["H (hue)", "S (saturation)", "V (brightness)"],
        OUT / "t1_hsv_channels.png",
    )
    save_row(
        [lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]],
        ["L (lightness)", "A (green-red)", "B (blue-yellow)"],
        OUT / "t1_lab_channels.png",
    )
    save_row([img, op_pick(img)], ["Original", "HSV colour pick"], OUT / "t1_picker.png")


def add_gaussian(gray, sigma=15):
    noise = np.random.normal(0, sigma, gray.shape)
    return np.clip(gray.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_sap(gray, amount=0.05):
    # salt and pepper: random pixels to 0 or 255
    out = gray.copy()
    n = out.size
    n_flip = int(amount * n)
    flat_idx = np.random.choice(n, n_flip, replace=False)
    coords = np.unravel_index(flat_idx, out.shape)
    half = n_flip // 2
    out[coords[0][:half], coords[1][:half]] = 0
    out[coords[0][half:], coords[1][half:]] = 255
    return out


def task2_noise():
    clean = cv2.imread(str(IMG / "clean_gray.png"), cv2.IMREAD_GRAYSCALE)
    g_noisy = add_gaussian(clean, 15)
    s_noisy = add_sap(clean, 0.05)
    save_row(
        [clean, g_noisy, s_noisy],
        ["Clean", "Gaussian sigma=15", "Salt and pepper 5%"],
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
                rows.append({
                    "noise": noise_name,
                    "filter": fname,
                    "ksize": k,
                    "psnr": round(psnr(clean, restored), 3),
                })

    with open(OUT / "t2_psnr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["noise", "filter", "ksize", "psnr"])
        w.writeheader()
        w.writerows(rows)

    # usually: gaussian filter for gaussian noise, median for salt-and-pepper
    save_row(
        [cv2.GaussianBlur(g_noisy, (5, 5), 0), cv2.medianBlur(s_noisy, 3)],
        ["Gauss noise -> Gaussian filter", "S&P -> median"],
        OUT / "t2_restored.png",
    )


def task3_sharpen():
    img = cv2.imread(str(IMG / "colourful.jpg"))
    save_row(
        [img, op_sharpen(img, 0.5), op_sharpen(img, 1.0), op_sharpen(img, 2.0)],
        ["Original", "amount=0.5", "amount=1.0", "amount=2.0 (halos)"],
        OUT / "t3_sharpen.png",
    )


def task4_edges():
    gray = cv2.imread(str(IMG / "clean_gray.png"), cv2.IMREAD_GRAYSCALE)

    mine = my_sobel(gray)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ref = np.sqrt(gx ** 2 + gy ** 2)

    # normalise both so I can compare shapes, not absolute scale
    mine_n = mine / (mine.max() + 1e-8)
    ref_n = ref / (ref.max() + 1e-8)
    max_diff = float(np.max(np.abs(mine_n - ref_n)))
    (OUT / "t4_sobel_maxdiff.txt").write_text(
        f"max_abs_diff_normalised={max_diff:.6f}\n", encoding="utf-8"
    )

    mine_u8 = np.clip(mine_n * 255, 0, 255).astype(np.uint8)
    ref_u8 = np.clip(ref_n * 255, 0, 255).astype(np.uint8)
    save_row([mine_u8, ref_u8], ["My Sobel", "OpenCV Sobel"], OUT / "t4_sobel.png")

    canny_imgs, titles = [], []
    for lo, hi in ((30, 90), (80, 160), (150, 250)):
        canny_imgs.append(cv2.Canny(gray, lo, hi))
        titles.append(f"Canny {lo}/{hi}")
    save_row(canny_imgs, titles, OUT / "t4_canny.png")

    # LoG = blur first, then Laplacian
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    log_u8 = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_64F))
    save_row([mine_u8, log_u8], ["Sobel magnitude", "LoG"], OUT / "t4_log.png")


def task5_morph():
    img = cv2.imread(str(IMG / "coins.jpg"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # small ellipse - earrings are small, big kernel would glue them together
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, kernel, iterations=1)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # count with size filter (same idea as process.op_count)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    h, w = gray.shape
    min_a = (h * w) * 0.0003
    max_a = (h * w) * 0.02
    count = sum(1 for i in range(1, n) if min_a <= stats[i, cv2.CC_STAT_AREA] <= max_a)

    (OUT / "t5_count.txt").write_text(
        f"object_count={count}\nstructuring_element=3x3 ellipse\n"
        "note=earrings on white card; area filter drops text speckles\n",
        encoding="utf-8",
    )
    save_row(
        [mask, eroded, dilated, opened, closed],
        ["Threshold", "Erode", "Dilate", "Open", f"Close -> count={count}"],
        OUT / "t5_morph.png",
    )
    vis, _ = op_count(img)
    cv2.imwrite(str(OUT / "t5_count_vis.png"), vis)


def main():
    np.random.seed(0)
    task1_colour()
    task2_noise()
    task3_sharpen()
    task4_edges()
    task5_morph()
    print("Practical 1 figures saved to", OUT)


if __name__ == "__main__":
    main()
