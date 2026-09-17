"""Task 5 — Stereo SGBM disparity; show numDisparities / blockSize effects."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
HERE = Path(__file__).resolve().parent
OUT = HERE / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def run():
    left = cv2.imread(str(IMG / "stereo_left.jpg"), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(IMG / "stereo_right.jpg"), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise FileNotFoundError("Need images/stereo_left.jpg and stereo_right.jpg")
    # Match sizes
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))

    outs = []
    params = [(64, 5), (96, 5), (64, 15), (128, 9)]
    for nd, bs in params:
        # numDisparities must be divisible by 16
        nd = max(16, (nd // 16) * 16)
        bs = bs if bs % 2 == 1 else bs + 1
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=nd,
            blockSize=bs,
            P1=8 * 3 * bs * bs,
            P2=32 * 3 * bs * bs,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2,
        )
        disp = stereo.compute(left, right).astype(np.float32) / 16.0
        vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colour = cv2.applyColorMap(vis, cv2.COLORMAP_PLASMA)
        label = colour.copy()
        cv2.putText(
            label,
            f"nd={nd} bs={bs}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        outs.append(label)
        cv2.imwrite(str(OUT / f"t5_disp_nd{nd}_bs{bs}.png"), label)

    # 2x2 grid
    top = np.hstack(outs[:2])
    bot = np.hstack(outs[2:4])
    if top.shape[1] != bot.shape[1]:
        bot = cv2.resize(bot, (top.shape[1], bot.shape[0]))
    grid = np.vstack([top, bot])
    pair = np.hstack([
        cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(right, cv2.COLOR_GRAY2BGR),
    ])
    cv2.imwrite(str(OUT / "t5_stereo_pair.png"), pair)
    cv2.imwrite(str(OUT / "t5_stereo.png"), grid)

    # Quick depth-ordering check: mean disparity in left-third vs right-third
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=9)
    disp = stereo.compute(left, right).astype(np.float32) / 16.0
    h, w = disp.shape
    left_mean = float(np.mean(disp[:, : w // 3][disp[:, : w // 3] > 0]))
    right_mean = float(np.mean(disp[:, 2 * w // 3 :][disp[:, 2 * w // 3 :] > 0]))
    (OUT / "t5_notes.txt").write_text(
        "Nearer objects → larger disparity → brighter in the normalised map.\n"
        f"mean_disp_left_third={left_mean:.2f} mean_disp_right_third={right_mean:.2f}\n"
        "Larger numDisparities searches farther matches (handles closer objects / bigger baseline) "
        "but costs more and can add noise. Larger blockSize smooths the map but melts fine edges.\n"
        "My pair is a rough side-shift (or published-style pair), not a calibrated stereo rig — "
        "imperfect rectification causes streaking / holes. A proper rig would fix epipolar "
        "alignment so matching stays on the same scanline.\n",
        encoding="utf-8",
    )
    print("Task5 stereo →", OUT)


if __name__ == "__main__":
    run()
