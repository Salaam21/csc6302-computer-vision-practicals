"""
CSC 6302 Practical 2 — Features that Matter

From-scratch: harris() structure tensor + NMS (Task 1).
OpenCV OK for ORB/SIFT/BFMatcher/findHomography — but NOT cv2.Stitcher.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
OUT = Path(__file__).resolve().parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def harris(gray, k=0.04, win_sigma=1.5, thresh_rel=0.01, nms_size=9):
    """
    Own Harris detector. Returns (N,2) array of (x, y) corner coordinates.
    Gradients use the same Sobel kernels as Practical 1 (NumPy / filter2D only
    for the 3x3 multiply-accumulate — not cv2.cornerHarris).
    """
    g = gray.astype(np.float64)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    # Same kernels as Prac1 my_sobel; filter2D is just convolution
    Ix = cv2.filter2D(g, -1, kx, borderType=cv2.BORDER_REFLECT_101)
    Iy = cv2.filter2D(g, -1, ky, borderType=cv2.BORDER_REFLECT_101)

    Ix2 = cv2.GaussianBlur(Ix * Ix, (0, 0), win_sigma)
    Iy2 = cv2.GaussianBlur(Iy * Iy, (0, 0), win_sigma)
    Ixy = cv2.GaussianBlur(Ix * Iy, (0, 0), win_sigma)

    det = Ix2 * Iy2 - Ixy * Ixy
    trace = Ix2 + Iy2
    R = det - k * (trace**2)

    thresh = thresh_rel * float(R.max()) if R.max() > 0 else 0.0
    mask = R > thresh
    dil = cv2.dilate(R, np.ones((nms_size, nms_size), np.uint8))
    nms = (R == dil) & mask
    ys, xs = np.where(nms)
    return np.column_stack([xs, ys]).astype(np.float64)


def match_ratio(des1, des2, ratio=0.75, norm=cv2.NORM_HAMMING):
    """Brute-force knn + Lowe's ratio test."""
    bf = cv2.BFMatcher(norm)
    pairs = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def overlay_corners(img, pts, colour=(0, 0, 255)):
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y in np.asarray(pts).reshape(-1, 2):
        cv2.circle(out, (int(x), int(y)), 3, colour, -1, lineType=cv2.LINE_AA)
    return out


def save_row(imgs, titles, path, figsize_scale=3.6):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(figsize_scale * n, figsize_scale))
    if n == 1:
        axes = [axes]
    for ax, im, t in zip(axes, imgs, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def task1():
    board = cv2.imread(str(IMG / "checkerboard.png"), cv2.IMREAD_GRAYSCALE)
    natural_bgr = cv2.imread(str(IMG / "prac2_natural.jpg"))
    if natural_bgr is None:
        natural_bgr = cv2.imread(str(IMG / "prac2_dusk.jpg"))
    natural = cv2.cvtColor(natural_bgr, cv2.COLOR_BGR2GRAY)

    lines = ["image,detector,count,ms"]
    for name, gray in [("checkerboard", board), ("natural", natural)]:
        t0 = time.perf_counter()
        my = harris(gray)
        t_harris = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        shi = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
        t_shi = (time.perf_counter() - t0) * 1000
        shi_pts = shi.reshape(-1, 2) if shi is not None else np.zeros((0, 2))

        fast = cv2.FastFeatureDetector_create(threshold=15)
        # Synthetic checkerboards are perfectly flat; FAST needs a little blur to fire
        gray_fast = cv2.GaussianBlur(gray, (3, 3), 0.8) if name == "checkerboard" else gray
        t0 = time.perf_counter()
        fast_kps = fast.detect(gray_fast, None)
        t_fast = (time.perf_counter() - t0) * 1000
        fast_pts = (
            np.array([kp.pt for kp in fast_kps], dtype=float)
            if fast_kps
            else np.zeros((0, 2))
        )

        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        save_row(
            [
                overlay_corners(base, my, (0, 0, 255)),
                overlay_corners(base, shi_pts, (0, 200, 0)),
                overlay_corners(base, fast_pts, (255, 128, 0)),
            ],
            [
                f"My Harris ({len(my)}) {t_harris:.0f}ms",
                f"Shi-Tomasi ({len(shi_pts)}) {t_shi:.0f}ms",
                f"FAST ({len(fast_pts)}) {t_fast:.0f}ms",
            ],
            OUT / f"t1_{name}_corners.png",
        )
        lines.append(f"{name},harris,{len(my)},{t_harris:.2f}")
        lines.append(f"{name},shi-tomasi,{len(shi_pts)},{t_shi:.2f}")
        lines.append(f"{name},fast,{len(fast_pts)},{t_fast:.2f}")

    (OUT / "t1_counts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # keep a colour natural preview for the report
    cv2.imwrite(str(OUT / "t1_natural_input.png"), natural_bgr)


def _transform_keypoints(kps, M):
    pts = np.float32([kp.pt for kp in kps]).reshape(-1, 1, 2)
    return cv2.transform(pts, M).reshape(-1, 2)


def task2_invariance():
    img = cv2.imread(str(IMG / "prac2_show.jpg"))
    if img is None:
        img = cv2.imread(str(IMG / "prac2_natural.jpg"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0

    detectors = {
        "ORB": (cv2.ORB_create(1200), cv2.NORM_HAMMING),
        "SIFT": (cv2.SIFT_create(nfeatures=1200), cv2.NORM_L2),
    }
    transforms = {
        "rot30": cv2.getRotationMatrix2D((cx, cy), 30, 1.0),
        "rot90": cv2.getRotationMatrix2D((cx, cy), 90, 1.0),
        "scale05": np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=float),
    }

    results = []
    for dname, (det, norm) in detectors.items():
        kp0, des0 = det.detectAndCompute(gray, None)
        for tname, M in transforms.items():
            warped = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR)
            kp1, des1 = det.detectAndCompute(warped, None)
            if des0 is None or des1 is None or len(kp0) == 0 or len(kp1) == 0:
                results.append((dname, tname, 0.0, 0, 0))
                continue
            good = match_ratio(des0, des1, 0.75, norm)
            pred = _transform_keypoints(kp0, M)
            correct = 0
            for m in good:
                px, py = pred[m.queryIdx]
                # skip predictions that left the canvas
                if not (0 <= px < w and 0 <= py < h):
                    continue
                qx, qy = kp1[m.trainIdx].pt
                if (px - qx) ** 2 + (py - qy) ** 2 <= 9.0:
                    correct += 1
            # repeatability = correct / original keypoints (brief)
            pct = 100.0 * correct / max(len(kp0), 1)
            results.append((dname, tname, pct, len(kp0), correct))

    with open(OUT / "t2_repeatability.csv", "w", encoding="utf-8") as f:
        f.write("detector,transform,repeatability_pct,orig_kps,correct_matches\n")
        for dname, tname, pct, n0, c in results:
            f.write(f"{dname},{tname},{pct:.2f},{n0},{c}\n")

    # quick visual: ORB vs SIFT keypoints on the show image
    orb = cv2.ORB_create(400)
    sift = cv2.SIFT_create(nfeatures=400)
    ko, _ = orb.detectAndCompute(gray, None)
    ks, _ = sift.detectAndCompute(gray, None)
    vo = cv2.drawKeypoints(img, ko, None, color=(0, 255, 0))
    vs = cv2.drawKeypoints(img, ks, None, color=(255, 128, 0))
    save_row([vo, vs], [f"ORB kps={len(ko)}", f"SIFT kps={len(ks)}"], OUT / "t2_descriptors.png")


def task3_matching():
    img1 = cv2.imread(str(IMG / "pano_left.jpg"))
    img2 = cv2.imread(str(IMG / "pano_mid.jpg"))
    # SIFT + L2 for denser geometric matches on outdoor texture
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_pairs = bf.knnMatch(des1, des2, k=2)
    raw = [m for m, _n in raw_pairs]

    counts = {}
    for ratio in (0.6, 0.75, 0.9):
        counts[ratio] = len(match_ratio(des1, des2, ratio, cv2.NORM_L2))

    good = match_ratio(des1, des2, 0.75, cv2.NORM_L2)
    raw_vis = cv2.drawMatches(img1, kp1, img2, kp2, raw[:100], None, flags=2)
    good_vis = cv2.drawMatches(img1, kp1, img2, kp2, good[:100], None, flags=2)
    save_row(
        [raw_vis, good_vis],
        [f"Raw NN (~{len(raw)} shown 100)", f"Ratio 0.75 ({len(good)} good)"],
        OUT / "t3_matches.png",
        figsize_scale=5.0,
    )
    with open(OUT / "t3_ratio_counts.txt", "w", encoding="utf-8") as f:
        for r, c in counts.items():
            f.write(f"ratio={r}: matches={c}\n")
        f.write("matcher=SIFT+BF+Lowe\n")
    return img1, img2, kp1, kp2, good


def stitch(img_src, img_dst, kp_src, kp_dst, good, feather=True):
    """
    Warp img_src into img_dst's frame via RANSAC homography, then composite.
    Feather blending in the overlap (not cv2.Stitcher).
    """
    if len(good) < 4:
        return img_dst.copy(), 0
    src = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return img_dst.copy(), 0
    inliers = int(mask.sum()) if mask is not None else 0

    h1, w1 = img_src.shape[:2]
    h2, w2 = img_dst.shape[:2]
    corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    warped_c = cv2.perspectiveTransform(corners, H)
    all_c = np.concatenate(
        [warped_c, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)],
        axis=0,
    )
    xmin, ymin = np.int32(all_c.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(all_c.max(axis=0).ravel() + 0.5)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=float)
    size = (int(xmax - xmin), int(ymax - ymin))

    warped = cv2.warpPerspective(img_src, T @ H, size)
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    # place destination image
    y0, x0 = -ymin, -xmin
    canvas[y0 : y0 + h2, x0 : x0 + w2] = img_dst

    if not feather:
        # simple overwrite where warped has content
        mask_w = (warped.sum(axis=2) > 0)
        canvas[mask_w] = warped[mask_w]
        return canvas, inliers

    # feather: average in overlap, prefer each image elsewhere
    mask_w = (warped.sum(axis=2) > 0).astype(np.float32)
    mask_c = (canvas.sum(axis=2) > 0).astype(np.float32)
    overlap = (mask_w * mask_c) > 0
    out = canvas.astype(np.float32)
    wimg = warped.astype(np.float32)
    only_w = (mask_w > 0) & (mask_c == 0)
    out[only_w] = wimg[only_w]
    out[overlap] = 0.5 * out[overlap] + 0.5 * wimg[overlap]
    return np.clip(out, 0, 255).astype(np.uint8), inliers


def task4_panorama(img1, img2, kp1, kp2, good):
    pano, inliers = stitch(img1, img2, kp1, kp2, good, feather=True)
    cv2.imwrite(str(OUT / "t4_panorama.png"), pano)
    lines = [f"pair_left_mid_ransac_inliers={inliers}", "blend=feather_average_in_overlap"]

    img3 = cv2.imread(str(IMG / "pano_right.jpg"))
    sift = cv2.SIFT_create(nfeatures=2000)
    kp_p, des_p = sift.detectAndCompute(pano, None)
    kp3, des3 = sift.detectAndCompute(img3, None)
    if des_p is not None and des3 is not None:
        good2 = match_ratio(des_p, des3, 0.75, cv2.NORM_L2)
        if len(good2) >= 8:
            # Rematch right -> pano so stitch(src=right, dst=pano)
            good_rp = match_ratio(des3, des_p, 0.75, cv2.NORM_L2)
            if len(good_rp) >= 8:
                wider, inl2 = stitch(img3, pano, kp3, kp_p, good_rp, feather=True)
                cv2.imwrite(str(OUT / "t4_panorama_wide.png"), wider)
                lines.append(f"third_image_ransac_inliers={inl2}")
            else:
                lines.append("third_image_ransac_inliers=0")
        else:
            lines.append("third_image_ransac_inliers=0")
    (OUT / "t4_inliers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def task5_robustness(img1, img2):
    """Degrade mid image; report RANSAC inliers for each degradation."""
    degraded = {
        "none": img2,
        "blur": cv2.GaussianBlur(img2, (15, 15), 4),
        "dark60": np.clip(img2.astype(np.float32) * 0.4, 0, 255).astype(np.uint8),
    }
    ok, enc = cv2.imencode(".jpg", img2, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
    degraded["jpeg10"] = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    lines = ["degradation,good_matches,ransac_inliers,panorama_survived"]
    for name, bad in degraded.items():
        kp2, des2 = sift.detectAndCompute(bad, None)
        if des1 is None or des2 is None:
            lines.append(f"{name},0,0,no")
            continue
        good = match_ratio(des1, des2, 0.75, cv2.NORM_L2)
        if len(good) < 8:
            lines.append(f"{name},{len(good)},0,no")
            continue
        pano, inliers = stitch(img1, bad, kp1, kp2, good, feather=True)
        survived = "yes" if inliers >= 12 else "no"
        lines.append(f"{name},{len(good)},{inliers},{survived}")
        if name != "none":
            cv2.imwrite(str(OUT / f"t5_{name}_pano.png"), pano)
            cv2.imwrite(str(OUT / f"t5_{name}_input.png"), bad)

    (OUT / "t5_robustness.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    # refresh overlapping crops if prepare script exists
    prep = Path(__file__).resolve().parent / "prepare_images.py"
    if prep.exists():
        import runpy

        runpy.run_path(str(prep))

    task1()
    task2_invariance()
    img1, img2, kp1, kp2, good = task3_matching()
    task4_panorama(img1, img2, kp1, kp2, good)
    task5_robustness(img1, img2)
    print("Practical 2 figures saved to", OUT)
    print((OUT / "t1_counts.txt").read_text(encoding="utf-8"))
    print((OUT / "t2_repeatability.csv").read_text(encoding="utf-8"))
    print((OUT / "t3_ratio_counts.txt").read_text(encoding="utf-8"))
    print((OUT / "t4_inliers.txt").read_text(encoding="utf-8"))
    print((OUT / "t5_robustness.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
