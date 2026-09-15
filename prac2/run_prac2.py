"""Run Practical 2 tasks and save figures for my report."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
OUT = Path(__file__).resolve().parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def harris(gray, k=0.04, win_sigma=1.5, thresh_rel=0.01, nms_size=9):
    # Harris corners from the notes: structure tensor then R = det - k*trace^2
    g = gray.astype(np.float64)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float)
    Ix = cv2.filter2D(g, -1, kx)
    Iy = cv2.filter2D(g, -1, ky)
    Ix2 = cv2.GaussianBlur(Ix * Ix, (0, 0), win_sigma)
    Iy2 = cv2.GaussianBlur(Iy * Iy, (0, 0), win_sigma)
    Ixy = cv2.GaussianBlur(Ix * Iy, (0, 0), win_sigma)
    det = Ix2 * Iy2 - Ixy * Ixy
    trace = Ix2 + Iy2
    R = det - k * (trace ** 2)
    thresh = thresh_rel * R.max()
    mask = R > thresh
    # non-max suppression so corners do not pile up
    dil = cv2.dilate(R, np.ones((nms_size, nms_size)))
    nms = (R == dil) & mask
    ys, xs = np.where(nms)
    return np.column_stack([xs, ys])


def match_ratio(des1, des2, ratio=0.75, norm=cv2.NORM_HAMMING):
    # Lowe ratio test from the notes
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
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), 3, colour, -1)
    return out


def save_pair(imgs, titles, path):
    fig, axes = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 3.5))
    if len(imgs) == 1:
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
    natural = cv2.imread(str(IMG / "colourful.jpg"), cv2.IMREAD_GRAYSCALE)
    lines = []
    for name, gray in [("checkerboard", board), ("natural", natural)]:
        my = harris(gray)
        shi = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10)
        shi_pts = shi.reshape(-1, 2) if shi is not None else np.zeros((0, 2))
        fast = cv2.FastFeatureDetector_create(threshold=25)
        fast_kps = fast.detect(gray, None)
        fast_pts = np.array([kp.pt for kp in fast_kps], float) if fast_kps else np.zeros((0, 2))
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        save_pair(
            [
                overlay_corners(base, my, (0, 0, 255)),
                overlay_corners(base, shi_pts, (0, 255, 0)),
                overlay_corners(base, fast_pts, (255, 0, 0)),
            ],
            [f"Harris ({len(my)})", f"Shi-Tomasi ({len(shi_pts)})", f"FAST ({len(fast_pts)})"],
            OUT / f"t1_{name}_corners.png",
        )
        lines.append(f"{name}: harris={len(my)} shi={len(shi_pts)} fast={len(fast_pts)}")
    (OUT / "t1_counts.txt").write_text("\n".join(lines) + "\n")


def _transform_keypoints(kps, M):
    pts = np.float32([kp.pt for kp in kps]).reshape(-1, 1, 2)
    pred = cv2.transform(pts, M).reshape(-1, 2)
    return pred


def task2_invariance():
    img = cv2.imread(str(IMG / "colourful.jpg"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = []
    detectors = {
        "ORB": (cv2.ORB_create(1000), cv2.NORM_HAMMING),
        "SIFT": (cv2.SIFT_create(), cv2.NORM_L2),
    }
    transforms = {
        "rot30": cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), 30, 1.0),
        "rot90": cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), 90, 1.0),
        "scale05": np.array([[0.5, 0, 0], [0, 0.5, 0]], float),
    }
    for dname, (det, norm) in detectors.items():
        kp0, des0 = det.detectAndCompute(gray, None)
        for tname, M in transforms.items():
            warped = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
            kp1, des1 = det.detectAndCompute(warped, None)
            if des0 is None or des1 is None or len(kp0) == 0 or len(kp1) == 0:
                results.append((dname, tname, 0.0))
                continue
            good = match_ratio(des0, des1, 0.75, norm)
            pred = _transform_keypoints(kp0, M)
            correct = 0
            for m in good:
                px, py = pred[m.queryIdx]
                qx, qy = kp1[m.trainIdx].pt
                if (px - qx) ** 2 + (py - qy) ** 2 <= 9:  # within 3 px
                    correct += 1
            pct = 100.0 * correct / max(len(kp0), 1)
            results.append((dname, tname, pct))
    with open(OUT / "t2_repeatability.csv", "w", encoding="utf-8") as f:
        f.write("detector,transform,repeatability_pct\n")
        for r in results:
            f.write(f"{r[0]},{r[1]},{r[2]:.2f}\n")


def task3_matching():
    img1 = cv2.imread(str(IMG / "pano_left.jpg"))
    img2 = cv2.imread(str(IMG / "pano_mid.jpg"))
    orb = cv2.ORB_create(1500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_pairs = bf.knnMatch(des1, des2, k=2)
    raw = [m for m, n in raw_pairs]
    counts = {}
    for ratio in (0.6, 0.75, 0.9):
        good = match_ratio(des1, des2, ratio, cv2.NORM_HAMMING)
        counts[ratio] = len(good)
    good = match_ratio(des1, des2, 0.75, cv2.NORM_HAMMING)
    raw_vis = cv2.drawMatches(img1, kp1, img2, kp2, raw[:80], None, flags=2)
    good_vis = cv2.drawMatches(img1, kp1, img2, kp2, good[:80], None, flags=2)
    save_pair([raw_vis, good_vis], [f"Raw NN (~{len(raw)} matches)", f"Ratio 0.75 ({len(good)})"], OUT / "t3_matches.png")
    with open(OUT / "t3_ratio_counts.txt", "w", encoding="utf-8") as f:
        for r, c in counts.items():
            f.write(f"ratio={r}: matches={c}\n")
    return img1, img2, kp1, kp2, good


def stitch(img1, img2, kp1, kp2, good):
    if len(good) < 4:
        return img2.copy(), 0
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return img2.copy(), 0
    inliers = int(mask.sum()) if mask is not None else 0
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    all_c = np.concatenate(
        (warped_corners, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)),
        axis=0,
    )
    [xmin, ymin] = np.int32(all_c.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_c.max(axis=0).ravel() + 0.5)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], float)
    canvas = cv2.warpPerspective(img1, T @ H, (xmax - xmin, ymax - ymin))
    # simple paste of img2 on top
    canvas[-ymin : h2 - ymin, -xmin : w2 - xmin] = img2
    return canvas, inliers


def task4_panorama(img1, img2, kp1, kp2, good):
    pano, inliers = stitch(img1, img2, kp1, kp2, good)
    cv2.imwrite(str(OUT / "t4_panorama.png"), pano)
    (OUT / "t4_inliers.txt").write_text(f"ransac_inliers={inliers}\n")
    img3 = cv2.imread(str(IMG / "pano_right.jpg"))
    orb = cv2.ORB_create(1500)
    kp_p, des_p = orb.detectAndCompute(pano, None)
    kp3, des3 = orb.detectAndCompute(img3, None)
    if des_p is not None and des3 is not None:
        good2 = match_ratio(des_p, des3, 0.75, cv2.NORM_HAMMING)
        if len(good2) >= 8:
            wider, inl2 = stitch(pano, img3, kp_p, kp3, good2)
            cv2.imwrite(str(OUT / "t4_panorama_wide.png"), wider)
            with open(OUT / "t4_inliers.txt", "a", encoding="utf-8") as f:
                f.write(f"third_image_inliers={inl2}\n")


def task5_robustness(img1, img2, kp1, des1_unused=None):
    # I mess up img2 on purpose and see if matching still works
    degraded = {
        "blur": cv2.GaussianBlur(img2, (11, 11), 3),
        "dark": (img2.astype(np.float32) * 0.4).astype(np.uint8),
        "jpeg10": None,
    }
    ok, enc = cv2.imencode(".jpg", img2, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
    degraded["jpeg10"] = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    orb = cv2.ORB_create(1500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    lines = ["degradation,inliers,survived"]
    for name, bad in degraded.items():
        kp2, des2 = orb.detectAndCompute(bad, None)
        if des1 is None or des2 is None:
            lines.append(f"{name},0,no")
            continue
        good = match_ratio(des1, des2, 0.75, cv2.NORM_HAMMING)
        if len(good) < 8:
            lines.append(f"{name},{0},no")
            continue
        _, inliers = stitch(img1, bad, kp1, kp2, good)
        lines.append(f"{name},{inliers},{'yes' if inliers >= 8 else 'no'}")
    (OUT / "t5_robustness.csv").write_text("\n".join(lines) + "\n")


def main():
    task1()
    task2_invariance()
    img1, img2, kp1, kp2, good = task3_matching()
    task4_panorama(img1, img2, kp1, kp2, good)
    task5_robustness(img1, img2, kp1)
    print("Practical 2 figures saved to", OUT)


if __name__ == "__main__":
    main()
