"""Run Practical 5 segmentation tasks and save figures for my report."""

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


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    # IoU from the notes - intersection over union
    a = (pred > 0).astype(np.uint8)
    b = (gt > 0).astype(np.uint8)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def save_row(imgs, titles, path):
    fig, axes = plt.subplots(1, len(imgs), figsize=(3.2 * len(imgs), 3.2))
    if len(imgs) == 1:
        axes = [axes]
    for ax, im, t in zip(axes, imgs, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def task1_threshold():
    doc = cv2.imread(str(IMG / "document.jpg"), cv2.IMREAD_GRAYSCALE)
    crop = doc[60:280, 40:360]
    _, gt = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    gt = cv2.morphologyEx(gt, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cv2.imwrite(str(OUT / "t1_handmade_mask.png"), gt)

    fixed = (crop < 140).astype(np.uint8) * 255
    _, otsu = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adapt_m = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10)
    adapt_g = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10)
    scores = {
        "fixed": iou(fixed, gt),
        "otsu": iou(otsu, gt),
        "adaptive_mean": iou(adapt_m, gt),
        "adaptive_gaussian": iou(adapt_g, gt),
    }
    with open(OUT / "t1_iou.csv", "w", encoding="utf-8") as f:
        f.write("method,iou\n")
        for k, v in scores.items():
            f.write(f"{k},{v:.4f}\n")
    save_row(
        [crop, fixed, otsu, adapt_g, gt],
        ["Crop", f"Fixed IoU={scores['fixed']:.2f}", f"Otsu {scores['otsu']:.2f}",
         f"AdaptG {scores['adaptive_gaussian']:.2f}", "Reference mask"],
        OUT / "t1_threshold.png",
    )


def task2_watershed():
    img = cv2.imread(str(IMG / "coins.jpg"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers_ws = markers.copy()
    cv2.watershed(img, markers_ws)
    vis = img.copy()
    vis[markers_ws == -1] = [0, 0, 255]
    count = n - 1
    (OUT / "t2_count.txt").write_text(f"watershed_objects~={count}\n", encoding="utf-8")
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save_row(
        [thresh, opening, dist_u8, sure_fg, vis],
        ["Thresh", "Open", "Distance", "Sure FG", f"Watershed (~{count})"],
        OUT / "t2_watershed.png",
    )


def task3_grabcut():
    img = cv2.imread(str(IMG / "portrait.jpg"))
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (40, 20, 240, 380)
    cv2.grabCut(img, mask, rect, bgd, fgd, 1, cv2.GC_INIT_WITH_RECT)
    mask1 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    out1 = img * mask1[:, :, None]
    cv2.grabCut(img, mask, rect, bgd, fgd, 4, cv2.GC_EVAL)
    mask5 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    out5 = img * mask5[:, :, None]
    mask[0:20, :] = cv2.GC_BGD
    mask[380:, :] = cv2.GC_BGD
    cv2.grabCut(img, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    maskh = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    outh = img * maskh[:, :, None]
    save_row([img, out1, out5, outh], ["Original", "1 iter", "5 iters", "With mask hint"], OUT / "t3_grabcut.png")


def colourise_semantic(classes: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, size=(21, 3), dtype=np.uint8)
    palette[0] = 0
    return palette[classes]


def task4_deep():
    try:
        import torch
        from torchvision import transforms
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    except Exception as e:
        (OUT / "t4_deep_SKIPPED.txt").write_text(f"torchvision models unavailable: {e}\n")
        return

    street = cv2.imread(str(IMG / "street.jpg"))
    rgb = cv2.cvtColor(street, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (320, 180))

    seg = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).eval()
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    t0 = time.perf_counter()
    with torch.no_grad():
        out = seg(tf(small).unsqueeze(0))["out"][0]
    sem_ms = 1000 * (time.perf_counter() - t0)
    classes = out.argmax(0).byte().numpy()
    colour = colourise_semantic(classes)
    overlay = (0.5 * small + 0.5 * colour).astype(np.uint8)
    cv2.imwrite(str(OUT / "t4_deeplab.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    ins = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = ins([torch.from_numpy(small / 255.0).permute(2, 0, 1).float()])[0]
    ins_ms = 1000 * (time.perf_counter() - t0)
    keep = pred["scores"] > 0.5
    masks = pred["masks"][keep, 0] > 0.5
    inst = small.copy()
    for i, m in enumerate(masks):
        colour = np.array([int(37 * (i + 1) % 255), int(99 * (i + 3) % 255), int(17 * (i + 5) % 255)])
        inst[m.numpy()] = (0.5 * inst[m.numpy()] + 0.5 * colour).astype(np.uint8)
    cv2.imwrite(str(OUT / "t4_maskrcnn.png"), cv2.cvtColor(inst, cv2.COLOR_RGB2BGR))
    (OUT / "t4_timing.txt").write_text(f"deeplab_ms={sem_ms:.1f}\nmaskrcnn_ms={ins_ms:.1f}\n")
    save_row(
        [cv2.cvtColor(small, cv2.COLOR_RGB2BGR), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), cv2.cvtColor(inst, cv2.COLOR_RGB2BGR)],
        ["Street", "DeepLab semantic", "Mask R-CNN instances"],
        OUT / "t4_semantic_vs_instance.png",
    )


def main():
    task1_threshold()
    task2_watershed()
    task3_grabcut()
    task4_deep()
    print("Practical 5 figures saved to", OUT)


if __name__ == "__main__":
    main()
