"""
CSC 6302 Practical 5 — Image Segmentation
Run: python prac5/prepare_images.py && python prac5/run_prac5.py
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
HERE = Path(__file__).resolve().parent
OUT = HERE / "report" / "figures"
INPUTS = HERE / "inputs"
OUT.mkdir(parents=True, exist_ok=True)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union in NumPy (my own implementation)."""
    a = pred > 0
    b = gt > 0
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def save_row(imgs, titles, path, figsize_scale=3.0):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(figsize_scale * n, figsize_scale))
    if n == 1:
        axes = [axes]
    for ax, im, t in zip(axes, imgs, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.shape[2] == 3 else im
            ax.imshow(rgb)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def task1_threshold():
    # Ensure inputs exist
    mask_path = INPUTS / "handmade_mask.png"
    crop_path = INPUTS / "t1_crop.png"
    if not mask_path.exists() or not crop_path.exists():
        from prepare_images import make_document, make_handmade_mask

        make_document()
        make_handmade_mask()

    crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert crop is not None and gt is not None
    if gt.shape != crop.shape:
        gt = cv2.resize(gt, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

    # (a) fixed global threshold chosen by eye (works on dark side, fails on bright)
    fixed_t = 90
    fixed = (crop < fixed_t).astype(np.uint8) * 255
    # (b) Otsu
    _, otsu = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # (c) adaptive mean + Gaussian — local windows follow the lighting ramp
    adapt_m = cv2.adaptiveThreshold(
        crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )
    adapt_g = cv2.adaptiveThreshold(
        crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )

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
    (OUT / "t1_note.txt").write_text(
        "Adaptive wins when lighting ramps across the page: a single threshold "
        "cannot mark dark-side ink and bright-side ink at once. Otsu picks one "
        "global cut; fixed is worse if my guess is off. Adaptive windows follow "
        "local contrast, so ink stays foreground under the ramp.\n"
        f"fixed_threshold={fixed_t}\n",
        encoding="utf-8",
    )
    cv2.imwrite(str(OUT / "t1_handmade_mask.png"), gt)
    cv2.imwrite(str(OUT / "t1_crop.png"), crop)
    save_row(
        [crop, fixed, otsu, adapt_m, adapt_g, gt],
        [
            "Crop",
            f"Fixed T={fixed_t}\nIoU={scores['fixed']:.3f}",
            f"Otsu\nIoU={scores['otsu']:.3f}",
            f"Adapt mean\nIoU={scores['adaptive_mean']:.3f}",
            f"Adapt Gauss\nIoU={scores['adaptive_gaussian']:.3f}",
            "Hand-made GT",
        ],
        OUT / "t1_threshold.png",
        figsize_scale=2.6,
    )
    print("Task1 IoU:", scores)


def task2_watershed():
    path = IMG / "coins.jpg"
    if not path.exists():
        from prepare_images import make_coins

        make_coins()
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Tray texture fools Otsu; coins are darker — fixed cut chosen on the histogram
    thresh = ((blur < 125).astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_cc, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers_ws = markers.copy()
    cv2.watershed(img, markers_ws)
    vis = img.copy()
    vis[markers_ws == -1] = [0, 0, 255]
    # Colourise regions
    coloured = img.copy()
    rng = np.random.default_rng(0)
    for lab in range(2, markers_ws.max() + 1):
        colour = rng.integers(40, 255, size=3).tolist()
        coloured[markers_ws == lab] = (
            0.45 * coloured[markers_ws == lab] + 0.55 * np.array(colour)
        ).astype(np.uint8)
    coloured[markers_ws == -1] = [0, 0, 255]
    count = int(markers_ws.max() - 1)  # labels 2..max are objects (1=bg)
    # Prefer connected-component marker count (n_cc - 1 background-less)
    count_markers = n_cc - 1
    manual = 15
    manual_file = INPUTS / "t2_manual_count.txt"
    if manual_file.exists():
        for line in manual_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("manual_count="):
                manual = int(line.split("=")[1])
    (OUT / "t2_count.txt").write_text(
        f"manual_count={manual}\n"
        f"watershed_markers={count_markers}\n"
        f"watershed_labels={count}\n"
        "note=Touching coins merge under connected components; distance peaks "
        "give one marker per coin when the peak threshold is tuned. Over-split "
        "happens if specular blobs create extra peaks; merges happen when two "
        "coins share one distance ridge below the sure-FG cut.\n",
        encoding="utf-8",
    )
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Save every intermediate (brief requirement)
    cv2.imwrite(str(OUT / "t2_01_gray.png"), gray)
    cv2.imwrite(str(OUT / "t2_02_thresh.png"), thresh)
    cv2.imwrite(str(OUT / "t2_03_opening.png"), opening)
    cv2.imwrite(str(OUT / "t2_04_sure_bg.png"), sure_bg)
    cv2.imwrite(str(OUT / "t2_05_distance.png"), dist_u8)
    cv2.imwrite(str(OUT / "t2_06_sure_fg.png"), sure_fg)
    cv2.imwrite(str(OUT / "t2_07_unknown.png"), unknown)
    cv2.imwrite(str(OUT / "t2_08_watershed.png"), coloured)
    save_row(
        [thresh, opening, dist_u8, sure_fg, coloured],
        ["Thresh", "Open", "Distance", "Sure FG", f"WS ~{count_markers} (manual {manual})"],
        OUT / "t2_watershed.png",
    )
    print(f"Task2 watershed markers={count_markers} manual={manual}")


def task3_grabcut():
    img = cv2.imread(str(IMG / "portrait.jpg"))
    if img is None:
        raise FileNotFoundError("images/portrait.jpg missing")
    # Shrink for speed while keeping quality for the figure
    scale = 640 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    h, w = img.shape[:2]
    # Bounding rect around the jar + hand (portrait is creatine bottle)
    rect = (int(0.18 * w), int(0.05 * h), int(0.62 * w), int(0.9 * h))
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd, fgd, 1, cv2.GC_INIT_WITH_RECT)
    mask1 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    out1 = img * mask1[:, :, None]
    cv2.grabCut(img, mask, rect, bgd, fgd, 4, cv2.GC_EVAL)
    mask5 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    out5 = img * mask5[:, :, None]
    # Mask hints: force top corners / bottom patterned cloth as definite background;
    # force centre of jar as definite foreground
    hint = mask.copy()
    hint[0 : int(0.08 * h), :] = cv2.GC_BGD
    hint[int(0.92 * h) :, :] = cv2.GC_BGD
    hint[:, 0 : int(0.08 * w)] = cv2.GC_BGD
    hint[:, int(0.92 * w) :] = cv2.GC_BGD
    cy, cx = int(0.45 * h), int(0.55 * w)
    cv2.circle(hint, (cx, cy), int(0.08 * min(h, w)), int(cv2.GC_FGD), -1)
    bgd2 = np.zeros((1, 65), np.float64)
    fgd2 = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, hint, None, bgd2, fgd2, 3, cv2.GC_INIT_WITH_MASK)
    maskh = np.where((hint == 2) | (hint == 0), 0, 1).astype("uint8")
    outh = img * maskh[:, :, None]
    # Draw rect on a copy for the figure
    rect_vis = img.copy()
    x, y, rw, rh = rect
    cv2.rectangle(rect_vis, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
    save_row(
        [rect_vis, out1, out5, outh],
        ["Rect seed", "1 iteration", "5 iterations", "Mask hints"],
        OUT / "t3_grabcut.png",
    )
    (OUT / "t3_note.txt").write_text(
        "GrabCut iterates a Gaussian Mixture Model (GMM) for foreground and "
        "background colours, then a graph-cut that labels each pixel. Extra "
        "iterations refine the GMMs; brush hints clamp some pixels as hard FG/BG.\n",
        encoding="utf-8",
    )
    print("Task3 GrabCut done")


# Pascal VOC 21 labels used by DeepLabV3 torchvision weights
VOC_CLASSES = [
    "bg", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "table", "dog", "horse", "motorbike", "person", "plant",
    "sheep", "sofa", "train", "tv",
]


def _voc_palette():
    rng = np.random.default_rng(42)
    palette = rng.integers(40, 230, size=(21, 3), dtype=np.uint8)
    palette[0] = 0
    # Distinct person colour
    palette[15] = np.array([220, 40, 40], dtype=np.uint8)
    return palette


def task4_deep():
    import torch
    from torchvision import transforms
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

    # Street scene with vehicles; group photo for 3+ people demo
    street = cv2.imread(str(IMG / "street.jpg"))
    group = cv2.imread(str(IMG / "group_faces.jpg"))
    if street is None or group is None:
        raise FileNotFoundError("Need images/street.jpg and images/group_faces.jpg")

    def prep_rgb(bgr, max_side=640):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        return rgb

    street_rgb = prep_rgb(street, 480)
    group_rgb = prep_rgb(group, 640)

    seg = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).eval()
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    palette = _voc_palette()

    def run_semantic(rgb):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = seg(tf(rgb).unsqueeze(0))["out"][0]
        ms = 1000 * (time.perf_counter() - t0)
        classes = out.argmax(0).byte().numpy()
        colour = palette[classes]
        overlay = (0.55 * rgb + 0.45 * colour).astype(np.uint8)
        return classes, overlay, ms

    classes_s, overlay_s, sem_ms_street = run_semantic(street_rgb)
    classes_g, overlay_g, sem_ms_group = run_semantic(group_rgb)

    ins = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()

    def run_instance(rgb, score_thresh=0.5):
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = ins([torch.from_numpy(rgb / 255.0).permute(2, 0, 1).float()])[0]
        ms = 1000 * (time.perf_counter() - t0)
        keep = pred["scores"] > score_thresh
        masks = pred["masks"][keep, 0] > 0.5
        labels = pred["labels"][keep]
        scores = pred["scores"][keep]
        inst = rgb.copy()
        rng = np.random.default_rng(1)
        for i, m in enumerate(masks):
            colour = rng.integers(30, 255, size=3)
            m_np = m.numpy()
            inst[m_np] = (0.45 * inst[m_np] + 0.55 * colour).astype(np.uint8)
        return inst, int(masks.shape[0]), ms, labels.tolist(), scores.tolist()

    inst_s, n_s, ins_ms_street, _, _ = run_instance(street_rgb)
    inst_g, n_g, ins_ms_group, labels_g, _ = run_instance(group_rgb)

    # Person pixels: semantic = one class blob; instance = separate people
    person_sem = (classes_g == 15).astype(np.uint8) * 255
    person_sem_vis = group_rgb.copy()
    person_sem_vis[person_sem > 0] = (
        0.4 * person_sem_vis[person_sem > 0] + 0.6 * np.array([220, 40, 40])
    ).astype(np.uint8)

    cv2.imwrite(str(OUT / "t4_deeplab_street.png"), cv2.cvtColor(overlay_s, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / "t4_maskrcnn_street.png"), cv2.cvtColor(inst_s, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / "t4_deeplab_group.png"), cv2.cvtColor(overlay_g, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / "t4_maskrcnn_group.png"), cv2.cvtColor(inst_g, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / "t4_person_semantic_blob.png"), cv2.cvtColor(person_sem_vis, cv2.COLOR_RGB2BGR))

    save_row(
        [
            cv2.cvtColor(group_rgb, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(person_sem_vis, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(inst_g, cv2.COLOR_RGB2BGR),
        ],
        [
            "Group (≥3 people)",
            "DeepLab: one 'person' class",
            f"Mask R-CNN: {n_g} instances",
        ],
        OUT / "t4_semantic_vs_instance.png",
    )
    save_row(
        [
            cv2.cvtColor(street_rgb, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(overlay_s, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(inst_s, cv2.COLOR_RGB2BGR),
        ],
        ["Street", "DeepLab semantic", f"Mask R-CNN ({n_s})"],
        OUT / "t4_street_both.png",
    )

    # Failure note: patterned cloth / chitenge-like table in lowlight breakfast photo
    fail_img = cv2.imread(str(IMG / "lowlight.jpg"))
    fail_note = (
        "Failure case: my patterned tablecloth (chitenge-like geometry) is not a VOC/COCO "
        "class. DeepLab maps busy fabric to 'background' or nearby furniture/plant bits; "
        "Mask R-CNN may hallucinate a 'dining table' box but cannot name the print. "
        "Models only know labels they were trained on.\n"
    )
    if fail_img is not None:
        fail_rgb = prep_rgb(fail_img, 480)
        _, fail_ov, fail_ms = run_semantic(fail_rgb)
        cv2.imwrite(str(OUT / "t4_failure_deeplab.png"), cv2.cvtColor(fail_ov, cv2.COLOR_RGB2BGR))
        fail_note += f"failure_deeplab_ms={fail_ms:.1f}\n"

    (OUT / "t4_timing.txt").write_text(
        f"deeplab_street_ms={sem_ms_street:.1f}\n"
        f"maskrcnn_street_ms={ins_ms_street:.1f}\n"
        f"deeplab_group_ms={sem_ms_group:.1f}\n"
        f"maskrcnn_group_ms={ins_ms_group:.1f}\n"
        f"group_instances={n_g}\n"
        f"street_instances={n_s}\n",
        encoding="utf-8",
    )
    (OUT / "t4_unet_note.txt").write_text(
        "DeepLabV3 is an encoder–decoder idea like U-Net: a ResNet encoder builds "
        "rich semantics at low resolution; ASPP + decoder upsample back to pixels. "
        "U-Net adds skip links from encoder to decoder; DeepLab leans on dilated "
        "convolutions / ASPP instead of dense skips, but the contract–expand story "
        "is the same lecture idea.\n",
        encoding="utf-8",
    )
    (OUT / "t4_failure.txt").write_text(fail_note, encoding="utf-8")
    print(
        f"Task4 DeepLab street {sem_ms_street:.0f}ms / MaskRCNN {ins_ms_street:.0f}ms; "
        f"group instances={n_g}"
    )


def main():
    task1_threshold()
    task2_watershed()
    task3_grabcut()
    task4_deep()
    print("Practical 5 figures →", OUT)


if __name__ == "__main__":
    main()
