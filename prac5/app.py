"""
CSC 6302 Practical 5 — Segmentation Studio (Gradio).
Modes: Otsu / adaptive / watershed / DeepLabV3.
DeepLab is lazy-loaded; expect a few seconds on CPU.
"""
from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

_deeplab = None
_deeplab_tf = None

VOC_CLASSES = [
    "bg", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "table", "dog", "horse", "motorbike", "person", "plant",
    "sheep", "sofa", "train", "tv",
]


def _palette():
    rng = np.random.default_rng(42)
    palette = rng.integers(40, 230, size=(21, 3), dtype=np.uint8)
    palette[0] = 0
    palette[15] = np.array([220, 40, 40], dtype=np.uint8)
    return palette


PALETTE = _palette()


def otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    overlay = img.copy()
    overlay[m > 0] = (0.4 * overlay[m > 0] + 0.6 * np.array([40, 200, 80])).astype(np.uint8)
    return overlay, "Otsu overlay (ink/foreground highlighted)"


def adaptive(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 8
    )
    overlay = img.copy()
    overlay[m > 0] = (0.4 * overlay[m > 0] + 0.6 * np.array([40, 120, 220])).astype(np.uint8)
    return overlay, "Adaptive Gaussian overlay"


def watershed(img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Prefer darker-object cut; fall back to Otsu INV for general uploads
    thresh = ((blur < 125).astype(np.uint8)) * 255
    if thresh.mean() < 5 or thresh.mean() > 200:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if thresh.mean() > 127:
            thresh = 255 - thresh
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    peak = 0.5 * dist.max() if dist.max() > 0 else 1
    _, sure_fg = cv2.threshold(dist, peak, 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(bgr, markers)
    vis = bgr.copy()
    rng = np.random.default_rng(0)
    for lab in range(2, int(markers.max()) + 1):
        colour = rng.integers(40, 255, size=3).tolist()
        vis[markers == lab] = (0.45 * vis[markers == lab] + 0.55 * np.array(colour)).astype(np.uint8)
    vis[markers == -1] = [0, 0, 255]
    count = n - 1
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), f"Watershed object count ≈ {count}"


def _get_deeplab():
    global _deeplab, _deeplab_tf
    if _deeplab is None:
        import torch
        from torchvision import transforms
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

        _deeplab = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).eval()
        _deeplab_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # warm-up silence
        _ = torch
    return _deeplab, _deeplab_tf


def deeplab(img):
    import torch

    model, tf = _get_deeplab()
    h, w = img.shape[:2]
    scale = 480 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
    with torch.no_grad():
        classes = model(tf(small).unsqueeze(0))["out"][0].argmax(0).byte().numpy()
    colour = PALETTE[classes]
    overlay = (0.55 * small + 0.45 * colour).astype(np.uint8)
    present = sorted({int(c) for c in np.unique(classes) if c > 0})
    legend = ", ".join(f"{i}:{VOC_CLASSES[i]}" for i in present[:12]) or "(background only)"
    return overlay, f"DeepLabV3 (VOC). Classes: {legend}. Note: first run downloads weights; CPU takes a few seconds."


def run(image, mode):
    if image is None:
        return None, "Upload an image first."
    img = np.asarray(image)
    if img.ndim != 3:
        return img, "Need a colour image."
    if mode == "Otsu":
        return otsu(img)
    if mode == "adaptive":
        return adaptive(img)
    if mode == "watershed":
        return watershed(img)
    return deeplab(img)


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="numpy", label="Upload image"),
        gr.Dropdown(
            ["Otsu", "adaptive", "watershed", "DeepLabV3"],
            value="watershed",
            label="Segmentation mode",
        ),
    ],
    outputs=[
        gr.Image(type="numpy", label="Segmentation overlay"),
        gr.Textbox(label="Notes / object count / class legend"),
    ],
    title="CSC 6302 Practical 5 — Segmentation Studio (SINYANI)",
    description=(
        "Munyeme Sinyani · University of Zambia. "
        "Classical modes are instant. **DeepLabV3 on free CPU takes a few seconds** "
        "(weights download on first use). Overlay + class legend for the deep model; "
        "watershed reports an object count."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
