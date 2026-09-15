"""Gradio segmentation demo for Practical 5."""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np


def otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2RGB), "Otsu binary mask"


def adaptive(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2RGB), "Adaptive Gaussian"


def watershed(img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(bgr, markers)
    vis = bgr.copy()
    vis[markers == -1] = [0, 0, 255]
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), f"Watershed objects ~ {n - 1}"


def deeplab(img):
    try:
        import torch
        from torchvision import transforms
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    except Exception:
        return img, "DeepLab needs torch and torchvision on this machine."

    small = cv2.resize(img, (320, 180))
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).eval()
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        classes = model(tf(small).unsqueeze(0))["out"][0].argmax(0).byte().numpy()
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, (21, 3), dtype=np.uint8)
    palette[0] = 0
    colour = palette[classes]
    overlay = (0.5 * small + 0.5 * colour).astype(np.uint8)
    return overlay, "DeepLab overlay (slow on CPU)"


def run(image, mode):
    if image is None:
        return None, "Upload an image first."
    if mode == "Otsu":
        return otsu(image)
    if mode == "adaptive":
        return adaptive(image)
    if mode == "watershed":
        return watershed(image)
    return deeplab(image)


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="numpy", label="Upload"),
        gr.Dropdown(["Otsu", "adaptive", "watershed", "DeepLabV3"], value="watershed", label="Mode"),
    ],
    outputs=[gr.Image(type="numpy"), gr.Textbox(label="Notes")],
    title="CSC 6302 Practical 5 - segmentation",
    description="DeepLab can take a while on CPU.",
)

if __name__ == "__main__":
    demo.launch()
