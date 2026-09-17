"""
CSC 6302 Practical 4 — Gradio bird classifier (local + Hugging Face).
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
DEVICE = "cpu"


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = ckpt["classes"]
    img_size = int(ckpt.get("img_size", 160))
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(classes))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return m, classes, tf


model, CLASSES, tf = load_model()


def predict(img: Image.Image):
    if img is None:
        return {}
    x = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0]
    top = torch.topk(probs, k=min(3, len(CLASSES)))
    return {CLASSES[i]: float(p) for p, i in zip(top.values, top.indices)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload bird / poultry photo"),
    outputs=gr.Label(num_top_classes=3),
    title="CSC 6302 Practical 4 — bird triage (SINYANI)",
    description=(
        "Munyeme Sinyani. Fine-tuned MobileNetV3-Small on chicken / duck / turkey / goose / quail. "
        "Returns top-3 class probabilities."
    ),
)

if __name__ == "__main__":
    demo.launch()
