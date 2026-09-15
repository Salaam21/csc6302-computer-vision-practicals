"""Gradio leaf classifier for Practical 4."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
DEVICE = "cpu"

tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = ckpt["classes"]
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(classes))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, classes


model, CLASSES = load_model()


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
    inputs=gr.Image(type="pil", label="Leaf photo"),
    outputs=gr.Label(num_top_classes=3),
    title="CSC 6302 Practical 4 - leaf triage",
    description="Uses my fine-tuned MobileNet on the leaf_dataset in images/.",
)

if __name__ == "__main__":
    demo.launch()
