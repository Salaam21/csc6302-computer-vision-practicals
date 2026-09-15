"""
Capstone - maize leaf triage (Agriculture track).
Upload a leaf photo, get class + confidence + short guidance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CLASSES = ["healthy", "rust", "blight"]
GUIDANCE = {
    "healthy": "Looks healthy. Keep monitoring; no fungicide needed from this photo alone.",
    "rust": "Possible rust. Isolate affected leaves if you can, and ask an agronomist before spraying.",
    "blight": "Possible blight pattern. Do not treat this as a diagnosis - get a field extension officer to confirm.",
}
DISCLAIMER = (
    "NOT FOR REAL FARM DECISIONS. This is a student coursework for CSC 6302. "
    "Wrong advice can waste money or miss a real outbreak."
)

_model = None
_torch_classes = None


def try_load_torch():
    global _model, _torch_classes
    ckpt_path = ROOT.parent / "prac4" / "model.pt"
    if not ckpt_path.exists():
        return False
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _torch_classes = ckpt["classes"]
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(_torch_classes))
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        _model = (m, transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))
        return True
    except Exception:
        return False


try_load_torch()


def blurry(img: Image.Image, thresh: float = 25.0) -> bool:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(arr)
    return float(gx.var() + gy.var()) < thresh


def _textured(colour, size=(128, 128), seed=0):
    # for tests - solid colour plus noise so blur gate passes
    rng = np.random.default_rng(seed)
    base = np.zeros((size[1], size[0], 3), np.uint8)
    base[:, :] = colour
    noise = rng.integers(-25, 26, base.shape, dtype=np.int16)
    return Image.fromarray(np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8))


def heuristic_predict(img: Image.Image) -> dict[str, float]:
    # fallback when prac4 model.pt is missing
    arr = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32)
    mean = arr.reshape(-1, 3).mean(0)
    scores = {
        "healthy": float(mean[1] - mean[0]),
        "rust": float(mean[2] * 0.3 + mean[0] - mean[1]),
        "blight": float(200 - mean.mean()),
    }
    exps = {k: np.exp(v / 20.0) for k, v in scores.items()}
    s = sum(exps.values())
    return {k: float(exps[k] / s) for k in CLASSES}


def torch_predict(img: Image.Image) -> dict[str, float]:
    import torch

    m, tf = _model
    x = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(m(x), 1)[0]
    return {_torch_classes[i]: float(probs[i]) for i in range(len(_torch_classes))}


def predict(img: Image.Image, conf_gate: float = 0.45):
    if img is None:
        return "No image", {}, DISCLAIMER
    if img.mode not in ("RGB", "RGBA", "L"):
        return "Unsupported file type. Please upload a normal photo (JPG/PNG).", {}, DISCLAIMER
    if blurry(img):
        return (
            "Image looks too blurry for a reliable read. Retake closer and steadier.",
            {},
            DISCLAIMER,
        )
    probs = torch_predict(img) if _model is not None else heuristic_predict(img)
    top = max(probs, key=probs.get)
    conf = probs[top]
    if conf < conf_gate:
        return (
            f"Unsure (top guess {top} at {conf:.0%}). Try a clearer leaf-centred photo.",
            probs,
            DISCLAIMER,
        )
    msg = f"Prediction: {top} ({conf:.0%}). {GUIDANCE[top]}"
    return msg, probs, DISCLAIMER


def preprocess_shape(img: Image.Image):
    arr = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)


def build_demo():
    import gradio as gr

    demo = gr.Blocks(title="CSC6302 Capstone - Maize leaf triage")
    with demo:
        gr.Markdown("# Maize leaf disease triage")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload leaf photo")
            with gr.Column():
                gate = gr.Slider(0.2, 0.9, value=0.45, label="Confidence gate")
                out_msg = gr.Textbox(label="Result")
                out_probs = gr.Label(label="Class probabilities")
                out_note = gr.Textbox(label="Must NOT be used for")
        btn = gr.Button("Analyse")
        btn.click(predict, [inp, gate], [out_msg, out_probs, out_note])
    return demo


if __name__ == "__main__":
    build_demo().launch()
