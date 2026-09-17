"""
Capstone - Road & transport track.
Vehicle counting / classification from roadside frames, plus a simple
pothole severity hint from dashcam-style photos.

Mobile companion (Allan's Car Counter — separate Flutter app, not Blindspot):
see folder allans-car-counter / package com.allans.car_counter
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
WEIGHTS = ROOT / "weights" / "yolov8n.pt"
VEHICLE_CLASSES = ("bicycle", "car", "motorcycle", "bus", "truck")
DISCLAIMER = (
    "COURSEWORK DEMO ONLY (CSC 6302). Counts are model estimates, not official "
    "traffic statistics. Pothole severity is a rough image heuristic — not a "
    "road-maintenance inspection."
)

_yolo = None


def ensure_weights() -> Path:
    """Use local weights if present; otherwise let Ultralytics download yolov8n."""
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if WEIGHTS.exists():
        return WEIGHTS
    return Path("yolov8n.pt")


def get_model():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO

        _yolo = YOLO(str(ensure_weights()))
        # cache under prac7/weights after first download if possible
        try:
            src = Path(_yolo.ckpt_path) if hasattr(_yolo, "ckpt_path") else None
        except (AttributeError, OSError, TypeError):
            src = None
        if src and src.exists() and not WEIGHTS.exists():
            try:
                WEIGHTS.write_bytes(src.read_bytes())
            except OSError:
                pass
    return _yolo


def blurry(img: Image.Image, thresh: float = 18.0) -> bool:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(arr)
    return float(gx.var() + gy.var()) < thresh


def _textured(colour, size=(256, 256), seed=0):
    rng = np.random.default_rng(seed)
    base = np.zeros((size[1], size[0], 3), np.uint8)
    base[:, :] = colour
    noise = rng.integers(-30, 31, base.shape, dtype=np.int16)
    return Image.fromarray(np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8))


def empty_counts() -> dict[str, int]:
    return {c: 0 for c in VEHICLE_CLASSES}


def orient_frame(img: Image.Image) -> Image.Image:
    """Fix EXIF orientation; many phone roadside shots are stored sideways."""
    return ImageOps.exif_transpose(img.convert("RGB"))


def _detect_on(model, rgb: Image.Image, conf_gate: float):
    arr = np.asarray(rgb)
    results = model.predict(arr, conf=conf_gate, verbose=False)
    counts = empty_counts()
    boxes_out: list[tuple[str, float, tuple[float, float, float, float]]] = []
    for r in results:
        names = r.names
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls.item())
            label = names.get(cls_id, str(cls_id))
            if label not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf.item())
            counts[label] += 1
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            boxes_out.append((label, conf, (x1, y1, x2, y2)))
    return counts, boxes_out


def annotate(rgb: Image.Image, boxes) -> Image.Image:
    out = rgb.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    for label, conf, (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(0, 180, 255), width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{label} {conf:.0%}", fill=(255, 220, 0), font=font)
    return out


def format_counts(counts: dict[str, int]) -> str:
    parts = [f"{n} {name}{'s' if n != 1 else ''}" for name, n in counts.items() if n]
    total = sum(counts.values())
    if not parts:
        return "No vehicles detected above the confidence gate."
    return f"Total vehicles: {total}. Breakdown: " + ", ".join(parts) + "."


def count_vehicles(img: Image.Image, conf_gate: float = 0.35):
    """Detect and count roadside vehicles. Returns message, counts, annotated PIL, note."""
    if img is None:
        return "No image", empty_counts(), None, DISCLAIMER
    # Reject clearly non-image / empty payloads
    try:
        w, h = img.size
    except (AttributeError, TypeError):
        return "Could not read image. Upload JPG/PNG.", empty_counts(), None, DISCLAIMER
    if w < 16 or h < 16:
        return "Image too small. Upload a clearer roadside frame.", empty_counts(), None, DISCLAIMER
    if img.mode not in ("RGB", "RGBA", "L", "P"):
        return "Unsupported file type. Upload JPG/PNG.", empty_counts(), None, DISCLAIMER
    if blurry(img):
        return (
            "Frame looks too blurry for a reliable count. Use a steadier roadside shot.",
            empty_counts(),
            img.convert("RGB"),
            DISCLAIMER,
        )

    model = get_model()
    base = orient_frame(img)
    candidates = [base]
    # sideways phone frames are common — keep the orientation with more vehicle hits
    for angle in (90, 270):
        candidates.append(base.rotate(angle, expand=True))

    best_counts, best_boxes, best_rgb = empty_counts(), [], base
    best_total = -1
    best_mean_conf = 0.0
    for cand in candidates:
        counts, boxes = _detect_on(model, cand, conf_gate)
        total = sum(counts.values())
        mean_conf = float(np.mean([c for _, c, _ in boxes])) if boxes else 0.0
        if total > best_total or (total == best_total and mean_conf > best_mean_conf):
            best_total = total
            best_mean_conf = mean_conf
            best_counts, best_boxes, best_rgb = counts, boxes, cand

    annotated = annotate(best_rgb, best_boxes)
    if best_total == 0:
        msg = (
            "UNSURE: no vehicles above the confidence gate. "
            "Try a closer roadside frame, lower the gate slightly, or check lighting."
        )
    elif best_mean_conf < conf_gate + 0.05 and best_total <= 1:
        msg = (
            f"UNSURE (low confidence ~{best_mean_conf:.0%}): "
            + format_counts(best_counts)
            + " Treat as a weak estimate."
        )
    else:
        msg = format_counts(best_counts)
    return msg, best_counts, annotated, DISCLAIMER


def rate_pothole(img: Image.Image):
    """
    Lightweight dashcam heuristic (not a trained pothole detector).
    Looks for dark irregular patches on a road-like lower region.
    """
    if img is None:
        return "No image", {}, None, DISCLAIMER
    rgb = orient_frame(img)
    arr = np.asarray(rgb)
    h, _w = arr.shape[:2]
    roi = arr[int(h * 0.45) :, :, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # dark patches relative to road
    thr = max(40, int(np.percentile(blur, 25)))
    mask = (blur < thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    frac = float(mask.mean() / 255.0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) > 0.002 * mask.size]
    n_blobs = len(big)

    if frac < 0.02 and n_blobs == 0:
        level, advice = "low", "Little dark-patch evidence in the lower frame."
    elif frac < 0.08 or n_blobs <= 2:
        level, advice = "medium", "Possible damaged patches — verify on site."
    else:
        level, advice = "high", "Large/many dark regions — treat as priority for a human check."

    vis = roi.copy()
    overlay = vis.copy()
    overlay[mask > 0] = (220, 60, 60)
    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
    full = arr.copy()
    full[int(h * 0.45) :, :, :] = vis
    out = Image.fromarray(full)
    scores = {"low": 0.0, "medium": 0.0, "high": 0.0, "dark_fraction": round(frac, 3), "blobs": n_blobs}
    scores[level] = 1.0
    msg = f"Pothole severity hint: {level.upper()} (dark_frac={frac:.1%}, blobs={n_blobs}). {advice}"
    return msg, scores, out, DISCLAIMER


def build_demo():
    import gradio as gr

    demo = gr.Blocks(title="CSC6302 Capstone - Vehicle counting")
    with demo:
        gr.Markdown("# Road & transport — vehicle counting")
        gr.Markdown(
            "Track: **vehicle counting and classification** from roadside footage, "
            "plus a **pothole severity hint** from dashcam-style images.\n\n"
            "Phone app: **Allan's Car Counter** (Flutter, on-device TFLite) — separate from Blindspot. "
            "This Gradio page uses YOLOv8n for the coursework demo."
        )
        gr.Markdown(DISCLAIMER)
        with gr.Tab("Vehicle count"):
            with gr.Row():
                vin = gr.Image(type="pil", label="Roadside / traffic frame")
                with gr.Column():
                    gate = gr.Slider(0.15, 0.8, value=0.35, label="Confidence gate")
                    vmsg = gr.Textbox(label="Count summary")
                    vcounts = gr.Label(label="Per-class counts")
                    vnote = gr.Textbox(label="Disclaimer")
            vout = gr.Image(type="pil", label="Annotated frame")
            vbtn = gr.Button("Count vehicles")
            vbtn.click(count_vehicles, [vin, gate], [vmsg, vcounts, vout, vnote])
        with gr.Tab("Pothole severity"):
            with gr.Row():
                pin = gr.Image(type="pil", label="Dashcam / road surface photo")
                with gr.Column():
                    pmsg = gr.Textbox(label="Severity hint")
                    pscores = gr.Label(label="Scores / diagnostics")
                    pnote = gr.Textbox(label="Disclaimer")
            pout = gr.Image(type="pil", label="Highlighted lower ROI")
            pbtn = gr.Button("Rate pothole hint")
            pbtn.click(rate_pothole, [pin], [pmsg, pscores, pout, pnote])
    return demo


if __name__ == "__main__":
    import os

    # SHARE=1 → temporary public *.gradio.live link (good for a quick demo)
    share = os.environ.get("SHARE", "0").strip() in ("1", "true", "True", "yes")
    build_demo().launch(share=share, server_name="127.0.0.1")
