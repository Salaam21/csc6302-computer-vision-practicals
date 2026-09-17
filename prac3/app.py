"""
CSC 6302 Practical 3 — Gradio detection app
Back-ends: Haar faces / YOLO objects
Launch: python prac3/app.py
"""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_yolo = None


def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO

        _yolo = YOLO("yolov8n.pt")  # downloads once (~6 MB), runs on CPU
    return _yolo


def detect(image, backend, conf):
    """Gradio callback. Input image is RGB numpy."""
    if image is None:
        return None
    img = np.asarray(image).copy()
    if img.ndim != 3:
        return img

    if backend == "Haar (faces)":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # conf slider unused for Haar — map loosely to minNeighbors
        min_neighbors = 3 if conf < 0.3 else (5 if conf < 0.6 else 8)
        boxes = face_cascade.detectMultiScale(gray, 1.1, min_neighbors, minSize=(30, 30))
        out = img.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(
                out,
                "face",
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2,
            )
        return out

    # YOLO expects RGB or BGR; ultralytics accepts numpy RGB fine
    yolo = get_yolo()
    results = yolo(img, conf=float(conf), verbose=False)
    plotted_bgr = results[0].plot()
    return cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="Upload image"),
        gr.Radio(
            ["Haar (faces)", "YOLO (objects)"],
            value="YOLO (objects)",
            label="Detector back-end",
        ),
        gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence / strictness"),
    ],
    outputs=gr.Image(type="numpy", label="Annotated output"),
    title="CSC 6302 Practical 3 — Object Detection",
    description=(
        "Haar cascade finds faces. YOLO finds everyday objects (person, car, bottle, …). "
        "For Haar, the slider maps to minNeighbors (stricter → fewer false faces). "
        "For YOLO, it is the confidence threshold."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
