"""Small Gradio app for Practical 3 - Haar vs YOLO."""

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
        _yolo = YOLO("yolov8n.pt")
    return _yolo


def detect(image, backend, conf):
    if image is None:
        return None
    # Gradio gives RGB not BGR
    img = image.copy()
    if backend == "Haar (faces)":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        boxes = face_cascade.detectMultiScale(gray, 1.1, 5)
        out = img.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(out, "face", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return out
    yolo = get_yolo()
    results = yolo(img, conf=float(conf), verbose=False)
    plotted_bgr = results[0].plot()
    return cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="Upload image"),
        gr.Radio(["Haar (faces)", "YOLO (objects)"], value="YOLO (objects)", label="Backend"),
        gr.Slider(0.05, 0.95, value=0.25, label="confidence"),
    ],
    outputs=gr.Image(type="numpy", label="Annotated"),
    title="CSC 6302 Practical 3 - object detection",
    description="Haar for faces, YOLO for general objects. Confidence slider is YOLO only.",
)

if __name__ == "__main__":
    demo.launch()
