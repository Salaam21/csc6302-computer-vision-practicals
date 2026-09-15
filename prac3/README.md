# CSC 6302 Practical 3 - Detection

**Student:** Munyeme Sinyani (SINYANI) - University of Zambia

Template matching, Haar cascades, and YOLO on the same kinds of scenes.

## Install

```bash
cd csc6302-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

First YOLO run downloads `yolov8n.pt` (~6 MB).

## Run

Figures and timing:

```bash
python prac3/run_prac3.py
```

Gradio:

```bash
python prac3/app.py
```

## Hugging Face Space

I create Space `csc6302-prac3-SINYANI` (Gradio), push `app.py` and `requirements.txt`, and set secrets if I use CI deploy.

## AI Use

Composer helped with ideas for connecting OpenCV, Ultralytics YOLO, and the Gradio UI. I picked confidence values and wrote up the timing CSV myself.
