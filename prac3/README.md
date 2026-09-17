# CSC 6302 Practical 3 — Detecting Things

**Student:** Munyeme Sinyani (SINYANI) · Computer number **2025024319** · University of Zambia

Template matching → Haar (Viola–Jones) → YOLOv8n → Gradio app.

## Hugging Face Spaces (what it is + bonus)

Hugging Face Spaces is a **free website that runs your Gradio app in the cloud**, so a marker can open a link without installing Python.

1. Go to [https://huggingface.co/spaces](https://huggingface.co/spaces) (sign in).
2. **Create new Space** → name e.g. `csc6302-prac3-SINYANI` → SDK **Gradio** → public.
3. Upload (or git-push) these files into the Space:
   - `app.py` (this folder’s app)
   - `requirements.txt` with:
     ```
     opencv-python-headless
     ultralytics
     gradio
     numpy
     ```
4. Wait for the build; copy the public URL into this README top line.

**Space link (fill after you host):** `https://huggingface.co/spaces/Salaam21/csc6302-prac3-SINYANI`

## Install (local)

```bash
cd csc6302-computer-vision-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r prac3/requirements.txt
```

First YOLO run downloads `yolov8n.pt` (~6 MB) if weights are not already under `prac7/weights/`.

## Media

| File | Role |
|------|------|
| `images/object.jpg` | Template matching + YOLO everyday object |
| `images/group_faces.jpg` / `images/group.jpg` | Consented group photo for Haar + YOLO |
| `images/street_clip.mp4` | Phone street video (Haar + YOLO frame timing / clips) |

## Reproduce

```bash
python prac3/run_prac3.py
python prac3/app.py
```

Report: `prac3/report/Practical3_Report.pdf`

## AI use

Composer helped with Gradio wiring and figure-script layout. Parameter choices, ethics reflection, and explanations are mine to defend.
