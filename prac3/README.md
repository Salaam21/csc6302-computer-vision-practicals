# CSC 6302 Practical 3 — Detecting Things

**Student:** Munyeme Sinyani (SINYANI) · Computer number **2025024319** · University of Zambia  

**For markers:** how to review this practical.

## What this practical covers

Template matching → Haar (Viola–Jones) faces → YOLOv8n objects → Gradio demo app.

## Quick review (preferred)

```bash
cd csc6302-computer-vision-practicals
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r prac3/requirements.txt
python prac3/run_prac3.py       # regenerates report figures
python prac3/app.py             # opens Gradio locally
```

- Report: `prac3/report/Practical3_Report.pdf`
- Figures: `prac3/report/figures/`

## Gradio app (local)

`python prac3/app.py` starts the interactive demo (template / Haar / YOLO tabs). First YOLO run may download `yolov8n.pt` (~6 MB).

## Hugging Face Space

A public Gradio Space was intended at:

https://huggingface.co/spaces/Salaam21/csc6302-prac3-SINYANI

Creating **Gradio Spaces** on this account currently returns **402 Payment Required** (Hugging Face PRO). Please use the **local Gradio** path above for marking. Same restriction as Practicals 4–5.

## Media used

| File | Role |
|------|------|
| `images/object.jpg` | Template matching + YOLO everyday object |
| `images/group_faces.jpg` / `images/group.jpg` | Consented group photo for Haar + YOLO |
| `images/street_clip.mp4` | Phone street video (timing / clips) |

## AI use

Composer helped with Gradio wiring and figure-script layout. Parameter choices, ethics reflection, and explanations are the student’s to defend.
