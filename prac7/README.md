# CSC 6302 Practical 7 - Capstone (Road & transport)

**Student:** Munyeme Sinyani (SINYANI) - University of Zambia  

**Track:** vehicle counting and classification from roadside footage; pothole severity hint from dashcam images.

**Mobile app (Android + TFLite SSD MobileNet):** [Salaam21/blindspot](https://github.com/Salaam21/blindspot) — branch `vehicle-counting`  
(Note: this is a native Android/Kotlin project with an on-device model, not Flutter.)

## Install

```bash
cd csc6302-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

First Gradio/YOLO run downloads `yolov8n.pt` (or caches under `prac7/weights/`).

## Run

```bash
cd prac7
pytest -q
cd ..
python prac7/run_prac7.py
python prac7/app.py
```

See `PROPOSAL.md` and `MOBILE.md`.

## AI Use

Composer helped with Gradio layout, YOLO counting glue, tests, and report wording. I chose the road-transport track and linked it to my existing Blindspot phone project.
