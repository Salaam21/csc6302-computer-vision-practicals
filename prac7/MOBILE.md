# Mobile companion — Blindspot

Repo: https://github.com/Salaam21/blindspot  

Branch used for this capstone: **`vehicle-counting`**

## What it already had
- Android (Kotlin) camera app
- On-device **SSD MobileNet v1** TFLite model (`app/src/main/ml/ssd_mobilenet_v1_1_metadata_1.tflite`)
- COCO labels including `car`, `bus`, `truck`, `motorcycle`, `bicycle`

## What the vehicle-counting branch changes
- Filters detections to **road vehicles only**
- Shows a live **count summary** on screen
- Speaks counts with TTS (e.g. “2 cars, 1 truck”)

## Coursework Gradio side
`prac7/app.py` mirrors the same problem on a laptop with **YOLOv8n** so markers can try uploads without an Android build.
