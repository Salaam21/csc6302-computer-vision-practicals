# Capstone proposal — Road & transport track

**Team:** Munyeme Sinyani  
**Coursework Gradio repo folder:** `prac7/` in `csc6302-computer-vision-practicals`  
**Mobile app repo:** [https://github.com/Salaam21/blindspot](https://github.com/Salaam21/blindspot) (branch `vehicle-counting`)

## Problem statement
Road agencies and students need a **phone-first / laptop demo** that:
1. Counts and classifies vehicles (car, bus, truck, motorcycle, bicycle) in a roadside frame.
2. Gives a **rough pothole severity hint** from a dashcam-style road photo.

Wrong counts must not be treated as official traffic stats. Pothole output is a **heuristic**, not a certified inspection.

## Dataset / demo media plan
- Roadside still: `images/street.jpg` for counting demos and report figures.
- Optional: short roadside clips on phone via Blindspot live camera.
- Held-out idea (if expanding): labelled Lusaka roadside frames; report precision/recall per class.

## Model plan
| Surface | Model | Role |
|---------|-------|------|
| Gradio coursework demo | YOLOv8n (Ultralytics; auto-download / `prac7/weights/`) | Vehicle detect + count + classify |
| Phone app (Blindspot) | SSD MobileNet v1 TFLite already in `app/src/main/ml/` | On-device detect; UI filtered to vehicles + live count |
| Pothole tab | Classical OpenCV dark-blob heuristic on lower ROI | Severity: low / medium / high |

## Team split
| Person | Owns |
|--------|------|
| Munyeme Sinyani | Blindspot vehicle-counting branch, Gradio app, tests, report |
| Partner (TBA) | Extra roadside data, HF Space deploy, metrics notebook |

## Risks
- Double-counting the same vehicle across frames if we do naive per-frame sums on video.
- Night / rain / glare kills both SSD and YOLO confidence — confidence gate + “unsure / blurry” paths.
- Pothole heuristic confuses shadows and wet patches — always shown with a disclaimer.

## Ethics
No identifiable number-plate OCR in this submission. Footage is for coursework demo only.
