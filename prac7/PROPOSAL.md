# Capstone proposal — Road & transport track

**Team:** Munyeme Sinyani  
**Coursework Gradio repo folder:** `prac7/` in `csc6302-computer-vision-practicals`  
**Mobile app (separate repo/folder):** `allans-car-counter` — **Allan's Car Counter** (Flutter + on-device TFLite)  
**Blindspot:** left alone as its own accessibility project (not this capstone app)

## Problem statement
Road agencies and students need a **phone-first / laptop demo** that:
1. Counts and classifies vehicles (car, bus, truck, motorcycle, bicycle) in a roadside frame.
2. Gives a **rough pothole severity hint** from a dashcam-style road photo.

Wrong counts must not be treated as official traffic stats. Pothole output is a **heuristic**, not a certified inspection.

## Dataset / demo media plan
- Roadside still: `images/street.jpg` for counting demos and report figures.
- Optional: short roadside clips on phone via Allan's Car Counter live camera.
- Held-out: labelled / synthetic scenes via `prac7/eval_heldout.py` (MAE on totals).

## Model plan
| Surface | Model | Role |
|---------|-------|------|
| Gradio / Streamlit coursework demo | YOLOv8n (Ultralytics) | Vehicle detect + count + classify |
| Phone — Allan's Car Counter | SSD MobileNet v1 TFLite | On-device detect; live count UI |
| Pothole tab | Classical OpenCV dark-blob heuristic | Severity: low / medium / high |

## Team split
| Person | Owns |
|--------|------|
| Munyeme Sinyani | Allan's Car Counter, Gradio/Streamlit app, tests, CI, report |
| Partner (TBA) | Extra Lusaka roadside labels, host unlock, seminar co-demo |

**Note:** Brief prefers teams of two. Ready for a second surname in `csc6302-capstone-SINYANI-<PARTNER>`.

## Risks
- Double-counting the same vehicle across frames if we do naive per-frame sums on video.
- Night / rain / glare kills confidence — confidence gate + “unsure / blurry” paths.
- Pothole heuristic confuses shadows and wet patches — always shown with a disclaimer.

## Ethics
No identifiable number-plate OCR in this submission. Footage is for coursework demo only.
