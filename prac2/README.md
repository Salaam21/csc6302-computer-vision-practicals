# CSC 6302 Practical 2 — Features that Matter

**Student:** Munyeme Sinyani (SINYANI) · Computer number **2025024319** · University of Zambia

Corners, descriptors, matching, and a hand-built panorama (no `cv2.Stitcher`).

## Install

```bash
cd csc6302-computer-vision-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r prac2/requirements.txt
```

## My photos (`images/`)

| File | Use |
|------|-----|
| `checkerboard.png` | Task 1 Harris on a printed/tiled pattern |
| `prac2_natural.jpg` | Task 1 natural scene (campus courtyard) |
| `prac2_show.jpg` | Tasks 2–5 source; overlapping crops below |
| `prac2_dusk.jpg` | Extra natural photo |
| `pano_left.jpg` / `pano_mid.jpg` / `pano_right.jpg` | Overlapping shots (~30%+) cut from `prac2_show.jpg` |

`python prac2/prepare_images.py` rebuilds the three `pano_*.jpg` crops.

## Reproduce everything

```bash
python prac2/prepare_images.py
python prac2/run_prac2.py
```

Figures and tables land in `prac2/report/figures/`.

From-scratch: `harris()` in `run_prac2.py` (structure tensor + R + NMS).  
Forbidden: `cv2.Stitcher` / all-in-one panorama helpers.

## Report

`prac2/report/Practical2_Report.pdf`

## AI use

Composer helped with stitching canvas math ideas. Harris response/NMS and the invariance interpretation I can explain myself.
