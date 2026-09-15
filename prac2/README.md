# CSC 6302 Practical 2 - Interest Points and Panorama

**Student:** Munyeme Sinyani (SINYANI) - University of Zambia

Corners, descriptor matching, homography, and a simple stitch.

## Install

```bash
cd csc6302-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python prac2/run_prac2.py
```

Figures go to `prac2/report/figures/`.

For the panorama I use three overlapping shots in `images/` (`pano_left.jpg`, `pano_mid.jpg`, `pano_right.jpg`).

## What I study

Harris R is in `run_prac2.py` from the notes. I report ratio-test matches and RANSAC inlier counts with my figures.

## AI Use

Composer helped with ideas for wiring the matching/stitch pipeline. I checked Harris and matching against lecture notes and can walk through the maths.
