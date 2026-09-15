# CSC 6302 Practical 2 - Interest Points and Panorama

**Student:** Munyeme Sinyani (SINYANI) - University of Zambia

Corners, descriptor matching, homography, and a simple stitch.

## Install

```bash
cd csc6302-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python shared/make_demo_images.py
```

## Run

```bash
python prac2/run_prac2.py
```

Figures go to `prac2/report/figures/`.

For a wide panorama I use three overlapping shots in `shared/images/` (`pano_left.jpg`, `pano_mid.jpg`, `pano_right.jpg`). I plan to replace the demos with campus photos.

## What I study

Harris R is in `run_prac2.py` from the notes. I report ratio-test matches and RANSAC inlier counts with my figures.

## AI Use

Cursor helped wire the pipeline. I checked Harris and matching against lecture notes and can walk through the maths. I will use my own panorama photos before final submission.
