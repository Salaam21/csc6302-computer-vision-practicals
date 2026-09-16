# CSC 6302 Practical 1 - Seeing Like a Machine

**Student:** Munyeme Sinyani (SINYANI) - University of Zambia

Pixels are numbers. I work through colour spaces, denoising, sharpening, edges, and morphology.

## Install

From the repo root:

```bash
cd csc6302-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Report figures:

```bash
python prac1/run_prac1.py
```

Output folder: `prac1/report/figures/`.

CLI on `process.py`:

```bash
python prac1/process.py --input images/colourful.jpg --op pick --output out.png
python prac1/process.py --input images/clean_gray.png --op edges --output edges.png
python prac1/process.py --input images/coins.jpg --op count --output count_vis.png
```

## Photos I used

My photos are in `images/`: `colourful.jpg` (outdoor scene), `lowlight.jpg` (darker indoor), `coins.jpg` (earrings for counting).

## AI Use

Composer helped me with ideas for how to structure `process.py` and `run_prac1.py`. From-scratch parts (`psnr`, `conv2d`, `my_sobel`) I can explain myself.
