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
python shared/make_demo_images.py
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

Optional master PDF:

```bash
python master_report/build_master_pdf.py
```

## Before I submit

My photos live in `images/` (`colourful.jpg`, `lowlight.jpg`, `coins.jpg`). I re-run `run_prac1.py` after swapping photos and write the report in my own words. I need to explain PSNR, conv2d, and my_sobel line by line.

## AI Use

Cursor (Composer) helped with `process.py`, `run_prac1.py`, and the master PDF walkthrough. From-scratch functions (`psnr`, `conv2d`, `my_sobel`) match the brief; I can explain every line. Demo images are stand-ins until I use my own captures.
