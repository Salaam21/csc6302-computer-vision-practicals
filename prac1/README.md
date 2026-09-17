# CSC 6302 Practical 1 — Seeing Like a Machine

**Student:** Munyeme Sinyani (SINYANI) · Computer number **2025024319** · University of Zambia

Image-processing toolbox: colour spaces, noise/smoothing, unsharp masking, edges, morphology + counting, and a CLI.

## Install (fresh machine)

```bash
cd csc6302-computer-vision-practicals
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r prac1/requirements.txt
```

Or from the repo root: `pip install -r requirements.txt` (includes these packages plus later practicals).

## My photos (`images/`)

| File | Role |
|------|------|
| `colourful.jpg` | Task 1 colour spaces + HSV picker (lime-green bike); Task 3 sharpening |
| `lowlight.jpg` | Low-light / noisy photo (shown in report) |
| `coins.jpg` | Earrings on a card — Task 5 morphology + counting |
| `clean_gray.png` | Regenerated from `colourful.jpg` by `run_prac1.py` for Tasks 2 & 4 |

## Reproduce every result

```bash
# All report figures + PSNR table + Sobel max-diff + object count
python prac1/run_prac1.py
```

Outputs go to `prac1/report/figures/`.

### CLI toolbox (`process.py`) — five operations

```bash
python prac1/process.py --input images/colourful.jpg --op pick --output out_pick.png
python prac1/process.py --input images/lowlight.jpg --op denoise --ksize 5 --output out_denoise.png
python prac1/process.py --input images/colourful.jpg --op sharpen --amount 1.0 --output out_sharp.png
python prac1/process.py --input images/clean_gray.png --op edges --output out_edges.png
python prac1/process.py --input images/coins.jpg --op count --output out_count.png
```

From-scratch implementations (brief rule): `psnr`, `conv2d`, `my_sobel` in `process.py`.

## Report

`prac1/report/Practical1_Report.pdf`

## AI use

Composer helped with script layout ideas. From-scratch PSNR/Sobel and the Task 6 reflection are mine to explain.
