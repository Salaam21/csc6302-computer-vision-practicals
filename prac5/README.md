# CSC 6302 Practical 5 — Image Segmentation

**Student:** Munyeme Sinyani (SINYANI) · Computer number **2025024319** · University of Zambia

Thresholding (IoU vs hand-made mask) → watershed on touching coins → GrabCut → DeepLabV3 vs Mask R-CNN → Gradio segmentation studio.

## Naming

| Item | Name |
|------|------|
| GitHub pack | `Salaam21/csc6302-computer-vision-practicals` |
| Suggested hand-in | `csc6302-prac5-SINYANI` |
| HF Space (Gradio) | https://huggingface.co/spaces/Salaam21/csc6302-prac5-SINYANI |
| HF model (app source fallback) | https://huggingface.co/Salaam21/csc6302-prac5-SINYANI |

*(Brief header sometimes says “CSC 6502”; this pack uses **CSC 6302**.)*

## Inputs

| File | Role |
|------|------|
| `images/document.jpg` | Unevenly-lit page (Task 1) |
| `images/prac5_handmade_mask.png` | My hand-made reference mask for the crop |
| `images/coins.jpg` | Touching coins (Task 2) |
| `images/portrait.jpg` | GrabCut subject (Task 3) |
| `images/street.jpg` / `images/group_faces.jpg` | Deep semantic vs instance (Task 4) |
| `images/lowlight.jpg` | Failure case (patterned cloth) |

Regenerate synthetic document/coins + mask:

```bash
python prac5/prepare_images.py
```

## Reproduce

```bash
cd csc6302-computer-vision-practicals
pip install -r prac5/requirements.txt
python prac5/prepare_images.py
python prac5/run_prac5.py
python prac5/build_report.py
python prac5/app.py
```

Figures: `prac5/report/figures/` · Report: `prac5/report/Practical5_Report.pdf`

IoU is implemented in `run_prac5.py` (`iou()`). Deep models download torchvision weights on first run (CPU: a few seconds per image).

## Hugging Face Space (Task 5)

App: upload + dropdown **Otsu / adaptive / watershed / DeepLabV3** → overlay (+ watershed count / DeepLab class legend). UI notes that DeepLab on free CPU takes a few seconds.

**Public app source (model repo):** https://huggingface.co/Salaam21/csc6302-prac5-SINYANI  

Creating a **Gradio Space** with this account returns **402 Payment Required** — Hugging Face now requires **PRO** for Gradio/Docker Spaces on free CPU (same restriction as Practical 4). Documented for markers: restricted by PRO. When PRO unlocks, Space name `csc6302-prac5-SINYANI` → https://huggingface.co/spaces/Salaam21/csc6302-prac5-SINYANI

Local demo: `python prac5/app.py`

## AI use

Composer helped with OpenCV/torchvision wiring and Gradio layout. Hand-made mask, IoU numbers, watershed discussion, and report wording are mine.
