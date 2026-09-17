# CSC 6302 Practical 4 — Deep Learning for Vision

**Student:** Munyeme Sinyani (SINYANI) · Computer number **2025024319** · University of Zambia

TinyCNN on CIFAR-10 → MobileNetV3 transfer on my farm-bird set → Grad-CAM → Gradio app.

## Naming

| Item | Name |
|------|------|
| GitHub pack | `Salaam21/csc6302-computer-vision-practicals` |
| Suggested hand-in | `csc6302-prac4-SINYANI` |
| HF model (app + weights) | https://huggingface.co/Salaam21/csc6302-prac4-SINYANI |
| Colab notebook | `prac4/CSC6302_Practical4_SINYANI.ipynb` |

## Dataset (Task 2)

Classes: **chicken, duck, turkey, goose, quail** (≥55 images each) under `images/zambia_birds/`, split **70/15/15** into `images/bird_dataset/`.

```bash
python prac4/prepare_bird_dataset.py
```

Augmentation: flip, ±15° rotation, colour jitter 0.2, ImageNet norm. Sources listed in `images/zambia_birds/SOURCES*.csv` (Commons + search). No photos of people.

## Reproduce

```bash
cd csc6302-computer-vision-practicals
pip install -r prac4/requirements.txt
python prac4/train_cifar_task1.py   # Task 1 CIFAR-10 figures
python prac4/train_prac4.py         # Task 2–3 birds + model.pt + Grad-CAM
python prac4/app.py                 # Task 4 local Gradio
```

Report: `prac4/report/Practical4_Report.pdf`

## For markers — how to run

Preferred: local Gradio — `python prac4/app.py`  
Weights / app also on Hugging Face model repo: https://huggingface.co/Salaam21/csc6302-prac4-SINYANI  

A public **Gradio Space** could not be created (Hugging Face returns **402 PRO required**). Please mark via local Gradio or the model repo above.

CPU inference (MobileNetV3-Small): about **0.1–0.5 s**/image after warm-up.

## AI use

Composer helped with wiring. Parameter choices and report wording are mine.
