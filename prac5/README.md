# CSC 6302 Practical 5 - Segmentation

**Student:** Munyeme Sinyani (SINYANI) - University of Zambia

Thresholding, watershed, GrabCut, and optional DeepLab / Mask R-CNN.

## Install

```bash
cd csc6302-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python prac5/run_prac5.py
python prac5/app.py
```

Deep models in task 4 are slow on CPU. I may get `t4_deep_SKIPPED.txt` if weights fail to download.

IoU is my own function in `run_prac5.py`. Photos for this practical are in `images/`.

## AI Use

Composer helped with ideas for the segmentation steps and Gradio app. I built the reference mask comparison and wrote the IoU discussion.
