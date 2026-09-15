# CSC 6302 Practical 4 - CNNs and Transfer Learning

**Student:** Munyeme Sinyani (SINYANI) - University of Zambia

TinyCNN from scratch, then MobileNetV3 transfer learning and Grad-CAM.

## Install

```bash
cd csc6302-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python shared/make_demo_images.py
```

## Run on my laptop (CPU)

```bash
python prac4/train_prac4.py
python prac4/app.py
```

This writes `prac4/model.pt` and figures under `prac4/report/figures/`.

The script uses the small `leaf_dataset` so training finishes on CPU. For the full CIFAR-10 story I copy the training loop to Colab with GPU and more epochs, and I put those numbers in my report only if I actually ran them.

## AI Use

Cursor helped with the PyTorch loop and Grad-CAM hooks. I chose epochs and read the curves myself. Class order follows `ImageFolder` alphabetical names. I retrain if I change the leaf photos.
