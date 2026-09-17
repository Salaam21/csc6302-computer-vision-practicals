---
title: CSC 6302 Prac5 Segmentation Studio
emoji: 🧩
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# CSC 6302 Practical 5 — Segmentation Studio (SINYANI)

**Student:** Munyeme Sinyani · Computer number **2025024319** · University of Zambia

Upload an image and choose **Otsu / adaptive / watershed / DeepLabV3**.

- Watershed returns an approximate **object count**.
- DeepLab returns a colour overlay + **VOC class legend**.
- **Note:** DeepLabV3 on free CPU takes a few seconds (weights download on first run).

If this Space could not be created on the free tier (Hugging Face PRO required for Gradio Spaces), the same `app.py` runs locally and is also uploaded to the model repo.
