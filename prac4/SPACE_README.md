# CSC 6302 Practical 4 — leaf triage (Hugging Face Space)

**Student:** Munyeme Sinyani (SINYANI) · Computer number **2025024319** · University of Zambia

Fine-tuned **MobileNetV3-Small** on my 3-class leaf-condition set (`blight` / `healthy` / `rust`). Upload a leaf photo; the app returns the top-3 class probabilities.

**Space URL:** restricted — Hugging Face Gradio Spaces currently require **PRO** on this account (I could not publish the live Space without paying). App source + `model.pt` are on https://huggingface.co/Salaam21/csc6302-prac4-SINYANI instead.

## Inference on the free CPU tier

If a Gradio Space is available, MobileNetV3-Small is typically about **0.1–0.5 s** per image after warm-up (cold start slower). Locally the same app runs with `python app.py`. For a real deployment I would use a warmer host or small GPU, export **ONNX** / **TorchScript**, and add rate limits.

## Repo / files

- `app.py` — Gradio UI (this Space)
- `model.pt` — fine-tuned weights + class names (hosted here; large file stays on the Space)
- `requirements.txt` — Space dependencies

Full training notebook and report live in the coursework GitHub pack (`csc6302-computer-vision-practicals` / Practical 4). Suggested standalone hand-in name: `csc6302-prac4-SINYANI`.

## AI use

Composer helped with Gradio wiring. Training choices and dataset notes are mine.
