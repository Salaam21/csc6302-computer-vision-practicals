# Capstone proposal — Agriculture track

**Team:** Munyeme Sinyani (+ partner TBA — register with lecturer by gate date)  
**Repo:** `csc6302-capstone-SINYANI-PARTNER`

## Problem statement
Smallholder farmers and students need a **phone-first screening aid** that looks at a maize leaf photo and suggests whether it looks healthy, rusty, or blighted — with an explicit confidence gate and a hard disclaimer that this is **not** a diagnosis.

## Dataset plan
- Preferred: field photos around Lusaka/UNZA farms (consent noted; no identifiable people).
- Dataset: leaf photos under `images/leaf_dataset` (healthy / rust / blight), split 70/15/15 by folder.
- Held-out evaluation: ≥30 test images; report accuracy + per-class F1.

## Model plan
- Fine-tune MobileNetV3-Small (Practical 4 style): freeze backbone → train head → light fine-tune.
- Classical HSV pre-check for “is this even a leaf-ish green image?” optional.
- Optional VLM explanation via OpenRouter free tier, cached; degrade to template text on 429.

## Team split
| Person | Owns |
|--------|------|
| Member A | Data collection, training notebook, metrics |
| Member B | Gradio app, tests, CI/CD, ethics page |

## Risks
False blight alarms waste spray money; missed rust can spread. We tune the confidence gate toward **“unsure”** rather than a confident wrong class.
