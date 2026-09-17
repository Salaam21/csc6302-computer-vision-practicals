# CSC 6302 Practical 7 — Capstone (Road & transport)

**Student:** Munyeme Sinyani (SINYANI) · Computer number **2025024319** · University of Zambia  

**Track:** vehicle counting and classification from roadside footage; pothole severity hint from dashcam images.

**Mobile app:** [Allan's Car Counter](../allans-car-counter) — Flutter, package `com.allans.car_counter` (separate from Blindspot)


**Suggested dedicated repo name (brief):** `csc6302-capstone-SINYANI` (currently lives as `prac7/` in the practicals pack).

## Brief checklist (honest)

| Requirement | Status |
|-------------|--------|
| R1 Vision core (YOLO / SSD / classical pothole) | Done |
| R1 Held-out ≥30 with metric (count MAE) | Done — `python prac7/eval_heldout.py` |
| R1 Optional VLM | Not wired (template text only) |
| R2 Gradio upload → counts → disclaimer | Done |
| R2 Graceful failure / UNSURE | Done |
| R3 ≥5 pytest | Done (6+) |
| R3 CI (ruff + pytest) | Done — `.github/workflows/ci-cd.yml` at **repo root** |
| R3 CD auto-deploy to HF Space | Workflow ready; **HF Gradio Spaces need PRO** on this account (402) — run locally |
| R3 Secrets out of repo | Done |
| R4 Ethics page in report | Done |
| Team of two + both in git log | **Gap** — solo / partner TBA |
| Live public URL at marking | **Blocked by HF PRO** unless alternative host |
| Report 6–8 pages | Rebuilt via `build_report.py` |
| PROPOSAL.md Week-12 gate | Present |

## Install

```bash
cd csc6302-computer-vision-practicals
pip install -r prac7/requirements.txt
```

## Run

```bash
cd prac7
pytest -q
cd ..
python prac7/run_prac7.py
python prac7/eval_heldout.py
python prac7/build_report.py
python prac7/app.py
```

## CI/CD

- Workflow: `.github/workflows/ci-cd.yml`
- Set GitHub secret `HF_TOKEN` to enable the deploy job (when Spaces are allowed).
- Until then: local Gradio is the marker path; PRO restriction documented (same as Prac 4–5).

## AI use

Composer helped with Gradio layout, eval harness, CI, and report scaffolding. Track choice, ethics, and disclaimers are mine.
