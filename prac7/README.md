# CSC 6302 Practical 7 — Capstone (Road & transport)

**Student:** Munyeme Sinyani (SINYANI) · Computer number **2025024319** · University of Zambia  

**For markers:** how to review the capstone (vehicle counting + pothole severity hint).

## Deployed web app (primary)

**Live Streamlit URL:**  
https://csc6302-computer-vision-practicalsgit-5wmff88oj3b8thqovqxukj.streamlit.app/

| | |
|--|--|
| GitHub | https://github.com/Salaam21/csc6302-computer-vision-practicals |
| Branch | `prac7-road-transport-vehicle-counting` |
| Entrypoint | `prac7/streamlit_app.py` |

Upload a roadside photo → vehicle counts; or a road-surface photo → pothole severity hint. First inference may take longer while YOLO weights load on CPU.

## Android APK (Allan's Car Counter)

Companion phone app (on-device TFLite). **Not** Blindspot.

1. Copy `prac7/apk/Allans_Car_Counter.apk` to an Android phone (or use `APK_FOR_PHONE/Allans_Car_Counter.apk` in the school-stuff folder).
2. On the phone: allow install from unknown sources if prompted.
3. Open the APK and install **Allan's Car Counter** (`com.allans.car_counter`).
4. Grant camera permission and point at vehicles to see live counts.

Source: sibling folder `allans-car-counter/` (Flutter). Details: `prac7/MOBILE.md`.

## Local review (optional)

```bash
cd csc6302-computer-vision-practicals
pip install -r prac7/requirements.txt
cd prac7
pytest -q
cd ..
python prac7/run_prac7.py
python prac7/eval_heldout.py
python prac7/app.py              # Gradio local
# or: streamlit run prac7/streamlit_app.py
```

- Report: `prac7/report/Practical7_Capstone_Report.pdf`
- Proposal: `prac7/PROPOSAL.md`
- Held-out count metric: `python prac7/eval_heldout.py`

## Brief coverage

| Item | Where |
|------|--------|
| YOLO vehicle count + class | Streamlit / Gradio / APK |
| Pothole severity hint | Streamlit / Gradio |
| Held-out ≥30, count MAE | `eval_heldout.py` + report |
| Tests + CI | `prac7/tests/`, `.github/workflows/ci-cd.yml` |
| Ethics | Capstone report |
| HF Gradio Space CD | Blocked by HF PRO (402); Streamlit used instead |

## AI use

Composer helped with Gradio/Streamlit layout, eval harness, CI, and report scaffolding. Track choice, ethics, and disclaimers are the student’s.
