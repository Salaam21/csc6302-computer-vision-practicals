"""
Streamlit UI for the capstone (Streamlit Community Cloud).
Same logic as app.py — upload → vehicle counts / pothole hint.
"""
from __future__ import annotations

import streamlit as st
from PIL import Image

from app import count_vehicles, rate_pothole

st.set_page_config(page_title="CSC 6302 Capstone — Road & transport", layout="wide")
st.title("Road & transport — vehicle counting")
st.caption("CSC 6302 · Munyeme Sinyani · YOLOv8n demo (CPU on Streamlit Cloud)")

tab_v, tab_p = st.tabs(["Vehicle count", "Pothole severity"])

with tab_v:
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded = st.file_uploader("Roadside / traffic frame", type=["jpg", "jpeg", "png", "webp"])
        gate = st.slider("Confidence gate", 0.15, 0.8, 0.35, 0.05)
        run = st.button("Count vehicles", type="primary")
    with col_b:
        if run and uploaded is not None:
            img = Image.open(uploaded)
            msg, counts, ann, _note = count_vehicles(img, conf_gate=gate)
            st.write(msg)
            st.json(counts)
            if ann is not None:
                st.image(ann, caption="Annotated frame", use_container_width=True)
        elif run:
            st.warning("Upload an image first.")

with tab_p:
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_p = st.file_uploader("Dashcam / road surface photo", type=["jpg", "jpeg", "png", "webp"], key="pothole")
        run_p = st.button("Rate pothole hint", type="primary")
    with col_b:
        if run_p and uploaded_p is not None:
            img = Image.open(uploaded_p)
            msg, scores, ann, _note = rate_pothole(img)
            st.write(msg)
            st.json(scores)
            if ann is not None:
                st.image(ann, caption="Highlighted lower ROI", use_container_width=True)
        elif run_p:
            st.warning("Upload an image first.")
