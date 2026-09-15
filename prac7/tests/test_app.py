"""Pytest checks for my capstone app."""

from __future__ import annotations

import numpy as np
from PIL import Image

from app import _textured, blurry, heuristic_predict, predict, preprocess_shape


def _flat(colour, size=(128, 128)):
    return Image.new("RGB", size, colour)


def test_preprocess_shape_dtype():
    img = _textured((40, 160, 60))
    arr = preprocess_shape(img)
    assert arr.shape == (3, 128, 128)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() <= arr.max() <= 1.0


def test_heuristic_healthy_leans_green():
    probs = heuristic_predict(_textured((40, 180, 50)))
    assert max(probs, key=probs.get) == "healthy"


def test_known_green_image_not_crash():
    msg, probs, note = predict(_textured((40, 180, 50)), conf_gate=0.1)
    assert "Prediction" in msg or "Unsure" in msg
    assert "NOT FOR REAL" in note
    assert abs(sum(probs.values()) - 1.0) < 1e-3


def test_blurry_rejected():
    img = _flat((120, 120, 120))
    assert blurry(img) is True
    msg, probs, _ = predict(img, conf_gate=0.45)
    assert "blurry" in msg.lower()
    assert probs == {}


def test_confidence_gate_unsure():
    msg, probs, _ = predict(_textured((40, 180, 50)), conf_gate=0.99)
    assert "Unsure" in msg
    assert probs
