"""Pytest checks for the road-transport capstone app."""

from __future__ import annotations

import numpy as np
from PIL import Image

from app import (
    VEHICLE_CLASSES,
    _textured,
    blurry,
    count_vehicles,
    empty_counts,
    format_counts,
    rate_pothole,
)


def _flat(colour, size=(128, 128)):
    return Image.new("RGB", size, colour)


def test_empty_counts_keys():
    c = empty_counts()
    assert set(c) == set(VEHICLE_CLASSES)
    assert sum(c.values()) == 0


def test_format_counts_none():
    assert "No vehicles" in format_counts(empty_counts())


def test_format_counts_mixed():
    c = empty_counts()
    c["car"] = 2
    c["bus"] = 1
    text = format_counts(c)
    assert "Total vehicles: 3" in text
    assert "2 cars" in text
    assert "1 bus" in text


def test_blurry_rejected():
    img = _flat((120, 120, 120))
    assert blurry(img) is True
    msg, counts, _, note = count_vehicles(img, conf_gate=0.35)
    assert "blurry" in msg.lower()
    assert sum(counts.values()) == 0
    assert "COURSEWORK" in note


def test_pothole_heuristic_runs():
    img = _textured((90, 90, 90), size=(320, 240), seed=3)
    # punch a dark hole in the lower half
    arr = np.asarray(img).copy()
    arr[160:220, 100:180] = 15
    img = Image.fromarray(arr)
    msg, scores, out, note = rate_pothole(img)
    assert "severity" in msg.lower() or "POTHOLE" in msg.upper() or "Pothole" in msg
    assert out is not None
    assert "COURSEWORK" in note
    assert scores.get("blobs", 0) >= 0
