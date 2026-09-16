"""Generate report figures for Practical 7 (vehicle counting + pothole hint)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app import count_vehicles, rate_pothole

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images" / "street.jpg"
OUT = Path(__file__).resolve().parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    if not IMG.exists():
        raise SystemExit(f"Missing demo frame: {IMG}")
    img = Image.open(IMG)
    msg, counts, ann, _ = count_vehicles(img, conf_gate=0.35)
    if ann is not None:
        ann.save(OUT / "t1_vehicle_count.png")
    (OUT / "t1_counts.json").write_text(json.dumps({"message": msg, "counts": counts}, indent=2), encoding="utf-8")

    pmsg, pscores, pann, _ = rate_pothole(img)
    if pann is not None:
        pann.save(OUT / "t2_pothole_hint.png")
    (OUT / "t2_pothole.json").write_text(
        json.dumps({"message": pmsg, "scores": pscores}, indent=2), encoding="utf-8"
    )
    print(f"Practical 7 figures saved to {OUT}")
    print(msg)
    print(pmsg)


if __name__ == "__main__":
    main()
