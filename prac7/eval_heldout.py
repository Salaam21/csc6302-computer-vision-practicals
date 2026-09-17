"""
Held-out evaluation for the capstone (≥30 images).
Synthetic scenes have known vehicle counts (exact MAE).
A few real roadside frames use my manual count labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app import count_vehicles, empty_counts

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / "report" / "figures"
DATA = ROOT / "eval_data"
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def _car_sprite(rng, w=90, h=45):
    img = Image.new("RGB", (w, h), (30, 30, 30))
    d = ImageDraw.Draw(img)
    colour = tuple(int(c) for c in rng.integers(40, 220, 3))
    d.rounded_rectangle([2, 8, w - 2, h - 6], radius=6, fill=colour)
    d.rectangle([w // 4, 2, 3 * w // 4, 14], fill=(60, 120, 200))
    d.ellipse([8, h - 14, 28, h - 2], fill=(20, 20, 20))
    d.ellipse([w - 28, h - 14, w - 8, h - 2], fill=(20, 20, 20))
    return img


def make_synthetic(n=28, size=(640, 360)):
    """Scenes with a known number of car-like sprites (ground-truth count)."""
    rng = np.random.default_rng(11)
    items = []
    for i in range(n):
        bg = Image.new("RGB", size, (70, 110, 70))
        d = ImageDraw.Draw(bg)
        d.rectangle([0, size[1] // 2, size[0], size[1]], fill=(55, 55, 55))
        # lane marks
        for x in range(20, size[0], 60):
            d.rectangle([x, size[1] // 2 + 40, x + 30, size[1] // 2 + 46], fill=(200, 200, 180))
        n_cars = int(rng.integers(1, 5))
        for _ in range(n_cars):
            sprite = _car_sprite(rng)
            x = int(rng.integers(10, size[0] - sprite.size[0] - 10))
            y = int(rng.integers(size[1] // 2 + 10, size[1] - sprite.size[1] - 5))
            bg.paste(sprite, (x, y))
        path = DATA / f"syn_{i:02d}.jpg"
        bg.save(path, quality=90)
        gt = empty_counts()
        gt["car"] = n_cars  # sprites are car-like; YOLO may call some truck/bus
        items.append({"path": str(path), "gt_total": n_cars, "gt": gt, "kind": "synthetic"})
    return items


def make_real_labelled():
    """A few real frames with my manual vehicle totals."""
    items = []
    # street.jpg — I count motorcycles/people scene: 2 motorcycles visible in full street crop
    street = REPO / "images" / "street.jpg"
    if street.exists():
        items.append(
            {
                "path": str(street),
                "gt_total": 2,  # manual: green sportbike + dark bike behind
                "gt": {"bicycle": 0, "car": 0, "motorcycle": 2, "bus": 0, "truck": 0},
                "kind": "real_manual",
            }
        )
    # Sample a couple of frames from street_clip if present
    clip = REPO / "images" / "street_clip.mp4"
    if clip.exists():
        import cv2

        cap = cv2.VideoCapture(str(clip))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for j, fi in enumerate([30, 90, 150]):
            if fi >= total:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            # resize for speed
            h, w = fr.shape[:2]
            scale = 640 / max(h, w)
            fr = cv2.resize(fr, (int(w * scale), int(h * scale)))
            path = DATA / f"real_clip_{j}.jpg"
            cv2.imwrite(str(path), fr)
            # Manual-ish: leave gt_total as -1 to skip MAE, but still run detection for qualitative
            items.append(
                {
                    "path": str(path),
                    "gt_total": -1,
                    "gt": empty_counts(),
                    "kind": "real_unlabelled",
                }
            )
        cap.release()
    return items


def run_eval(conf_gate: float = 0.35):
    items = make_synthetic(28) + make_real_labelled()
    # Ensure ≥30
    while len([x for x in items if x["gt_total"] >= 0]) < 30:
        extra = make_synthetic(5)
        items.extend(extra)

    labelled = [x for x in items if x["gt_total"] >= 0]
    abs_err = []
    pred_totals = []
    gt_totals = []
    per_image = []
    for it in labelled[: max(30, len(labelled))]:
        img = Image.open(it["path"])
        msg, counts, _, _ = count_vehicles(img, conf_gate=conf_gate)
        pred = int(sum(counts.values()))
        gt = int(it["gt_total"])
        err = abs(pred - gt)
        abs_err.append(err)
        pred_totals.append(pred)
        gt_totals.append(gt)
        per_image.append(
            {
                "path": Path(it["path"]).name,
                "kind": it["kind"],
                "gt_total": gt,
                "pred_total": pred,
                "abs_err": err,
                "counts": counts,
                "message": msg,
            }
        )

    mae = float(np.mean(abs_err)) if abs_err else None
    within1 = float(np.mean([e <= 1 for e in abs_err])) if abs_err else None
    summary = {
        "n_labelled": len(abs_err),
        "mae_vehicle_count": round(mae, 3) if mae is not None else None,
        "frac_abs_err_le_1": round(within1, 3) if within1 is not None else None,
        "conf_gate": conf_gate,
        "note": (
            "Synthetic sprites are car-like drawings; YOLO may under-count stylised art "
            "or mislabel as truck. Real frames use my manual totals. Metric = MAE on total vehicles."
        ),
        "per_image": per_image,
    }
    (OUT / "eval_heldout.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "eval_heldout.txt").write_text(
        f"n_labelled={summary['n_labelled']}\n"
        f"mae_vehicle_count={summary['mae_vehicle_count']}\n"
        f"frac_abs_err_le_1={summary['frac_abs_err_le_1']}\n"
        f"conf_gate={conf_gate}\n"
        f"{summary['note']}\n",
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in summary if k != "per_image"}, indent=2))
    return summary


if __name__ == "__main__":
    run_eval()
