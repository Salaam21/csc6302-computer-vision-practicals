"""Build 70/15/15 ImageFolder splits from zambia_birds (5 poultry classes)."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "images" / "zambia_birds"
DST = ROOT / "images" / "bird_dataset"
# 5 distinctive farm-bird classes (brief: 3–5 classes, ≥60 ideally; we have ~55+)
CLASSES = ["chicken", "duck", "turkey", "goose", "quail"]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SEED = 42


def main() -> None:
    random.seed(SEED)
    if DST.exists():
        shutil.rmtree(DST)
    counts = {}
    for cls in CLASSES:
        files = [
            p
            for p in (SRC / cls).iterdir()
            if p.is_file() and p.suffix.lower() in EXTS
        ]
        random.shuffle(files)
        n = len(files)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        splits = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }
        counts[cls] = {k: len(v) for k, v in splits.items()}
        for split, paths in splits.items():
            out = DST / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(paths):
                shutil.copy2(p, out / f"{cls}_{i:03d}{p.suffix.lower()}")
    note = DST / "DATASET.txt"
    lines = [
        "CSC 6302 Practical 4 — Zambian / farm bird dataset",
        "Classes: " + ", ".join(CLASSES),
        "Split: 70/15/15 by random shuffle (seed=42) within each class folder.",
        "Sources: Wikimedia Commons + web image search; see images/zambia_birds/SOURCES*.csv",
        "No photographs of people.",
        "",
        "Per-class counts (train/val/test):",
    ]
    for cls, c in counts.items():
        lines.append(f"  {cls}: {c['train']}/{c['val']}/{c['test']} (total {sum(c.values())})")
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(note.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
