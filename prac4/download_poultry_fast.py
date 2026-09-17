"""Fast poultry image download via DuckDuckGo image search (deduped by hash)."""
from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path

from ddgs import DDGS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "zambia_birds"
OUT.mkdir(parents=True, exist_ok=True)

QUERIES = {
    "village_chicken": [
        "village chicken Africa",
        "free range indigenous chicken",
        "African backyard chicken",
    ],
    "broiler": ["broiler chicken farm", "white broiler chicken", "broiler hen"],
    "duck": ["domestic farm duck", "Pekin duck", "farmyard duck"],
    "black_duck": ["American black duck", "black duck bird", "Anas rubripes"],
    "quail": ["Japanese quail", "Coturnix quail bird", "common quail bird"],
    "turkey": ["domestic turkey farm", "Meleagris turkey", "farm turkey bird"],
    "goose": ["domestic goose farm", "Embden goose", "farm goose bird"],
}

TARGET = 55
UA = "Mozilla/5.0 (educational CSC6302)"
seen: set[str] = set()

for p in OUT.rglob("*"):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        seen.add(hashlib.sha1(p.read_bytes()).hexdigest())

meta = OUT / "SOURCES_poultry.csv"
lines = ["class,filename,source_url"]
if meta.exists():
    lines = meta.read_text(encoding="utf-8").strip().splitlines()


def fetch(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read()
    except Exception:
        return False
    if len(data) < 6000:
        return False
    h = hashlib.sha1(data).hexdigest()
    if h in seen:
        return False
    seen.add(h)
    dest.write_bytes(data)
    return True


def main() -> None:
    total_new = 0
    with DDGS() as ddgs:
        for cls, qlist in QUERIES.items():
            cdir = OUT / cls
            cdir.mkdir(parents=True, exist_ok=True)
            have = len(
                [
                    p
                    for p in cdir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                ]
            )
            need = max(0, TARGET - have)
            print(cls, "have", have, "need", need, flush=True)
            if need == 0:
                continue
            urls: list[str] = []
            for q in qlist:
                try:
                    for hit in ddgs.images(q, max_results=80):
                        u = hit.get("image") or hit.get("url")
                        if u and u not in urls:
                            urls.append(u)
                except Exception as e:
                    print("  search err", e, flush=True)
                time.sleep(0.4)
            n = have
            added = 0
            for u in urls:
                if added >= need:
                    break
                ext = ".jpg"
                low = u.lower()
                if ".png" in low:
                    ext = ".png"
                elif ".webp" in low:
                    ext = ".webp"
                fname = f"{cls}_{n:03d}{ext}"
                dest = cdir / fname
                ok = fetch(u, dest)
                if not ok:
                    if dest.exists():
                        dest.unlink()
                    continue
                lines.append(f"{cls},{fname},{u}")
                n += 1
                added += 1
                total_new += 1
                if added % 10 == 0:
                    print(" ", cls, added, flush=True)
            print(cls, "added", added, "now", n, flush=True)

    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("NEW", total_new, flush=True)
    grand = 0
    for p in sorted(OUT.iterdir()):
        if p.is_dir():
            c = len(
                [
                    f
                    for f in p.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                ]
            )
            grand += c
            print(p.name, c, flush=True)
    print("GRAND_TOTAL", grand, flush=True)


if __name__ == "__main__":
    main()
