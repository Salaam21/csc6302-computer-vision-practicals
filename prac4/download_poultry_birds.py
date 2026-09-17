"""Download unique poultry images from Wikimedia Commons (educational use)."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "zambia_birds"
OUT.mkdir(parents=True, exist_ok=True)

# Farm / village poultry common in Zambia + user request
SPECIES = {
    "chicken": ["domestic chicken", "Gallus gallus domesticus hen", "rooster chicken"],
    "village_chicken": [
        "free range chicken",
        "village chicken Africa",
        "indigenous chicken",
        "scavenging chicken",
    ],
    "broiler": ["broiler chicken", "broiler hen", "meat chicken broiler"],
    "duck": ["domestic duck", "Anas platyrhynchos domesticus", "farm duck"],
    "black_duck": ["American black duck", "Anas rubripes", "black domestic duck"],
    "quail": ["Coturnix quail", "common quail", "Japanese quail"],
    "turkey": ["domestic turkey", "Meleagris gallopavo", "farm turkey"],
    "goose": ["domestic goose", "Embden goose", "farm goose"],
}

UA = "CSC6302StudentBot/1.0 (educational; University of Zambia)"
TARGET_PER_CLASS = 55
seen_hashes: set[str] = set()

# Seed with already-downloaded files so we do not duplicate across classes
for p in OUT.rglob("*"):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        seen_hashes.add(hashlib.sha1(p.read_bytes()).hexdigest())

meta_path = OUT / "SOURCES.csv"
if meta_path.exists():
    meta_lines = meta_path.read_text(encoding="utf-8").strip().splitlines()
else:
    meta_lines = ["class,filename,source_url,page_url,title"]


def api_get(params: dict) -> dict:
    params = dict(params)
    params["format"] = "json"
    params["origin"] = "*"
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if "429" in str(e) or "Too many" in str(e):
                time.sleep(2.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("api_get failed after retries")


def search_file_titles(query: str, limit: int = 100) -> list[str]:
    titles: list[str] = []
    cont = None
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"filetype:bitmap {query}",
            "srnamespace": 6,
            "srlimit": min(50, limit - len(titles)),
        }
        if cont is not None:
            params["sroffset"] = cont
        data = api_get(params)
        batch = [h["title"] for h in data.get("query", {}).get("search", [])]
        if not batch:
            break
        titles.extend(batch)
        cont = data.get("continue", {}).get("sroffset")
        if cont is None:
            break
        time.sleep(0.25)
    return titles


def imageinfo(titles: list[str]) -> list[dict]:
    out: list[dict] = []
    for i in range(0, len(titles), 40):
        batch = titles[i : i + 40]
        data = api_get(
            {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|sha1",
                "iiurlwidth": 640,
            }
        )
        pages = data.get("query", {}).get("pages", {})
        for p in pages.values():
            if "imageinfo" not in p:
                continue
            ii = p["imageinfo"][0]
            mime = ii.get("mime", "")
            if not mime.startswith("image/") or mime in ("image/svg+xml", "image/gif"):
                continue
            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue
            title = p.get("title", "")
            out.append(
                {
                    "title": title,
                    "url": url,
                    "page": "https://commons.wikimedia.org/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_")),
                    "sha1": ii.get("sha1"),
                }
            )
        time.sleep(0.35)
    return out


def download(url: str, dest: Path) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            break
        except Exception as e:
            if "429" in str(e):
                time.sleep(3 * (attempt + 1))
                continue
            raise
    else:
        return None
    h = hashlib.sha1(data).hexdigest()
    if h in seen_hashes or len(data) < 8000:
        return None
    seen_hashes.add(h)
    dest.write_bytes(data)
    return h


def main() -> None:
    total_new = 0
    for cls, queries in SPECIES.items():
        cdir = OUT / cls
        cdir.mkdir(parents=True, exist_ok=True)
        already = len(
            [
                p
                for p in cdir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
        )
        need = max(0, TARGET_PER_CLASS - already)
        if need == 0:
            print(cls, "already", already, "skip", flush=True)
            continue
        titles: list[str] = []
        for q in queries:
            titles.extend(search_file_titles(q, limit=80))
            time.sleep(0.3)
        seen_t: set[str] = set()
        uniq = []
        for t in titles:
            if t not in seen_t:
                seen_t.add(t)
                uniq.append(t)
        infos = imageinfo(uniq)
        print(cls, "have", already, "need", need, "candidates", len(infos), flush=True)
        n = already
        added = 0
        for info in infos:
            if added >= need:
                break
            sha = info.get("sha1")
            if sha and sha in seen_hashes:
                continue
            u = info["url"].lower()
            ext = ".jpg"
            if ".png" in u:
                ext = ".png"
            elif ".webp" in u:
                ext = ".webp"
            fname = f"{cls}_{n:03d}{ext}"
            dest = cdir / fname
            try:
                h = download(info["url"], dest)
                time.sleep(0.2)
            except Exception as e:
                print("  skip", e, flush=True)
                time.sleep(1.0)
                continue
            if h is None:
                if dest.exists():
                    dest.unlink()
                continue
            if sha:
                seen_hashes.add(sha)
            safe_title = info["title"].replace('"', "")
            meta_lines.append(
                f'{cls},{fname},{info["url"]},{info["page"]},"{safe_title}"'
            )
            n += 1
            added += 1
            total_new += 1
        print(cls, "added", added, "now", n, flush=True)

    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
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
