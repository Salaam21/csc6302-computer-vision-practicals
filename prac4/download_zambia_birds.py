"""Download unique Zambian bird images from Wikimedia Commons (educational use)."""
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

SPECIES = {
    "african_fish_eagle": ["African fish eagle", "Haliaeetus vocifer"],
    "lilac_breasted_roller": ["Lilac-breasted roller", "Coracias caudatus"],
    "carmine_bee_eater": [
        "Southern carmine bee-eater",
        "Merops nubicoides",
        "Carmine bee-eater",
    ],
    "grey_crowned_crane": ["Grey crowned crane", "Balearica regulorum"],
}

UA = "CSC6302StudentBot/1.0 (educational; University of Zambia)"
TARGET_PER_CLASS = 70
seen_hashes: set[str] = set()
meta_lines = ["class,filename,source_url,page_url,title"]


def api_get(params: dict) -> dict:
    params = dict(params)
    params["format"] = "json"
    params["origin"] = "*"
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def search_file_titles(query: str, limit: int = 120) -> list[str]:
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
        time.sleep(0.12)
    return out


def download(url: str, dest: Path) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    h = hashlib.sha1(data).hexdigest()
    if h in seen_hashes or len(data) < 8000:
        return None
    seen_hashes.add(h)
    dest.write_bytes(data)
    return h


def main() -> None:
    total = 0
    for cls, queries in SPECIES.items():
        cdir = OUT / cls
        cdir.mkdir(parents=True, exist_ok=True)
        titles: list[str] = []
        for q in queries:
            titles.extend(search_file_titles(q, limit=90))
            time.sleep(0.15)
        seen_t: set[str] = set()
        uniq = []
        for t in titles:
            if t not in seen_t:
                seen_t.add(t)
                uniq.append(t)
        infos = imageinfo(uniq)
        print(cls, "candidates", len(infos), flush=True)
        n = 0
        for info in infos:
            if n >= TARGET_PER_CLASS:
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
                time.sleep(0.05)
            except Exception as e:
                print("  skip", e, flush=True)
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
            total += 1
        print(cls, "saved", n, flush=True)

    (OUT / "SOURCES.csv").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    (OUT / "README.txt").write_text(
        "Zambian bird images from Wikimedia Commons for CSC 6302 Practical 4.\n"
        "Deduplicated by content hash / Commons sha1. See SOURCES.csv.\n"
        "Licence: check each Commons file page (often CC BY / CC BY-SA / PD).\n",
        encoding="utf-8",
    )
    print("TOTAL", total, flush=True)
    for p in sorted(OUT.iterdir()):
        if p.is_dir():
            print(p.name, len(list(p.glob("*"))), flush=True)


if __name__ == "__main__":
    main()
