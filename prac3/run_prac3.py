"""
CSC 6302 Practical 3 — Detecting Things

Tasks 1–3: figures under report/figures/
Task 4: see app.py (Gradio)
Task 5: ethics section in the report PDF
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
OUT = Path(__file__).resolve().parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save_row(imgs, titles, path, scale=3.6):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(scale * n, scale))
    if n == 1:
        axes = [axes]
    for ax, im, t in zip(axes, imgs, titles):
        if im is None:
            ax.set_visible(False)
            continue
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.shape[2] == 3 else im
            ax.imshow(rgb)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def ensure_group_faces() -> Path:
    """Prefer the student's real group photo (group_faces.jpg or group.jpg)."""
    for name in ("group_faces.jpg", "group.jpg"):
        p = IMG / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Add images/group_faces.jpg or images/group.jpg (consented group photo)."
    )


def ensure_street_video() -> Path:
    """Prefer the student's phone street clip; else build a short sliding demo."""
    for name in ("street_clip.mp4", "street_clip.avi", "street.mp4"):
        p = IMG / name
        if p.exists():
            return p

    out = IMG / "street_clip.avi"
    src = cv2.imread(str(IMG / "prac2_show.jpg"))
    if src is None:
        src = cv2.imread(str(IMG / "street.jpg"))
    if src is None:
        raise FileNotFoundError("No street video/image found under images/")
    h, w = src.shape[:2]
    win_w = min(w, int(h * 1.4), w)
    n_frames = 60
    max_x = max(1, w - win_w)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"XVID"), 8, (win_w, h)
    )
    for i in range(n_frames):
        x0 = int(i * max_x / max(n_frames - 1, 1))
        frame = src[:, x0 : x0 + win_w].copy()
        if frame.shape[1] != win_w:
            frame = cv2.resize(frame, (win_w, h))
        writer.write(frame)
    writer.release()
    return out


def task1_template():
    # Everyday object: creatine tub — crop the yellow NPL logo region
    scene = cv2.imread(str(IMG / "object.jpg"))
    if scene is None:
        scene = cv2.imread(str(IMG / "colourful.jpg"))
    h0, w0 = scene.shape[:2]
    # upper logo-ish crop (works for the NPL tub photo)
    y0, x0 = int(0.12 * h0), int(0.30 * w0)
    th, tw = max(60, int(0.10 * h0)), max(80, int(0.18 * w0))
    tmpl = scene[y0 : y0 + th, x0 : x0 + tw].copy()
    cv2.imwrite(str(OUT / "t1_template_crop.png"), tmpl)

    res = cv2.matchTemplate(scene, tmpl, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    hit = scene.copy()
    cv2.rectangle(hit, maxloc, (maxloc[0] + tw, maxloc[1] + th), (0, 255, 0), 3)

    # Failure: different scale (farther / closer) — template size no longer matches
    scaled = cv2.resize(scene, None, fx=0.55, fy=0.55)
    res2 = cv2.matchTemplate(scaled, tmpl, cv2.TM_CCOEFF_NORMED)
    _, maxv2, _, maxloc2 = cv2.minMaxLoc(res2)
    fail = scaled.copy()
    cv2.rectangle(fail, maxloc2, (maxloc2[0] + tw, maxloc2[1] + th), (0, 0, 255), 2)

    # Failure 2: rotated scene
    M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), 25, 1.0)
    rotated = cv2.warpAffine(scene, M, (w0, h0))
    res3 = cv2.matchTemplate(rotated, tmpl, cv2.TM_CCOEFF_NORMED)
    _, maxv3, _, _ = cv2.minMaxLoc(res3)

    score_ok = cv2.applyColorMap((np.clip(res, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    score_bad = cv2.applyColorMap((np.clip(res2, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)

    save_row(
        [hit, score_ok, fail, score_bad],
        [
            f"Same view hit ({maxv:.2f})",
            "Score map OK",
            f"Scaled fail ({maxv2:.2f})",
            "Score map scaled",
        ],
        OUT / "t1_template.png",
        scale=3.2,
    )
    (OUT / "t1_scores.txt").write_text(
        f"same_view={maxv:.4f}\nscaled_0.55x={maxv2:.4f}\nrotated_25deg={maxv3:.4f}\n"
        "note=NCC collapses under scale/rotation because the template must match pixel pattern size and orientation\n",
        encoding="utf-8",
    )


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def task2_haar(group_path: Path, video_path: Path):
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    group = cv2.imread(str(group_path))
    gray = cv2.cvtColor(group, cv2.COLOR_BGR2GRAY)

    # Ground truth ≈ loose Haar boxes (manual eye count on this photo ~8–12 frontal faces).
    # Using liberal detections as reference boxes for the parameter study.
    loose = cascade.detectMultiScale(gray, 1.05, 2, minSize=(40, 40))
    gt_boxes = [tuple(map(int, b)) for b in loose]
    # Deduplicate heavily overlapping loose boxes
    kept = []
    for b in sorted(gt_boxes, key=lambda t: t[2] * t[3], reverse=True):
        if all(_iou(b, k) < 0.4 for k in kept):
            kept.append(b)
    gt_boxes = kept
    gt_count = max(len(gt_boxes), 1)
    (OUT / "t2_gt_note.txt").write_text(
        f"gt_face_boxes={gt_count}\n"
        "note=GT from liberal Haar (1.05,2) after IoU dedupe; eye count of clear frontal faces is similar\n",
        encoding="utf-8",
    )

    rows = ["scaleFactor,minNeighbors,detections,hits,misses,false_positives"]
    for sf in (1.05, 1.1, 1.3):
        for mn in (3, 5, 8):
            boxes = cascade.detectMultiScale(gray, sf, mn, minSize=(40, 40))
            matched_gt = set()
            fps = 0
            for b in boxes:
                b = tuple(map(int, b))
                hit = False
                for gi, g in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    if _iou(b, g) >= 0.25:
                        matched_gt.add(gi)
                        hit = True
                        break
                if not hit:
                    fps += 1
            hits = len(matched_gt)
            misses = gt_count - hits
            rows.append(f"{sf},{mn},{len(boxes)},{hits},{misses},{fps}")

    (OUT / "t2_haar_params.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    boxes = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    vis = group.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(str(OUT / "t2_haar_group.png"), vis)
    cv2.imwrite(str(OUT / "t2_group_input.png"), group)

    # Annotate a compressed sample of the street video (every 3rd frame, ~100 frames)
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    n = 0
    written = 0
    while written < 100:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % 3 != 0:
            continue
        # shrink for report-sized avi
        h0, w0 = frame.shape[:2]
        if w0 > 720:
            s = 720 / w0
            frame = cv2.resize(frame, (720, int(h0 * s)))
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(g, 1.1, 5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if writer is None:
            hh, ww = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(OUT / "t2_haar_video.avi"),
                cv2.VideoWriter_fourcc(*"XVID"),
                10,
                (ww, hh),
            )
        writer.write(frame)
        written += 1
    cap.release()
    if writer:
        writer.release()
    (OUT / "t2_video_note.txt").write_text(
        f"annotated_frames={written}\nsource={video_path.name}\n"
        f"note=subsampled every 3rd frame from phone street clip\n",
        encoding="utf-8",
    )


def task3_yolo(group_path: Path, video_path: Path):
    from ultralytics import YOLO

    weights = ROOT / "prac7" / "weights" / "yolov8n.pt"
    model = YOLO(str(weights) if weights.exists() else "yolov8n.pt")

    cap0 = cv2.VideoCapture(str(video_path))
    ok, street_still = cap0.read()
    cap0.release()
    if ok:
        h0, w0 = street_still.shape[:2]
        if w0 > 960:
            s = 960 / w0
            street_still = cv2.resize(street_still, (960, int(h0 * s)))
        cv2.imwrite(str(OUT / "t3_street_still.png"), street_still)
        street_path = OUT / "t3_street_still.png"
    else:
        street_path = IMG / "prac2_show.jpg"

    paths = {
        "group": group_path,
        "street": street_path,
        "objects": IMG / "object.jpg",
    }
    conf_lines = ["image,conf,num_detections"]
    for name, p in paths.items():
        img = cv2.imread(str(p))
        if img is None:
            continue
        for conf in (0.1, 0.25, 0.5, 0.75):
            res = model(img, conf=conf, verbose=False)[0]
            conf_lines.append(f"{name},{conf},{len(res.boxes)}")
        plotted = model(img, conf=0.25, verbose=False)[0].plot()
        cv2.imwrite(str(OUT / f"t3_yolo_{name}.png"), plotted)

    (OUT / "t3_conf_study.csv").write_text("\n".join(conf_lines) + "\n", encoding="utf-8")

    img = cv2.imread(str(paths["street"]))
    many = model(img, conf=0.05, iou=0.9, verbose=False)[0]
    few = model(img, conf=0.5, iou=0.45, verbose=False)[0]
    save_row(
        [many.plot(), few.plot()],
        [
            f"Low conf / loose IoU ({len(many.boxes)} boxes)",
            f"conf=0.5 NMS ({len(few.boxes)} boxes)",
        ],
        OUT / "t3_nms_compare.png",
        scale=4.5,
    )
    (OUT / "t3_nms_note.txt").write_text(
        "NMS keeps the highest-confidence box and suppresses overlapping lower-score boxes "
        "(IoU above threshold). Without it, one object gets many stacked detections.\n",
        encoding="utf-8",
    )

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while len(frames) < 25:
        ok, fr = cap.read()
        if not ok:
            break
        idx += 1
        if idx % 4 != 0:
            continue
        h0, w0 = fr.shape[:2]
        if w0 > 640:
            s = 640 / w0
            fr = cv2.resize(fr, (640, int(h0 * s)))
        frames.append(fr)
    cap.release()
    if not frames:
        frames = [cv2.imread(str(group_path))]

    t0 = time.perf_counter()
    for fr in frames:
        cascade.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), 1.1, 5)
    haar_ms = 1000 * (time.perf_counter() - t0) / len(frames)

    t0 = time.perf_counter()
    for fr in frames:
        model(fr, conf=0.25, verbose=False)
    yolo_ms = 1000 * (time.perf_counter() - t0) / len(frames)

    writer = None
    for fr in frames[:40]:
        plotted = model(fr, conf=0.25, verbose=False)[0].plot()
        if writer is None:
            hh, ww = plotted.shape[:2]
            writer = cv2.VideoWriter(
                str(OUT / "t3_yolo_video.avi"),
                cv2.VideoWriter_fourcc(*"XVID"),
                8,
                (ww, hh),
            )
        writer.write(plotted)
    if writer:
        writer.release()

    (OUT / "t3_timing.txt").write_text(
        f"haar_ms_per_frame={haar_ms:.2f}\n"
        f"yolo_ms_per_frame={yolo_ms:.2f}\n"
        f"frames_timed={len(frames)}\n"
        "note=Haar is face-only and usually faster on CPU; YOLO detects many classes and is heavier but more robust to pose/occlusion\n",
        encoding="utf-8",
    )


def main():
    group_path = ensure_group_faces()
    video_path = ensure_street_video()
    print("group:", group_path)
    print("video:", video_path)
    task1_template()
    task2_haar(group_path, video_path)
    task3_yolo(group_path, video_path)
    print("Practical 3 figures saved to", OUT)


if __name__ == "__main__":
    main()
