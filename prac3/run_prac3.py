"""Run Practical 3 tasks and save figures for my report."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "shared" / "images"
OUT = Path(__file__).resolve().parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save_pair(imgs, titles, path):
    fig, axes = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 3.5))
    if len(imgs) == 1:
        axes = [axes]
    for ax, im, t in zip(axes, imgs, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.shape[2] == 3 else im)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def task1_template():
    scene = cv2.imread(str(IMG / "colourful.jpg"))
    # I crop the green circle as my template
    tmpl = scene[70:210, 410:550].copy()
    res = cv2.matchTemplate(scene, tmpl, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    hit = scene.copy()
    h, w = tmpl.shape[:2]
    cv2.rectangle(hit, maxloc, (maxloc[0] + w, maxloc[1] + h), (0, 255, 0), 2)

    # template matching fails when scale changes
    scaled = cv2.resize(scene, None, fx=0.6, fy=0.6)
    res2 = cv2.matchTemplate(scaled, tmpl, cv2.TM_CCOEFF_NORMED)
    _, maxv2, _, _ = cv2.minMaxLoc(res2)
    score_vis = (res * 255).astype(np.uint8)
    score_vis2 = (res2 * 255).astype(np.uint8)
    save_pair(
        [hit, score_vis, score_vis2],
        [f"Found (score={maxv:.2f})", "Score map OK", f"Scaled scene score max={maxv2:.2f}"],
        OUT / "t1_template.png",
    )
    (OUT / "t1_scores.txt").write_text(f"same_view={maxv:.4f}\nscaled_view={maxv2:.4f}\n")


def task2_haar():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    group = cv2.imread(str(IMG / "group.jpg"))
    gray = cv2.cvtColor(group, cv2.COLOR_BGR2GRAY)
    rows = ["scaleFactor,minNeighbors,detections"]
    for sf in (1.05, 1.1, 1.3):
        for mn in (3, 5, 8):
            boxes = cascade.detectMultiScale(gray, sf, mn)
            rows.append(f"{sf},{mn},{len(boxes)}")
    (OUT / "t2_haar_params.csv").write_text("\n".join(rows) + "\n")
    boxes = cascade.detectMultiScale(gray, 1.1, 5)
    vis = group.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(str(OUT / "t2_haar_group.png"), vis)

    # short clip from motion frames
    frames_dir = IMG / "motion_frames"
    writer = None
    for i, fp in enumerate(sorted(frames_dir.glob("frame_*.png"))[:30]):
        frame = cv2.imread(str(fp))
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(g, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if writer is None:
            h0, w0 = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(OUT / "t2_haar_video.avi"),
                cv2.VideoWriter_fourcc(*"XVID"),
                10,
                (w0, h0),
            )
        writer.write(frame)
    if writer:
        writer.release()


def task3_yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        (OUT / "t3_yolo_SKIPPED.txt").write_text(
            "ultralytics not installed. pip install ultralytics\n"
        )
        return

    model = YOLO("yolov8n.pt")
    paths = {
        "group": IMG / "group.jpg",
        "street": IMG / "street.jpg",
        "objects": IMG / "colourful.jpg",
    }
    conf_lines = ["image,conf,num_detections"]
    for name, p in paths.items():
        img = cv2.imread(str(p))
        for conf in (0.1, 0.25, 0.5, 0.75):
            res = model(img, conf=conf, verbose=False)[0]
            conf_lines.append(f"{name},{conf},{len(res.boxes)}")
        plotted = model(img, conf=0.25, verbose=False)[0].plot()
        cv2.imwrite(str(OUT / f"t3_yolo_{name}.png"), plotted)
    (OUT / "t3_conf_study.csv").write_text("\n".join(conf_lines) + "\n")

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    frames = [cv2.imread(str(fp)) for fp in sorted((IMG / "motion_frames").glob("frame_*.png"))[:20]]
    t0 = time.perf_counter()
    for fr in frames:
        cascade.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), 1.1, 5)
    haar_ms = 1000 * (time.perf_counter() - t0) / len(frames)
    t0 = time.perf_counter()
    for fr in frames:
        model(fr, conf=0.25, verbose=False)
    yolo_ms = 1000 * (time.perf_counter() - t0) / len(frames)
    (OUT / "t3_timing.txt").write_text(
        f"haar_ms_per_frame={haar_ms:.2f}\nyolo_ms_per_frame={yolo_ms:.2f}\n"
    )


def main():
    task1_template()
    task2_haar()
    task3_yolo()
    print("Practical 3 figures saved to", OUT)


if __name__ == "__main__":
    main()
