"""Task 1 — Sparse LK trails + dense Farneback + brightness-constancy failure."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _frames():
    paths = sorted((DATA / "flow_frames").glob("frame_*.png"))
    if len(paths) < 5:
        from prepare_assets import extract_flow_frames

        extract_flow_frames()
        paths = sorted((DATA / "flow_frames").glob("frame_*.png"))
    return [cv2.imread(str(p)) for p in paths]


def run():
    frames = _frames()
    old = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(old, maxCorners=120, qualityLevel=0.01, minDistance=8)
    trail = frames[0].copy()
    # accumulate trails across a stretch of the clip
    n = min(25, len(frames) - 1)
    for fr in frames[1 : n + 1]:
        new = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            old, new, pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if nxt is None:
            break
        good_new = nxt[st.ravel() == 1]
        good_old = pts[st.ravel() == 1]
        for p0, p1 in zip(good_old, good_new):
            a, b = tuple(p0.ravel().astype(int)), tuple(p1.ravel().astype(int))
            cv2.line(trail, a, b, (0, 255, 0), 1)
            cv2.circle(trail, b, 2, (0, 0, 255), -1)
        pts = good_new.reshape(-1, 1, 2).astype(np.float32)
        old = new
        if len(pts) < 20:
            pts = cv2.goodFeaturesToTrack(old, 120, 0.01, 8)
            if pts is None:
                break
    cv2.imwrite(str(OUT / "t1_lk_trails.png"), trail)

    # Dense Farneback on a pair a few frames apart
    i0, i1 = 5, min(12, len(frames) - 1)
    g0 = cv2.cvtColor(frames[i0], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frames[i1], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*g0.shape, 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    dense = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(OUT / "t1_farneback.png"), dense)
    cv2.imwrite(str(OUT / "t1_farneback_pair.png"), np.hstack([frames[i0], frames[i1], dense]))

    # Break brightness constancy: sudden gain / “lighting change”
    base = frames[min(15, len(frames) - 1)]
    bright = np.clip(base.astype(np.float32) * 2.4 + 20, 0, 255).astype(np.uint8)
    ga = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
    flow_bad = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_b, ang_b = cv2.cartToPolar(flow_bad[..., 0], flow_bad[..., 1])
    hsv2 = np.zeros((*ga.shape, 3), np.uint8)
    hsv2[..., 0] = ang_b * 180 / np.pi / 2
    hsv2[..., 1] = 255
    hsv2[..., 2] = cv2.normalize(mag_b, None, 0, 255, cv2.NORM_MINMAX)
    bad = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    # Also motion-blur failure
    blur = cv2.GaussianBlur(base, (21, 21), 0)
    gc = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    flow_blur = cv2.calcOpticalFlowFarneback(ga, gc, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag_c, ang_c = cv2.cartToPolar(flow_blur[..., 0], flow_blur[..., 1])
    hsv3 = np.zeros_like(hsv2)
    hsv3[..., 0] = ang_c * 180 / np.pi / 2
    hsv3[..., 1] = 255
    hsv3[..., 2] = cv2.normalize(mag_c, None, 0, 255, cv2.NORM_MINMAX)
    bad_blur = cv2.cvtColor(hsv3, cv2.COLOR_HSV2BGR)

    cv2.imwrite(str(OUT / "t1_flow_failure.png"), np.hstack([base, bright, bad]))
    cv2.imwrite(str(OUT / "t1_flow_failure_blur.png"), np.hstack([base, blur, bad_blur]))
    (OUT / "t1_note.txt").write_text(
        "LK assumes brightness constancy, spatial coherence, and small motion. "
        "A sudden lighting gain invents large false flow even with no camera motion — "
        "pixels change intensity without moving. Heavy blur also kills corner texture "
        "that Shi-Tomasi / LK need.\n",
        encoding="utf-8",
    )
    print("Task1 optical flow →", OUT)


if __name__ == "__main__":
    run()
