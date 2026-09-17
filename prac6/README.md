# CSC 6302 Practical 6 — Motion, Tracking, Depth & AR

**Student:** Munyeme Sinyani (SINYANI) · Computer number **2025024319** · University of Zambia

Optical flow → own Kalman filter through occlusion → camera calibration → AR cube → stereo SGBM.

No Hugging Face Space this time (webcam/video pipelines). Marker runs locally; demo video is in the report folder.

## Naming

| Item | Name |
|------|------|
| GitHub pack | `Salaam21/csc6302-computer-vision-practicals` |
| Suggested hand-in | `csc6302-prac6-SINYANI` |

## Install

```bash
cd csc6302-computer-vision-practicals
pip install -r prac6/requirements.txt
```

Use **opencv-contrib** (not plain opencv-python) so CSRT is available.

## Exact commands

```bash
# 1) Build inputs (flow frames from street_clip, occlusion ball, calib views, AR pan)
python prac6/prepare_assets.py

# 2) One script per task
python prac6/task1_optical_flow.py
python prac6/task2_kalman.py
python prac6/task3_calibrate.py
python prac6/task4_ar.py
python prac6/task5_stereo.py

# Or everything:
python prac6/run_prac6.py

# Report + captioned demo reel
python prac6/build_report.py
python prac6/make_demo.py
```

## Outputs

| Path | What |
|------|------|
| `prac6/report/figures/` | All stills, KF plot, calib numbers, disparity maps |
| `prac6/report/figures/t2_kalman.mp4` | Annotated KF track |
| `prac6/report/figures/t4_ar_cube.mp4` | AR cube pan |
| `prac6/report/figures/camera.npz` | Intrinsics for Task 4 |
| `prac6/report/Practical6_Report.pdf` | 4–6 page report |
| `prac6/report/Prac6_Demo.mp4` | Captioned demo (~3–5 min class length at 12 fps) |
| `prac6/data/` | Intermediate videos / calib images |

## Inputs

- Flow: sampled from `images/street_clip.mp4` (my phone)
- Occlusion: generated red ball + pillar in `prac6/data/occlusion_ball.mp4`
- Calibration / AR: 9×6 chessboard views in `prac6/data/calib_images/` (+ printable `images/checkerboard.png`)
- Stereo: `images/stereo_left.jpg`, `images/stereo_right.jpg`

## Kalman (from scratch)

Class `KF` in `task2_kalman.py` — state `[x,y,vx,vy]`, predict `x=Fx`, `P=FPF'+Q`, update with innovation / gain. `update(None)` coasts during occlusion. Tuned `q=5e-2`, `r=12`.

## AI use

Composer helped structure scripts and the demo stitcher. Kalman equations, noise tuning, and report wording are mine.
