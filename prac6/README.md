# CSC 6302 Practical 6 - Motion, Tracking, Geometry

**Student:** Munyeme Sinyani (SINYANI) - University of Zambia

Optical flow, Kalman filter, calibration, AR cube, stereo disparity.

## Install

```bash
cd csc6302-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python prac6/run_prac6.py
```

Outputs include `t2_kalman.avi`, `camera.npz`, and stereo maps in `prac6/report/figures/`.

Kalman predict/update is in class `KF` with state `[x, y, vx, vy]`. When measurement is `None` I coast on predict only.

Motion frames and stereo inputs are under `images/`.

## AI Use

Composer helped with ideas for structuring flow plots and the tracker loop. I tuned `q` and `r` and wrote about flow failure under brightness change myself.
