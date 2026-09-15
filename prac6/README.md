# CSC 6302 Practical 6 - Motion, Tracking, Geometry

**Student:** Munyeme Sinyani (SINYANI) - University of Zambia

Optical flow, Kalman filter, calibration, AR cube, stereo disparity.

## Install

```bash
cd csc6302-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python shared/make_demo_images.py
```

## Run

```bash
python prac6/run_prac6.py
```

Outputs include `t2_kalman.avi`, `camera.npz`, and stereo maps in `prac6/report/figures/`.

Kalman predict/update is in class `KF` with state `[x, y, vx, vy]`. When measurement is `None` I coast on predict only.

Motion frames are synthetic. A phone clip of a coloured object works better for my Kalman write-up. The stereo pair is a teaching stub.

## AI Use

Cursor helped structure flow plots and the tracker loop. I tuned `q` and `r` and wrote about flow failure under brightness change myself. I will use real motion/stereo captures in the final report where I can.
