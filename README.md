# CSC 6302 - Computer Vision Practicals (1-7)

**Student:** Munyeme Sinyani (SINYANI) - University of Zambia

My code and figures for practicals 1 through 7, plus a master walkthrough PDF in `master_report/`.

## How I set up

```bash
cd csc6302-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python shared/make_demo_images.py
```

## How I run the figure scripts

```bash
python prac1/run_prac1.py
python prac2/run_prac2.py
python prac3/run_prac3.py
python prac4/train_prac4.py
python prac5/run_prac5.py
python prac6/run_prac6.py
cd prac7 && pytest -q && cd ..
python master_report/build_master_pdf.py
```

Practical 3 needs `ultralytics` for YOLO. Practical 4 training takes a few minutes on CPU.

## Gradio apps

```bash
python prac3/app.py
python prac4/app.py
python prac5/app.py
python prac7/app.py
```

## Repo names (course sheet)

- Individual: `csc6302-pracN-SINYANI`
- Capstone: `csc6302-capstone-SINYANI-PARTNER`

## AI Use

I used Cursor (Composer) to help layout scripts, fix OpenCV/PyTorch glue, and draft READMEs. From-scratch parts (PSNR, Sobel, Harris, IoU, Kalman update) follow the brief and I can explain them in a viva. Demo images in `shared/images/` are synthetic until I swap in my own phone photos.
