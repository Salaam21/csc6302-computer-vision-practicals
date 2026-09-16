# CSC 6302 - Computer Vision Practicals (1-7)

**Student:** Munyeme Sinyani (SINYANI) - University of Zambia

My code, photos, figures and reports for practicals 1 through 7.

## How I set up

```bash
cd csc6302-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Photos are in `images/`.

## How I run the figure scripts

```bash
python prac1/run_prac1.py
python prac2/run_prac2.py
python prac3/run_prac3.py
python prac4/train_prac4.py
python prac5/run_prac5.py
python prac6/run_prac6.py
cd prac7 && pytest -q && cd ..
python prac7/run_prac7.py
python prac7/app.py
```

Practical 7 is the **Road & transport** capstone (vehicle counting + pothole hint). Phone companion: [Salaam21/blindspot](https://github.com/Salaam21/blindspot) branch `vehicle-counting`.

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

I used Composer to help with ideas for script layout and OpenCV/PyTorch glue. Working photos are in `images/`.
