# CSC 6302 Practical 7 - Capstone (Maize Leaf Triage)

**Student:** Munyeme Sinyani (SINYANI) - University of Zambia  
**Team repo:** `csc6302-capstone-SINYANI-PARTNER` (partner TBA)

Phone upload: healthy / rust / blight guess, confidence gate, disclaimer.

## Install

```bash
cd csc6302-practicals
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python prac4/train_prac4.py
```

Training prac4 is optional but loads real torch weights into the capstone app.

## Run

```bash
cd prac7
pytest -q
cd ..
python prac7/app.py
```

See `PROPOSAL.md` for problem statement and team split.

I still need to register my partner, push the capstone repo, and set up Hugging Face with my own token. The disclaimer in `app.py` stays visible.

## AI Use

Composer helped with ideas for Gradio layout, blur gate, tests, and CI yaml. I wrote the guidance text and set the confidence gate to say "unsure" when it should.
