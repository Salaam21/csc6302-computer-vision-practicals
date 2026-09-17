"""Build Practical 7 capstone report (6–8 pages) matching the brief."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent
FIG = HERE / "report" / "figures"
OUT = HERE / "report" / "Practical7_Capstone_Report.pdf"


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=14, leading=17, spaceAfter=6),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=12, leading=15, spaceBefore=9, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=10.5, leading=13, spaceBefore=7, spaceAfter=3),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=5),
        "cap": ParagraphStyle("cap", parent=base["Normal"], fontSize=8, leading=10, spaceAfter=6, textColor="#333333"),
        "meta": ParagraphStyle("meta", parent=base["Normal"], fontSize=9, leading=11, spaceAfter=3),
    }


def fig(path: Path, width=16 * cm):
    st = styles()
    if not path.exists():
        return Paragraph(f"[missing {path.name}]", st["cap"])
    from PIL import Image as PILImage

    im = PILImage.open(path)
    w, h = im.size
    height = width * (h / w)
    if height > 10 * cm:
        height = 10 * cm
        width = height * (w / h)
    return Image(str(path), width=width, height=height)


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def build():
    st = styles()
    story = []
    story.append(Paragraph("CSC 6302 Practical 7 — Capstone Report", st["title"]))
    story.append(
        Paragraph(
            "Road &amp; transport: vehicle counting / classification; pothole severity hint",
            st["meta"],
        )
    )
    story.append(
        Paragraph(
            "Munyeme Sinyani (SINYANI) · Computer number 2025024319 · University of Zambia · "
            "Solo submission (partner TBA — lecturer note: team-of-two preferred; this pack is ready for a second name).",
            st["meta"],
        )
    )
    story.append(
        Paragraph(
            "Mobile companion: <font face='Courier'>https://github.com/Salaam21/blindspot</font> "
            "(branch <font face='Courier'>vehicle-counting</font>). "
            "Coursework Gradio: <font face='Courier'>prac7/</font> in "
            "<font face='Courier'>csc6302-computer-vision-practicals</font>.",
            st["meta"],
        )
    )

    story.append(Paragraph("1. Problem and track", st["h1"]))
    story.append(
        Paragraph(
            "I chose the <b>Road &amp; transport</b> track. Road agencies and students need a phone-first "
            "tool that (1) counts and classifies vehicles (car, bus, truck, motorcycle, bicycle) in a roadside "
            "frame, and (2) gives a rough pothole severity hint from a dashcam-style photo. "
            "Wrong counts must not be treated as official traffic statistics; the pothole output is a "
            "heuristic, not a certified inspection.",
            st["body"],
        )
    )

    story.append(Paragraph("2. Vision core (R1)", st["h1"]))
    story.append(
        Paragraph(
            "Substantive CV component: a pretrained detector adapted to the task (Practical 3/5 style). "
            "Laptop/Gradio uses <b>YOLOv8n</b> (Ultralytics COCO weights), filtered to five vehicle classes, "
            "with a confidence gate (default 0.35). The Android Blindspot app uses on-device "
            "<b>SSD MobileNet v1 TFLite</b> already in the repo — same problem, different runtime. "
            "Pothole tab: classical OpenCV dark-blob heuristic on the lower ROI (justified because we lack "
            "a labelled Zambian pothole set for fine-tuning). "
            "Optional VLM explanation was not wired for this hand-in (OpenRouter quota); the CV pipeline "
            "stands alone with template plain-language summaries.",
            st["body"],
        )
    )

    story.append(Paragraph("3. Application (R2)", st["h1"]))
    story.append(
        Paragraph(
            "Gradio Blocks UI: upload → per-class counts + annotated boxes → plain-language summary + "
            "visible coursework disclaimer stating what the tool must <b>not</b> be used for. "
            "Graceful failure: tiny/unsupported images rejected; blurry frames rejected via gradient variance; "
            "zero detections or marginal confidence return an <b>UNSURE</b> message instead of a confident wrong total.",
            st["body"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("4. Evaluation (held-out ≥30)", st["h1"]))
    ev = read(FIG / "eval_heldout.txt") or "Run python prac7/eval_heldout.py to refresh metrics."
    story.append(
        Paragraph(
            "Metric for counting: mean absolute error (MAE) on total vehicles vs ground truth. "
            "I built ≥30 labelled scenes (synthetic car sprites with known counts + my manual totals on real frames). "
            "YOLO may under-count stylised sprites — that failure mode is intentional honesty for free-tier COCO weights "
            "without Lusaka-specific fine-tuning.",
            st["body"],
        )
    )
    story.append(Paragraph(ev.replace("\n", "<br/>"), st["body"]))
    story.append(
        Paragraph(
            "On the demo roadside still (<font face='Courier'>images/street.jpg</font>) the live path reported "
            "the counts in <font face='Courier'>t1_counts.json</font> (orientation-corrected). "
            "That single-frame demo is separate from the held-out MAE table.",
            st["body"],
        )
    )
    if (FIG / "t1_vehicle_count.png").exists():
        story.append(fig(FIG / "t1_vehicle_count.png"))
        story.append(Paragraph("Figure 1. Vehicle counting annotation on the demo roadside frame.", st["cap"]))
    if (FIG / "t2_pothole_hint.png").exists():
        story.append(fig(FIG / "t2_pothole_hint.png"))
        story.append(Paragraph("Figure 2. Pothole severity heuristic overlay on the lower ROI.", st["cap"]))

    story.append(Paragraph("4.1 Method detail — vehicle path", st["h2"]))
    story.append(
        Paragraph(
            "Pipeline: EXIF transpose → optional 90/270 rotation search → YOLOv8n predict at the confidence gate → "
            "keep only bicycle/car/motorcycle/bus/truck → draw boxes → format a sentence for non-programmers. "
            "Blur gate uses spatial gradient variance; below threshold returns a helpful reject, not a count of zero "
            "disguised as certainty. The rotation search exists because many phone roadside shots are stored sideways; "
            "I keep the orientation with the strongest vehicle total (ties broken by mean box confidence).",
            st["body"],
        )
    )
    story.append(Paragraph("4.2 Method detail — pothole path", st["h2"]))
    story.append(
        Paragraph(
            "Lower 55% of the frame is treated as road. Gaussian blur, percentile dark threshold, open/close morphology, "
            "then blob count and dark-pixel fraction map to low / medium / high. "
            "This classical choice is deliberate: without a licensed Zambian pothole dataset I refuse to pretend a "
            "fine-tuned detector exists. The UI labels the result a hint and repeats the disclaimer.",
            st["body"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("5. Engineering, tests, CI/CD (R3)", st["h1"]))
    story.append(
        Paragraph(
            "Repository: coursework monorepo <font face='Courier'>Salaam21/csc6302-computer-vision-practicals</font> "
            "(suggested dedicated name <font face='Courier'>csc6302-capstone-SINYANI</font> when splitting). "
            "pytest suite (≥5 meaningful tests): count schema, formatting, blur reject, tiny-image reject, "
            "confidence-gate unsure path, pothole heuristic. "
            "GitHub Actions workflow at repo root <font face='Courier'>.github/workflows/ci-cd.yml</font> "
            "installs deps, runs ruff + pytest on push/PR. "
            "CD job pushes Gradio files to a Hugging Face Space when <font face='Courier'>HF_TOKEN</font> "
            "is set as a GitHub secret — never committed. "
            "<b>Limitation:</b> this Hugging Face account currently gets HTTP 402 PRO for new Gradio Spaces "
            "(same as Practicals 4–5). Until PRO/education unlock, markers run "
            "<font face='Courier'>python prac7/app.py</font> locally; CD is automated in workflow form and "
            "documented. No API keys in the repo.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "Test design deliberately avoids <font face='Courier'>assert True</font>. "
            "Blur and tiny-image tests check reject messages; formatting tests check pluralisation; "
            "the confidence-gate test checks that an empty roadside-like texture returns UNSURE/zero without crashing. "
            "Heavy “model classifies this JPEG as car” golden tests are left to the held-out eval script so CI stays "
            "fast on free GitHub runners (YOLO weights download once on first local eval).",
            st["body"],
        )
    )

    story.append(Paragraph("6. Free-tier performance", st["h1"]))
    story.append(
        Paragraph(
            "YOLOv8n on CPU is a few hundred ms to ~2 s per 640-px frame after warm-up. "
            "First run downloads weights. Phone SSD TFLite is real-time on mid-range Android (Infinix test install). "
            "Night/rain/glare drop confidence — the UI prefers UNSURE/blurry over inventing counts. "
            "For a production Lusaka deploy I would quantise YOLO or keep TFLite on-device and only sync aggregates.",
            st["body"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("7. Ethics &amp; impact (R4)", st["h1"]))
    story.append(
        Paragraph(
            "This section is the one-page ethics statement required by the brief, written for a Zambian roadside "
            "deployment context rather than a generic checklist.",
            st["body"],
        )
    )
    story.append(Paragraph("Who could be harmed?", st["h2"]))
    story.append(
        Paragraph(
            "If counts were used for enforcement or contractor payments in Zambia, under-counting buses "
            "on a busy Lusaka corridor could under-fund a route; over-counting could waste budget. "
            "A high pothole hint on wet asphalt could divert crews from a real crater elsewhere. "
            "Drivers filmed without consent could feel surveilled if plates were read — this system "
            "does <b>not</b> OCR plates. Students using the demo for a traffic assignment must cite it as "
            "an estimate, not as city data.",
            st["body"],
        )
    )
    story.append(Paragraph("Error asymmetry", st["h2"]))
    story.append(
        Paragraph(
            "For traffic planning, a false negative (missed vehicle) is often worse than a false positive "
            "because demand is under-estimated. I keep the default confidence gate moderate (0.35) rather "
            "than very high. For potholes, a false positive (shadow → “high”) wastes a site visit; a false "
            "negative leaves a hazard — I label output as a hint and require human verification before any "
            "maintenance decision.",
            st["body"],
        )
    )
    story.append(Paragraph("Data provenance, licensing, consent", st["h2"]))
    story.append(
        Paragraph(
            "Demo stills are my own phone photos / clip. YOLOv8n and SSD MobileNet use COCO-trained "
            "public weights (Ultralytics / TF Hub lineage) — not trained on private Zambian CCTV. "
            "Synthetic eval sprites are generated in-repo. No third-party faces are labelled for identity. "
            "If a future version ingested council CCTV, a data-sharing agreement and retention limit would be required first.",
            st["body"],
        )
    )
    story.append(Paragraph("Bias risks", st["h2"]))
    story.append(
        Paragraph(
            "COCO under-represents minibus taxis, heavily loaded trucks, and dusty dry-season roads common "
            "locally. Night markets and matatus may be missed. Training data lacks Zambian dashcam potholes — "
            "the heuristic will confuse shadows and patches. Skin tone / clothing bias is less central for vehicle "
            "boxes, but pedestrian filtering must stay off to avoid implying people-tracking.",
            st["body"],
        )
    )
    story.append(Paragraph("Before real decisions", st["h2"]))
    story.append(
        Paragraph(
            "I would require: (1) a labelled Lusaka roadside set with agreement metrics; (2) human-in-the-loop "
            "review for any operational use; (3) written scope that this is a screening aid; "
            "(4) privacy review if cameras face private property; (5) no plate OCR without legal basis; "
            "(6) a published failure dashboard (night / rain / glare).",
            st["body"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("8. Limitations and failure cases", st["h1"]))
    story.append(
        Paragraph(
            "Per-frame counting double-counts the same vehicle across video frames (no tracker yet — Practical 6 "
            "Kalman / ByteTrack-style IDs could fuse later). Extreme blur and glare trip the blur gate or UNSURE path. "
            "Pothole heuristic fires on shadows. Synthetic MAE will look poor if YOLO ignores drawings — "
            "that shows the domain gap, not a hidden fine-tune. "
            "Live HF Space URL is the remaining submission risk on this account because Gradio Spaces currently "
            "require PRO; the mitigation is a working local app + CI green + documented CD job.",
            st["body"],
        )
    )

    story.append(Paragraph("9. Demo &amp; seminar plan", st["h1"]))
    story.append(
        Paragraph(
            "Five-minute demo: (1) open Gradio, upload street frame, show counts + boxes; (2) show blurry reject; "
            "(3) pothole tab on a road photo; (4) open Blindspot phone count screen; (5) open CI badge / pytest. "
            "Seminar: either member (when partner joins) must be able to change the confidence gate live and explain "
            "UNSURE behaviour.",
            st["body"],
        )
    )

    story.append(Paragraph("10. Reproducibility", st["h1"]))
    story.append(
        Paragraph(
            "<font face='Courier'>pip install -r prac7/requirements.txt</font><br/>"
            "<font face='Courier'>cd prac7 &amp;&amp; pytest -q</font><br/>"
            "<font face='Courier'>python prac7/run_prac7.py</font><br/>"
            "<font face='Courier'>python prac7/eval_heldout.py</font><br/>"
            "<font face='Courier'>python prac7/build_report.py</font><br/>"
            "<font face='Courier'>python prac7/app.py</font><br/>"
            "Proposal gate file: <font face='Courier'>prac7/PROPOSAL.md</font>. "
            "Mobile notes: <font face='Courier'>prac7/MOBILE.md</font>.",
            st["body"],
        )
    )

    story.append(Paragraph("11. Contributions &amp; AI use", st["h1"]))
    story.append(
        Paragraph(
            "Munyeme Sinyani: Blindspot vehicle-counting branch, Gradio app, tests, CI workflow, evaluation, report. "
            "Partner TBA for extra roadside labels / Space unlock / seminar co-demo. "
            "Composer helped with Gradio glue, eval harness, and report layout; track choice, disclaimers, and ethics wording are mine.",
            st["body"],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "References: Ultralytics YOLOv8 docs; COCO dataset; OpenCV morphology; CSC 6302 Practicals 3–6; "
            "course ethics lecture notes on dual-use and consent.",
            st["body"],
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    ).build(story)
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
