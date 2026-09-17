"""Build Practical 5 report PDF (Munyeme Sinyani voice)."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FIG = HERE / "report" / "figures"
OUT = HERE / "report" / "Practical5_Report.pdf"


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t", parent=base["Title"], fontSize=16, leading=20, spaceAfter=8
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=11, leading=14, spaceBefore=8, spaceAfter=4
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=10, leading=13, alignment=TA_JUSTIFY, spaceAfter=6
        ),
        "caption": ParagraphStyle(
            "cap", parent=base["Normal"], fontSize=8, leading=10, spaceAfter=8, textColor="#333333"
        ),
        "meta": ParagraphStyle(
            "meta", parent=base["Normal"], fontSize=9, leading=12, spaceAfter=4
        ),
    }


def fig(path: Path, width=16 * cm):
    if not path.exists():
        return Paragraph(f"[missing figure: {path.name}]", styles()["caption"])
    # Keep aspect
    from PIL import Image as PILImage

    im = PILImage.open(path)
    w, h = im.size
    height = width * (h / w)
    if height > 9 * cm:
        height = 9 * cm
        width = height * (w / h)
    return Image(str(path), width=width, height=height)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def build():
    st = styles()
    story = []
    story.append(Paragraph("CSC 6302 Practical 5 — Image Segmentation", st["title"]))
    story.append(
        Paragraph(
            "Munyeme Sinyani (SINYANI) · Computer number 2025024319 · "
            "University of Zambia · School of Natural and Applied Sciences · "
            "Department of Computing and Informatics",
            st["meta"],
        )
    )
    story.append(
        Paragraph(
            "Detection draws a box; segmentation answers which pixels belong. "
            "I work through global and adaptive thresholding with a hand-made mask and my own "
            "IoU, marker-based watershed on touching coins, GrabCut on a portrait object, "
            "then DeepLabV3 (semantic) versus Mask R-CNN (instance), and a Gradio segmentation studio.",
            st["body"],
        )
    )

    # Task 1
    story.append(Paragraph("Task 1 — Thresholding, honestly evaluated", st["h1"]))
    iou_txt = read(FIG / "t1_iou.csv")
    story.append(
        Paragraph(
            "I built an unevenly-lit document page (strong left–right brightness ramp) so a single "
            "threshold has to fail somewhere. For a representative crop I painted a reference mask "
            "myself (stroke regions + ink darkening), saved as <font face='Courier'>images/prac5_handmade_mask.png</font>. "
            "IoU is my NumPy function: intersection of binary masks over their union.",
            st["body"],
        )
    )
    rows = [r for r in iou_txt.strip().splitlines()[1:] if r]
    table_bits = ", ".join(r.replace(",", " = ") for r in rows)
    story.append(Paragraph(f"IoU results: {table_bits}.", st["body"]))
    story.append(
        Paragraph(
            read(FIG / "t1_note.txt") or (
                "Adaptive wins on this ramp because each window chooses a local cut; "
                "Otsu and fixed pick one global value that cannot serve both the dark and bright sides."
            ),
            st["body"],
        )
    )
    story.append(fig(FIG / "t1_threshold.png"))
    story.append(Paragraph("Figure: crop, fixed, Otsu, adaptive mean/Gaussian, hand-made GT.", st["caption"]))

    # Task 2
    story.append(Paragraph("Task 2 — Watershed for touching objects", st["h1"]))
    count = read(FIG / "t2_count.txt")
    story.append(
        Paragraph(
            "Touching coins defeat plain connected components: the binary blob is one component. "
            "Marker-based watershed: threshold → morphological opening → distance transform → "
            "sure-foreground peaks → <font face='Courier'>cv2.watershed</font>. "
            "Intermediates are saved under <font face='Courier'>report/figures/t2_*.png</font>.",
            st["body"],
        )
    )
    story.append(Paragraph(count.replace("\n", "<br/>"), st["body"]))
    story.append(fig(FIG / "t2_watershed.png"))
    story.append(Paragraph("Figure: threshold, opening, distance, sure FG, watershed boundaries.", st["caption"]))
    # show a couple of intermediates if present
    for name, cap in [
        ("t2_05_distance.png", "Distance transform"),
        ("t2_06_sure_fg.png", "Sure-foreground markers"),
        ("t2_08_watershed.png", "Final coloured watershed"),
    ]:
        p = FIG / name
        if p.exists():
            story.append(fig(p, width=12 * cm))
            story.append(Paragraph(f"Figure: {cap}.", st["caption"]))

    story.append(PageBreak())

    # Task 3
    story.append(Paragraph("Task 3 — GrabCut", st["h1"]))
    story.append(
        Paragraph(
            "I cut the creatine jar (and hand) from <font face='Courier'>portrait.jpg</font> with a bounding rectangle, "
            "then compared 1 vs 5 iterations, then added definite background strokes at the borders and a "
            "definite foreground circle on the label.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            read(FIG / "t3_note.txt")
            or "GrabCut iterates GMMs for FG/BG colour and a graph-cut labelling.",
            st["body"],
        )
    )
    story.append(fig(FIG / "t3_grabcut.png"))
    story.append(Paragraph("Figure: rect seed, 1 iter, 5 iters, mask hints.", st["caption"]))

    # Task 4
    story.append(Paragraph("Task 4 — Semantic vs instance", st["h1"]))
    timing = read(FIG / "t4_timing.txt")
    story.append(
        Paragraph(
            "I ran torchvision DeepLabV3-ResNet50 (VOC semantic) and Mask R-CNN-ResNet50-FPN (COCO instance) "
            "on my street scene and on a group street photo with many people. "
            "Semantic segmentation paints every person pixel with one 'person' class; "
            "instance segmentation gives each person a separate coloured mask.",
            st["body"],
        )
    )
    story.append(Paragraph(timing.replace("\n", "<br/>"), st["body"]))
    story.append(Paragraph(read(FIG / "t4_unet_note.txt"), st["body"]))
    story.append(fig(FIG / "t4_semantic_vs_instance.png"))
    story.append(Paragraph("Figure: group photo — one person blob vs separate instances.", st["caption"]))
    if (FIG / "t4_street_both.png").exists():
        story.append(fig(FIG / "t4_street_both.png"))
        story.append(Paragraph("Figure: street scene — DeepLab vs Mask R-CNN.", st["caption"]))
    story.append(Paragraph("Failure case", st["h2"]))
    story.append(Paragraph(read(FIG / "t4_failure.txt"), st["body"]))
    if (FIG / "t4_failure_deeplab.png").exists():
        story.append(fig(FIG / "t4_failure_deeplab.png", width=12 * cm))
        story.append(Paragraph("Figure: patterned cloth — classes the model was never taught.", st["caption"]))

    story.append(PageBreak())

    # Task 5
    story.append(Paragraph("Task 5 — Deployed segmentation studio", st["h1"]))
    story.append(
        Paragraph(
            "Gradio app (<font face='Courier'>prac5/app.py</font>): image upload; dropdown "
            "Otsu / adaptive / watershed / DeepLabV3; overlay output; watershed object count; "
            "DeepLab class legend; UI note that DeepLab on CPU takes a few seconds. "
            "App source: https://huggingface.co/Salaam21/csc6302-prac5-SINYANI . "
            "Creating a live Gradio Space returned HTTP 402 — restricted by Hugging Face PRO "
            "(same as Practical 4). Local: <font face='Courier'>python prac5/app.py</font>.",
            st["body"],
        )
    )

    story.append(Paragraph("Reproducibility & integrity", st["h1"]))
    story.append(
        Paragraph(
            "Reproduce: <font face='Courier'>pip install -r prac5/requirements.txt</font>, "
            "then <font face='Courier'>python prac5/prepare_images.py</font>, "
            "<font face='Courier'>python prac5/run_prac5.py</font>, "
            "<font face='Courier'>python prac5/app.py</font>. "
            "Composer (Cursor) helped wire OpenCV/torchvision and the Gradio shell; "
            "IoU, mask painting choices, watershed discussion, and this report wording are mine. "
            "The hand-made mask is my annotation for this crop — not shared with other students.",
            st["body"],
        )
    )
    story.append(
        Paragraph(
            "References: OpenCV docs (threshold, watershed, GrabCut); "
            "torchvision DeepLabV3 / Mask R-CNN model cards; course lectures on U-Net and segmentation.",
            st["body"],
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    doc.build(story)
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
