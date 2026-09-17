"""Build Practical 6 report PDF (Munyeme Sinyani)."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate

HERE = Path(__file__).resolve().parent
FIG = HERE / "report" / "figures"
OUT = HERE / "report" / "Practical6_Report.pdf"


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=15, leading=18, spaceAfter=8),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=12, leading=15, spaceBefore=10, spaceAfter=5),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=5),
        "cap": ParagraphStyle("cap", parent=base["Normal"], fontSize=8, leading=10, spaceAfter=7, textColor="#333333"),
        "meta": ParagraphStyle("meta", parent=base["Normal"], fontSize=9, leading=11, spaceAfter=4),
    }


def fig(path: Path, width=15.5 * cm):
    st = styles()
    if not path.exists():
        return Paragraph(f"[missing {path.name}]", st["cap"])
    from PIL import Image as PILImage

    im = PILImage.open(path)
    w, h = im.size
    height = width * (h / w)
    if height > 8.5 * cm:
        height = 8.5 * cm
        width = height * (w / h)
    return Image(str(path), width=width, height=height)


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def build():
    st = styles()
    story = []
    story.append(Paragraph("CSC 6302 Practical 6 — Motion, Tracking, Depth & AR", st["title"]))
    story.append(
        Paragraph(
            "Munyeme Sinyani (SINYANI) · Computer number 2025024319 · University of Zambia · "
            "Computing and Informatics",
            st["meta"],
        )
    )
    story.append(
        Paragraph(
            "This practical adds time and geometry: optical flow on my street clip, a from-scratch "
            "Kalman filter that coasts through occlusion, camera calibration, an AR cube locked to "
            "a chessboard, and stereo disparity. No cloud host — everything runs locally.",
            st["body"],
        )
    )

    story.append(Paragraph("Task 1 — Optical flow", st["h1"]))
    story.append(
        Paragraph(
            "On frames sampled from <font face='Courier'>street_clip.mp4</font> I detect Shi-Tomasi "
            "corners and track them with pyramidal Lucas–Kanade, drawing green trails. Dense Farneback "
            "flow is shown in HSV (hue = direction, value = magnitude). I then break brightness "
            "constancy with a sudden gain: the flow map lights up even though nothing moved — exactly "
            "what the LK assumptions forbid.",
            st["body"],
        )
    )
    story.append(Paragraph(read(FIG / "t1_note.txt"), st["body"]))
    story.append(fig(FIG / "t1_lk_trails.png"))
    story.append(Paragraph("Figure: LK motion trails on the street clip.", st["cap"]))
    story.append(fig(FIG / "t1_farneback_pair.png" if (FIG / "t1_farneback_pair.png").exists() else FIG / "t1_farneback.png"))
    story.append(Paragraph("Figure: Farneback dense flow (HSV).", st["cap"]))
    story.append(fig(FIG / "t1_flow_failure.png"))
    story.append(Paragraph("Figure: lighting change invents false flow.", st["cap"]))

    story.append(Paragraph("Task 2 — Kalman through occlusion", st["h1"]))
    story.append(
        Paragraph(
            "I detect a bright red ball with an HSV picker (largest contour centroid) and feed "
            "measurements into my own constant-velocity Kalman filter (state x, y, vx, vy). "
            "Predict and update are written in NumPy — no <font face='Courier'>cv2.KalmanFilter</font>. "
            "When the ball hides behind the pillar, <font face='Courier'>update(None)</font> coasts on prediction.",
            st["body"],
        )
    )
    story.append(Paragraph(read(FIG / "t2_kf_tune.txt").replace("\n", "<br/>"), st["body"]))
    story.append(fig(FIG / "t2_kf_plot.png"))
    story.append(Paragraph("Figure: x-position — measurements (green), KF (red), occlusion band.", st["cap"]))
    story.append(Paragraph(read(FIG / "t2_csrt.txt").replace("\n", "<br/>"), st["body"]))

    story.append(PageBreak())
    story.append(Paragraph("Task 3 — Camera calibration", st["h1"]))
    story.append(
        Paragraph(
            "From 15+ chessboard views (9×6 inner corners) I run "
            "<font face='Courier'>findChessboardCorners</font> + <font face='Courier'>calibrateCamera</font>. "
            "RMS reprojection error target is &lt; 1 px. Focal length in mm uses an assumed ~5.6 mm sensor width.",
            st["body"],
        )
    )
    story.append(Paragraph(f"<font face='Courier' size='8'>{read(FIG / 't3_calib.txt').replace(chr(10), '<br/>')}</font>", st["body"]))
    story.append(fig(FIG / "t3_undistort.png"))
    story.append(Paragraph("Figure: before / after undistortion.", st["cap"]))

    story.append(Paragraph("Task 4 — AR cube", st["h1"]))
    story.append(
        Paragraph(
            read(FIG / "t4_note.txt")
            or "solvePnP recovers pose; calibration must come first so projection uses the right K.",
            st["body"],
        )
    )
    story.append(fig(FIG / "t4_ar_cube.png"))
    story.append(Paragraph("Figure: virtual cube + XYZ axes anchored to the board.", st["cap"]))

    story.append(Paragraph("Task 5 — Stereo depth", st["h1"]))
    story.append(Paragraph(read(FIG / "t5_notes.txt").replace("\n", "<br/>"), st["body"]))
    story.append(fig(FIG / "t5_stereo.png"))
    story.append(Paragraph("Figure: disparity maps for two numDisparities / blockSize settings.", st["cap"]))

    story.append(Paragraph("Demo video & integrity", st["h1"]))
    story.append(
        Paragraph(
            "Commands are in <font face='Courier'>prac6/README.md</font>. "
            "A captioned demo reel is built by <font face='Courier'>python prac6/make_demo.py</font> → "
            "<font face='Courier'>prac6/report/Prac6_Demo.mp4</font>. "
            "Composer helped wire scripts; Kalman equations, tuning, and this write-up are mine. "
            "Calibration views are generated for my simulated phone camera in this repo — not copied from classmates.",
            st["body"],
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
    ).build(story)
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
