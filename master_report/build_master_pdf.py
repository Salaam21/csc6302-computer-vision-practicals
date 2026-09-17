"""
Build CSC6302_All_Practicals_Walkthrough.pdf — Munyeme Sinyani student notes pack.

Run from repo root:
    python master_report/build_master_pdf.py

Requires: reportlab (see requirements.txt). Does NOT train models.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = Path(__file__).resolve().parent / "CSC6302_All_Practicals_Walkthrough.pdf"
PAGE_W, PAGE_H = A4
MARGIN = 2 * cm


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontSize=16,
            leading=20,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#1a365d"),
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=13,
            leading=17,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#2c5282"),
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            leftIndent=14,
            bulletIndent=0,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            backColor=colors.HexColor("#f4f4f4"),
            borderPadding=6,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
    }
    return styles


def P(text: str, style: str, styles: dict) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), styles[style])


def bullets(items: list[str], styles: dict) -> list:
    return [P(f"• {item}", "bullet", styles) for item in items]


def code_block(text: str, styles: dict) -> Preformatted:
    return Preformatted(text.strip(), styles["code"])


def embed_figures(fig_dir: Path, patterns: list[str], styles: dict, max_w=16 * cm) -> list:
    """Embed PNG figures if they exist; skip missing files gracefully."""
    story = []
    if not fig_dir.is_dir():
        story.append(P(f"<i>No figures folder yet — run the practical script first: {fig_dir}</i>", "body", styles))
        return story

    seen = set()
    for pat in patterns:
        for fp in sorted(fig_dir.glob(pat)):
            if fp.suffix.lower() != ".png" or fp in seen:
                continue
            seen.add(fp)
            try:
                img = RLImage(str(fp))
                iw, ih = img.drawWidth, img.drawHeight
                scale = min(max_w / iw, (10 * cm) / ih, 1.0)
                img.drawWidth = iw * scale
                img.drawHeight = ih * scale
                story.append(img)
                story.append(P(fp.name, "caption", styles))
            except Exception as exc:
                story.append(P(f"<i>Could not embed {fp.name}: {exc}</i>", "body", styles))
    if not seen:
        story.append(P(f"<i>No PNG figures found in {fig_dir} yet.</i>", "body", styles))
    return story


# ---------------------------------------------------------------------------
# Content sections
# ---------------------------------------------------------------------------

CODE_PSNR = """def psnr(clean, restored):
    mse = np.mean((clean.astype(float) - restored.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)"""

CODE_SOBEL = """# Gx and Gy are 3x3 Sobel kernels; convolve, then:
mag = np.sqrt(gx ** 2 + gy ** 2)"""

CODE_HARRIS = """# After Gaussian-smoothed Ix2, Iy2, Ixy:
R = (Ix2 * Iy2 - Ixy ** 2) - k * (Ix2 + Iy2) ** 2
# Keep local maxima of R above a threshold → corner points"""

CODE_IOU = """inter = np.logical_and(pred > 0, gt > 0).sum()
union = np.logical_or(pred > 0, gt > 0).sum()
iou = inter / union if union else 1.0"""

CODE_KALMAN = """# Predict (constant-velocity model):
x = F @ x
P = F @ P @ F.T + Q
# Update when measurement z arrives:
K = P @ H.T @ inv(H @ P @ H.T + R)
x = x + K @ (z - H @ x)
P = (I - K @ H) @ P
# If z is None (occlusion): skip update, coast on prediction"""


def section_cover(styles: dict) -> list:
    return [
        Spacer(1, 3 * cm),
        P("CSC 6302 Computer Vision", "title", styles),
        P("Practicals 1–7 — My Walkthrough Pack", "title", styles),
        Spacer(1, 0.5 * cm),
        P("Munyeme Sinyani · University of Zambia · 2025/26", "subtitle", styles),
        P(f"Generated {date.today():%d %B %Y}", "subtitle", styles),
        Spacer(1, 1.5 * cm),
        P(
            "This PDF is my own study notes — written like I am explaining to a confused classmate "
            "who missed the lab. Same vibe as Assignment 1: a computer does not see a picture, "
            "it sees a grid of numbers. Every practical here is about what those numbers mean and "
            "what we do to them.",
            "body",
            styles,
        ),
        PageBreak(),
    ]


def section_how_to_use(styles: dict) -> list:
    story = [
        P("How to use this pack", "h1", styles),
        P(
            "Think of this repo as three layers: (1) runnable scripts that produce report figures, "
            "(2) small Gradio apps where the brief asks for a demo, and (3) this PDF that tells you "
            "why each line exists. Do not submit this PDF blindly — use it to understand, then write "
            "your own report in your own words.",
            "body",
            styles,
        ),
        P("Suggested order", "h2", styles),
    ]
    story += bullets(
        [
            "Create a venv and <font face='Courier'>pip install -r requirements.txt</font>.",
            "Run <font face='Courier'># photos already in images/</font> once (my photos in images/).",
            "Run each <font face='Courier'>pracN/run_pracN.py</font> (or <font face='Courier'>train_prac4.py</font>) to fill <font face='Courier'>pracN/report/figures/</font>.",
            "Open the PNGs while reading the matching section below — the pictures are the proof.",
            "Photos live in images/.",
            "Re-run this script to refresh the master PDF after new figures exist.",
        ],
        styles,
    )
    story.append(PageBreak())
    return story


def section_general_instructions(styles: dict) -> list:
    story = [
        P("General course instructions (distilled)", "h1", styles),
        P("The arc across the semester", "h2", styles),
        P(
            "The practicals are deliberately stacked. We start where Assignment 1 left off — pixels as "
            "numbers — then climb: edges and morphology (Prac 1), interest points and stitching (Prac 2), "
            "detection (Prac 3), learning features (Prac 4), segmentation (Prac 5), motion and geometry "
            "(Prac 6), and finally a deployed capstone (Prac 7). Each week assumes you still remember "
            "the previous layer.",
            "body",
            styles,
        ),
        P("From-scratch rules (non-negotiable)", "h2", styles),
    ]
    story += bullets(
        [
            "When the brief says implement X yourself, calling the library one-liner scores zero — same rule as Assignment 1 banning <font face='Courier'>cv2.Sobel</font> for your own Sobel task.",
            "In this repo, the from-scratch pieces are: PSNR, 2-D convolution + Sobel magnitude, Harris R, IoU, and Kalman predict/update. I must be able to explain every line in a viva.",
            "OpenCV/PyTorch are fine for glue (load image, show result, train a head) unless the task explicitly forbids it.",
        ],
        styles,
    )
    story.append(P("Submission conventions", "h2", styles))
    story += bullets(
        [
            "Individual practical repo name: <font face='Courier'>csc6302-pracN-SINYANI</font> (N = 1…7).",
            "Capstone team repo: <font face='Courier'>csc6302-capstone-SINYANI-PARTNER</font>.",
            "Typical hand-in: code + short PDF report + figure folder + link to Hugging Face Space where required.",
            "Zip naming often follows <font face='Courier'>SINYANI_ComputerNumber_PracticalN.zip</font> — check the LMS post for the exact string.",
            "Commit little and often; markers like to see history, not one giant upload on deadline night.",
        ],
        styles,
    )
    story.append(P("AI use — declare honestly", "h2", styles))
    story.append(
        P(
            "I used Composer to scaffold pipelines, debug OpenCV/PyTorch glue, and draft this "
            "walkthrough. That is allowed only if I declare it and still understand the work. The "
            "intellectual core — problem framing, parameter choices, what the figures prove — must be mine. "
            "Copy-pasting AI text into the marked report without acknowledgement is academic dishonesty.",
            "body",
            styles,
        )
    )
    story.append(PageBreak())
    return story


def section_practical(n: int, styles: dict) -> list:
    """Return story elements for practical n (1–7)."""
    fig_dir = ROOT / f"prac{n}" / "report" / "figures"
    meta = PRACTICALS[n]
    story = [P(f"Practical {n} — {meta['title']}", "h1", styles)]

    story.append(P("Why this practical exists", "h2", styles))
    story.append(P(meta["why"], "body", styles))

    story.append(P("Task-by-task (baby steps)", "h2", styles))
    for task_title, task_body in meta["tasks"]:
        story.append(P(task_title, "h2", styles))
        story.append(P(task_body, "body", styles))

    if meta.get("code_snippet"):
        story.append(P("Core formula / snippet to know", "h2", styles))
        story.append(code_block(meta["code_snippet"], styles))

    story.append(P("How marks usually work", "h2", styles))
    story += bullets(meta["marks"], styles)

    story.append(P("How to run", "h2", styles))
    story.append(code_block(meta["run_cmd"], styles))

    story.append(P("Figures to inspect", "h2", styles))
    story += bullets(meta["figures"], styles)
    story += embed_figures(fig_dir, meta.get("fig_globs", ["*.png"]), styles)
    story.append(PageBreak())
    return story


PRACTICALS = {
    1: {
        "title": "Seeing Like a Machine",
        "why": (
            "Before deep learning magic, we need the boring truth: colour is three number channels, "
            "noise is just wrong numbers, and 'finding edges' means asking where brightness changes fast. "
            "This practical connects Assignment 1 (images as arrays) to the pipelines we will reuse all "
            "semester — threshold, filter, measure."
        ),
        "tasks": [
            (
                "Task 1 — Colour spaces (HSV / LAB)",
                "RGB is how the camera stores bytes; HSV splits hue (what colour) from saturation and "
                "value (how strong / how bright). LAB is closer to human perception. We split channels "
                "so we can pick objects by hue — e.g. mask the red region with <font face='Courier'>cv2.inRange</font> "
                "on H, because red wraps around 0° in hue.",
            ),
            (
                "Task 2 — Noise and PSNR",
                "We corrupt a clean grayscale image with Gaussian noise (add random values) and salt-and-pepper "
                "(random pixels to 0 or 255). Then we try box, Gaussian, and median filters. PSNR tells us "
                "how close the restored image is to the clean original — higher is better. We implement PSNR "
                "ourselves because the brief wants the maths, not <font face='Courier'>cv2.PSNR</font>.",
            ),
            (
                "Task 3 — Sharpening",
                "Sharpen = original + amount × (original − blurred). The blur estimate removes low frequencies; "
                "adding the difference back boosts edges. Too much amount creates ugly halos — that is a feature, "
                "not a bug, for the report discussion.",
            ),
            (
                "Task 4 — Edges (own Sobel + Canny + LoG)",
                "Sobel estimates partial derivatives Gx and Gy with 3×3 kernels, then magnitude √(Gx²+Gy²). "
                "I wrote <font face='Courier'>conv2d</font> with loops so I can point at each multiply-add in a viva. "
                "Canny is the industrial version: blur, gradient, non-max suppression, hysteresis thresholds. "
                "LoG (Laplacian of Gaussian) finds zero-crossings — another edge story.",
            ),
            (
                "Task 5 — Morphology + counting",
                "Threshold → erode/dilate/open/close cleans a binary mask. Connected components counts separate "
                "blobs — our coin counter. The CLI in <font face='Courier'>process.py</font> wraps the same ops "
                "for marker testing.",
            ),
        ],
        "code_snippet": CODE_PSNR + "\n\n" + CODE_SOBEL,
        "marks": [
            "Correct colour-space discussion with figures for each channel.",
            "PSNR table comparing filters — explain why median wins on salt-and-pepper.",
            "Working from-scratch Sobel with comparison to OpenCV reference.",
            "Morphology pipeline with defensible object count.",
            "CLI runs all ops; report explains failures (halos, broken edges).",
        ],
        "run_cmd": "# photos already in images/\npython prac1/run_prac1.py\n# CLI examples:\npython prac1/process.py --input images/colourful.jpg --op pick --output out.png\npython prac1/process.py --input images/coins.jpg --op count --output count_vis.png",
        "figures": [
            "t1_hsv_channels.png, t1_lab_channels.png, t1_picker.png — colour space intuition.",
            "t2_noise.png, t2_restored.png, t2_psnr.csv — denoising evidence.",
            "t3_sharpen.png — halo trade-off at amount=2.0.",
            "t4_sobel.png, t4_canny.png, t4_log.png — edge detectors side by side.",
            "t5_morph.png, t5_count_vis.png — morphology and count.",
        ],
        "fig_globs": ["t1_*.png", "t2_*.png", "t3_*.png", "t4_*.png", "t5_*.png"],
    },
    2: {
        "title": "Interest Points, Matching, Panorama",
        "why": (
            "How does Google Photos stitch your camera roll? It finds the same corners in two images, "
            "matches descriptors, rejects outliers with RANSAC, then warps. This week is that story — "
            "from Harris corners to homography."
        ),
        "tasks": [
            (
                "Task 1 — Corner detectors",
                "Corners are points where gradient changes in all directions (Harris R). Flat regions and "
                "straight edges score badly. We implement Harris from scratch and compare counts to "
                "Shi-Tomasi and FAST on checkerboard vs natural texture.",
            ),
            (
                "Task 2 — Descriptor invariance",
                "ORB (binary, fast) vs SIFT (float, heavier) under rotation and scale. We measure repeatability: "
                "after warping, what fraction of keypoints match within a few pixels? SIFT usually wins on hard "
                "transforms; ORB is the budget option.",
            ),
            (
                "Task 3 — Matching with ratio test",
                "Nearest-neighbour matching alone is noisy. Lowe's ratio test keeps a match only if the best "
                "descriptor is clearly better than the second-best — kills ambiguous matches.",
            ),
            (
                "Task 4 — Panorama / homography",
                "Matched points give a homography H via RANSAC. Warp image A into B's plane, blend. "
                "Inlier count tells you if the stitch is trustworthy.",
            ),
            (
                "Task 5 — Robustness",
                "Blur, darkness, JPEG compression — repeat matching. Good features survive moderate degradation; "
                "bad lighting breaks naive pipelines. Document where yours fails.",
            ),
        ],
        "code_snippet": CODE_HARRIS,
        "marks": [
            "From-scratch Harris with sensible corner overlay figures.",
            "Repeatability table (ORB vs SIFT) with short interpretation.",
            "Ratio-test match visualisation; explain false matches without ratio.",
            "Panorama with RANSAC inlier count reported.",
            "Robustness CSV under at least three degradations.",
        ],
        "run_cmd": "python prac2/run_prac2.py",
        "figures": [
            "t1_checkerboard_corners.png, t1_natural_corners.png — detector comparison.",
            "t3_matches.png — raw vs ratio-filtered matches.",
            "t4_panorama.png, t4_panorama_wide.png — stitch results.",
            "t2_repeatability.csv, t5_robustness.csv — numbers for the report.",
        ],
        "fig_globs": ["t1_*.png", "t3_*.png", "t4_*.png"],
    },
    3: {
        "title": "Object Detection — Classical to YOLO",
        "why": (
            "Detection = draw a box and a label. Classical methods (template, Haar) are fast and interpretable "
            "but brittle. Deep detectors (YOLO) generalise but need GPU and data. This practical is the "
            "before/after story every CV student needs."
        ),
        "tasks": [
            (
                "Task 1 — Template matching",
                "Slide a template across the image; best correlation peak = location. Works when scale and "
                "rotation match; fails when they do not — we show that with a scaled scene.",
            ),
            (
                "Task 2 — Haar cascades",
                "Viola-Jones scans rectangular Haar features at many scales. Parameters "
                "<font face='Courier'>scaleFactor</font> and <font face='Courier'>minNeighbors</font> trade "
                "speed vs false positives. We sweep them and annotate a short video.",
            ),
            (
                "Task 3 — YOLOv8",
                "One forward pass predicts many boxes + classes. Confidence threshold study shows how "
                "low conf floods you with junk detections. Compare timing vs Haar on motion frames.",
            ),
        ],
        "code_snippet": "# Template: cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)\n# Haar: cascade.detectMultiScale(gray, scaleFactor, minNeighbors)\n# YOLO: model(img, conf=0.25)",
        "marks": [
            "Template success + intentional failure case explained.",
            "Haar parameter table and annotated still/video.",
            "YOLO detections at multiple conf values; timing comparison.",
            "Gradio app (<font face='Courier'>prac3/app.py</font>) demoing Haar vs YOLO.",
            "HF Space link if the brief requires hosted demo.",
        ],
        "run_cmd": "pip install ultralytics   # if not already\npython prac3/run_prac3.py\npython prac3/app.py",
        "figures": [
            "t1_template.png — match vs scaled failure.",
            "t2_haar_group.png — face boxes (may be zero on synthetic group).",
            "t3_yolo_group.png, t3_yolo_street.png, t3_yolo_objects.png — YOLO overlays.",
            "t3_conf_study.csv, t3_timing.txt — numbers for discussion.",
        ],
        "fig_globs": ["t1_*.png", "t2_*.png", "t3_*.png"],
    },
    4: {
        "title": "CNNs and Transfer Learning",
        "why": (
            "Hand-crafting features got us here; learning features gets us further. We train a tiny CNN "
            "from scratch, then swap in MobileNetV3 pretrained on ImageNet — freeze backbone, train head, "
            "then lightly fine-tune. Same story as CIFAR-10 in the brief, but we use a small leaf dataset "
            "locally so CPU training finishes in minutes."
        ),
        "tasks": [
            (
                "Task 1 — Scratch TinyCNN",
                "Three conv blocks + pooling learn filters from random init. Expect lower accuracy and longer "
                "training — that is the point of comparison.",
            ),
            (
                "Task 2 — Transfer learning",
                "ImageNet weights already encode edges/textures. We replace the classifier head for our 3 classes "
                "(healthy / rust / blight), train head only, then unfreeze last block for fine-tune.",
            ),
            (
                "Task 3 — Grad-CAM",
                "Which pixels pushed the prediction? Backprop to a conv layer, weight activations by gradients, "
                "overlay heatmap. Sanity-check: model should look at leaf-ish regions, not background.",
            ),
        ],
        "code_snippet": "# Transfer head swap (PyTorch):\n# m.classifier[3] = nn.Linear(in_features, n_classes)\n# train only m.classifier.parameters() first",
        "marks": [
            "Training curves for scratch vs transfer.",
            "Test accuracy + confusion matrix.",
            "Comparison JSON with training time — honest about CPU vs Colab GPU.",
            "Grad-CAM figures for correct and wrong predictions.",
            "Gradio / HF Space classifier app loading saved <font face='Courier'>model.pt</font>.",
        ],
        "run_cmd": "# CPU demo (minutes):\npython prac4/train_prac4.py\npython prac4/app.py\n\n# For real CIFAR-10 marks: re-run notebook on Colab GPU\n# with more epochs — same pipeline, bigger data.",
        "figures": [
            "t1_curves_scratch.png — scratch training loss/acc.",
            "t2_curves_finetune.png — fine-tune phase.",
            "t1_confusion.csv, t2_comparison.json — metrics.",
            "t3_gradcam_correct_*.png, t3_gradcam_miss_*.png — interpretability.",
        ],
        "fig_globs": ["t1_*.png", "t2_*.png", "t3_*.png"],
    },
    5: {
        "title": "Segmentation",
        "why": (
            "Detection draws boxes; segmentation labels every pixel. We climb from thresholding through "
            "watershed and GrabCut to optional DeepLab / Mask R-CNN — classical vs deep, semantic vs instance."
        ),
        "tasks": [
            (
                "Task 1 — Thresholding + IoU",
                "Fixed threshold, Otsu (automatic), adaptive mean/Gaussian. We score against a hand-made "
                "reference mask with IoU — intersection over union, implemented ourselves.",
            ),
            (
                "Task 2 — Watershed",
                "Treat gradients like a landscape; flood from markers to split touching objects. Good for "
                "separated blobs (coins), bad when markers wrong.",
            ),
            (
                "Task 3 — GrabCut",
                "Iterative graph cut with a rectangle (and optional scribble mask). More iterations = cleaner "
                "boundaries around foreground.",
            ),
            (
                "Task 4 — Deep segmentation (optional on CPU)",
                "DeepLabV3 semantic labels every pixel; Mask R-CNN adds instance masks. Slow on CPU — "
                "report timing honestly.",
            ),
        ],
        "code_snippet": CODE_IOU,
        "marks": [
            "IoU table for threshold methods with visual comparison.",
            "Watershed pipeline figure + object count note.",
            "GrabCut iterations vs mask hints.",
            "Deep vs classical timing if Task 4 runs.",
            "Gradio segmentation studio (<font face='Courier'>prac5/app.py</font>).",
        ],
        "run_cmd": "python prac5/run_prac5.py\npython prac5/app.py",
        "figures": [
            "t1_threshold.png, t1_handmade_mask.png, t1_iou.csv.",
            "t2_watershed.png — distance transform story.",
            "t3_grabcut.png — iteration comparison.",
            "t4_deeplab.png, t4_maskrcnn.png, t4_semantic_vs_instance.png.",
        ],
        "fig_globs": ["t1_*.png", "t2_*.png", "t3_*.png", "t4_*.png"],
    },
    6: {
        "title": "Motion, Tracking, Geometry",
        "why": (
            "Video is a stack of images plus time. Optical flow estimates motion; Kalman filters smooth noisy "
            "measurements; calibration undistorts lenses; AR overlays 3D points; stereo estimates depth. "
            "This is where CV meets physics and geometry."
        ),
        "tasks": [
            (
                "Task 1 — Optical flow",
                "Lucas-Kanade tracks sparse corner points frame-to-frame. Farneback gives dense flow coloured "
                "by direction/magnitude. Brightness constancy breaks when lighting jumps — we show that failure.",
            ),
            (
                "Task 2 — Kalman filter tracking",
                "Detect red blob centroid (measurement), predict with constant-velocity model, update when seen, "
                "coast when occluded. Compare to CSRT tracker if available.",
            ),
            (
                "Task 3 — Camera calibration",
                "Multiple chessboard views → intrinsic matrix K and distortion coeffs → undistort. "
                "Report RMS reprojection error in pixels.",
            ),
            (
                "Task 4 — AR overlay",
                "solvePnP gives pose; projectPoints draws a 3D cube on the board plane.",
            ),
            (
                "Task 5 — Stereo depth",
                "Semi-global block matching on a rectified pair; disparity map brightness ≈ nearer objects.",
            ),
        ],
        "code_snippet": CODE_KALMAN,
        "marks": [
            "LK trail figure + Farneback colour flow.",
            "Kalman video/plot showing coast through occlusion.",
            "Calibration RMS + undistort before/after.",
            "AR cube on chessboard.",
            "Stereo disparity with parameter note.",
        ],
        "run_cmd": "python prac6/run_prac6.py",
        "figures": [
            "t1_lk_trails.png, t1_farneback.png, t1_flow_failure.png.",
            "t2_kalman.avi, t2_kf_plot.png — tracking story.",
            "t3_undistort.png, t3_calib.txt.",
            "t4_ar_cube.png.",
            "t5_stereo.png, t5_disp_nd*.png.",
        ],
        "fig_globs": ["t1_*.png", "t2_*.png", "t3_*.png", "t4_*.png", "t5_*.png"],
    },
    7: {
        "title": "Capstone — Road &amp; transport (vehicle counting)",
        "why": (
            "Everything merges: detection from Prac 3, Gradio/Streamlit UI from Prac 3/4/5, tests + CI from "
            "software engineering expectations. My track is <b>Road &amp; transport</b>: count/classify vehicles "
            "in roadside frames and give a rough pothole severity hint — not official traffic stats."
        ),
        "tasks": [
            (
                "Live web app",
                "Streamlit Community Cloud (HF Gradio Spaces need PRO on this account). "
                "URL: <font face='Courier'>https://csc6302-computer-vision-practicalsgit-5wmff88oj3b8thqovqxukj.streamlit.app/</font>",
            ),
            (
                "Vision core",
                "YOLOv8n for car/bus/truck/motorcycle/bicycle counts; classical dark-blob heuristic for pothole "
                "low/medium/high on the lower ROI.",
            ),
            (
                "Quality gates",
                "Blur check rejects shaky photos. Confidence gate + <b>UNSURE</b> when detections are weak.",
            ),
            (
                "Android APK",
                "Allan's Car Counter (<font face='Courier'>prac7/apk/Allans_Car_Counter.apk</font>) — on-device "
                "TFLite. Separate from Blindspot. Install on Android and open the APK.",
            ),
            (
                "Engineering",
                "<font face='Courier'>pytest</font> + GitHub Actions CI; GitHub branch "
                "<font face='Courier'>prac7-road-transport-vehicle-counting</font>.",
            ),
        ],
        "code_snippet": (
            "# Capstone vehicle path:\n"
            "# boxes = YOLO(image, conf=gate)\n"
            "# keep vehicle classes → counts + boxes\n"
            "# if blurry or empty/weak: return 'UNSURE'\n"
            "# pothole: dark blobs on lower ROI → low/medium/high"
        ),
        "marks": [
            "Working Streamlit (or Gradio) upload → counts / pothole hint.",
            "Tests pass in CI; held-out count MAE documented.",
            "Proposal aligns with road-transport scope.",
            "Public Streamlit URL + GitHub repo with commit history.",
            "Optional Android APK for phone demo.",
        ],
        "run_cmd": (
            "cd prac7 && pytest -q && cd ..\n"
            "python prac7/run_prac7.py\n"
            "python prac7/eval_heldout.py\n"
            "python prac7/app.py\n"
            "# or open Streamlit URL / streamlit run prac7/streamlit_app.py"
        ),
        "figures": [
            "prac7/report/figures — vehicle count + pothole hint stills; capstone PDF has ethics page.",
            "APK path: prac7/apk/Allans_Car_Counter.apk (also APK_FOR_PHONE/).",
        ],
        "fig_globs": ["t1_*.png", "t2_*.png"],
    },
}


def section_capstone_proposal(styles: dict) -> list:
    story = [
        P("Capstone proposal summary — Road &amp; transport (vehicle counting)", "h1", styles),
        P("Problem statement", "h2", styles),
        P(
            "Road agencies and students need a phone-first / laptop demo that (1) counts and classifies "
            "vehicles (car, bus, truck, motorcycle, bicycle) in a roadside frame, and (2) gives a rough "
            "pothole severity hint from a dashcam-style photo. Counts are model estimates — not official "
            "traffic statistics. Pothole output is a heuristic, not a certified inspection.",
            "body",
            styles,
        ),
        P("How markers review", "h2", styles),
    ]
    story += bullets(
        [
            "Live Streamlit: <font face='Courier'>https://csc6302-computer-vision-practicalsgit-5wmff88oj3b8thqovqxukj.streamlit.app/</font>",
            "GitHub: <font face='Courier'>https://github.com/Salaam21/csc6302-computer-vision-practicals</font> "
            "(branch <font face='Courier'>prac7-road-transport-vehicle-counting</font>)",
            "Android: install <font face='Courier'>prac7/apk/Allans_Car_Counter.apk</font> "
            "(<font face='Courier'>https://github.com/Salaam21/allans-car-counter</font>)",
            "Report: <font face='Courier'>prac7/report/Practical7_Capstone_Report.pdf</font>",
        ],
        styles,
    )
    story.append(P("Model plan", "h2", styles))
    story += bullets(
        [
            "Web: YOLOv8n (Ultralytics) filtered to five vehicle classes + confidence gate.",
            "Phone: SSD MobileNet v1 TFLite in Allan's Car Counter (not Blindspot).",
            "Pothole: classical dark-blob heuristic on lower ROI.",
            "Held-out ≥30 scenes; metric = count MAE (<font face='Courier'>python prac7/eval_heldout.py</font>).",
        ],
        styles,
    )
    story.append(P("Team", "h2", styles))
    tbl = Table(
        [
            ["Munyeme Sinyani", "Allan's Car Counter, Streamlit/Gradio, tests, CI, eval, report"],
            ["Partner", "TBA (solo pack ready for a second name)"],
        ],
        colWidths=[4 * cm, 11 * cm],
    )
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 0.3 * cm))
    story.append(P("Risks", "h2", styles))
    story.append(
        P(
            "Under-counting buses under-funds a route; over-counting wastes budget. High pothole hints on "
            "shadows waste crew visits. The UI prefers UNSURE / blur reject over inventing totals. "
            "HF Gradio Spaces return 402 PRO — Streamlit is the public host.",
            "body",
            styles,
        )
    )
    story.append(PageBreak())
    return story



def section_checklist(styles: dict) -> list:
    story = [
        P("Marker quick links &amp; hand-in notes", "h1", styles),
        P(
            "For markers: each pracN/README.md has run steps. Capstone live URL and APK paths are above. "
            "Individual reports live under pracN/report/. Hugging Face Gradio Spaces for Prac 3–5/7 are "
            "blocked by PRO (402) — use local Gradio or the Streamlit link for Prac 7.",
            "body",
            styles,
        ),
    ]
    items = [
        "Prac 1–6: run scripts in each README; open PracticalN_Report.pdf + figures.",
        "Prac 3–5: local Gradio (python pracN/app.py); HF Space optional / PRO-blocked.",
        "Prac 6: demo video Prac6_Demo.mp4 (no HF Space required).",
        "Prac 7 Streamlit: https://csc6302-computer-vision-practicalsgit-5wmff88oj3b8thqovqxukj.streamlit.app/",
        "Prac 7 APK: prac7/apk/Allans_Car_Counter.apk — install on Android.",
        "Repos: https://github.com/Salaam21/csc6302-computer-vision-practicals · allans-car-counter · car-counter-sinyani.",
        "AI declaration is in each practical README / report.",
        "From-scratch viva: PSNR, Sobel, Harris R, IoU, Kalman update.",
    ]

    story += bullets(items, styles)
    story.append(Spacer(1, 0.5 * cm))
    story.append(
        P(
            "If a figure folder is empty when you read this PDF, that only means I have not run the generator "
            "yet — not that the practical is optional. Run the script, then rebuild this PDF.",
            "body",
            styles,
        )
    )
    return story


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_pdf():
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="CSC6302 All Practicals Walkthrough",
        author="Munyeme Sinyani",
    )

    story: list = []
    story += section_cover(styles)
    story += section_how_to_use(styles)
    story += section_general_instructions(styles)
    for n in range(1, 8):
        story += section_practical(n, styles)
    story += section_capstone_proposal(styles)
    story += section_checklist(styles)

    doc.build(story)
    print(f"Wrote {OUT_PDF} ({OUT_PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build_pdf()
