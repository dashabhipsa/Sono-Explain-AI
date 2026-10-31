import base64
import io
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import sys

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.services.model_service import ModelService

MAX_UPLOAD_MB = 10

def checkpoint_location():
    configured = os.getenv("SONOEXPLAIN_CHECKPOINT")
    if configured:
        return Path(configured)
    try:
        configured = st.secrets.get("SONOEXPLAIN_CHECKPOINT")
        if configured:
            return Path(configured)
    except Exception:
        pass
    return Path(__file__).resolve().parents[2] / "checkpoints" / "E3" / "best_model_e3.pth"


@st.cache_resource(show_spinner="Loading the SonoExplain AI model…")
def load_model_service():
    checkpoint = checkpoint_location()
    return ModelService(str(checkpoint), device="cpu")

st.set_page_config(
    page_title="SonoExplain AI | Ultrasound Research",
    page_icon="assets/sono_icon.png" if os.path.exists("assets/sono_icon.png") else "🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#17243a; --muted:#66758b; --teal:#147d82; --teal-dark:#0d6268; --line:#e4eaf0; --wash:#f5f8fa; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
    .stApp { background:linear-gradient(180deg,#f8fbfc 0%,#f5f7fa 55%,#f8fafc 100%); }
    [data-testid="stHeader"] { background:rgba(248,251,252,.88); }
    .block-container { max-width:1200px; padding-top:1.35rem; padding-bottom:3rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; color:var(--ink); }
    h1 { font-weight:800 !important; }
    .brand-row { display:flex; align-items:center; gap:12px; margin:0 0 1.3rem 0; }
    .brand-mark { width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,#147d82,#35a7a0);display:flex;align-items:center;justify-content:center;color:white;font-size:22px;box-shadow:0 8px 20px #147d822b; }
    .brand-name { font:800 18px 'Manrope',sans-serif; letter-spacing:-.04em; color:#17243a; }
    .brand-sub { color:#7b8798;font-size:11px;letter-spacing:.11em;text-transform:uppercase;margin-top:1px; }
    .hero { background:linear-gradient(115deg,#102d42 0%,#155c67 58%,#168c88 100%); border-radius:24px; padding:34px 38px; color:white; margin:0 0 24px 0; box-shadow:0 18px 45px #173d4a1a; position:relative; overflow:hidden; }
    .hero:after { content:''; position:absolute; width:260px;height:260px;border:1px solid #ffffff22;border-radius:50%;right:-75px;top:-120px;box-shadow:0 0 0 28px #ffffff08,0 0 0 58px #ffffff06; }
    .hero-kicker { color:#b9e7e2;font-size:11px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;margin-bottom:10px; }
    .hero-title { color:white;font:800 clamp(28px,4vw,42px) 'Manrope',sans-serif;letter-spacing:-.05em;line-height:1.12;margin-bottom:10px; }
    .hero-copy { color:#d7ecec;font-size:15px;max-width:690px;line-height:1.65; }
    .hero-pill { display:inline-block;margin-top:18px;padding:7px 11px;border:1px solid #ffffff38;border-radius:30px;color:#e5f7f5;font-size:11px;letter-spacing:.04em; }
    .section-label { color:#147d82;font-size:11px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;margin:4px 0 8px; }
    .section-title { font:750 23px 'Manrope',sans-serif;color:#17243a;letter-spacing:-.04em;margin:0 0 5px; }
    .section-copy { color:#718096;font-size:13px;line-height:1.6;margin-bottom:17px; }
    .panel { background:white;border:1px solid var(--line);border-radius:18px;padding:21px 22px;box-shadow:0 5px 20px #20394b08; }
    .soft-note { background:#edf7f6;border:1px solid #d7eeeb;border-radius:13px;padding:13px 15px;color:#315e61;font-size:12px;line-height:1.55; }
    .warning-note { background:#fff8ec;border:1px solid #f3e3c4;border-radius:13px;padding:13px 15px;color:#765a2d;font-size:12px;line-height:1.6; }
    .result-banner { border-radius:17px;padding:20px 23px;background:#eff7f6;border:1px solid #d9ebe8; }
    .result-label { color:#477b79;font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase; }
    .result-value { color:#153f48;font:800 29px 'Manrope',sans-serif;letter-spacing:-.04em;margin-top:3px; }
    .metric-card { background:#fff;border:1px solid var(--line);border-radius:15px;padding:16px 17px;height:100%; }
    .metric-label { color:#718096;font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase; }
    .metric-value { color:#17243a;font:750 24px 'Manrope',sans-serif;margin-top:5px; }
    .metric-foot { color:#8491a2;font-size:11px;margin-top:3px; }
    div[data-testid="stFileUploader"] { background:#fff;border:1.5px dashed #b8cbd0;border-radius:16px;padding:13px; }
    div[data-testid="stFileUploader"] section { border:0; }
    .stButton>button[kind="primary"] { background:linear-gradient(100deg,#147d82,#1a9690);border:0;border-radius:11px;font-weight:700;min-height:46px;box-shadow:0 7px 16px #147d8225; }
    .stButton>button[kind="primary"]:hover { background:#0d6268;border:0; }
    .stDownloadButton>button { border-radius:10px;border:1px solid #cbd9df;font-weight:600; }
    div[data-testid="stTabs"] button { font-weight:650; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color:#147d82;border-bottom-color:#147d82; }
    .footer { border-top:1px solid #e4eaf0;margin-top:35px;padding-top:17px;color:#8793a2;font-size:11px;line-height:1.7; }
    #MainMenu, footer { visibility:hidden; }
    @media(max-width:700px){ .hero{padding:25px 22px}.block-container{padding-left:1rem;padding-right:1rem}.panel{padding:16px} }
    </style>
    """,
    unsafe_allow_html=True,
)




def decode_png(value):
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGBA")


def metric_card(label, value, foot=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def make_probability_chart(probabilities):
    names = [name.title() for name in probabilities]
    values = [float(value) * 100 for value in probabilities.values()]
    colors = ["#168c88" if name.lower() == "malignant" else "#8eb9bd" for name in probabilities]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=colors,
        text=[f"{v:.2f}%" for v in values], textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        height=220, margin=dict(l=5, r=55, t=10, b=10),
        xaxis=dict(range=[0, 108], title="Model score", ticksuffix="%", showgrid=True, gridcolor="#edf1f4"),
        yaxis=dict(title=None, autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="DM Sans", color="#526176", size=12),
    )
    return fig


# Header
st.markdown(
    '<div class="brand-row"><div class="brand-mark">⌁</div><div>'
    '<div class="brand-name">SonoExplain AI</div>'
    '</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">Interpretable analysis of<br>breast ultrasound images.</div>'
    '<div class="hero-copy">A research prototype combining image classification, lesion segmentation, Grad-CAM visualisation and predictive uncertainty.</div>'
    '</div>',
    unsafe_allow_html=True,
)

# Load the model once per Streamlit process; no separate API server is required.
model_service = None
model_error = None
try:
    model_service = load_model_service()
    service_ok = True
except Exception as exc:
    service_ok = False
    model_error = str(exc)
    st.markdown(
        '<div class="warning-note">● &nbsp; The model could not be initialised. Check the application logs for details.</div>',
        unsafe_allow_html=True,
    )

st.write("")
left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.markdown('<div class="section-label">01 · Image input</div><div class="section-title">Upload an image</div><div class="section-copy">Upload a breast ultrasound image for analysis. Supported formats: PNG, JPG, JPEG, BMP and WEBP. Maximum file size: 10 MB.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Select an ultrasound image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed",
        help="Upload only de-identified images that you are authorised to use.",
    )
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error("The selected image exceeds the 10 MB limit. Choose a smaller file.")
                image = None
        except Exception:
            st.error("The image could not be opened. Check the file and try again.")
            image = None
        if image:
            st.image(image, caption="Input ultrasound image", use_container_width=True)
            st.caption(f"{image.width} × {image.height} pixels")
    else:
        image = None
        st.markdown('<div class="panel"><div style="font-size:30px;margin-bottom:8px">＋</div><b>Image preview</b><div style="font-size:12px;color:#8290a0;margin-top:5px">The selected image will be displayed here.</div></div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">02 · Analysis configuration</div><div class="section-title">Select the analysis outputs</div><div class="section-copy">Select the outputs to generate. Enabling additional analyses may increase processing time.</div>', unsafe_allow_html=True)
    with st.container():
        explain = st.toggle("Grad-CAM visualisation", value=True)
        st.caption("Displays image regions associated with the predicted class.")
        st.divider()
        uncertainty = st.toggle("Monte Carlo dropout uncertainty", value=True)
        st.caption("Estimates prediction variability across repeated stochastic forward passes. It is not a clinical risk score.")
    st.markdown('<div class="soft-note"><b>Data handling</b><br>Use de-identified images only. Do not submit identifiable patient information.</div>', unsafe_allow_html=True)
    run_analysis = st.button("Run analysis", type="primary", use_container_width=True, disabled=(not service_ok or image is None))

if run_analysis and image is not None:
    with st.spinner("Running model inference and generating the selected outputs…"):
        try:
            analysis_result = model_service.analyze(
                image,
                explain=explain,
                estimate_uncertainty=uncertainty,
            )
            st.session_state["sono_result"] = analysis_result
            st.session_state["sono_image"] = image.copy()
            st.session_state["sono_filename"] = uploaded.name
        except Exception as exc:
            st.error("Analysis failed. Review the application logs or try another image.")
            st.exception(exc)

result = st.session_state.get("sono_result")
if result:
    st.write("")
    st.markdown('<div class="section-label">03 · Analysis results</div><div class="section-title">Prediction summary</div>', unsafe_allow_html=True)
    predicted = result.get("predicted_class", "unknown").title()
    confidence = result.get("calibrated_confidence")
    confidence_text = f"{confidence * 100:.1f}%" if confidence is not None else "Not available"
    st.markdown(
        f'<div class="result-banner"><div class="result-label">Predicted class</div>'
        f'<div class="result-value">{predicted}</div>'
        f'<div style="color:#5d777b;font-size:12px;margin-top:5px">Calibrated model confidence: {confidence_text}</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    tab_overview, tab_visuals, tab_reliability = st.tabs(["Prediction summary", "Image-based outputs", "Uncertainty estimates"])
    with tab_overview:
        c1, c2 = st.columns([1.1, 0.9], gap="large")
        with c1:
            st.markdown("**Class-wise prediction scores**")
            st.plotly_chart(make_probability_chart(result.get("class_probabilities", {})), use_container_width=True, config={"displayModeBar": False})
            st.caption("Scores represent the model output distribution; they should not be interpreted as an individual-level probability of disease.")
        with c2:
            st.markdown("**Model output summary**")
            metric_card("Calibrated top-class score", confidence_text, "Temperature-scaled model output")
            st.write("")
            metric_card("Segmentation map size", f'{result.get("segmentation_mask_width", 224)} × {result.get("segmentation_mask_height", 224)}', "Output dimensions in pixels")
            st.write("")
            st.markdown('<div class="warning-note">This model has not been validated for clinical diagnosis, screening or treatment decisions. Ultrasound findings must be assessed by a qualified healthcare professional.</div>', unsafe_allow_html=True)
    with tab_visuals:
        original = st.session_state["sono_image"]
        mask = decode_png(result["segmentation_mask_png_base64"]).convert("L")
        # Model input uses aspect-ratio-preserving padding; reproduce that display geometry.
        display_image = ImageOps.pad(original, mask.size, method=Image.Resampling.BILINEAR, color=(0, 0, 0))
        mask_array = np.asarray(mask)
        alpha = (mask_array > 0).astype(np.uint8) * 105
        tint = Image.new("RGBA", mask.size, (20, 153, 145, 0))
        tint.putalpha(Image.fromarray(alpha))
        mask_overlay = Image.alpha_composite(display_image.convert("RGBA"), tint)
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("**Predicted segmentation mask**")
            st.image(mask, use_container_width=True)
        with col_b:
            st.markdown("**Segmentation overlay**")
            st.image(mask_overlay, use_container_width=True)
        st.caption("Highlighted pixels show the region assigned to the foreground class by the segmentation model.")
        cam_b64 = result.get("gradcam_png_base64")
        if cam_b64:
            cam = decode_png(cam_b64).convert("L").resize(display_image.size)
            heat = Image.new("RGBA", cam.size, (235, 87, 76, 0))
            heat.putalpha(cam.point(lambda p: int(p * 0.55)))
            cam_overlay = Image.alpha_composite(display_image.convert("RGBA"), heat)
            st.markdown("**Grad-CAM class-activation map**")
            st.image(cam_overlay, use_container_width=True)
            st.caption("Grad-CAM indicates image regions contributing to the selected class score.")
        else:
            st.info("Grad-CAM was not generated for this run.")
    with tab_reliability:
        entropy = result.get("predictive_entropy")
        variance = result.get("mean_predictive_variance")
        if entropy is not None:
            a, b = st.columns(2)
            with a:
                metric_card("Predictive entropy", f"{float(entropy):.4f}", "Summarises the spread of the predicted class distribution")
            with b:
                metric_card("Mean predictive variance", f"{float(variance or 0):.6f}", "Variation across repeated stochastic predictions")
            mc = result.get("mc_mean_probabilities", {})
            if mc:
                st.markdown("**Mean class scores across stochastic passes**")
                st.plotly_chart(make_probability_chart(mc), use_container_width=True, config={"displayModeBar": False})
            st.markdown('<div class="soft-note"></div>', unsafe_allow_html=True)
        else:
            st.info("Monte Carlo dropout uncertainty was not requested for this analysis.")

    # Export a user-facing report without internal checkpoint/training metadata.
    report = {
        "application": "SonoExplain AI",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_filename": st.session_state.get("sono_filename", "uploaded_image"),
        "predicted_class": result.get("predicted_class"),
        "class_probabilities": result.get("class_probabilities", {}),
        "calibrated_probabilities": result.get("calibrated_probabilities", {}),
        "calibrated_confidence": result.get("calibrated_confidence"),
        "predictive_entropy": result.get("predictive_entropy"),
        "mean_predictive_variance": result.get("mean_predictive_variance"),
        "segmentation_threshold": result.get("segmentation_threshold"),
        "disclaimer": "Research use only. Not intended for diagnosis or clinical decision-making.",
    }
    st.write("")
    st.markdown("**Export analysis results**")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Export results (JSON)", json.dumps(report, indent=2), "sonoexplain_summary.json", "application/json", use_container_width=True)
    with d2:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            pdf = io.BytesIO()
            page = canvas.Canvas(pdf, pagesize=letter)
            y = 745
            page.setTitle("SonoExplain AI Research Summary")
            page.setFont("Helvetica-Bold", 18)
            page.drawString(48, y, "SonoExplain AI")
            y -= 24
            page.setFont("Helvetica", 10)
            page.drawString(48, y, "Breast ultrasound research analysis summary")
            y -= 30
            page.setFont("Helvetica-Bold", 12)
            page.drawString(48, y, f"Predicted class: {predicted}")
            y -= 24
            page.setFont("Helvetica", 10)
            page.drawString(48, y, f"Calibrated model confidence: {confidence_text}")
            y -= 25
            page.setFont("Helvetica-Bold", 11)
            page.drawString(48, y, "Class-wise prediction scores")
            y -= 18
            page.setFont("Helvetica", 10)
            for name, value in result.get("class_probabilities", {}).items():
                page.drawString(60, y, f"{name.title()}: {float(value) * 100:.3f}%")
                y -= 16
            y -= 12
            page.drawString(48, y, f"Generated (UTC): {report['generated_at_utc']}")
            y -= 28
            page.setFont("Helvetica-Bold", 10)
            page.drawString(48, y, "Important limitation")
            y -= 17
            page.setFont("Helvetica", 9)
            for line in [
                "This is an experimental research tool, not a medical device.",
                "Outputs are not validated for diagnosis or treatment decisions.",
                "Segmentation and Grad-CAM visualizations are model-generated estimates.",
                "Please consult a qualified healthcare professional for medical interpretation.",
            ]:
                page.drawString(48, y, line)
                y -= 14
            page.save()
            with d2:
                st.download_button("Export report (PDF)", pdf.getvalue(), "sonoexplain_summary.pdf", "application/pdf", use_container_width=True)
        except ImportError:
            with d2:
                st.info("PDF export is unavailable. The JSON file can still be downloaded.")

    if st.button("Analyse another image", use_container_width=True):
        for key in ["sono_result", "sono_image", "sono_filename"]:
            st.session_state.pop(key, None)
        st.rerun()

st.markdown(
    '<div class="footer">Use de-identified images only. This research application is not intended for clinical diagnosis or treatment decisions.</div>',
    unsafe_allow_html=True,
)
