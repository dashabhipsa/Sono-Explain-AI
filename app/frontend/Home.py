import base64
import io
import json
import os
from datetime import datetime, timezone

import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageOps

try:
    API_URL = (
        os.getenv("SONOEXPLAIN_API_URL")
        or st.secrets.get("SONOEXPLAIN_API_URL", "http://localhost:8000")
    ).rstrip("/")
except Exception:
    API_URL = os.getenv("SONOEXPLAIN_API_URL", "http://localhost:8000").rstrip("/")
MAX_UPLOAD_MB = 10

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


def api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=4)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


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
        xaxis=dict(range=[0, 108], title="Model probability", ticksuffix="%", showgrid=True, gridcolor="#edf1f4"),
        yaxis=dict(title=None, autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="DM Sans", color="#526176", size=12),
    )
    return fig


# Header
st.markdown(
    '<div class="brand-row"><div class="brand-mark">⌁</div><div>'
    '<div class="brand-name">SonoExplain AI</div><div class="brand-sub">Breast ultrasound research platform</div>'
    '</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><div class="hero-kicker">AI-assisted image analysis · Research prototype</div>'
    '<div class="hero-title">A clearer view of<br>ultrasound AI.</div>'
    '<div class="hero-copy">Explore model-generated breast ultrasound classification, lesion segmentation, visual explanations and predictive uncertainty — in one workspace.</div>'
    '<span class="hero-pill">MULTISCALE DEEP LEARNING · EXPLAINABILITY · UNCERTAINTY</span></div>',
    unsafe_allow_html=True,
)

# Compact service status; intentionally no checkpoint or training metadata is shown to users.
service_ok = api_health()
if service_ok:
    st.markdown('<div class="soft-note">● &nbsp; Analysis service is connected and ready.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="warning-note">● &nbsp; The analysis service is currently unavailable. Please try again in a moment.</div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.markdown('<div class="section-label">01 · Add an image</div><div class="section-title">Start an analysis</div><div class="section-copy">Upload one breast ultrasound image. Supported formats: PNG, JPG, JPEG, BMP and WEBP. Maximum file size: 10 MB.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose an ultrasound image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed",
        help="Use a de-identified image that you are permitted to share.",
    )
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error("This image is larger than 10 MB. Please choose a smaller image.")
                image = None
        except Exception:
            st.error("We couldn't read this image. Please upload a valid PNG or JPEG.")
            image = None
        if image:
            st.image(image, caption="Uploaded ultrasound image", use_container_width=True)
            st.caption(f"{image.width} × {image.height} pixels")
    else:
        image = None
        st.markdown('<div class="panel"><div style="font-size:30px;margin-bottom:8px">＋</div><b>Your image will appear here</b><div style="font-size:12px;color:#8290a0;margin-top:5px">Choose an image above to preview it before analysis.</div></div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">02 · Choose analysis options</div><div class="section-title">What would you like to see?</div><div class="section-copy">Both options are enabled by default. You can switch either off to shorten analysis time.</div>', unsafe_allow_html=True)
    with st.container():
        explain = st.toggle("Visual explanation (Grad-CAM)", value=True)
        st.caption("Highlights image regions associated with the model's predicted class. It is not a causal explanation.")
        st.divider()
        uncertainty = st.toggle("Predictive uncertainty", value=True)
        st.caption("Uses repeated dropout predictions to estimate model variability. It is not a clinical risk score.")
    st.markdown('<div class="soft-note"><b>Privacy reminder</b><br>Use de-identified images only. This demo is not designed for storing patient records.</div>', unsafe_allow_html=True)
    run_analysis = st.button("Analyze image", type="primary", use_container_width=True, disabled=(not service_ok or image is None))

if run_analysis and image is not None:
    with st.spinner("Analyzing the image… This may take a little while."):
        try:
            response = requests.post(
                f"{API_URL}/api/v1/predict",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                data={"explain": str(explain).lower(), "estimate_uncertainty": str(uncertainty).lower()},
                timeout=300,
            )
            response.raise_for_status()
            st.session_state["sono_result"] = response.json()
            st.session_state["sono_image"] = image.copy()
            st.session_state["sono_filename"] = uploaded.name
        except requests.RequestException as exc:
            detail = "The analysis could not be completed. Please try again."
            if getattr(exc, "response", None) is not None:
                try:
                    detail = exc.response.json().get("detail", detail)
                except Exception:
                    pass
            st.error(detail)

result = st.session_state.get("sono_result")
if result:
    st.write("")
    st.markdown('<div class="section-label">03 · Your results</div><div class="section-title">Analysis overview</div><div class="section-copy">These are outputs from the research model for the image you submitted. They should be interpreted cautiously.</div>', unsafe_allow_html=True)
    predicted = result.get("predicted_class", "unknown").title()
    confidence = result.get("calibrated_confidence")
    confidence_text = f"{confidence * 100:.1f}%" if confidence is not None else "Not available"
    st.markdown(
        f'<div class="result-banner"><div class="result-label">Model-predicted category</div>'
        f'<div class="result-value">{predicted}</div>'
        f'<div style="color:#5d777b;font-size:12px;margin-top:5px">Calibrated model confidence: {confidence_text}</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    tab_overview, tab_visuals, tab_reliability = st.tabs(["Overview", "Visual analysis", "Reliability"])
    with tab_overview:
        c1, c2 = st.columns([1.1, 0.9], gap="large")
        with c1:
            st.markdown("**Class probability distribution**")
            st.plotly_chart(make_probability_chart(result.get("class_probabilities", {})), use_container_width=True, config={"displayModeBar": False})
            st.caption("Probabilities describe the model's output distribution. They are not the probability that a person has a condition.")
        with c2:
            st.markdown("**At a glance**")
            metric_card("Calibrated confidence", confidence_text, "Temperature-scaled output")
            st.write("")
            metric_card("Segmentation output", f'{result.get("segmentation_mask_width", 224)} × {result.get("segmentation_mask_height", 224)}', "Predicted mask dimensions")
            st.write("")
            st.markdown('<div class="warning-note">This research model is not validated for clinical diagnosis, screening or treatment decisions. A qualified healthcare professional must interpret ultrasound images.</div>', unsafe_allow_html=True)
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
            st.markdown("**Predicted lesion mask**")
            st.image(mask, use_container_width=True)
        with col_b:
            st.markdown("**Mask over ultrasound**")
            st.image(mask_overlay, use_container_width=True)
        st.caption("The colored area is a model-predicted foreground mask at the configured threshold. It is not a verified lesion boundary.")
        cam_b64 = result.get("gradcam_png_base64")
        if cam_b64:
            cam = decode_png(cam_b64).convert("L").resize(display_image.size)
            heat = Image.new("RGBA", cam.size, (235, 87, 76, 0))
            heat.putalpha(cam.point(lambda p: int(p * 0.55)))
            cam_overlay = Image.alpha_composite(display_image.convert("RGBA"), heat)
            st.markdown("**Grad-CAM activation overlay**")
            st.image(cam_overlay, use_container_width=True)
            st.caption("Grad-CAM highlights regions associated with the selected class. It does not establish why the model made its prediction.")
        else:
            st.info("Visual explanation was not requested for this analysis.")
    with tab_reliability:
        entropy = result.get("predictive_entropy")
        variance = result.get("mean_predictive_variance")
        if entropy is not None:
            a, b = st.columns(2)
            with a:
                metric_card("Predictive entropy", f"{float(entropy):.4f}", "Lower values indicate a more concentrated class distribution")
            with b:
                metric_card("Mean predictive variance", f"{float(variance or 0):.6f}", "Variation across stochastic predictions")
            mc = result.get("mc_mean_probabilities", {})
            if mc:
                st.markdown("**Average probabilities across repeated predictions**")
                st.plotly_chart(make_probability_chart(mc), use_container_width=True, config={"displayModeBar": False})
            st.markdown('<div class="soft-note">Uncertainty estimates describe variability in this model under the selected inference procedure. They do not establish whether an individual result is safe or correct.</div>', unsafe_allow_html=True)
        else:
            st.info("Predictive uncertainty was not requested for this analysis.")

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
    st.markdown("**Save this analysis**")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download summary (JSON)", json.dumps(report, indent=2), "sonoexplain_summary.json", "application/json", use_container_width=True)
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
            page.drawString(48, y, f"Model-predicted category: {predicted}")
            y -= 24
            page.setFont("Helvetica", 10)
            page.drawString(48, y, f"Calibrated model confidence: {confidence_text}")
            y -= 25
            page.setFont("Helvetica-Bold", 11)
            page.drawString(48, y, "Class probability distribution")
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
                st.download_button("Download PDF summary", pdf.getvalue(), "sonoexplain_summary.pdf", "application/pdf", use_container_width=True)
        except ImportError:
            with d2:
                st.info("PDF export is unavailable. JSON summary is ready.")

    with st.expander("How should I interpret these results?"):
        st.markdown(
            "- **Predicted category:** the class with the highest model score.\n"
            "- **Segmentation:** a model-generated foreground estimate, not a ground-truth contour.\n"
            "- **Grad-CAM:** a coarse visualization of class-associated activations, not a causal explanation.\n"
            "- **Uncertainty:** an estimate of prediction variability, not a clinical risk score.\n\n"
            "This tool has been developed using a small BUSI dataset with an image-level split, no patient identifiers and no external validation cohort."
        )
    if st.button("Start a new analysis", use_container_width=True):
        for key in ["sono_result", "sono_image", "sono_filename"]:
            st.session_state.pop(key, None)
        st.rerun()

st.markdown(
    '<div class="footer">SonoExplain AI · Experimental research prototype<br>'
    'Do not upload identifiable patient information. This application is not intended for clinical diagnosis or treatment decisions.</div>',
    unsafe_allow_html=True,
)
