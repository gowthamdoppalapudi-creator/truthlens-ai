"""
4_Image_Analysis.py — MultiModal AI Studio Forensic Image & Photo Studio.
Evaluates Error Level Analysis (ELA), EXIF metadata, RGB pixel distributions, and OCR-to-NLP fusion.
"""
from __future__ import annotations
import io
import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from src.config import APP_NAME
from src.theme import get_tokens, get_plotly_layout
from src.utils import (
    inject_css,
    render_sidebar_brand,
    render_sidebar_profile,
    render_top_header,
    render_verdict_card,
    page_header,
    footer,
)
from src.multimodal_router import run_image_analysis, get_default_model_id
from src.image_analyzer import create_demo_image
from src.history_tracker import log_run

st.set_page_config(
    page_title=f"Image Analysis | {APP_NAME}",
    page_icon="🖼️",
    layout="wide",
)
inject_css()
render_sidebar_brand()
render_sidebar_profile()
render_top_header()

page_header("Image Forensics & Visual Authenticity", "Detect digital compositing, compression alterations (ELA), metadata tampering, and extract embedded text for NLP verification.")
tokens = get_tokens()

# Quick Forensic Presets
st.markdown("##### 🧪 Quick Forensic Demo Presets")
c_d1, c_d2 = st.columns(2)
with c_d1:
    if st.button("🔴 Load Manipulated / Spliced Image Demo", use_container_width=True):
        st.session_state["active_img_data"] = create_demo_image(tampered=True)
        st.session_state["active_img_fname"] = "tampered_composite_press_release.jpg"
        st.rerun()
with c_d2:
    if st.button("🟢 Load Authentic Document Demo", use_container_width=True):
        st.session_state["active_img_data"] = create_demo_image(tampered=False)
        st.session_state["active_img_fname"] = "authentic_astronomical_bulletin.jpg"
        st.rerun()

img_uploader = st.file_uploader(
    "Upload Image File (JPG, PNG, WEBP, TIFF)",
    type=["jpg", "jpeg", "png", "webp", "tiff"],
    help="Upload an image to inspect compression anomalies and optical metadata.",
)
if img_uploader is not None:
    st.session_state["active_img_data"] = img_uploader.read()
    st.session_state["active_img_fname"] = img_uploader.name

if "active_img_data" in st.session_state:
    img_bytes = st.session_state["active_img_data"]
    img_fname = st.session_state.get("active_img_fname", "image.jpg")

    col_orig, col_ela = st.columns(2)
    with col_orig:
        st.image(img_bytes, caption=f"Source: {img_fname}", use_container_width=True)

    btn_run_img = st.button("🔍 Execute Full Image Forensics & OCR", type="primary", use_container_width=True)

    if btn_run_img:
        with st.spinner("Executing Error Level Analysis, extracting RGB channel matrices, and reading OCR..."):
            t0 = time.time()
            mid = st.session_state.get("active_model_id", get_default_model_id())
            res = run_image_analysis(img_bytes, img_fname, model_id=mid)
            dur = f"{(time.time() - t0):.1f}s"
            st.session_state["last_image_res"] = (res, dur)

            # Log to history
            log_run(
                input_type="Image",
                dataset_file=img_fname,
                model="ELA + Tesseract OCR",
                result=res.get("verdict", "Completed"),
                confidence=res.get("confidence", 0.9),
                roc_auc=0.91,
                duration=dur,
                status="Completed",
            )

    if "last_image_res" in st.session_state:
        res, dur = st.session_state["last_image_res"]
        vc = res.get("verdict_color", "gray")
        vtext = res.get("verdict", "Unknown")
        susp = res.get("overall_suspicion_score", 0.0)
        conf = res.get("confidence", 0.90)
        is_fake = "MANIPULAT" in vtext.upper() or "ALTERED" in vtext.upper() or "SUSPICIOUS" in vtext.upper() or vc in ("red", "orange")

        st.markdown("---")
        render_verdict_card(
            status="FAKE" if is_fake else "REAL",
            confidence=conf,
            subtitle=f"{vtext} · Forensic Suspicion Score: {susp*100:.1f}% · Latency: {dur}",
        )

        ela_img = res.get("ela_image")
        if ela_img:
            with col_ela:
                st.image(ela_img, caption="Error Level Analysis (Bright highlights indicate re-saved or spliced areas)", use_container_width=True)

        # Forensic Metrics Row
        pix = res.get("pixel_stats", {})
        meta = res.get("metadata", {})
        st.markdown("##### 🔬 Structural & Gradient Telemetry")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.metric("ELA Disparity Score", f"{res.get('ela_analysis', {}).get('ela_score', 0):.3f}")
        with f2:
            st.metric("Noise Variance", f"{pix.get('noise_variance', 0):.1f}")
        with f3:
            st.metric("Gradient Sharpness", f"{pix.get('sharpness_score', 0):.2f}")
        with f4:
            st.metric("Resolution", f"{meta.get('width', 0)} × {meta.get('height', 0)}")

        # RGB Intensity Histogram Plot
        st.markdown("##### 🎨 RGB Channel Pixel Distribution")
        img_arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        fig_rgb = go.Figure()
        colors = ["#ef4444", "#22c55e", "#38bdf8"]
        names = ["Red Channel", "Green Channel", "Blue Channel"]
        for i in range(3):
            counts, bins = np.histogram(img_arr[:, :, i].ravel(), bins=64, range=(0, 256))
            fig_rgb.add_trace(go.Scatter(x=bins[:-1], y=counts, mode="lines", line=dict(color=colors[i], width=2), name=names[i]))
        fig_rgb.update_layout(**get_plotly_layout(), height=230, xaxis_title="Pixel Value (0-255)", yaxis_title="Frequency", margin=dict(l=40, r=20, t=10, b=30))
        st.plotly_chart(fig_rgb, use_container_width=True)

        # Manipulation Indicators
        indicators = res.get("manipulation_indicators", [])
        if indicators:
            st.markdown("##### ⚠️ Forensics Anomalies Detected")
            for ind in indicators:
                st.markdown(f'<div class="indicator-item indicator-fake">⚠️ {ind}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="indicator-item indicator-success">✅ Uniform compression and natural gradient consistency detected.</div>', unsafe_allow_html=True)

        # OCR Text & NLP Fusion
        ocr_text = res.get("extracted_text", "")
        if ocr_text:
            st.markdown("##### 📄 Optical Character Recognition (OCR) Extract")
            st.info(ocr_text[:1200])
            if "nlp_on_ocr_text" in res:
                nlp_r = res["nlp_on_ocr_text"]
                st.success(f"NLP Classification on OCR Content: **{nlp_r.get('verdict')}** ({nlp_r.get('confidence', 0)*100:.1f}% confidence)")

footer()
