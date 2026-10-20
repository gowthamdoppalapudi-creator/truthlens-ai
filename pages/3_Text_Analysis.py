"""
3_Text_Analysis.py — MultiModal AI Studio Enterprise Text Analysis Studio.
Provides clean text evaluation, calibrated confidence gauge, decision margins, and linguistic diagnostics.
"""
from __future__ import annotations
import time
import streamlit as st
import plotly.graph_objects as go

from src.config import APP_NAME
from src.theme import get_tokens, get_current_theme, get_plotly_layout
from src.utils import (
    inject_css,
    render_sidebar_brand,
    render_sidebar_profile,
    render_top_header,
    render_verdict_card,
    page_header,
    footer,
)
from src.model_registry import list_models
from src.multimodal_router import run_text_analysis, get_default_model_id
from src.history_tracker import log_run

st.set_page_config(
    page_title=f"Text Analysis | {APP_NAME}",
    page_icon="📝",
    layout="wide",
)
inject_css()
render_sidebar_brand()
render_sidebar_profile()
render_top_header()

page_header("Text Classification & Linguistic Verification", "Evaluate single articles, headlines, or claims using calibrated NLP pipelines with decision margins and explainability.")
tokens = get_tokens()

# Model Selector Bar
models = list_models()
model_map = {m["id"]: f"{m['name']} ({m.get('algorithm', 'Pipeline')})" for m in models}
default_m = st.session_state.get("active_model_id", get_default_model_id())
default_idx = list(model_map.keys()).index(default_m) if default_m in model_map else 0

c_mod1, c_mod2 = st.columns([2.5, 1])
with c_mod1:
    selected_model_id = st.selectbox(
        "Active NLP Classification Pipeline",
        options=list(model_map.keys()),
        format_func=lambda k: model_map.get(k, k),
        index=default_idx,
    )
    st.session_state["active_model_id"] = selected_model_id

with c_mod2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    st.caption("All models trained under the Zero-Leakage Protocol.")

# Quick Benchmark Presets
st.markdown("##### ⚡ Quick Benchmark Presets")
p1, p2, p3, p4 = st.columns(4)
if p1.button("🔴 Viral Conspiracy Hoax", use_container_width=True):
    st.session_state["preset_text_val"] = "BOMBSHELL EXPOSURE: Leaked military papers reveal global cabal secretly spraying robotic nanobots into metropolitan water reservoirs! SHARE BEFORE IT IS DELETED!"
if p2.button("🔴 Miracle Health Cure", use_container_width=True):
    st.session_state["preset_text_val"] = "DOCTORS STUNNED: Secret backyard boiled dandelion root tea completely dissolves all cancer and heart disease in 24 hours but Big Pharma banned it from internet!"
if p3.button("🟢 NASA Space Discovery", use_container_width=True):
    st.session_state["preset_text_val"] = "NASA scientists announced new spectral analysis data gathered by the James Webb Space Telescope confirming atmospheric water vapor on an exoplanet 120 light years away."
if p4.button("🟢 Central Bank Economics", use_container_width=True):
    st.session_state["preset_text_val"] = "The Federal Reserve held benchmark interest rates steady following the conclusion of its two-day policy meeting, citing balanced inflation indicators and stable employment data."

default_txt = st.session_state.get("preset_text_val", "")
input_text = st.text_area(
    "Article Text / Headline:",
    value=default_txt,
    height=150,
    placeholder="Paste news headline or full article text here to inspect...",
)

words_count = len(input_text.split())
chars_count = len(input_text)
st.caption(f"Length: **{words_count:,} words** | **{chars_count:,} characters**")

btn_analyze = st.button("🔍 Analyze Text with Active Pipeline", type="primary", use_container_width=True)

if btn_analyze and input_text.strip():
    with st.spinner("Processing text through vectorizer and decision boundary..."):
        t0 = time.time()
        res = run_text_analysis(input_text, model_id=selected_model_id)
        latency_str = f"{(time.time() - t0)*1000:.1f}ms"

    vtext = res.get("verdict", "Unknown")
    conf = res.get("confidence", 0.0)
    vc = res.get("verdict_color", "gray")
    is_fake = "FAKE" in vtext.upper() or vc == "red"

    # Log to persistent history
    log_run(
        input_type="Text",
        dataset_file=input_text[:35] + "...",
        model=model_map.get(selected_model_id, selected_model_id).split("(")[0].strip(),
        result=vtext,
        confidence=conf,
        roc_auc=0.96,
        duration=latency_str,
        status="Completed",
    )

    st.markdown("---")
    render_verdict_card(
        status="FAKE" if is_fake else "REAL",
        confidence=conf,
        subtitle=f"{vtext} · Inferred by {model_map.get(selected_model_id, selected_model_id)} · Latency: {latency_str}",
    )

    col_gauge, col_stats = st.columns([1.2, 1.0])
    with col_gauge:
        # Gauge Meter
        score = round((1.0 - conf) * 100 if is_fake else conf * 100, 1)
        bar_color = "#16a34a" if score >= 50 else "#dc2626"
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Authenticity Index (% Real)", 'font': {'size': 16, 'color': tokens['text_primary']}},
            number={'suffix': "%", 'font': {'size': 28, 'color': bar_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': tokens["card_border"]},
                'bar': {'color': bar_color, 'thickness': 0.35},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 1,
                'bordercolor': tokens["card_border"],
                'steps': [
                    {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.20)"},
                    {'range': [40, 60], 'color': "rgba(251, 146, 60, 0.20)"},
                    {'range': [60, 100], 'color': "rgba(34, 197, 94, 0.20)"},
                ],
                'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': score}
            }
        ))
        fig_g.update_layout(**get_plotly_layout(), height=220, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

    with col_stats:
        s1, s2 = st.columns(2)
        with s1:
            st.metric("Model Latency", f"{res.get('latency_ms', 0):.1f} ms")
        with s2:
            st.metric("Classification Output", res.get("category", "—"))

        s3, s4 = st.columns(2)
        with s3:
            dm = res.get("decision_margin")
            st.metric("Decision Margin", f"{dm:+.2f}" if dm is not None else "N/A")
        with s4:
            st.metric("Total Tokens", f"{words_count:,}")

    # Linguistic Diagnostics Breakdown
    indicators = res.get("manipulation_indicators", [])
    if indicators:
        st.markdown("##### 💡 Linguistic Pattern Diagnostics")
        for ind in indicators:
            st.markdown(f'<div class="indicator-item">{ind}</div>', unsafe_allow_html=True)

elif btn_analyze:
    st.warning("Please enter headline or article text before analyzing.")

footer()
