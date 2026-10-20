"""
12_Settings.py — MultiModal AI Studio Platform Configuration & System Settings.
Manages theme modes, model defaults, forensic hyperparameters, and hardware telemetry.
"""
from __future__ import annotations
import sys
import platform
import streamlit as st

from src.config import APP_NAME, APP_VERSION, RESEARCHER_NAME
from src.theme import get_tokens, get_current_theme, toggle_theme
from src.utils import (
    inject_css,
    render_sidebar_brand,
    render_sidebar_profile,
    render_top_header,
    page_header,
    footer,
)
from src.model_registry import list_models

st.set_page_config(
    page_title=f"Settings | {APP_NAME}",
    page_icon="⚙️",
    layout="wide",
)
inject_css()
render_sidebar_brand()
render_sidebar_profile()
render_top_header()

page_header("System Settings & Environment Configuration", "Configure theme appearance, default classification pipelines, forensic thresholds, and monitor system resources.")
tokens = get_tokens()

# Theme Settings Card
st.markdown("##### 🎨 Appearance & Display Theme")
c_th1, c_th2 = st.columns([2, 1])
with c_th1:
    st.markdown(
        f"""
        <div class="enterprise-card">
            <div style="font-weight: 700; color: {tokens['primary']}; font-size: 1rem;">Theme Preference</div>
            <div style="font-size: 0.85rem; color: {tokens['text_secondary']}; margin-top: 0.2rem;">
                Currently active: <strong>{get_current_theme().capitalize()} Theme</strong> (Sage-Green & Cream SaaS palette).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c_th2:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button(f"Switch to {'Dark' if get_current_theme() == 'light' else 'Light'} Theme", type="primary", use_container_width=True):
        toggle_theme()
        st.rerun()

st.markdown("---")

# Pipeline Defaults
st.markdown("##### ⚙️ Default Model & Inference Parameters")
models = list_models()
m_opts = {m["id"]: f"{m['name']} ({m.get('algorithm')})" for m in models}
curr_mid = st.session_state.get("active_model_id", list(m_opts.keys())[0] if m_opts else "")

col_s1, col_s2 = st.columns(2)
with col_s1:
    new_mid = st.selectbox("Global Default Model", options=list(m_opts.keys()), format_func=lambda x: m_opts.get(x, x), index=list(m_opts.keys()).index(curr_mid) if curr_mid in m_opts else 0)
    st.session_state["active_model_id"] = new_mid

with col_s2:
    conf_thresh = st.slider("Misinformation Warning Threshold (%)", 50, 95, 75, 5)

st.markdown("---")

# Multimodal Engine Toggles
st.markdown("##### 🔬 Multimodal Forensics Engine Hyperparameters")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    ela_q = st.slider("ELA Re-compression Quality (%)", 70, 98, 90, 2)
with col_f2:
    vid_frames = st.slider("Video Max Frame Slices", 10, 50, 25, 5)
with col_f3:
    ocr_lang = st.selectbox("OCR Language Engine", ["English (eng)", "Multilingual (eng+deu+fra)"])

st.markdown("---")

# System Telemetry
st.markdown("##### 💻 Environment Telemetry & Architecture")
t1, t2, t3, t4 = st.columns(4)
with t1:
    st.metric("Platform Version", APP_VERSION)
with t2:
    st.metric("Python Runtime", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
with t3:
    st.metric("Host OS", platform.system())
with t4:
    st.metric("Active Models Registered", f"{len(models):,}")

footer()
