"""
utils.py — MultiModal AI Studio Enterprise UI Components & Layout Utilities.
Matches the Sage-Green and Cream SaaS mockup aesthetic.
"""
from __future__ import annotations
import base64
from pathlib import Path
import re
import streamlit as st

from src.config import APP_NAME, APP_SUBTITLE, APP_ICON, RESEARCHER_NAME
from src.theme import inject_theme_css, get_tokens, get_current_theme, toggle_theme, get_plotly_layout

_LOGO_PATH = Path(__file__).parent.parent / "assets" / "truthlens_logo.png"


def get_logo_base64() -> str:
    """Return base64 encoded string of the official TruthLens logo."""
    if _LOGO_PATH.exists():
        try:
            with open(_LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


def inject_css():
    """Inject dynamic enterprise design system stylesheet."""
    inject_theme_css()


def render_sidebar_brand():
    """Render the top logo, brand identity, and bottom profile in the sidebar."""
    tokens = get_tokens()
    logo_b64 = get_logo_base64()

    # Register native st.logo so it is pinned at the top above page navigation
    if _LOGO_PATH.exists():
        try:
            st.logo(str(_LOGO_PATH))
        except Exception:
            pass

    if logo_b64:
        logo_img_tag = f'<img src="data:image/png;base64,{logo_b64}" style="width: 38px; height: 38px; object-fit: contain; border-radius: 8px; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.35));" />'
    else:
        logo_img_tag = f'<div style="width: 38px; height: 38px; border-radius: 9px; background: {tokens["accent_sage"]}; display: flex; align-items: center; justify-content: center; font-size: 1.25rem;">🌿</div>'

    with st.sidebar:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.2rem; padding: 0.2rem 0;">
                {logo_img_tag}
                <div>
                    <div style="font-size: 1.05rem; font-weight: 800; color: {tokens['sidebar_text']}; letter-spacing: -0.01em; line-height: 1.2;">
                        {APP_NAME}
                    </div>
                    <div style="font-size: 0.74rem; color: {tokens['sidebar_muted']}; font-weight: 600;">
                        {APP_SUBTITLE}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f'<div style="height: 1px; background: {tokens["sidebar_border"]}; margin: 0.8rem 0;"></div>', unsafe_allow_html=True)


def render_sidebar_profile():
    """Render the user profile card at the bottom of the sidebar."""
    tokens = get_tokens()
    with st.sidebar:
        st.markdown(f'<div style="height: 1px; background: {tokens["sidebar_border"]}; margin: 1.5rem 0 0.8rem 0;"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between;
                        padding: 0.5rem 0.6rem; border-radius: 8px; background: {tokens['card_bg']}; border: 1px solid {tokens['card_border']};">
                <div style="display: flex; align-items: center; gap: 0.65rem;">
                    <div style="width: 32px; height: 32px; border-radius: 50%; background: {tokens['primary']}; color: #ffffff;
                                display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.78rem;">
                        DG
                    </div>
                    <div>
                        <div style="font-size: 0.82rem; font-weight: 700; color: {tokens['sidebar_text']};">{RESEARCHER_NAME}</div>
                        <div style="font-size: 0.72rem; color: {tokens['sidebar_muted']}; font-weight: 500;">Student & Lead</div>
                    </div>
                </div>
                <div style="color: {tokens['sidebar_muted']}; font-size: 0.85rem;">›</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_top_header():
    """
    Renders top search bar, theme toggle, and notification pills matching the mockup.
    """
    tokens = get_tokens()
    theme_name = get_current_theme()
    logo_b64 = get_logo_base64()

    if logo_b64:
        header_logo_tag = f'<img src="data:image/png;base64,{logo_b64}" style="width: 30px; height: 30px; object-fit: contain; border-radius: 6px;" />'
    else:
        header_logo_tag = f'<span style="font-size: 1.2rem;">🌿</span>'

    c_brand, c_search, c_theme, c_notif, c_user = st.columns([2.5, 4.6, 1.2, 0.5, 0.6])
    with c_brand:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.65rem; height: 38px;">
                {header_logo_tag}
                <div>
                    <div style="font-size: 0.95rem; font-weight: 800; color: {tokens['text_primary']}; line-height: 1.1;">{APP_NAME}</div>
                    <div style="font-size: 0.70rem; color: {tokens['text_secondary']}; font-weight: 500;">{APP_SUBTITLE}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_search:
        st.text_input(
            "Search",
            placeholder="🔍 Search datasets, results, or models...",
            label_visibility="collapsed",
            key="global_search_input",
        )
    with c_theme:
        # Functional Theme Switch
        theme_icon = "🌙 Dark" if theme_name == "light" else "☀️ Light"
        if st.button(theme_icon, key="theme_toggle_btn", help="Switch between Light & Dark themes", use_container_width=True):
            toggle_theme()
            st.rerun()
    with c_notif:
        st.markdown(
            f"""
            <div style="height: 38px; display: flex; align-items: center; justify-content: center;
                        background: {tokens['card_bg']}; border: 1px solid {tokens['card_border']};
                        border-radius: 8px; color: {tokens['text_secondary']}; font-size: 1.1rem; cursor: pointer;">
                🔔
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_user:
        st.markdown(
            f"""
            <div style="height: 38px; display: flex; align-items: center; justify-content: center;
                        background: {tokens['primary']}; border-radius: 8px; color: #ffffff;
                        font-weight: 700; font-size: 0.85rem;">
                DG
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hero_greeting(title: str = "Welcome Back, Gowtham 👋", subtitle: str = "Your Multimodal Machine Learning Platform", action_label: str = "+ New Analysis"):
    """Render hero greeting section with the '+ New Analysis' action button."""
    tokens = get_tokens()
    col_text, col_action = st.columns([4, 1.3])
    with col_text:
        st.markdown(f"<h2 style='margin-bottom: 0.15rem; font-size: 1.75rem;'>{title}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {tokens['text_secondary']}; font-size: 0.95rem; margin-top: 0;'>{subtitle}</p>", unsafe_allow_html=True)
    with col_action:
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        new_analysis = st.button(action_label, type="primary", use_container_width=True, key="hero_new_analysis_btn")
        if new_analysis:
            st.switch_page("pages/2_Data_Upload.py")


def render_metric_card(icon: str, label: str, value: str, trend: str, subtext: str):
    """Render an individual enterprise metric card matching mockup."""
    tokens = get_tokens()
    card_html = f"""
    <div class="metric-card-custom">
        <div>
            <div class="metric-header">
                <div class="metric-icon-box">{icon}</div>
                <div class="metric-label">{label}</div>
            </div>
            <div class="metric-value-row">
                <div class="metric-val">{value}</div>
                <div class="metric-pill-green">{trend}</div>
            </div>
        </div>
        <div class="metric-subtext">{subtext}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    """Standard page title and description."""
    tokens = get_tokens()
    st.markdown(f"<h2 style='margin-bottom: 0.2rem; font-size: 1.7rem;'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='color: {tokens['text_secondary']}; font-size: 0.95rem; margin-top: 0; margin-bottom: 1.2rem;'>{subtitle}</p>", unsafe_allow_html=True)


def stat_card(value: str, label: str, color_class: str = "stat-accent"):
    """Legacy stat card compatibility."""
    tokens = get_tokens()
    st.markdown(
        f"""
        <div class="enterprise-card" style="padding: 1rem 1.2rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: {tokens['primary']};">{value}</div>
            <div style="font-size: 0.8rem; font-weight: 600; color: {tokens['text_secondary']}; text-transform: uppercase; margin-top: 0.2rem;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge_row(badges: list[str]):
    """Horizontal pill badges."""
    tokens = get_tokens()
    chips = "".join(f'<span class="preset-chip" style="background: {tokens["accent_sage_light"]}; color: {tokens["primary"]}; border: 1px solid {tokens["card_border"]};">{b}</span>' for b in badges)
    st.markdown(f'<div style="margin: 0.4rem 0 1rem 0; display: flex; flex-wrap: wrap; gap: 0.4rem;">{chips}</div>', unsafe_allow_html=True)


def footer():
    """Clean enterprise footer."""
    tokens = get_tokens()
    st.markdown(
        f"""
        <div style="border-top: 1px solid {tokens['card_border']}; margin-top: 2.5rem; padding-top: 1rem;
                    text-align: center; font-size: 0.8rem; color: {tokens['text_muted']};">
            <strong>{APP_NAME}</strong> · {APP_SUBTITLE} &nbsp;|&nbsp; Enterprise Machine Learning Platform &nbsp;|&nbsp; Developed by {RESEARCHER_NAME}
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_text_basic(text: str) -> str:
    """Fast basic text normalisation."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_chars: int = 120) -> str:
    return text[:max_chars] + "…" if len(text) > max_chars else text


def plotly_defaults(fig, height: int = 380):
    """Apply active theme's layout to Plotly figures."""
    fig.update_layout(height=height, **get_plotly_layout())


def render_verdict_card(status: str, confidence: float, subtitle: str = ""):
    """
    Renders an unmistakable, prominent 'STATUS: FAKE' or 'STATUS: REAL' verdict banner.
    """
    tokens = get_tokens()
    theme = get_current_theme()
    is_fake = "FAKE" in status.upper() or "ALTERED" in status.upper() or "MANIPULATED" in status.upper() or "SUSPICIOUS" in status.upper()

    if is_fake:
        status_label = "STATUS: FAKE"
        icon = "🚨"
        desc = subtitle or "Manipulated / Deceptive Content Detected"
        if theme == "dark":
            bg = "rgba(239, 68, 68, 0.16)"
            border = "rgba(239, 68, 68, 0.45)"
            text_color = "#fca5a5"
            pill_bg = "#dc2626"
            pill_text = "#ffffff"
        else:
            bg = "#fef2f2"
            border = "#fca5a5"
            text_color = "#991b1b"
            pill_bg = "#dc2626"
            pill_text = "#ffffff"
    else:
        status_label = "STATUS: REAL"
        icon = "🛡️"
        desc = subtitle or "Authentic / Verifiable Content Verified"
        if theme == "dark":
            bg = "rgba(34, 197, 94, 0.16)"
            border = "rgba(34, 197, 94, 0.45)"
            text_color = "#86efac"
            pill_bg = "#16a34a"
            pill_text = "#ffffff"
        else:
            bg = "#f0fdf4"
            border = "#86efac"
            text_color = "#166534"
            pill_bg = "#16a34a"
            pill_text = "#ffffff"

    card_html = f"""
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;
                padding: 1.15rem 1.5rem; background: {bg}; border: 1.5px solid {border};
                border-radius: 12px; margin: 1.25rem 0; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
        <div style="display: flex; align-items: center; gap: 1.1rem;">
            <div style="font-size: 2.2rem; line-height: 1;">{icon}</div>
            <div>
                <div style="display: inline-block; padding: 0.25rem 0.75rem; border-radius: 6px;
                            background: {pill_bg}; color: {pill_text}; font-weight: 800; font-size: 1rem; letter-spacing: 0.04em;">
                    {status_label}
                </div>
                <div style="font-size: 0.95rem; font-weight: 600; color: {text_color}; margin-top: 0.35rem;">
                    {desc}
                </div>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.78rem; font-weight: 700; color: {text_color}; text-transform: uppercase; letter-spacing: 0.04em;">
                Confidence Score
            </div>
            <div style="font-size: 1.85rem; font-weight: 900; color: {text_color}; line-height: 1.1;">
                {confidence*100:.1f}%
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
