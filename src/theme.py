"""
theme.py — MultiModal AI Studio Enterprise Design System.
Defines design tokens, color palettes, and CSS injection for Light & Dark themes.
Palette: Sage-Green & Cream Enterprise SaaS style.
"""
from __future__ import annotations
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Design Tokens
# ──────────────────────────────────────────────────────────────────────────────

THEME_TOKENS = {
    "light": {
        "page_bg": "#f7f5f0",           # Warm off-white / muted cream
        "sidebar_bg": "#ede9e1",        # Crisp light cream sidebar matching Light Mode
        "sidebar_border": "#ded7ca",
        "sidebar_text": "#14261c",      # Crisp dark forest text
        "sidebar_muted": "#556b5f",     # Medium sage
        "sidebar_active_bg": "#d5e6db", # Soft fresh sage active pill
        "sidebar_active_text": "#0d2818",# Deep dark forest text
        "sidebar_hover_bg": "#e2ded5",
        "card_bg": "#ffffff",           # Crisp white surface
        "card_border": "#e5e0d5",       # Subtle warm gray border
        "card_shadow": "0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03)",
        "text_primary": "#1a211d",     # Dark charcoal
        "text_secondary": "#495a51",   # Muted slate/charcoal (high contrast)
        "text_muted": "#6f7e76",       # Medium gray
        "primary": "#193a2c",          # Deep forest green
        "primary_hover": "#224e3b",
        "accent_sage": "#588157",      # Refined sage green
        "accent_sage_light": "#e9f0eb", # Very light sage tint
        "badge_green_bg": "#e6f4ea",
        "badge_green_text": "#137333",
        "badge_red_bg": "#fce8e6",
        "badge_red_text": "#c5221f",
        "chart_font": "#1a211d",
        "grid_color": "#e2ded4",
    },
    "dark": {
        "page_bg": "#0f1412",           # Deep charcoal
        "sidebar_bg": "#08140e",        # Deep dark forest obsidian
        "sidebar_border": "rgba(255, 255, 255, 0.08)",
        "sidebar_text": "#f0f4f1",      # Bright clean white text
        "sidebar_muted": "#8ca395",     # Muted light sage
        "sidebar_active_bg": "rgba(82, 183, 136, 0.25)",
        "sidebar_active_text": "#ffffff",
        "sidebar_hover_bg": "rgba(255, 255, 255, 0.12)",
        "card_bg": "#171f1b",           # Dark graphite surface
        "card_border": "#25312a",       # Subtle dark sage border
        "card_shadow": "0 2px 6px rgba(0, 0, 0, 0.35)",
        "text_primary": "#e8ebe9",     # Warm white
        "text_secondary": "#9ca8a2",   # Muted gray
        "text_muted": "#6b7771",
        "primary": "#3a6f52",          # Forest sage
        "primary_hover": "#4a8a66",
        "accent_sage": "#7fa48d",      # Muted sage
        "accent_sage_light": "#1b2821",
        "badge_green_bg": "rgba(34, 197, 94, 0.15)",
        "badge_green_text": "#4ade80",
        "badge_red_bg": "rgba(239, 68, 68, 0.15)",
        "badge_red_text": "#f87171",
        "chart_font": "#ffffff",
        "grid_color": "#212d26",
    }
}


def get_current_theme() -> str:
    """Return 'light' or 'dark' from session state, defaulting to 'light'."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    return st.session_state["theme"]


def toggle_theme():
    """Toggle between light and dark themes."""
    current = get_current_theme()
    st.session_state["theme"] = "dark" if current == "light" else "light"


def get_tokens() -> dict:
    """Return dictionary of theme colors for the active theme."""
    return THEME_TOKENS[get_current_theme()]


def inject_theme_css():
    """Inject dynamic CSS reflecting the active light or dark theme."""
    tokens = get_tokens()

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Base App Container ── */
    html, body, [data-testid="stAppViewContainer"], .stApp {{
        background-color: {tokens["page_bg"]} !important;
        color: {tokens["text_primary"]} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }}

    /* ── Header Visibility (Ensure Top Nav & Greeting are Fully Visible) ── */
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
        z-index: 10 !important;
    }}

    /* ── Main Content Area Padding ── */
    .block-container {{
        padding-top: 4.6rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px !important;
    }}

    /* ── Sidebar Styling (Matches Light & Dark Theme Dynamically) ── */
    section[data-testid="stSidebar"] {{
        background-color: {tokens["sidebar_bg"]} !important;
        border-right: 1px solid {tokens["sidebar_border"]} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {tokens["sidebar_text"]} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
        padding-top: 0.5rem !important;
    }}
    /* Hide duplicate 'app' navigation link */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul li a[href="/"] {{
        display: none !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a {{
        border-radius: 8px !important;
        padding: 0.55rem 0.9rem !important;
        margin: 3px 0 !important;
        background-color: transparent !important;
        transition: all 0.15s ease-in-out !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a span {{
        color: {tokens["sidebar_text"]} !important;
        font-size: 0.93rem !important;
        font-weight: 550 !important;
        font-family: 'Inter', sans-serif !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a:hover {{
        background-color: {tokens["sidebar_hover_bg"]} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a:hover span {{
        color: {tokens["sidebar_active_text"]} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a[aria-current="page"] {{
        background-color: {tokens["sidebar_active_bg"]} !important;
        border-left: 3.5px solid {tokens["accent_sage"]} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a[aria-current="page"] span {{
        color: {tokens["sidebar_active_text"]} !important;
        font-weight: 700 !important;
    }}

    /* ── Remove Broken Icon Ligature Text (keyboard_double_arrow) ── */
    [data-testid="stSidebarCollapseButton"] button span {{
        display: none !important;
    }}
    [data-testid="stSidebarCollapseButton"] button::after {{
        content: "◀" !important;
        font-size: 13px !important;
        color: {tokens["sidebar_text"]} !important;
    }}
    [data-testid="collapsedControl"] button span {{
        display: none !important;
    }}
    [data-testid="collapsedControl"] button::after {{
        content: "▶" !important;
        font-size: 13px !important;
        color: {tokens["text_primary"]} !important;
    }}

    /* ── Typography & Global Text Legibility ── */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', sans-serif !important;
        color: {tokens["text_primary"]} !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }}
    .main p, .main label, .main td, .main th {{
        font-family: 'Inter', sans-serif !important;
        color: {tokens["text_primary"]};
    }}
    .main span:not([class*="material"]):not([data-testid*="Icon"]) {{
        font-family: 'Inter', sans-serif;
    }}
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {{
        color: {tokens["text_primary"]} !important;
    }}
    div[data-testid="stCaptionContainer"] p, .stCaption {{
        color: {tokens["text_secondary"]} !important;
    }}
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span {{
        color: {tokens["text_primary"]} !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }}

    /* ── Streamlit Native Metrics ── */
    div[data-testid="stMetric"] {{
        background-color: {tokens["card_bg"]} !important;
        border: 1px solid {tokens["card_border"]} !important;
        border-radius: 10px !important;
        padding: 0.85rem 1.1rem !important;
        box-shadow: {tokens["card_shadow"]} !important;
    }}
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span {{
        color: {tokens["text_secondary"]} !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
    }}
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricValue"] span {{
        color: {tokens["text_primary"]} !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
    }}
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] > div {{
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }}

    /* ── Enterprise Cards ── */
    .enterprise-card {{
        background: {tokens["card_bg"]} !important;
        border: 1px solid {tokens["card_border"]} !important;
        border-radius: 10px !important;
        box-shadow: {tokens["card_shadow"]} !important;
        padding: 1.25rem 1.4rem !important;
        margin-bottom: 1rem !important;
        transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
    }}
    .enterprise-card:hover {{
        border-color: {tokens["accent_sage"]} !important;
    }}

    /* ── Custom Metric Card ── */
    .metric-card-custom {{
        background: {tokens["card_bg"]} !important;
        border: 1px solid {tokens["card_border"]} !important;
        border-radius: 10px !important;
        box-shadow: {tokens["card_shadow"]} !important;
        padding: 1.1rem 1.25rem !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }}
    .metric-header {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.4rem;
    }}
    .metric-icon-box {{
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: {tokens["accent_sage_light"]};
        color: {tokens["accent_sage"]};
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }}
    .metric-label {{
        font-size: 0.82rem;
        font-weight: 600;
        color: {tokens["text_secondary"]};
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .metric-value-row {{
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin: 0.3rem 0;
    }}
    .metric-val {{
        font-size: 1.85rem;
        font-weight: 800;
        color: {tokens["text_primary"]};
        line-height: 1.1;
    }}
    .metric-pill-green {{
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.18rem 0.5rem;
        border-radius: 999px;
        background: {tokens["badge_green_bg"]};
        color: {tokens["badge_green_text"]};
    }}
    .metric-subtext {{
        font-size: 0.78rem;
        color: {tokens["text_muted"]};
        margin-top: 0.25rem;
    }}

    /* ── Top Header Search & Profile Bar ── */
    .top-header-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.25rem;
    }}

    /* ── Action Buttons ── */
    button[kind="primary"], .stButton > button[kind="primary"] {{
        background-color: {tokens["primary"]} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.1rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        transition: background-color 0.15s ease !important;
    }}
    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {{
        background-color: {tokens["primary_hover"]} !important;
        color: #ffffff !important;
    }}
    button[kind="secondary"], .stButton > button[kind="secondary"] {{
        background-color: {tokens["card_bg"]} !important;
        color: {tokens["text_primary"]} !important;
        border: 1px solid {tokens["card_border"]} !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }}
    button[kind="secondary"]:hover, .stButton > button[kind="secondary"]:hover {{
        border-color: {tokens["accent_sage"]} !important;
        color: {tokens["accent_sage"]} !important;
    }}

    /* ── Inputs & Selects ── */
    input, textarea, [data-baseweb="select"] {{
        background-color: {tokens["card_bg"]} !important;
        color: {tokens["text_primary"]} !important;
        border-color: {tokens["card_border"]} !important;
        border-radius: 8px !important;
    }}
    [data-baseweb="input"] {{
        background-color: {tokens["card_bg"]} !important;
        border-color: {tokens["card_border"]} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {{
        background-color: {tokens["card_bg"]} !important;
        border: 1px solid {tokens["card_border"]} !important;
    }}
    li[role="option"] {{
        color: {tokens["text_primary"]} !important;
    }}
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {{
        background-color: {tokens["accent_sage_light"]} !important;
        color: {tokens["primary"]} !important;
    }}
    
    /* ── File Uploader (Fix dark box & overlapping text) ── */
    [data-testid="stFileUploader"] {{
        background-color: transparent !important;
    }}
    [data-testid="stFileUploader"] section {{
        background-color: {tokens["card_bg"]} !important;
        border: 1.5px dashed {tokens["card_border"]} !important;
        border-radius: 10px !important;
        padding: 1rem 1.25rem !important;
        color: {tokens["text_primary"]} !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {tokens["card_bg"]} !important;
        border: 1.5px dashed {tokens["card_border"]} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{
        color: {tokens["text_primary"]} !important;
    }}
    [data-testid="stFileUploaderDropzone"] span {{
        color: {tokens["text_secondary"]} !important;
    }}
    [data-testid="stFileUploaderDropzone"] small {{
        color: {tokens["text_muted"]} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {tokens["primary"]} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.45rem 1.2rem !important;
        cursor: pointer !important;
    }}
    [data-testid="stFileUploaderDropzone"] button * {{
        color: #ffffff !important;
    }}
    [data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"],
    [data-testid="stFileUploaderDropzone"] button i,
    [data-testid="stFileUploaderDropzone"] button span:has(span) {{
        display: none !important;
    }}

    /* ── Dataframes & Tables ── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {tokens["card_border"]} !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem !important;
        border-bottom: 2px solid {tokens["card_border"]} !important;
        background-color: transparent !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px 6px 0 0 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: {tokens["text_secondary"]} !important;
        padding: 0.6rem 1rem !important;
        background-color: transparent !important;
        border: none !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {tokens["primary"]} !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {tokens["primary"]} !important;
        border-bottom-color: {tokens["accent_sage"]} !important;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        background-color: {tokens["accent_sage"]} !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {tokens["accent_sage"]} !important;
    }}

    /* ── Expanders ── */
    div[data-testid="stExpander"] {{
        background-color: {tokens["card_bg"]} !important;
        border: 1px solid {tokens["card_border"]} !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stExpander"] summary span {{
        color: {tokens["text_primary"]} !important;
        font-weight: 600 !important;
    }}

    /* ── Status Badges ── */
    .badge-status {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    .badge-completed {{
        background: {tokens["badge_green_bg"]};
        color: {tokens["badge_green_text"]};
    }}
    .badge-processing {{
        background: rgba(56, 189, 248, 0.15);
        color: #0284c7;
    }}
    .badge-failed {{
        background: {tokens["badge_red_bg"]};
        color: {tokens["badge_red_text"]};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def get_plotly_layout(**custom_overrides) -> dict:
    """Return layout parameters for Plotly matching the active theme."""
    tokens = get_tokens()
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=tokens["chart_font"], family="Inter, sans-serif"),
    )
    layout.update(custom_overrides)
    return layout
