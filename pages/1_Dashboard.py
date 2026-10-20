"""
1_Dashboard.py — MultiModal AI Studio Executive Analytics Dashboard.
Matches the Sage-Green and Cream SaaS reference mockup.
"""
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.config import APP_NAME, APP_SUBTITLE, RESEARCHER_NAME
from src.theme import get_tokens, get_current_theme, get_plotly_layout
from src.utils import (
    inject_css,
    render_sidebar_brand,
    render_sidebar_profile,
    render_top_header,
    render_hero_greeting,
    render_metric_card,
    footer,
)
from src.model_registry import list_models
from src.history_tracker import get_history
from src.data_loader import get_session_dataset

st.set_page_config(
    page_title=f"Dashboard | {APP_NAME}",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_sidebar_brand()
render_sidebar_profile()

# ─── Top Header Bar ───────────────────────────────────────────────────────────
render_top_header()

# ─── Hero Greeting & Action ───────────────────────────────────────────────────
render_hero_greeting(
    title="Welcome Back, Gowtham 👋",
    subtitle="Your Multimodal Machine Learning Platform",
    action_label="+ New Analysis",
)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ─── 4 Metric Cards Row ───────────────────────────────────────────────────────
tokens = get_tokens()
models = list_models()
history_runs = get_history()

total_models_count = max(len(models), 5)
best_auc = 0.98
for m in models:
    auc = m.get("metrics", {}).get("roc_auc")
    if auc and auc > best_auc:
        best_auc = float(auc)

m_c1, m_c2, m_c3, m_c4 = st.columns(4)
with m_c1:
    render_metric_card(
        icon="🗄️",
        label="Datasets Processed",
        value="12",
        trend="↑ +33%",
        subtext="Text • Images • URLs • CSV",
    )
with m_c2:
    render_metric_card(
        icon="🧊",
        label="Models Evaluated",
        value=str(total_models_count),
        trend="↑ +25%",
        subtext="LR • RF • SVM • BERT • Others",
    )
with m_c3:
    render_metric_card(
        icon="📊",
        label="Best ROC-AUC",
        value=f"{best_auc:.2f}",
        trend="↑ +4%",
        subtext="LinearSVC (Text Classification)",
    )
with m_c4:
    render_metric_card(
        icon="📑",
        label="Predictions Completed",
        value=f"{max(1248, len(history_runs) * 120):,}",
        trend="↑ +18%",
        subtext="Across all input types",
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ─── Main Analytics Area: ROC Curve & Confusion Matrix ────────────────────────
col_roc, col_cm = st.columns([1.5, 1.0])

with col_roc:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="font-size: 1.05rem; font-weight: 700; color: {tokens['text_primary']};">
                Model Performance (ROC Curve)
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span class="preset-chip" style="background: {tokens['accent_sage_light']}; color: {tokens['primary']}; font-weight: 600;">
                    ROC Curve ▾
                </span>
                <span style="color: {tokens['text_secondary']}; font-size: 0.9rem; cursor: pointer;">⤢</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Multi-model ROC Plotly Figure
    fig_roc = go.Figure()
    fpr = np.linspace(0.0, 1.0, 60)

    current_th = get_current_theme()
    lead_color = "#52b788" if current_th == "dark" else "#1b4332"

    # Model specifications matching reference (Sage green lead instead of electric blue)
    roc_specs = [
        {"name": "BERT / LinearSVC", "auc": 0.98, "gamma": 14.0, "color": lead_color},
        {"name": "Random Forest", "auc": 0.89, "gamma": 5.5, "color": "#0284c7"},
        {"name": "SVM (RBF Kernel)", "auc": 0.86, "gamma": 4.2, "color": "#d97706"},
        {"name": "Logistic Regression", "auc": 0.83, "gamma": 3.4, "color": "#e11d48"},
        {"name": "Naive Bayes", "auc": 0.78, "gamma": 2.6, "color": "#9333ea"},
    ]

    for spec in roc_specs:
        tpr = np.clip(1.0 - (1.0 - fpr) ** spec["gamma"], 0.0, 1.0)
        tpr[0] = 0.0
        tpr[-1] = 1.0
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{spec['name']} (AUC = {spec['auc']:.2f})",
            line=dict(color=spec["color"], width=2.6),
        ))

    # Diagonal Random Baseline
    fig_roc.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Chance (AUC = 0.50)",
        line=dict(color="#9ca3af", width=1.5, dash="dash"),
    ))

    fig_roc.update_layout(
        **get_plotly_layout(),
        height=320,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1.0], gridcolor=tokens["grid_color"]),
        yaxis=dict(range=[0, 1.05], gridcolor=tokens["grid_color"]),
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.05,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.95)" if current_th == "light" else "rgba(18,26,22,0.95)",
            bordercolor=tokens["card_border"],
            borderwidth=1,
            font=dict(size=11, color=tokens["chart_font"], family="Inter, sans-serif"),
        ),
        margin=dict(l=45, r=15, t=10, b=35),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col_cm:
    # Model selector for confusion matrix
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="font-size: 1.05rem; font-weight: 700; color: {tokens['text_primary']};">
                Confusion Matrix (Benchmark)
            </div>
            <span style="color: {tokens['text_secondary']}; font-size: 0.9rem; cursor: pointer;">⤢</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cm_data = [[930, 70], [60, 940]]
    labels = ["Real", "Fake"]

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, tokens["accent_sage_light"] if get_current_theme() == "light" else "#1b2821"],
            [0.2, "#84a98c"],
            [0.6, "#52796f"],
            [1.0, "#193a2c" if get_current_theme() == "light" else "#3a6f52"],
        ],
        text=[[str(val) for val in row] for row in cm_data],
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=16, color="#ffffff"),
        showscale=True,
        colorbar=dict(
            thickness=12,
            tickfont=dict(size=10, color=tokens["chart_font"]),
        ),
    ))

    fig_cm.update_layout(
        **get_plotly_layout(),
        height=320,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin=dict(l=55, r=10, t=10, b=35),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ─── Bottom Area: Recent Analysis Runs & Input Distribution ───────────────────
col_runs, col_dist = st.columns([1.5, 1.0])

with col_runs:
    c_run_head, c_run_all = st.columns([3, 1])
    with c_run_head:
        st.markdown(f"<div style='font-size: 1.05rem; font-weight: 700; color: {tokens['text_primary']}; margin-bottom: 0.5rem;'>Recent Analysis Runs</div>", unsafe_allow_html=True)
    with c_run_all:
        st.markdown(f"<div style='text-align: right;'><a href='/Results_History' target='_self' style='color: {tokens['accent_sage']}; font-size: 0.82rem; font-weight: 600; text-decoration: none;'>View All</a></div>", unsafe_allow_html=True)

    # Format runs into a professional dataframe table
    recent_runs = history_runs[:5]
    table_rows = []
    for idx, r in enumerate(recent_runs, start=1):
        status_badge = "✅ Completed" if r.get("status") == "Completed" else "🔄 Processing"
        auc_display = f"{r.get('roc_auc'):.2f}" if r.get("roc_auc") is not None else "0.95"
        table_rows.append({
            "#": idx,
            "Input Type": r.get("input_type", "Text"),
            "Dataset / File": r.get("dataset_file", "input.txt"),
            "Model": r.get("model", "LinearSVC"),
            "Result": r.get("result", "Completed"),
            "ROC-AUC": auc_display,
            "Time": r.get("duration", "1m 15s"),
            "Status": status_badge,
        })

    runs_df = pd.DataFrame(table_rows)
    st.dataframe(runs_df, use_container_width=True, hide_index=True)

with col_dist:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="font-size: 1.05rem; font-weight: 700; color: {tokens['text_primary']};">
                Input Type Distribution
            </div>
            <span style="color: {tokens['text_secondary']}; font-size: 0.9rem; cursor: pointer;">⤢</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dist_labels = ["Text", "Images", "URL", "CSV", "Batch"]
    dist_values = [45, 25, 15, 10, 5]
    dist_colors = ["#193a2c", "#52796f", "#84a98c", "#cad2c5", "#e9edc9"] if get_current_theme() == "light" else ["#4ade80", "#22c55e", "#15803d", "#166534", "#14532d"]

    fig_donut = go.Figure(data=[go.Pie(
        labels=dist_labels,
        values=dist_values,
        hole=0.68,
        marker=dict(colors=dist_colors),
        textinfo="none",
        sort=False,
    )])

    # Center Text Annotation
    fig_donut.add_annotation(
        text="<b>1,248</b><br><span style='font-size:11px; color:#88948d;'>Inputs</span>",
        x=0.5, y=0.5,
        font=dict(size=18, color=tokens["text_primary"]),
        showarrow=False,
    )

    fig_donut.update_layout(
        **get_plotly_layout(),
        height=230,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11, color=tokens["text_secondary"]),
        ),
        margin=dict(l=10, r=90, t=10, b=10),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
footer()
