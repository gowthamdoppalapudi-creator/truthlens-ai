"""
pages/7_Batch_Analyzer.py — High-throughput batch article analysis with CSV exports.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import APP_NAME, PLOTLY_LAYOUT, COLOR_REAL, COLOR_FAKE
from src.utils import inject_css, page_header, stat_card, badge_row, footer
from src.model_registry import list_models
from src.prediction import predict_batch_dataframe, DISCLAIMER_TEXT
from src.data_loader import load_csv, get_session_dataset


st.set_page_config(
    page_title=f"Batch Analyzer | {APP_NAME}",
    page_icon="📦",
    layout="wide",
)
inject_css()

page_header(
    "High-Throughput Batch News Analyzer",
    "Process multi-article CSV files through trained pipelines with confidence scoring and downloadable reports.",
)
badge_row([
    "Batch CSV Ingestion",
    "Preserved Metadata",
    "Confidence Estimation",
    "Exportable Reports",
])

models = list_models()
if not models:
    st.error("No registered models found. Please train a model in the Training Studio.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Data Source Selection
# ──────────────────────────────────────────────────────────────────────────────
tab_active, tab_upload = st.tabs(["📂 Use Active Workspace Dataset", "📤 Upload New Evaluation CSV"])

batch_df = None
df_source_name = ""

with tab_active:
    session_df = get_session_dataset()
    if session_df is not None:
        st.success(f"Active workspace dataset available: `{st.session_state.get('active_df_name')}` ({len(session_df):,} rows).")
        if st.checkbox("Use active workspace dataset for batch inference", value=True):
            batch_df = session_df.copy()
            df_source_name = st.session_state.get("active_df_name", "workspace_data.csv")
    else:
        st.info("No active dataset in workspace. Upload a CSV below or load one in Dataset Manager.")

with tab_upload:
    batch_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"], key="batch_uploader")
    if batch_file is not None:
        try:
            batch_df = load_csv(batch_file)
            df_source_name = batch_file.name
            st.success(f"Loaded `{batch_file.name}` ({len(batch_df):,} rows).")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")

if batch_df is None:
    st.info("Please select or upload a dataset to begin batch analysis.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Model & Column Configuration
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚙️ Batch Execution Settings")

col_cfg1, col_cfg2 = st.columns(2)

with col_cfg1:
    model_options = {m["id"]: f"{m['name']} ({m.get('algorithm', 'Unknown')})" for m in models}
    selected_model_id = st.selectbox(
        "Select Pipeline Model for Batch Scoring",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
    )

with col_cfg2:
    text_column_candidates = batch_df.columns.tolist()
    default_c_idx = 0
    for idx, c in enumerate(text_column_candidates):
        if any(term in c.lower() for term in ["text", "article", "content", "news", "headline"]):
            default_c_idx = idx
            break

    selected_text_col = st.selectbox(
        "Select Column Containing Article Text",
        options=text_column_candidates,
        index=default_c_idx,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Execute Batch Inference
# ──────────────────────────────────────────────────────────────────────────────
if st.button("🚀 Process Batch Predictions", type="primary", use_container_width=True):
    prog_bar = st.progress(0, text="Initializing batch processing pipeline...")

    def update_progress(pct):
        prog_bar.progress(int(pct * 100), text=f"Scoring articles: {int(pct*100)}% complete...")

    with st.spinner("Classifying corpus..."):
        scored_df = predict_batch_dataframe(
            df=batch_df,
            text_col=selected_text_col,
            model_id=selected_model_id,
            progress_callback=update_progress,
        )

    prog_bar.progress(100, text="Batch processing finished!")
    st.session_state["last_scored_df"] = scored_df
    st.success("Batch classification complete!")

# ──────────────────────────────────────────────────────────────────────────────
# Display Batch Results
# ──────────────────────────────────────────────────────────────────────────────
scored_df = st.session_state.get("last_scored_df")
if scored_df is not None:
    st.markdown("---")
    st.markdown("### 📊 Batch Diagnostics & Summary")

    total_count = len(scored_df)
    fake_count = (scored_df["prediction_verdict"] == "Likely Fake").sum()
    real_count = (scored_df["prediction_verdict"] == "Likely Real").sum()
    other_count = total_count - fake_count - real_count
    elapsed = scored_df.attrs.get("batch_time_seconds", 0.0)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        stat_card(f"{total_count:,}", "Total Processed", "stat-accent")
    with s2:
        stat_card(f"{fake_count:,} ({fake_count/max(1, total_count)*100:.1f}%)", "Predicted Fake", "stat-red")
    with s3:
        stat_card(f"{real_count:,} ({real_count/max(1, total_count)*100:.1f}%)", "Predicted Real", "stat-green")
    with s4:
        stat_card(f"{elapsed:.2f}s", "Batch Processing Time", "stat-purple")

    # Visual distribution
    pred_counts = scored_df["prediction_verdict"].value_counts()
    fig_dist = px.bar(
        x=pred_counts.index,
        y=pred_counts.values,
        labels={"x": "Predicted Verdict", "y": "Number of Articles"},
        color=pred_counts.index,
        color_discrete_map={"Likely Real": COLOR_REAL, "Likely Fake": COLOR_FAKE},
        title="Batch Predicted Class Distribution",
    )
    fig_dist.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Searchable & Filterable Results Table
    st.markdown("#### 🔎 Filterable Results Corpus")
    filter_choice = st.radio(
        "Filter by Verdict:",
        options=["All", "Likely Fake", "Likely Real"],
        horizontal=True,
    )
    view_df = scored_df if filter_choice == "All" else scored_df[scored_df["prediction_verdict"] == filter_choice]

    st.dataframe(view_df, use_container_width=True)

    # Download Button
    csv_out = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Complete Scored Results (CSV)",
        data=csv_out,
        file_name=f"scored_predictions_{df_source_name}.csv",
        mime="text/csv",
        type="primary",
    )

    st.warning(f"⚠️ **Scientific Disclaimer:** {DISCLAIMER_TEXT}")

st.markdown("---")
footer()
