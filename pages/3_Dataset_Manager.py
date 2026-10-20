"""
pages/3_Dataset_Manager.py — Dataset upload, interactive schema mapping, rigorous validation, and cleaning.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd

from src.config import APP_NAME
from src.utils import inject_css, page_header, stat_card, badge_row, footer
from src.data_loader import (
    load_csv,
    get_session_dataset,
    set_session_dataset,
    clear_session_dataset,
    create_sample_research_dataset,
)
from src.data_validation import validate_dataset, clean_dataset


st.set_page_config(
    page_title=f"Dataset Manager | {APP_NAME}",
    page_icon="📁",
    layout="wide",
)
inject_css()

page_header(
    "Dataset Studio & CSV Pipeline",
    "Upload raw news corpora, inspect schema integrity, validate class distributions, and prepare clean training splits.",
)
badge_row([
    "Custom CSV Ingestion",
    "Interactive Label Mapping",
    "Data Leakage Detection",
    "Non-Destructive Cleaning",
])

# ──────────────────────────────────────────────────────────────────────────────
# Ingestion Options: File Upload or Sample Generator
# ──────────────────────────────────────────────────────────────────────────────
tab_upload, tab_sample = st.tabs(["📤 Upload Custom CSV", "🔬 Load Built-in Research Corpus"])

uploaded_file = None
with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing news articles",
        type=["csv"],
        help="Upload standard CSV with text and label columns. Common encodings (UTF-8, Latin-1) are automatically handled.",
    )

with tab_sample:
    st.markdown("Need a quick benchmark dataset to test the platform without uploading external files?")
    if st.button("📥 Load Built-in Research Dataset (40 Curated Articles)", type="secondary"):
        sample_df = create_sample_research_dataset()
        set_session_dataset(
            sample_df,
            name="builtin_research_benchmark.csv",
            metadata={"source": "TruthLens Research Synthetic Benchmark", "total_rows": len(sample_df)},
        )
        st.success("Loaded built-in benchmark dataset into active workspace!")
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Processing Uploaded File
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        raw_df = load_csv(uploaded_file)
        st.session_state["raw_upload_df"] = raw_df
        st.session_state["raw_upload_filename"] = uploaded_file.name
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

raw_df = st.session_state.get("raw_upload_df", None)
raw_filename = st.session_state.get("raw_upload_filename", "custom_dataset.csv")

if raw_df is not None:
    st.markdown("---")
    st.markdown(f"### 📋 Dataset Ingestion: `{raw_filename}`")
    st.caption(f"Loaded **{len(raw_df):,}** total rows across **{len(raw_df.columns)}** columns.")

    # Preview raw rows
    with st.expander("🔎 Preview Raw Data (First 5 Rows)", expanded=True):
        st.dataframe(raw_df.head(5), use_container_width=True)

    st.markdown("#### ⚙️ Column & Label Configuration")
    col_cols = raw_df.columns.tolist()

    # Heuristic column defaults
    default_text_idx = 0
    for idx, c in enumerate(col_cols):
        if any(term in c.lower() for term in ["text", "article", "content", "body", "news"]):
            default_text_idx = idx
            break

    default_label_idx = 1 if len(col_cols) > 1 else 0
    for idx, c in enumerate(col_cols):
        if any(term in c.lower() for term in ["label", "target", "class", "verdict"]):
            default_label_idx = idx
            break

    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        text_column = st.selectbox(
            "Select Text Feature Column",
            options=col_cols,
            index=default_text_idx,
            help="Column containing the news article text or headline.",
        )
    with cfg_col2:
        label_column = st.selectbox(
            "Select Ground-Truth Label Column",
            options=col_cols,
            index=default_label_idx,
            help="Column containing class indicators (e.g., 0/1, fake/real, true/false).",
        )

    # Label Mapping
    unique_labels = raw_df[label_column].dropna().unique().tolist()
    st.markdown("##### Map Distinct Values to Binary Classes")
    st.caption(
        "TruthLens models train on binary classifications: **FAKE** and **REAL**. "
        "Assign each unique value found in your label column."
    )

    mapping_cols = st.columns(min(len(unique_labels), 4) if unique_labels else 1)
    label_mapping = {}
    for i, val in enumerate(unique_labels):
        col_target = mapping_cols[i % len(mapping_cols)]
        with col_target:
            str_val = str(val).strip().upper()
            default_choice = 0  # FAKE
            if str_val in ("1", "REAL", "TRUE", "FACT"):
                default_choice = 1  # REAL

            mapped_choice = st.selectbox(
                f"Value: '{val}'",
                options=["FAKE", "REAL", "EXCLUDE"],
                index=default_choice,
                key=f"lbl_map_{val}",
            )
            if mapped_choice != "EXCLUDE":
                label_mapping[val] = mapped_choice

    # ──────────────────────────────────────────────────────────────────────────
    # Validation Execution
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧪 Dataset Quality & Integrity Report")

    report = validate_dataset(
        df=raw_df,
        text_col=text_column,
        label_col=label_column,
        label_mapping=label_mapping,
    )

    v1, v2, v3, v4 = st.columns(4)
    with v1:
        stat_card(f"{report.total_rows:,}", "Total Input Rows", "stat-accent")
    with v2:
        stat_card(f"{report.valid_rows:,}", "Valid Usable Rows", "stat-green")
    with v3:
        stat_card(f"{report.duplicate_rows:,}", "Duplicate Text Records", "stat-red" if report.duplicate_rows > 0 else "stat-green")
    with v4:
        stat_card(f"{report.short_text:,}", "Short / Empty Texts (<10 ch)", "stat-accent")

    # Errors & Warnings
    if report.errors:
        for err in report.errors:
            st.error(f"❌ **Validation Error:** {err}")
    if report.warnings:
        for warn in report.warnings:
            st.warning(f"⚠️ **Integrity Warning:** {warn}")

    # Class balance display
    st.markdown("#### Class Distribution Summary")
    if report.class_counts:
        st.write(pd.DataFrame(list(report.class_counts.items()), columns=["Target Class", "Count"]))
    else:
        st.info("No mapped classes found. Please configure the label mapping above.")

    # ──────────────────────────────────────────────────────────────────────────
    # Cleaning & Workspace Activation
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧹 Non-Destructive Cleaning & Workspace Activation")
    st.caption("Apply cleaning transformations to remove corrupt records before activating the dataset for modeling.")

    c_opt1, c_opt2 = st.columns(2)
    with c_opt1:
        drop_dups = st.checkbox("Deduplicate identical text entries (prevents data leakage)", value=True)
    with c_opt2:
        drop_short = st.checkbox("Drop excessively short text entries (< 10 characters)", value=True)

    if st.button("✨ Clean Dataset & Set as Active Workspace Data", type="primary", use_container_width=True):
        cleaned_df = clean_dataset(
            df=raw_df,
            text_col=text_column,
            label_col=label_column,
            label_mapping=label_mapping,
            drop_duplicates=drop_dups,
            drop_short=drop_short,
        )

        metadata = {
            "name": raw_filename,
            "text_column": text_column,
            "label_column": label_column,
            "label_mapping": label_mapping,
            "total_rows": len(cleaned_df),
            "unique_texts": int(cleaned_df[text_column].nunique()),
            "class_counts": cleaned_df[label_column].value_counts().to_dict(),
            "duplicate_rows": report.duplicate_rows if drop_dups else 0,
        }
        set_session_dataset(cleaned_df, name=raw_filename, metadata=metadata)
        st.success(f"Cleaned dataset activated! {len(cleaned_df):,} verified records ready for modeling.")
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Active Workspace Dataset Management
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🗃️ Current Active Workspace Dataset")
active_df = get_session_dataset()

if active_df is not None:
    act_meta = st.session_state.get("active_df_meta", {})
    st.success(f"**Active Dataset:** `{st.session_state.get('active_df_name', 'dataset.csv')}` with **{len(active_df):,}** validated records.")
    
    with st.expander("👁️ View First 10 Records of Active Dataset", expanded=False):
        st.dataframe(active_df.head(10), use_container_width=True)

    act_btn_col1, act_btn_col2, act_btn_col3 = st.columns(3)
    with act_btn_col1:
        # Download Cleaned CSV
        csv_bytes = active_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Cleaned CSV",
            data=csv_bytes,
            file_name=f"cleaned_{st.session_state.get('active_df_name', 'dataset.csv')}",
            mime="text/csv",
            use_container_width=True,
        )
    with act_btn_col2:
        if st.button("📊 Go to Exploratory Data Analysis", use_container_width=True):
            st.switch_page("pages/4_Data_Exploration.py")
    with act_btn_col3:
        if st.button("🗑️ Reset Active Dataset", use_container_width=True):
            clear_session_dataset()
            st.rerun()
else:
    st.info("No active dataset in workspace. Upload a CSV file above or load the built-in research benchmark.")

st.markdown("---")
footer()
