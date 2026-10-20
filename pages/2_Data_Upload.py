"""
2_Data_Upload.py — MultiModal AI Studio Enterprise Data Ingestion & Quality Studio.
Supports CSV, TXT, JSON, and structured datasets with automatic schema detection and cleaning.
"""
from __future__ import annotations
import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import APP_NAME, APP_SUBTITLE, DATA_DIR
from src.theme import get_tokens, get_plotly_layout
from src.utils import (
    inject_css,
    render_sidebar_brand,
    render_sidebar_profile,
    render_top_header,
    page_header,
    footer,
)
from src.data_loader import get_session_dataset, set_session_dataset, load_csv, create_sample_research_dataset
from src.data_validation import validate_dataset, clean_dataset

st.set_page_config(
    page_title=f"Data Upload | {APP_NAME}",
    page_icon="📥",
    layout="wide",
)
inject_css()
render_sidebar_brand()
render_sidebar_profile()
render_top_header()

page_header("Data Ingestion & Dataset Studio", "Upload, inspect, validate, and clean tabular and textual datasets for model training and analysis.")
tokens = get_tokens()

tab_upload, tab_builtin, tab_eda = st.tabs(["📤 Upload Dataset", "🔬 Built-in Benchmark Corpora", "📊 Exploratory Data Analysis"])

# ─── TAB 1: UPLOAD DATASET ───────────────────────────────────────────────────
with tab_upload:
    st.markdown("##### Upload Structured or Textual Corpus")
    st.caption("Supported formats: **CSV, TSV, TXT, JSON**. Maximum recommended file size: **200 MB**.")

    uploaded_file = st.file_uploader(
        "Choose File",
        type=["csv", "tsv", "txt", "json"],
        help="Upload standard dataset with text content and optional labels.",
    )

    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name
            file_bytes = uploaded_file.read()
            
            if file_name.endswith(".json"):
                df_raw = pd.read_json(io.BytesIO(file_bytes))
            elif file_name.endswith(".txt"):
                lines = file_bytes.decode("utf-8", errors="replace").splitlines()
                df_raw = pd.DataFrame({"text": [l.strip() for l in lines if l.strip()]})
            else:
                df_raw = load_csv(io.BytesIO(file_bytes))

            st.success(f"Successfully loaded `{file_name}` — **{len(df_raw):,} rows**, **{len(df_raw.columns)} columns**.")

            cols = list(df_raw.columns)
            text_candidates = ["text", "article", "content", "statement", "body", "news", "headline", "title"]
            def_t_idx = next((i for i, c in enumerate(cols) if any(kw in str(c).lower() for kw in text_candidates)), 0)

            label_candidates = ["label", "target", "class", "category", "ground_truth"]
            def_l_idx = next((i + 1 for i, c in enumerate(cols) if any(kw in str(c).lower() for kw in label_candidates)), 0)
            label_opts = ["None (Unlabeled Dataset)"] + cols

            st.markdown("---")
            st.markdown("##### ⚙️ Feature Schema Mapping")
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                sel_text = st.selectbox("Text Feature Column", options=cols, index=def_t_idx)
            with c_m2:
                sel_label_raw = st.selectbox("Target / Label Column", options=label_opts, index=def_l_idx)
                sel_label = None if sel_label_raw == "None (Unlabeled Dataset)" else sel_label_raw

            # Run Data Quality Audit
            if sel_label:
                report = validate_dataset(df_raw, text_col=sel_text, label_col=sel_label)
                st.markdown("##### 🛡️ Automated Data Quality Audit")
                q1, q2, q3, q4 = st.columns(4)
                with q1:
                    st.metric("Total Rows", f"{report.total_rows:,}")
                with q2:
                    st.metric("Duplicate Rows", f"{report.duplicate_rows:,}", delta="- Leaks" if report.duplicate_rows > 0 else "Clean", delta_color="inverse")
                with q3:
                    st.metric("Missing Content", f"{report.missing_text + report.missing_label:,}", delta="Invalid" if (report.missing_text + report.missing_label) > 0 else "Zero", delta_color="inverse")
                with q4:
                    st.metric("Short Texts (<10 chars)", f"{getattr(report, 'short_text', getattr(report, 'short_text_count', 0)):,}")

                col_clean, col_act = st.columns(2)
                with col_clean:
                    if st.button("🧹 Non-Destructive Cleaning & Deduplication", type="secondary", use_container_width=True):
                        cleaned_df = clean_dataset(df_raw, text_col=sel_text, label_col=sel_label)
                        set_session_dataset(cleaned_df, name=file_name, metadata={"text_column": sel_text, "label_column": sel_label})
                        st.success(f"Cleaned dataset: {len(cleaned_df):,} verified rows activated!")
                        st.rerun()
                with col_act:
                    if st.button("💾 Set as Active Workspace Dataset", type="primary", use_container_width=True):
                        set_session_dataset(df_raw, name=file_name, metadata={"text_column": sel_text, "label_column": sel_label})
                        st.success("Dataset activated across all workspace modules!")
            else:
                if st.button("💾 Set as Active Unlabeled Corpus", type="primary", use_container_width=True):
                    set_session_dataset(df_raw, name=file_name, metadata={"text_column": sel_text, "label_column": None})
                    st.success("Unlabeled dataset activated for inference!")

            with st.expander("🔎 Browse Raw Data Preview (First 25 Rows)", expanded=False):
                st.dataframe(df_raw.head(25), use_container_width=True)

        except Exception as exc:
            st.error(f"Error reading file: {exc}")

# ─── TAB 2: BUILT-IN CORPORA ─────────────────────────────────────────────────
with tab_builtin:
    st.markdown("##### Research Benchmark Corpora")
    st.caption("Pre-configured research corpora for rapid diagnostic validation and testing without external file preparation.")

    c_c1, c_c2 = st.columns(2)
    with c_c1:
        st.markdown(
            f"""
            <div class="enterprise-card">
                <div style="font-weight: 700; font-size: 1rem; color: {tokens['primary']};">TruthLens Curated Benchmark Corpus</div>
                <div style="font-size: 0.82rem; color: {tokens['text_secondary']}; margin: 0.3rem 0 0.8rem 0;">
                    70 fully balanced articles spanning global politics, science, health, technology, and viral hoaxes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📥 Load Curated Benchmark Corpus (70 Rows)", type="primary", use_container_width=True):
            corpus_p = DATA_DIR / "truthlens_research_corpus.csv"
            if corpus_p.exists():
                df_curated = pd.read_csv(corpus_p)
            else:
                df_curated = create_sample_research_dataset()
            set_session_dataset(df_curated, name="truthlens_research_corpus.csv", metadata={"text_column": "text", "label_column": "label"})
            st.success("Loaded TruthLens Research Corpus into active workspace!")
            st.rerun()

    with c_c2:
        st.markdown(
            f"""
            <div class="enterprise-card">
                <div style="font-weight: 700; font-size: 1rem; color: {tokens['primary']};">Synthetic Minimal Demonstration Set</div>
                <div style="font-size: 0.82rem; color: {tokens['text_secondary']}; margin: 0.3rem 0 0.8rem 0;">
                    40 curated sample texts designed for rapid unit test validation and fast interactive prototyping.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📥 Load Synthetic Minimal Set (40 Rows)", type="secondary", use_container_width=True):
            df_synth = create_sample_research_dataset()
            set_session_dataset(df_synth, name="synthetic_sample.csv", metadata={"text_column": "text", "label_column": "label"})
            st.success("Loaded Synthetic Minimal Set into active workspace!")
            st.rerun()

# ─── TAB 3: EXPLORATORY DATA ANALYSIS ─────────────────────────────────────────
with tab_eda:
    active_df = get_session_dataset()
    if active_df is None:
        st.info("No active dataset loaded. Please upload a file or load a benchmark corpus to view exploratory analytics.")
    else:
        st.markdown(f"##### Exploratory Analytics: `{st.session_state.get('active_df_name', 'Active Corpus')}` ({len(active_df):,} rows)")
        
        meta = st.session_state.get("active_df_meta", {})
        t_col = meta.get("text_column", active_df.columns[0])
        l_col = meta.get("label_column")

        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            if l_col and l_col in active_df.columns:
                fig_pie = px.pie(
                    active_df,
                    names=l_col,
                    title="Target Class Distribution",
                    hole=0.5,
                    color_discrete_sequence=["#193a2c", "#84a98c", "#e9edc9"],
                )
                fig_pie.update_layout(**get_plotly_layout(), height=280)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Unlabeled corpus: Target class balance unavailable.")

        with col_eda2:
            lengths = active_df[t_col].astype(str).str.len()
            fig_hist = px.histogram(
                lengths,
                nbins=30,
                title="Character Length Distribution",
                color_discrete_sequence=[tokens["accent_sage"]],
            )
            fig_hist.update_layout(**get_plotly_layout(), height=280, xaxis_title="Characters", yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)

footer()
