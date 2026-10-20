"""
9_Early_Detection.py — MultiModal AI Studio Early Fake News Detection Research Module.
Empirical investigation of classifier sensitivity across truncated article stages (10% to 100%).
"""
from __future__ import annotations
import streamlit as st
import pandas as pd

from src.config import APP_NAME
from src.theme import get_tokens
from src.utils import (
    inject_css,
    render_sidebar_brand,
    render_sidebar_profile,
    render_top_header,
    page_header,
    badge_row,
    footer,
)
from src.data_loader import get_session_dataset
from src.early_detection import run_early_detection_experiment, DEFAULT_STAGES
from src.training import ALGORITHM_REGISTRY

st.set_page_config(
    page_title=f"Early Detection | {APP_NAME}",
    page_icon="⚡",
    layout="wide",
)
inject_css()
render_sidebar_brand()
render_sidebar_profile()
render_top_header()

page_header(
    "Early Fake News Detection Research Engine",
    "Investigate classifier behavior when exposed strictly to early article segments (10% to 100% token availability).",
)
tokens = get_tokens()

badge_row([
    "Truncation Diagnostics",
    "Identical Held-Out Test Set",
    "Stage-Specific vs Distribution Shift",
    "Research-Grade Methodology",
])

df = get_session_dataset()
if df is None:
    st.warning("⚠️ No active dataset found. Please load or upload a dataset in the Data Ingestion Studio first.")
    if st.button("📥 Open Data Ingestion Studio", type="primary"):
        st.switch_page("pages/2_Data_Upload.py")
    st.stop()

meta = st.session_state.get("active_df_meta", {})
text_col = meta.get("text_column", "text" if "text" in df.columns else df.columns[0])
label_col = meta.get("label_column", "label" if "label" in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0]))

st.info(f"📂 Research Dataset: **{st.session_state.get('active_df_name', 'dataset.csv')}** ({len(df):,} articles). Text: `{text_col}`, Label: `{label_col}`.")

# ─── Experimental Design Selection ───────────────────────────────────────────
st.markdown("##### 1. Select Methodological Experimental Design")

exp_c1, exp_c2 = st.columns(2)

with exp_c1:
    st.markdown(
        f"""
        <div class="enterprise-card">
            <h5 style="margin:0; color:{tokens['primary']};">🔬 Experiment A: Stage-Specific Training</h5>
            <p style="color:{tokens['text_secondary']}; font-size:0.85rem; margin-top:0.4rem;">
                A separate classifier is trained on each partial text stage and evaluated strictly
                against held-out test articles truncated to the same percentage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    exp_a_chosen = st.checkbox("Run Experiment A (Stage-Specific Models)", value=True)

with exp_c2:
    st.markdown(
        f"""
        <div class="enterprise-card">
            <h5 style="margin:0; color:{tokens['accent_sage']};">📉 Experiment B: Full-Text Distribution Shift</h5>
            <p style="color:{tokens['text_secondary']}; font-size:0.85rem; margin-top:0.4rem;">
                A single model is trained on 100% full-length articles and evaluated against partially truncated
                held-out articles. Evaluates real-world degradation when a standard model receives partial early text.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    exp_b_chosen = st.checkbox("Run Experiment B (Full-Text Model Under Truncation)", value=False)

st.markdown("---")
st.markdown("##### 2. Experimental Parameters")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    algo_name = st.selectbox("Underlying Classifier", list(ALGORITHM_REGISTRY.keys()), index=0)

with col_p2:
    selected_stages = st.multiselect(
        "Truncation Stages (% of words)",
        options=[5, 10, 20, 25, 40, 50, 75, 90, 100],
        default=DEFAULT_STAGES,
    )
    if not selected_stages:
        selected_stages = DEFAULT_STAGES

with col_p3:
    test_ratio = st.slider("Held-out Test Ratio (%)", 10, 40, 20, step=5) / 100.0

# ─── Execution ───────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🚀 Execute Early Detection Experiment", type="primary", use_container_width=True):
    chosen_types = []
    if exp_a_chosen:
        chosen_types.append("A")
    if exp_b_chosen:
        chosen_types.append("B")

    if not chosen_types:
        st.error("Please select at least one experimental design (Experiment A or B).")
        st.stop()

    results_dict = {}
    for exp_type in chosen_types:
        exp_name = "Experiment A (Stage-Specific)" if exp_type == "A" else "Experiment B (Distribution Shift)"
        p_bar = st.progress(0, text=f"Launching {exp_name}...")

        def cb(pct, msg):
            p_bar.progress(int(pct * 100), text=f"{exp_name}: {msg}")

        exp_result = run_early_detection_experiment(
            df=df,
            text_col=text_col,
            label_col=label_col,
            experiment_type=exp_type,
            stages=selected_stages,
            algorithm_name=algo_name,
            test_size=test_ratio,
            random_seed=42,
            progress_callback=cb,
        )
        p_bar.progress(100, text=f"{exp_name} finished!")
        results_dict[exp_type] = exp_result

    st.session_state["last_early_detection"] = results_dict
    st.success("Early detection experiments successfully executed!")

# ─── Results & Visualizations ────────────────────────────────────────────────
results_dict = st.session_state.get("last_early_detection")
if results_dict:
    st.markdown("---")
    st.markdown("##### 📊 Empirical Truncation Results")

    for exp_type, res in results_dict.items():
        st.markdown(f"###### {'Experiment A: Stage-Specific Training' if exp_type == 'A' else 'Experiment B: Full-Text Model Under Truncation'}")

        st.plotly_chart(res["figure"], use_container_width=True)

        summary_df = res["summary_df"]
        st.dataframe(
            summary_df.style.format({
                "accuracy": "{:.2%}",
                "f1_macro": "{:.2%}",
                "precision_macro": "{:.2%}",
                "recall_macro": "{:.2%}",
                "roc_auc": "{:.4f}",
            }),
            use_container_width=True,
        )

        csv_data = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"💾 Download Results (Experiment {exp_type} CSV)",
            data=csv_data,
            file_name=f"early_detection_experiment_{exp_type}.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("##### 🔬 Methodological Discussion & Limitations")
    with st.expander("Read Key Limitations & Scientific Disclaimers", expanded=True):
        st.markdown(
            """
            - **No Premature Verification:** Early detection experiments evaluate statistical model sensitivity to truncated token lengths. They **do not** prove that an event can be verified before its facts are established or prior to publication.
            - **Distribution Shift in Experiment B:** When a model trained on full 500-word articles receives only 30 words, the sudden sparsity of TF-IDF feature overlap often causes an artificial drop in confidence, which is a known artifact of Bag-of-Words and TF-IDF representations.
            - **Source & Format Leakage:** Articles that open with specific editorial structures (e.g., Reuters bureau datelines like *"WASHINGTON (Reuters) - ..."*) can allow the model to recognize legitimate news from the opening 10 words due to style conventions, not semantic factuality.
            - **Clickbait Volatility:** Misleading articles often concentrate sensational adjectives in headlines and opening sentences, leading to high early lexical detection that may plateau or shift as the article develops.
            """
        )

footer()
