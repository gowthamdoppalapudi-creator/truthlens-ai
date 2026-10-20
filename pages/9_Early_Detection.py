"""
9_Early_Detection.py — MultiModal AI Studio Early Fake News Detection Research Module.
Empirical investigation of classifier sensitivity across truncated article stages (10% to 100%).
"""
from __future__ import annotations
import streamlit as st
import pandas as pd

from src.config import APP_NAME, DATA_DIR
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
from src.data_loader import get_session_dataset, set_session_dataset, create_sample_research_dataset
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
    st.markdown(
        f"""
        <div class="enterprise-card" style="margin-bottom: 1.2rem;">
            <div style="font-weight: 700; color: {tokens['primary']}; font-size: 1.05rem;">No Active Dataset in Workspace</div>
            <div style="font-size: 0.85rem; color: {tokens['text_secondary']}; margin: 0.3rem 0 1rem 0;">
                Early detection experiments require a labeled dataset with text content and ground truth labels. Load the built-in curated corpus or upload your own.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        if st.button("📥 Load TruthLens Curated Benchmark Corpus (70 Rows)", type="primary", use_container_width=True):
            corpus_p = DATA_DIR / "truthlens_research_corpus.csv"
            if corpus_p.exists():
                df_curated = pd.read_csv(corpus_p)
            else:
                df_curated = create_sample_research_dataset()
            set_session_dataset(df_curated, name="truthlens_research_corpus.csv", metadata={"text_column": "text", "label_column": "label"})
            st.success("Loaded TruthLens Research Corpus into active workspace!")
            st.rerun()
    with col_l2:
        if st.button("📂 Open Data Ingestion Studio", type="secondary", use_container_width=True):
            st.switch_page("pages/2_Data_Upload.py")
    st.stop()

# ─── Dataset & Column Mapping ────────────────────────────────────────────────
cols = list(df.columns)
text_candidates = ["text", "article", "content", "statement", "body", "news", "headline", "title"]
def_t_idx = next((i for i, c in enumerate(cols) if any(kw in str(c).lower() for kw in text_candidates)), 0)

label_candidates = ["label", "target", "class", "category", "ground_truth"]
def_l_idx = next((i for i, c in enumerate(cols) if any(kw in str(c).lower() for kw in label_candidates)), min(1, len(cols) - 1))

meta = st.session_state.get("active_df_meta", {})
preferred_text = meta.get("text_column")
preferred_label = meta.get("label_column")

if preferred_text and preferred_text in cols:
    def_t_idx = cols.index(preferred_text)
if preferred_label and preferred_label in cols:
    def_l_idx = cols.index(preferred_label)

st.markdown("##### 📁 Active Corpus Configuration")
col_sel1, col_sel2, col_switch = st.columns([2, 2, 1.2])
with col_sel1:
    text_col = st.selectbox("Text Feature Column", options=cols, index=def_t_idx)
with col_sel2:
    label_col = st.selectbox("Target Label Column", options=cols, index=def_l_idx)
with col_switch:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Switch / Upload", use_container_width=True):
        st.switch_page("pages/2_Data_Upload.py")

st.caption(f"Active Dataset: **{st.session_state.get('active_df_name', 'dataset.csv')}** ({len(df):,} rows)")

# ─── Experimental Design Selection ───────────────────────────────────────────
st.markdown("---")
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

        try:
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
        except Exception as exc:
            p_bar.empty()
            st.error(f"❌ Error during {exp_name}: {exc}")
            st.stop()

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
