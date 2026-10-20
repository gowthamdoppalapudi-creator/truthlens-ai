"""
pages/9_Explainability.py — Model explainability, global feature coefficients, and per-article term attribution.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd

from src.config import APP_NAME
from src.utils import inject_css, page_header, stat_card, badge_row, footer
from src.model_registry import list_models, get_model
from src.explainability import get_global_feature_importance, explain_single_prediction


st.set_page_config(
    page_title=f"Explainability | {APP_NAME}",
    page_icon="💡",
    layout="wide",
)
inject_css()

page_header(
    "Model Explainability & Feature Attribution",
    "Inspect global linear coefficients and examine how specific vocabulary tokens drive individual classification decisions.",
)
badge_row([
    "Global TF-IDF Coefficients",
    "Local Feature Attribution",
    "Transparent Decision Margins",
    "Responsible AI Integrity",
])

models = list_models()
if not models:
    st.error("No registered models found. Please train a model in the Training Studio.")
    st.stop()

# Model Selector
model_options = {m["id"]: f"{m['name']} ({m.get('algorithm', 'Unknown')})" for m in models}
default_id = st.session_state.get("active_model_id", models[0]["id"])
if default_id not in model_options:
    default_id = models[0]["id"]

selected_model_id = st.selectbox(
    "Select Model for Interpretability Diagnostics",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=list(model_options.keys()).index(default_id),
)
st.session_state["active_model_id"] = selected_model_id
model_entry = get_model(selected_model_id)

tab_global, tab_local = st.tabs(["🌐 Global Feature Weights", "🔎 Local Single-Article Attribution"])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Global Feature Weights
# ──────────────────────────────────────────────────────────────────────────────
with tab_global:
    st.markdown("### Global Vocabulary Feature Influences")
    st.caption("Visualizing the highest magnitude linear coefficients associated with REAL versus FAKE classifications.")

    top_n = st.slider("Number of Top Features to Display per Class", min_value=10, max_value=30, value=15)

    with st.spinner("Extracting model coefficients and vocabulary..."):
        feat_res = get_global_feature_importance(selected_model_id, top_n=top_n)

    if feat_res["status"] == "success":
        st.plotly_chart(feat_res["figure"], use_container_width=True)

        col_pos_tbl, col_neg_tbl = st.columns(2)
        with col_pos_tbl:
            st.markdown(f"#### Features Favoring: **{feat_res['pos_label']}**")
            st.dataframe(feat_res["pos_features"].style.format({"weight": "{:+.4f}"}), use_container_width=True)

        with col_neg_tbl:
            st.markdown(f"#### Features Favoring: **{feat_res['neg_label']}**")
            st.dataframe(feat_res["neg_features"].style.format({"weight": "{:+.4f}"}), use_container_width=True)
    elif feat_res["status"] == "unsupported":
        st.warning(feat_res["message"])
    else:
        st.error(feat_res["message"])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Local Single-Article Attribution
# ──────────────────────────────────────────────────────────────────────────────
with tab_local:
    st.markdown("### Per-Article Vocabulary Attribution")
    st.caption("Deconstruct a specific article to see which tokens directly influenced the decision margin.")

    last_text = st.session_state.get("last_analyzed_text", "")
    sample_input = st.text_area(
        "Enter text to explain (auto-filled with your last prediction):",
        value=last_text,
        height=180,
        placeholder="Paste text here to inspect its token contributions...",
    )

    if st.button("🔍 Explain Decision for This Text", type="primary"):
        if not sample_input or len(sample_input.strip()) < 10:
            st.warning("Please enter at least 10 characters to analyze feature matches.")
        else:
            with st.spinner("Calculating token TF-IDF activations and weights..."):
                local_res = explain_single_prediction(sample_input, selected_model_id)

            if local_res["status"] == "success":
                st.success(f"Matched **{local_res['total_active_features']}** active vocabulary terms in model dictionary.")
                st.caption(f"Cleaned token string: `{local_res['cleaned_text']}`")

                df_c = local_res["contributions_df"]
                st.dataframe(
                    df_c.style.format({
                        "tfidf_score": "{:.4f}",
                        "model_weight": "{:+.4f}",
                        "contribution": "{:+.4f}",
                    }),
                    use_container_width=True,
                )
            elif local_res["status"] == "no_features":
                st.info(local_res["message"])
            else:
                st.warning(local_res.get("message", "Unable to compute local feature explanation."))

# ──────────────────────────────────────────────────────────────────────────────
# Scientific Disclaimers
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Scientific Interpretation & Ethics")
st.info(
    "💡 **Key Scientific Distinctions:**\n"
    "1. **Correlation vs. Factuality:** A word like *'reuters'* or *'stated'* having a high positive weight reflects editorial conventions in the training corpus, not absolute truth. Conversely, emotional adjectives like *'shocking'* frequently occur in tabloid clickbait.\n"
    "2. **Context Blindness:** Linear TF-IDF models treat n-grams as additive weights and cannot understand nuance, irony, or negation (e.g., 'not a hoax').\n"
    "3. **Never Cite as Proof:** High feature attribution must never be cited as forensic proof of factual fraud or authenticity."
)

st.markdown("---")
footer()
