"""
pages/2_Predict_News.py — Interactive single article and headline classification with confidence semantics.
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go

from src.config import APP_NAME, COLOR_REAL, COLOR_FAKE, PLOTLY_LAYOUT
from src.utils import inject_css, page_header, badge_row, footer
from src.model_registry import list_models, get_model
from src.prediction import predict_single_text, DISCLAIMER_TEXT


st.set_page_config(
    page_title=f"Predict News | {APP_NAME}",
    page_icon="🔍",
    layout="wide",
)
inject_css()

page_header(
    "Real-Time News Classifier",
    "Analyze articles and headlines using calibrated machine learning models.",
)
badge_row([
    "Calibrated Confidence",
    "Style-Pattern Analysis",
    "Leakage-Free Models",
    "Decision Support Only",
])

# ──────────────────────────────────────────────────────────────────────────────
# Model Selector
# ──────────────────────────────────────────────────────────────────────────────
models = list_models()
if not models:
    st.error("No trained models found in registry. Please train a model in the Training Studio.")
    st.stop()

model_options = {m["id"]: f"{m['name']} ({m.get('algorithm', 'Unknown')})" for m in models}
default_model_id = st.session_state.get("active_model_id", models[0]["id"])
if default_model_id not in model_options:
    default_model_id = models[0]["id"]

col_sel, col_info = st.columns([2, 1])
with col_sel:
    selected_model_id = st.selectbox(
        "Select Active Classification Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(default_model_id),
    )
    st.session_state["active_model_id"] = selected_model_id

with col_info:
    m_meta = get_model(selected_model_id)
    if m_meta:
        st.caption(f"**Model ID:** `{selected_model_id}`")
        st.caption(f"**Algorithm:** {m_meta.get('algorithm', 'N/A')}")
        acc = m_meta.get("metrics", {}).get("accuracy") or m_meta.get("metrics", {}).get("test_accuracy")
        if acc:
            st.caption(f"**Held-out Test Accuracy:** {float(acc)*100:.1f}%")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Real Benchmark Examples
# ──────────────────────────────────────────────────────────────────────────────
EXAMPLES = {
    "— Pick an example or enter custom article text —": "",
    "📰 Real: Obama & Guantanamo Protocol (Reuters)": (
        "WASHINGTON (Reuters) - U.S. President Barack Obama is still committed to closing "
        "the U.S. prison for terrorism suspects at Guantanamo Bay, Cuba, and his "
        "administration expects to complete the task, White House said on Thursday. "
        "We are going to do our best to try to get this closed and it is our expectation "
        "that that is what we will do, White House spokesman Josh Earnest said at a news briefing."
    ),
    "📰 Real: Diplomatic Negotiations on Carbon Standards (Geneva)": (
        "GENEVA - International environmental delegates concluded multilateral negotiations "
        "establishing unified technical criteria for national carbon emissions audits. "
        "Representatives from 42 countries signed the procedural framework, which mandates "
        "independent satellite verification beginning in the next fiscal calendar."
    ),
    "🚩 Fake: Sensational Clickbait Conspiracy": (
        "SHOCKING TRUTH REVEALED! Secret global council discovered injecting tracking nanobots "
        "into regular commercial tap water systems! Banned scientists reveal the mind control frequencies "
        "are activated by full moons! Share this viral report everywhere before federal agents delete it!"
    ),
    "🚩 Fake: Fabricated Miraculous Medical Cure": (
        "DOCTORS ARE TERRIFIED! A simple common backyard weed boiling trick completely destroys all "
        "chronic human diseases in less than 24 hours! Big pharmaceutical conglomerates are paying millions "
        "to remove this leaked video from search engines! Read the secret formula here right now!"
    ),
}

col_input, col_sidebar_help = st.columns([2, 1])

with col_input:
    example_key = st.selectbox("Explore Benchmark Examples", list(EXAMPLES.keys()))
    default_text = EXAMPLES[example_key]

    user_text = st.text_area(
        "Article Content or Headline",
        value=default_text,
        height=240,
        placeholder="Paste full article text, body paragraphs, or headline to analyze...",
    )
    analyze_btn = st.button("🚀 Analyze Content", type="primary", use_container_width=True)

with col_sidebar_help:
    st.markdown("#### 🔬 Methodological Guidance")
    st.markdown(
        """
        - **Linguistic Markers:** The model evaluates lexical distribution, sensationalism cues, syntax patterns, and TF-IDF feature alignments.
        - **Calibrated Probability:** Probability scores reflect statistical confidence in stylistic classification, not objective factual truth.
        - **Decision Boundary:** For Linear SVMs, scores represent distance from the separating hyperplane.
        """
    )
    st.info("💡 **Tip:** Test both partial paragraphs and complete articles to study length effects on confidence.")

# ──────────────────────────────────────────────────────────────────────────────
# Inference Execution & Results Display
# ──────────────────────────────────────────────────────────────────────────────
if analyze_btn:
    if not user_text or len(user_text.strip()) < 10:
        st.warning("Please provide at least a sentence or two of text for reliable pattern analysis.")
    else:
        with st.spinner("Executing model pipeline..."):
            res = predict_single_text(user_text, selected_model_id)

        if res["status"] == "error":
            st.error(res["message"])
        else:
            st.write("")
            category = res["category"]
            confidence = res.get("confidence")
            verdict = res["verdict"]

            # Verdict Banner
            if "REAL" in category:
                st.markdown(
                    f"""
                    <div class="verdict-real">
                        <h2 style="margin:0; color:#22c55e;">✅ Prediction: {verdict}</h2>
                        <p style="color:#a7f3d0; margin-top:0.4rem; font-size:1.05rem;">
                            {res['interpretation']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-fake">
                        <h2 style="margin:0; color:#ef4444;">🚩 Prediction: {verdict}</h2>
                        <p style="color:#fca5a5; margin-top:0.4rem; font-size:1.05rem;">
                            {res['interpretation']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            res_c1, res_c2 = st.columns([1, 1])

            with res_c1:
                # Gauge Chart
                conf_pct = (confidence * 100) if confidence is not None else 50.0
                bar_color = COLOR_REAL if "REAL" in category else COLOR_FAKE
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=conf_pct,
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": bar_color},
                            "steps": [
                                {"range": [0, 50], "color": "#1f2430"},
                                {"range": [50, 100], "color": "#262b36"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 2},
                                "thickness": 0.75,
                                "value": conf_pct,
                            },
                        },
                        title={"text": "Statistical Confidence Index"},
                    )
                )
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            with res_c2:
                st.markdown("#### 📋 Diagnostic Attributes")
                diag_data = {
                    "Model Identifier": res["model_id"],
                    "Algorithm Architecture": res["algorithm"],
                    "Inference Latency": f"{res['latency_ms']} ms",
                    "Character Count": f"{res['char_count']:,}",
                    "Word Count": f"{res['word_count']:,}",
                    "Score Output Mode": res["score_type"].title(),
                }
                if res.get("decision_margin") is not None:
                    diag_data["Linear Decision Margin"] = f"{res['decision_margin']:+.3f}"
                st.table(diag_data)

            st.warning(f"⚠️ **Mandatory Scientific Notice:** {res['disclaimer']}")

            # Save text to session state for one-click explainability
            st.session_state["last_analyzed_text"] = user_text
            st.markdown("---")
            col_exp_btn, _ = st.columns([1, 2])
            with col_exp_btn:
                if st.button("🔍 Explain This Prediction in Model Explainability", use_container_width=True):
                    st.switch_page("pages/9_Explainability.py")

st.markdown("---")
footer()
