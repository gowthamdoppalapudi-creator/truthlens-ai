"""
10_Research_Reports.py — MultiModal AI Studio Scientific Research Report Generator.
Export structured, peer-review-ready technical reports documenting methodology, reproducibility seeds, empirical metrics, and limitations.
"""
from __future__ import annotations
import json
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
from src.model_registry import list_models, get_model
from src.report_generator import generate_markdown_report, save_research_report

st.set_page_config(
    page_title=f"Research Reports | {APP_NAME}",
    page_icon="📑",
    layout="wide",
)
inject_css()
render_sidebar_brand()
render_sidebar_profile()
render_top_header()

page_header(
    "Scientific Research Report Generator",
    "Export structured, peer-review-ready technical reports documenting methodology, reproducibility seeds, empirical metrics, and limitations.",
)
tokens = get_tokens()

badge_row([
    "Academic Markdown Deliverables",
    "Machine-Readable JSON Bundles",
    "Complete Provenance",
    "Reproducibility Records",
])

models = list_models()
if not models:
    st.error("No registered models found. Train or evaluate a model in the Model Evaluation Lab to compile a research report.")
    st.stop()

# ─── Experiment / Model Selector ─────────────────────────────────────────────
st.markdown("##### 1. Select Experiment or Model Checkpoint")

model_options = {m["id"]: f"{m['name']} — {m.get('algorithm', 'Model')} (ID: {m['id']})" for m in models}
default_id = st.session_state.get("active_model_id", models[0]["id"])
if default_id not in model_options:
    default_id = models[0]["id"]

selected_id = st.selectbox(
    "Available Experiment Checkpoints",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=list(model_options.keys()).index(default_id),
)
model_entry = get_model(selected_id)

early_data = st.session_state.get("last_early_detection", {})
early_df = None
if early_data:
    first_key = list(early_data.keys())[0]
    early_df = early_data[first_key].get("summary_df")

experiment_package = {
    "experiment_id": model_entry.get("id"),
    "timestamp": model_entry.get("timestamp"),
    "algorithm_name": model_entry.get("algorithm", model_entry.get("name")),
    "metrics": model_entry.get("metrics", {}),
    "tfidf_config": model_entry.get("tfidf_config", {}),
    "dataset_info": model_entry.get("dataset_info", {
        "name": st.session_state.get("active_df_name", "Pre-packaged Corpus"),
        "total_rows": (model_entry.get("n_train") or 0) + (model_entry.get("n_test") or 0),
        "class_counts": model_entry.get("classes", []),
    }),
    "n_train": model_entry.get("n_train"),
    "n_test": model_entry.get("n_test"),
    "text_col": model_entry.get("text_col", "text"),
    "label_col": "label",
    "class_weight": "balanced",
    "random_seed": model_entry.get("random_seed", 42),
    "test_size": model_entry.get("test_size", 0.20),
    "training_time": model_entry.get("metrics", {}).get("training_time", 0.0),
    "prediction_time": model_entry.get("metrics", {}).get("prediction_time", 0.0),
    "early_detection_results": early_df,
}

# ─── Report Compilation & Export ─────────────────────────────────────────────
st.markdown("---")
st.markdown("##### 2. Export Structured Research Deliverables")

report_md = generate_markdown_report(experiment_package)
md_path, json_path = save_research_report(experiment_package)

btn_c1, btn_c2, btn_c3 = st.columns(3)

with btn_c1:
    st.download_button(
        label="📥 Download Technical Report (Markdown)",
        data=report_md,
        file_name=f"TruthLens_Report_{selected_id}.md",
        mime="text/markdown",
        type="primary",
        use_container_width=True,
    )

with btn_c2:
    json_bytes = json.dumps(experiment_package, indent=2, default=str).encode("utf-8")
    st.download_button(
        label="💾 Download Reproducibility Metadata (JSON)",
        data=json_bytes,
        file_name=f"TruthLens_Metadata_{selected_id}.json",
        mime="application/json",
        use_container_width=True,
    )

with btn_c3:
    if "classification_report" in model_entry.get("metrics", {}):
        cr_df = pd.DataFrame(model_entry["metrics"]["classification_report"]).transpose()
        csv_cr = cr_df.to_csv().encode("utf-8")
        st.download_button(
            label="📊 Download Metrics Table (CSV)",
            data=csv_cr,
            file_name=f"TruthLens_Metrics_{selected_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.caption(f"📁 Reports automatically archived to workspace: `{md_path}` and `{json_path}`")

# ─── Interactive Markdown Preview ────────────────────────────────────────────
st.markdown("---")
st.markdown("##### 3. Report Live Preview")

with st.container():
    st.markdown(report_md)

footer()
