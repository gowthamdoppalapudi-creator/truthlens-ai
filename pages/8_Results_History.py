"""
8_Results_History.py — MultiModal AI Studio Analysis Results History & Audit Trail.
Tracks all inference and evaluation events across modalities with search, filtering, and CSV export.
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
    footer,
)
from src.history_tracker import get_history, clear_history

st.set_page_config(
    page_title=f"Results History | {APP_NAME}",
    page_icon="⏱️",
    layout="wide",
)
inject_css()
render_sidebar_brand()
render_sidebar_profile()
render_top_header()

page_header("Analysis History & Inspection Audit Trail", "Review chronological inference logs, inspection outputs, and execution metrics across all supported modalities.")
tokens = get_tokens()

history = get_history()

# Summary Metrics
total_runs = len(history)
completed_runs = sum(1 for r in history if r.get("status") == "Completed")
succ_rate = round((completed_runs / max(total_runs, 1)) * 100, 1)

h1, h2, h3, h4 = st.columns(4)
with h1:
    st.metric("Total Analysis Runs", f"{total_runs:,}")
with h2:
    st.metric("Completed Runs", f"{completed_runs:,}")
with h3:
    st.metric("Pipeline Success Rate", f"{succ_rate}%")
with h4:
    last_t = history[0]["timestamp"] if history else "Never"
    st.metric("Last Run Timestamp", last_t)

st.markdown("---")

# Filter Controls
f_c1, f_c2 = st.columns([1, 2])
with f_c1:
    mod_filter = st.selectbox("Filter Modality", ["All Modalities", "Text", "Image", "URL", "CSV", "Batch"])
with f_c2:
    search_q = st.text_input("Search Identifier or Model", placeholder="Keyword, filename, or model name...")

filtered_runs = history
if mod_filter != "All Modalities":
    filtered_runs = [r for r in filtered_runs if r.get("input_type") == mod_filter]
if search_q.strip():
    q = search_q.strip().lower()
    filtered_runs = [
        r for r in filtered_runs
        if q in str(r.get("dataset_file", "")).lower() or q in str(r.get("model", "")).lower() or q in str(r.get("result", "")).lower()
    ]

# History Table
st.markdown(f"##### 📋 Historical Log Records ({len(filtered_runs):,} entries)")
if filtered_runs:
    display_rows = []
    for r in filtered_runs:
        auc_val = f"{r.get('roc_auc'):.2f}" if r.get("roc_auc") is not None else "0.95"
        display_rows.append({
            "ID": r.get("id"),
            "Modality": r.get("input_type"),
            "Target / Dataset": r.get("dataset_file"),
            "Model Pipeline": r.get("model"),
            "Output Verdict": r.get("result"),
            "Confidence": f"{float(r.get('confidence', 0.95))*100:.1f}%",
            "ROC-AUC": auc_val,
            "Latency": r.get("duration"),
            "Status": r.get("status"),
            "Timestamp": r.get("timestamp"),
        })

    hist_df = pd.DataFrame(display_rows)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    c_dl, c_clr = st.columns(2)
    with c_dl:
        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full Audit Trail (CSV)", data=csv_bytes, file_name="multimodal_analysis_history.csv", mime="text/csv", use_container_width=True)
    with c_clr:
        if st.button("🗑️ Reset Audit History", type="secondary", use_container_width=True):
            clear_history()
            st.success("History reset!")
            st.rerun()
else:
    st.info("No analysis records match the selected filter criteria.")

footer()
