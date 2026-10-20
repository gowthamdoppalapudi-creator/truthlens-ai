"""
11_Multimodal_Detection.py — TruthLens AI
Dynamic Multimodal & Dataset Fake-News Detection Platform:
  • 📊 Dataset / CSV       — Dynamic dataset batch scoring, auto column discovery, interactive filters, evaluation & metrics
  • 📝 Text / Article      — Instant NLP classifier, probability margins, linguistic indicators, quick benchmark presets
  • 🖼️ Image / Photo       — Error Level Analysis (ELA), EXIF metadata, RGB pixel distributions, OCR → NLP fusion
  • 🎬 Video               — Frame extraction, temporal divergence, frame ELA, speech transcript → NLP
  • 🌐 Web URL             — BeautifulSoup web article scraping, domain credibility check & multi-field verification
  • 📦 Multi-File Batch    — Ingest and analyze multiple mixed modalities simultaneously with portfolio risk metrics
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import APP_NAME, APP_ICON, COLOR_FAKE, COLOR_REAL, COLOR_ACCENT, PLOTLY_LAYOUT, DATA_DIR
from src.model_registry import list_models
from src.multimodal_router import (
    detect_modality,
    run_text_analysis,
    run_image_analysis,
    run_video_analysis,
    run_csv_analysis,
    run_url_analysis,
    batch_route,
    get_default_model_id,
)
from src.image_analyzer import create_demo_image
from src.video_analyzer import create_demo_video

st.set_page_config(
    page_title=f"Multimodal & Dataset Detection — {APP_NAME}",
    page_icon="🛡️",
    layout="wide",
)

# ─── Custom Styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
.verdict-banner {
    padding: 1.1rem 1.5rem;
    border-radius: 12px;
    font-size: 1.25rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.verdict-green  { background: rgba(34,197,94,.18);  border: 2px solid #22c55e; color:#22c55e; }
.verdict-red    { background: rgba(239,68,68,.18);   border: 2px solid #ef4444; color:#ef4444; }
.verdict-orange { background: rgba(251,146,60,.18);  border: 2px solid #fb923c; color:#fb923c; }
.verdict-blue   { background: rgba(96,165,250,.18);  border: 2px solid #60a5fa; color:#60a5fa; }
.verdict-gray   { background: rgba(156,163,175,.18); border: 2px solid #9ca3af; color:#9ca3af; }

.indicator-item {
    padding: .5rem .9rem;
    border-left: 4px solid #38bdf8;
    background: rgba(56,189,248,.08);
    border-radius: 0 8px 8px 0;
    margin: .4rem 0;
    font-size: .9rem;
}
.indicator-fake {
    border-left-color: #ef4444;
    background: rgba(239,68,68,.08);
}
.indicator-success {
    border-left-color: #22c55e;
    background: rgba(34,197,94,.08);
}

.stat-card {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
.stat-value { font-size: 1.7rem; font-weight: 700; }
.stat-label { font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .75; margin-top: .2rem; }

.preset-chip {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    margin: 0.2rem;
    border-radius: 6px;
    background: rgba(56,189,248,.12);
    border: 1px solid rgba(56,189,248,.3);
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Helper Visualization Functions ──────────────────────────────────────────
def plot_authenticity_gauge(confidence: float, is_fake: bool) -> go.Figure:
    score = round((1.0 - confidence) * 100 if is_fake else confidence * 100, 1)
    bar_color = "#22c55e" if score >= 50 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Authenticity Index (% REAL)", 'font': {'size': 16, 'color': '#e5e7eb'}},
        number={'suffix': "%", 'font': {'size': 28, 'color': bar_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4b5563"},
            'bar': {'color': bar_color, 'thickness': 0.35},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 1,
            'bordercolor': "#374151",
            'steps': [
                {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.20)"},
                {'range': [40, 60], 'color': "rgba(251, 146, 60, 0.20)"},
                {'range': [60, 100], 'color': "rgba(34, 197, 94, 0.20)"},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


def plot_rgb_histogram(image_bytes: bytes) -> go.Figure:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    fig = go.Figure()
    colors = ["#ef4444", "#22c55e", "#38bdf8"]
    names = ["Red Channel", "Green Channel", "Blue Channel"]
    for i in range(3):
        counts, bins = np.histogram(arr[:, :, i].ravel(), bins=64, range=(0, 256))
        fig.add_trace(go.Scatter(
            x=bins[:-1], y=counts, mode="lines",
            line=dict(color=colors[i], width=2),
            name=names[i]
        ))
    fig.update_layout(
        title="RGB Color Channel Intensity Distribution",
        xaxis_title="Pixel Value (0-255)",
        yaxis_title="Frequency",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_temporal_divergence(diffs: list[float], threshold: float = 15.0) -> go.Figure:
    fig = go.Figure()
    frames = list(range(1, len(diffs) + 1))
    fig.add_trace(go.Scatter(
        x=frames, y=diffs, mode="lines+markers",
        line=dict(color="#38bdf8", width=2),
        marker=dict(size=6, color=["#ef4444" if d > threshold else "#22c55e" for d in diffs]),
        name="Inter-Frame Divergence"
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444",
                  annotation_text="Anomaly Threshold", annotation_position="bottom right")
    fig.update_layout(
        title="Temporal Frame Divergence & Splice Screening",
        xaxis_title="Frame Transition Index",
        yaxis_title="Mean Pixel Disparity",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown(f"# {APP_ICON} Multimodal & Dataset Fake-News Detection")
st.markdown(
    "Dynamically analyze **tabular datasets (CSV), raw text, photographs, or video files**. "
    "TruthLens AI auto-detects input modalities, applies specialized visual, audio, and linguistic inspection pipelines, "
    "and produces calibrated authenticity verdicts."
)

# Operational status badges
c_b1, c_b2, c_b3, c_b4 = st.columns(4)
with c_b1:
    st.markdown('<span class="preset-chip">🖼️ <strong>ELA Visual Engine</strong>: Active</span>', unsafe_allow_html=True)
with c_b2:
    st.markdown('<span class="preset-chip">🎬 <strong>Temporal OpenCV Flow</strong>: Active</span>', unsafe_allow_html=True)
with c_b3:
    st.markdown('<span class="preset-chip">🌐 <strong>Web Extractor (BS4)</strong>: Active</span>', unsafe_allow_html=True)
with c_b4:
    st.markdown('<span class="preset-chip">📄 <strong>Tesseract OCR</strong>: Integrated</span>', unsafe_allow_html=True)

st.divider()

# ─── Sidebar Model Selector ───────────────────────────────────────────────────
models = list_models()
model_map = {m["id"]: f"{m['name']} ({m.get('algorithm', 'Pipeline')})" for m in models}

with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    
    if models:
        current_active = st.session_state.get("active_model_id", get_default_model_id())
        default_idx = 0
        if current_active in model_map:
            default_idx = list(model_map.keys()).index(current_active)
            
        selected_mid = st.selectbox(
            "Active NLP Classification Model",
            options=list(model_map.keys()),
            format_func=lambda k: model_map.get(k, k),
            index=default_idx,
            help="Select the model used for classifying text, OCR snippets, transcripts, and CSV datasets.",
        )
        st.session_state["active_model_id"] = selected_mid
    else:
        selected_mid = "legacy_svc"
        st.info("Using baseline linear support vector classifier.")

    st.divider()
    st.markdown("### 🔬 Multimodal Modules")
    run_ela_toggle = st.toggle("Error Level Analysis (ELA)", value=True, help="Detects image compression alterations")
    run_ocr_toggle = st.toggle("Optical Character Recognition (OCR)", value=True, help="Extracts text embedded in images")
    run_video_audio = st.toggle("Speech Transcription (Video)", value=False, help="Transcribes spoken audio for NLP classification")

active_model_id = st.session_state.get("active_model_id", selected_mid)

# ─── Navigation Tabs ─────────────────────────────────────────────────────────
tab_dataset, tab_text, tab_image, tab_video, tab_url, tab_batch = st.tabs([
    "📊 Dataset / CSV", "📝 Text / Article", "🖼️ Image / Photo", "🎬 Video", "🌐 Web URL", "📦 Multi-File Batch"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET / CSV (DYNAMIC MODEL ON DATA)
# ══════════════════════════════════════════════════════════════════════════════
with tab_dataset:
    st.markdown("### 📊 Dynamic Dataset Ingestion & Batch Evaluation")
    st.caption(
        "Upload any news dataset (e.g. `shuffled_data.csv`). TruthLens AI automatically discovers the text content "
        "and ground-truth label columns, computes model predictions across the dataset, generates authenticity metrics, "
        "and highlights high-risk misinformation."
    )

    # Preset / Built-in Benchmark loader
    col_demo_ds, col_upload_ds = st.columns([1, 2])
    with col_demo_ds:
        st.markdown("##### 🔬 Quick Benchmark")
        if st.button("📥 Load Built-in Research Corpus (70 Articles)", type="secondary", use_container_width=True):
            corpus_p = DATA_DIR / "truthlens_research_corpus.csv"
            if corpus_p.exists():
                with open(corpus_p, "rb") as f_ds:
                    st.session_state["active_dataset_bytes"] = f_ds.read()
                    st.session_state["active_dataset_name"] = "truthlens_research_corpus.csv"
                    st.success("Loaded TruthLens Research Corpus!")
                    st.rerun()

    with col_upload_ds:
        st.markdown("##### 📤 Upload Custom CSV")
        csv_file = st.file_uploader(
            "Upload CSV Dataset",
            type=["csv", "tsv"],
            key="dataset_tab_uploader",
            label_visibility="collapsed",
            help="Upload CSV files containing news articles, statements, or claims.",
        )
        if csv_file:
            st.session_state["active_dataset_bytes"] = csv_file.read()
            st.session_state["active_dataset_name"] = csv_file.name

    if "active_dataset_bytes" in st.session_state:
        file_bytes = st.session_state["active_dataset_bytes"]
        ds_name = st.session_state.get("active_dataset_name", "dataset.csv")
        file_mb = len(file_bytes) / (1024 * 1024)
        
        # Read a quick preview to display column options
        try:
            preview_df = pd.read_csv(io.BytesIO(file_bytes), nrows=5)
            cols = list(preview_df.columns)
            
            # Auto-detect defaults
            text_candidates = ["text", "article", "content", "statement", "body", "news", "headline", "title"]
            default_t_idx = 0
            for idx, c in enumerate(cols):
                if any(kw in c.lower() for kw in text_candidates):
                    default_t_idx = idx
                    break

            label_candidates = ["label", "target", "class", "category", "ground_truth"]
            default_l_idx = 0
            label_options = ["None (Unlabeled Dataset)"] + cols
            for idx, c in enumerate(cols):
                if any(kw in c.lower() for kw in label_candidates):
                    default_l_idx = idx + 1
                    break

            st.markdown(f"**Loaded File:** `{ds_name}` ({file_mb:.2f} MB)")
            c_cfg1, c_cfg2, c_cfg3 = st.columns([2, 2, 2])
            with c_cfg1:
                sel_text_col = st.selectbox("Article Text Column", options=cols, index=default_t_idx)
            with c_cfg2:
                sel_label_col_raw = st.selectbox("Ground Truth Label Column (Optional)", options=label_options, index=default_l_idx)
                sel_label_col = None if sel_label_col_raw == "None (Unlabeled Dataset)" else sel_label_col_raw
            with c_cfg3:
                sample_mode = st.selectbox(
                    "Execution Scope",
                    options=["Full Dataset", "Fast Sample (First 2,000 rows)", "Quick Sample (First 500 rows)"],
                )
                sample_limit = None
                if "2,000" in sample_mode:
                    sample_limit = 2000
                elif "500" in sample_mode:
                    sample_limit = 500

            c_btn1, c_btn2 = st.columns([1, 1])
            with c_btn1:
                btn_analyze_csv = st.button("⚡ Run Dynamic Inference on Dataset", type="primary", use_container_width=True)
            with c_btn2:
                btn_adapt_model = st.button(
                    "🚀 Dynamically Train Model on this Dataset",
                    type="secondary",
                    use_container_width=True,
                    help="Instantly fits and registers a new high-accuracy model directly on this dataset.",
                    disabled=(sel_label_col is None),
                )

            if btn_adapt_model and sel_label_col:
                with st.spinner(f"Dynamically training and calibrating model on `{ds_name}`…"):
                    try:
                        from src.dynamic_engine import quick_adapt_to_dataset
                        full_train_df = pd.read_csv(io.BytesIO(file_bytes))
                        adapt_res = quick_adapt_to_dataset(
                            df=full_train_df,
                            text_col=sel_text_col,
                            label_col=sel_label_col,
                            model_name=f"Dynamic ({ds_name[:20]})",
                        )
                        st.session_state["active_model_id"] = adapt_res["model_id"]
                        st.session_state["last_adapted_model"] = adapt_res
                        st.success(
                            f"🎉 Dynamic Model Successfully Trained & Activated! "
                            f"Held-out Accuracy: **{adapt_res['accuracy']*100:.2f}%** | F1: **{adapt_res['f1']*100:.2f}%**"
                        )
                        time.sleep(1.2)
                        st.rerun()
                    except Exception as e_adapt:
                        st.error(f"Dynamic Training Error: {e_adapt}")

            if btn_analyze_csv:
                with st.spinner(f"Processing dataset ({file_mb:.1f} MB) using '{active_model_id}'…"):
                    analysis_res = run_csv_analysis(
                        file_bytes=file_bytes,
                        filename=ds_name,
                        model_id=active_model_id,
                        text_col=sel_text_col,
                        label_col=sel_label_col,
                        sample_limit=sample_limit,
                    )

                if analysis_res.get("error"):
                    st.error(f"Dataset Analysis Error: {analysis_res['error']}")
                else:
                    st.session_state["last_csv_analysis"] = analysis_res

            # Display saved analysis results if present
            if "last_csv_analysis" in st.session_state:
                res = st.session_state["last_csv_analysis"]
                
                # Verdict Banner
                vc = res.get("verdict_color", "blue")
                verdict_txt = res.get("verdict", "Completed")
                st.markdown(
                    f'<div class="verdict-banner verdict-{vc}">{verdict_txt}</div>',
                    unsafe_allow_html=True,
                )

                # Top Stats Row
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value">{res["scored_rows"]:,}</div>'
                        f'<div class="stat-label">Articles Analyzed</div></div>',
                        unsafe_allow_html=True,
                    )
                with sc2:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value" style="color:#ef4444">{res["fake_pct"]:.1f}%</div>'
                        f'<div class="stat-label">Fake News ({res["fake_count"]:,})</div></div>',
                        unsafe_allow_html=True,
                    )
                with sc3:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value" style="color:#22c55e">{res["real_pct"]:.1f}%</div>'
                        f'<div class="stat-label">Authentic News ({res["real_count"]:,})</div></div>',
                        unsafe_allow_html=True,
                    )
                with sc4:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value" style="color:#38bdf8">{res["avg_confidence"]*100:.1f}%</div>'
                        f'<div class="stat-label">Mean Confidence</div></div>',
                        unsafe_allow_html=True,
                    )

                # Ground Truth Evaluation KPIs & Confusion Heatmap
                eval_metrics = res.get("evaluation_metrics", {})
                if eval_metrics.get("has_ground_truth"):
                    st.markdown("---")
                    st.markdown("#### 🎯 Ground-Truth Diagnostic Evaluation")
                    ec1, ec2, ec3, ec4 = st.columns(4)
                    with ec1:
                        st.metric("Test Accuracy", f"{eval_metrics['accuracy']*100:.2f}%")
                    with ec2:
                        st.metric("Precision (Fake)", f"{eval_metrics['precision']*100:.2f}%")
                    with ec3:
                        st.metric("Recall (Fake)", f"{eval_metrics['recall']*100:.2f}%")
                    with ec4:
                        st.metric("F1 Score", f"{eval_metrics['f1']*100:.2f}%")

                st.markdown("---")
                
                # Visualizations
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    fig_dist = go.Figure(data=[go.Pie(
                        labels=["Fake News", "Authentic Real"],
                        values=[res["fake_count"], res["real_count"]],
                        hole=0.55,
                        marker_colors=[COLOR_FAKE, COLOR_REAL],
                        textinfo="label+percent",
                    )])
                    fig_dist.update_layout(
                        title="Dataset Class Balance",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e5e7eb"),
                        height=280,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                with col_chart2:
                    ann_df = res.get("annotated_df")
                    if ann_df is not None and "model_confidence" in ann_df:
                        fig_conf = px.histogram(
                            ann_df,
                            x="model_confidence",
                            color="predicted_label",
                            color_discrete_map={"FAKE": COLOR_FAKE, "REAL": COLOR_REAL, "0": COLOR_FAKE, "1": COLOR_REAL},
                            nbins=25,
                            title="Prediction Confidence Distribution",
                        )
                        fig_conf.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e5e7eb"),
                            height=280,
                            margin=dict(l=20, r=20, t=40, b=20),
                            xaxis_title="Confidence (0.0 to 1.0)",
                            yaxis_title="Article Count",
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)

                # Interactive Filter & Search Controls
                st.markdown("#### 🔎 Interactive Dataset Explorer & Filtering")
                if ann_df is not None:
                    fil_c1, fil_c2, fil_c3 = st.columns([1, 1, 2])
                    with fil_c1:
                        lbl_filter = st.selectbox("Filter by Verdict", ["All", "FAKE Only", "REAL Only"], key="ds_lbl_fil")
                    with fil_c2:
                        min_conf = st.slider("Min Confidence", 0.50, 1.00, 0.50, 0.05, key="ds_conf_fil")
                    with fil_c3:
                        search_kw = st.text_input("Search Text Keywords", placeholder="Filter by word or topic...", key="ds_kw_fil")

                    filtered_view = ann_df.copy()
                    if lbl_filter == "FAKE Only":
                        filtered_view = filtered_view[filtered_view["predicted_label"].astype(str).isin(["FAKE", "0"])]
                    elif lbl_filter == "REAL Only":
                        filtered_view = filtered_view[filtered_view["predicted_label"].astype(str).isin(["REAL", "1"])]

                    if "model_confidence" in filtered_view:
                        filtered_view = filtered_view[filtered_view["model_confidence"] >= min_conf]

                    if search_kw.strip():
                        filtered_view = filtered_view[
                            filtered_view[sel_text_col].astype(str).str.contains(search_kw.strip(), case=False, na=False)
                        ]

                    st.caption(f"Showing **{len(filtered_view):,}** of **{len(ann_df):,}** articles matching filters.")
                    st.dataframe(filtered_view.head(100), use_container_width=True)

                # High-Risk Misinformation Samples
                high_risk = res.get("high_risk_samples", [])
                if high_risk:
                    st.markdown("#### 🚨 High-Confidence Fake News Flagged")
                    for hr in high_risk[:4]:
                        st.markdown(
                            f'<div class="indicator-item indicator-fake">'
                            f'<strong>{hr["verdict"]}</strong> ({hr["confidence"]*100:.1f}% confidence)<br>'
                            f'<em>"{hr["text"]}"</em>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Action Buttons: Download & Activate
                st.markdown("---")
                col_act1, col_act2 = st.columns(2)
                
                with col_act1:
                    if ann_df is not None:
                        csv_data = ann_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"⬇️ Download Predictions CSV ({len(ann_df):,} rows)",
                            data=csv_data,
                            file_name=f"truthlens_predictions_{ds_name}",
                            mime="text/csv",
                            use_container_width=True,
                        )

                with col_act2:
                    if st.button("💾 Set as Active Workspace Dataset", use_container_width=True):
                        st.session_state["active_df"] = ann_df
                        st.session_state["active_df_name"] = ds_name
                        st.session_state["active_df_meta"] = {
                            "text_col": sel_text_col,
                            "label_col": sel_label_col or "predicted_label",
                            "total_samples": len(ann_df),
                            "class_counts": {
                                "FAKE": res["fake_count"],
                                "REAL": res["real_count"],
                            },
                        }
                        st.success(f"Dataset `{ds_name}` activated across all workspace modules!")

        except Exception as exc:
            st.error(f"Error parsing uploaded file: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TEXT / ARTICLE
# ══════════════════════════════════════════════════════════════════════════════
with tab_text:
    st.markdown("### 📝 Single Text / Article Verification")
    st.caption("Paste an article headline, statement, or full body text. TruthLens AI evaluates linguistic patterns, exclamation density, and decision margins.")

    # One-click quick presets
    st.markdown("##### ⚡ Quick Benchmark Examples")
    p1, p2, p3, p4 = st.columns(4)
    if p1.button("🔴 Viral Conspiracy Hoax", use_container_width=True):
        st.session_state["preset_text"] = "BOMBSHELL EXPOSURE: Leaked military papers reveal global cabal secretly spraying robotic nanobots into metropolitan water reservoirs! SHARE BEFORE IT IS DELETED!"
    if p2.button("🔴 Miracle Health Cure", use_container_width=True):
        st.session_state["preset_text"] = "DOCTORS STUNNED: Secret backyard boiled dandelion root tea completely dissolves all cancer and heart disease in 24 hours but Big Pharma banned it from internet!"
    if p3.button("🟢 NASA Space Discovery", use_container_width=True):
        st.session_state["preset_text"] = "NASA scientists announced new spectral analysis data gathered by the James Webb Space Telescope confirming atmospheric water vapor on an exoplanet 120 light years away."
    if p4.button("🟢 Central Bank Economics", use_container_width=True):
        st.session_state["preset_text"] = "The Federal Reserve held benchmark interest rates steady following the conclusion of its two-day policy meeting, citing balanced inflation indicators and stable employment data."

    default_txt = st.session_state.get("preset_text", "")
    txt_input = st.text_area(
        "Enter Headline & Body Text:",
        value=default_txt,
        height=160,
        placeholder="Paste news headline or article body here to evaluate...",
        key="mm_text_area",
    )

    btn_text = st.button("🔍 Analyze Text with NLP Engine", type="primary", key="btn_run_text")

    if btn_text and txt_input.strip():
        with st.spinner("Analyzing text with NLP pipeline…"):
            r = run_text_analysis(txt_input, model_id=active_model_id)

        vc = r.get("verdict_color", "gray")
        vtext = r.get("verdict", "Unknown")
        conf = r.get("confidence", 0.0)
        is_fake = "FAKE" in vtext.upper() or vc == "red"

        st.markdown(
            f'<div class="verdict-banner verdict-{vc}">{vtext} &nbsp;|&nbsp; Calibrated Confidence: {conf*100:.1f}%</div>',
            unsafe_allow_html=True,
        )

        col_gauge, col_stats = st.columns([1, 1])
        with col_gauge:
            fig_gauge = plot_authenticity_gauge(conf, is_fake)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_stats:
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value">{r.get("word_count", 0)}</div>'
                    f'<div class="stat-label">Word Count</div></div>',
                    unsafe_allow_html=True,
                )
            with tc2:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value">{r.get("latency_ms", 0):.1f} ms</div>'
                    f'<div class="stat-label">Model Latency</div></div>',
                    unsafe_allow_html=True,
                )

            tc3, tc4 = st.columns(2)
            with tc3:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value">{r.get("category", "—")}</div>'
                    f'<div class="stat-label">Model Prediction</div></div>',
                    unsafe_allow_html=True,
                )
            with tc4:
                margin_val = r.get("decision_margin")
                margin_str = f"{margin_val:+.2f}" if margin_val is not None else "N/A"
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value">{margin_str}</div>'
                    f'<div class="stat-label">Linear Decision Margin</div></div>',
                    unsafe_allow_html=True,
                )

        # Linguistic indicators
        indicators = r.get("manipulation_indicators", [])
        if indicators:
            st.markdown("#### 💡 Linguistic Diagnostics")
            for ind in indicators:
                st.markdown(f'<div class="indicator-item">{ind}</div>', unsafe_allow_html=True)

    elif btn_text:
        st.warning("Please provide article text first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IMAGE / PHOTO
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    st.markdown("### 🖼️ Image & Photo Forensics")
    st.caption("Upload a photograph or screenshot. TruthLens AI evaluates compression artifacts (ELA), inspects EXIF metadata, runs RGB pixel statistics, and extracts OCR text for NLP verification.")

    # One-click demo images
    st.markdown("##### 🧪 Quick Forensic Demo Presets")
    c_img_d1, c_img_d2 = st.columns(2)
    with c_img_d1:
        if st.button("🔴 Load Manipulated / Spliced Image Demo", use_container_width=True):
            st.session_state["active_img_bytes"] = create_demo_image(tampered=True)
            st.session_state["active_img_name"] = "tampered_composite_press_release.jpg"
            st.rerun()
    with c_img_d2:
        if st.button("🟢 Load Authentic Scientific Document Demo", use_container_width=True):
            st.session_state["active_img_bytes"] = create_demo_image(tampered=False)
            st.session_state["active_img_name"] = "authentic_astronomical_bulletin.jpg"
            st.rerun()

    img_file = st.file_uploader(
        "Upload Image File",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"],
        key="mm_img_uploader",
    )
    if img_file:
        st.session_state["active_img_bytes"] = img_file.read()
        st.session_state["active_img_name"] = img_file.name

    if "active_img_bytes" in st.session_state:
        img_bytes = st.session_state["active_img_bytes"]
        img_name = st.session_state.get("active_img_name", "image.jpg")

        col_img_prev, col_img_ela = st.columns(2)
        with col_img_prev:
            st.image(img_bytes, caption=f"Analyzed Image: {img_name}", use_container_width=True)

        if st.button("🔍 Run Full Image Forensics & OCR", type="primary", key="btn_run_img", use_container_width=True):
            with st.spinner("Analyzing image compression, noise patterns, and OCR…"):
                r = run_image_analysis(img_bytes, img_name, model_id=active_model_id)
                st.session_state["last_image_analysis"] = r

        if "last_image_analysis" in st.session_state:
            r = st.session_state["last_image_analysis"]
            vc = r.get("verdict_color", "gray")
            vtext = r.get("verdict", "Unknown")
            susp = r.get("overall_suspicion_score", 0.0)
            
            st.markdown(
                f'<div class="verdict-banner verdict-{vc}">{vtext} &nbsp;|&nbsp; Suspicion Index: {susp*100:.1f}%</div>',
                unsafe_allow_html=True,
            )

            ela_img = r.get("ela_image")
            if ela_img:
                with col_img_ela:
                    st.image(ela_img, caption="Error Level Analysis (Bright/Uneven highlights = Modified / Composited)", use_container_width=True)

            # Forensic Stats
            pix = r.get("pixel_stats", {})
            st.markdown("#### 🔬 Forensic Signal Diagnostics")
            c_f1, c_f2, c_f3, c_f4 = st.columns(4)
            with c_f1:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value">{r.get("ela_analysis", {}).get("ela_score", 0):.3f}</div>'
                    f'<div class="stat-label">ELA Disparity Score</div></div>',
                    unsafe_allow_html=True,
                )
            with c_f2:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value">{pix.get("noise_variance", 0):.1f}</div>'
                    f'<div class="stat-label">Noise Variance</div></div>',
                    unsafe_allow_html=True,
                )
            with c_f3:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value">{pix.get("sharpness_score", 0):.2f}</div>'
                    f'<div class="stat-label">Gradient Sharpness</div></div>',
                    unsafe_allow_html=True,
                )
            with c_f4:
                meta = r.get("metadata", {})
                mp = meta.get("megapixels", 0)
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value">{mp} MP</div>'
                    f'<div class="stat-label">Resolution ({meta.get("width", 0)}x{meta.get("height", 0)})</div></div>',
                    unsafe_allow_html=True,
                )

            # RGB Color Channel Distribution Chart
            st.plotly_chart(plot_rgb_histogram(img_bytes), use_container_width=True)

            # Manipulation Indicators
            inds = r.get("manipulation_indicators", [])
            if inds:
                st.markdown("#### ⚠️ Visual & Structural Forensic Signals")
                for ind in inds:
                    st.markdown(f'<div class="indicator-item indicator-fake">⚠️ {ind}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="indicator-item indicator-success">✅ Clean uniform pixel gradients and compression levels.</div>', unsafe_allow_html=True)

            # OCR Extract & NLP Fusion
            ocr_text = r.get("extracted_text", "")
            if ocr_text:
                st.markdown("#### 📄 Extracted Embedded Text (OCR & NLP Verification)")
                st.info(ocr_text[:1200])
                if "nlp_on_ocr_text" in r:
                    nlp_r = r["nlp_on_ocr_text"]
                    st.success(f"NLP Classification on OCR Text: **{nlp_r.get('verdict')}** ({nlp_r.get('confidence', 0)*100:.1f}% confidence)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown("### 🎬 Video Forensics & Deepfake Screening")
    st.caption("Inspect video clips for inter-frame temporal divergence, frame-level compression anomalies, and transcript audio matching.")

    # One-click synthetic demo video
    st.markdown("##### 🧪 Quick Video Forensic Demo")
    if st.button("🔴 Load Synthetic Temporal Spliced Video Demo", use_container_width=True):
        st.session_state["active_vid_bytes"] = create_demo_video(tampered=True)
        st.session_state["active_vid_name"] = "synthetic_temporal_splice_demo.mp4"
        st.rerun()

    vid_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "webm"], key="mm_vid_uploader")
    if vid_file:
        st.session_state["active_vid_bytes"] = vid_file.read()
        st.session_state["active_vid_name"] = vid_file.name

    if "active_vid_bytes" in st.session_state:
        vid_bytes = st.session_state["active_vid_bytes"]
        vid_name = st.session_state.get("active_vid_name", "video.mp4")
        st.caption(f"Active File: **{vid_name}** | Size: **{len(vid_bytes)/(1024*1024):.2f} MB**")

        st.video(vid_bytes)

        if st.button("🔍 Run Video Temporal Forensics", type="primary", key="btn_run_vid", use_container_width=True):
            with st.spinner("Decoding video stream and calculating temporal divergence…"):
                r = run_video_analysis(vid_bytes, vid_name, model_id=active_model_id)
                st.session_state["last_vid_analysis"] = r

        if "last_vid_analysis" in st.session_state:
            r = st.session_state["last_vid_analysis"]
            if r.get("error"):
                st.error(r["error"])
            else:
                vc = r.get("verdict_color", "gray")
                vtext = r.get("verdict", "Unknown")
                susp = r.get("overall_suspicion_score", 0.0)
                st.markdown(
                    f'<div class="verdict-banner verdict-{vc}">{vtext} &nbsp;|&nbsp; Temporal Suspicion Index: {susp*100:.1f}%</div>',
                    unsafe_allow_html=True,
                )

                v_info = r.get("video_info", {})
                vc1, vc2, vc3, vc4 = st.columns(4)
                with vc1:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value">{v_info.get("total_frames", 0)}</div>'
                        f'<div class="stat-label">Total Frames</div></div>',
                        unsafe_allow_html=True,
                    )
                with vc2:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value">{v_info.get("fps", 0)}</div>'
                        f'<div class="stat-label">FPS Rate</div></div>',
                        unsafe_allow_html=True,
                    )
                with vc3:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value">{v_info.get("duration_s", 0):.1f}s</div>'
                        f'<div class="stat-label">Duration</div></div>',
                        unsafe_allow_html=True,
                    )
                with vc4:
                    td = r.get("temporal_divergence", {})
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value">{td.get("anomalous_transitions", 0)}</div>'
                        f'<div class="stat-label">Anomalous Transitions</div></div>',
                        unsafe_allow_html=True,
                    )

                # Temporal Divergence Graph
                frame_diffs = td.get("frame_differences", [])
                if frame_diffs:
                    st.plotly_chart(plot_temporal_divergence(frame_diffs), use_container_width=True)

                # Keyframe Thumbnail
                thumb = r.get("thumbnail")
                if thumb is not None:
                    from PIL import Image as PILImage
                    st.image(PILImage.fromarray(thumb), caption="Analyzed Video Frame Sample", width=360)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WEB URL
# ══════════════════════════════════════════════════════════════════════════════
with tab_url:
    st.markdown("### 🌐 Web URL Article Scraper & Fact Verifier")
    st.caption("Scrape live news articles directly from the web using BeautifulSoup. TruthLens AI extracts metadata, title bylines, and runs NLP verification.")

    st.markdown("##### ⚡ Test Preset News URLs")
    u1, u2, u3 = st.columns(3)
    if u1.button("BBC World News", use_container_width=True):
        st.session_state["preset_url"] = "https://www.bbc.com/news"
    if u2.button("Reuters World Brief", use_container_width=True):
        st.session_state["preset_url"] = "https://www.reuters.com"
    if u3.button("NASA Science News Release", use_container_width=True):
        st.session_state["preset_url"] = "https://www.nasa.gov/news-release"

    default_url = st.session_state.get("preset_url", "")
    url_input = st.text_input("Enter Web News Article URL:", value=default_url, placeholder="https://example.com/news-story")

    if st.button("🔍 Fetch & Verify Article URL", type="primary", key="btn_run_url", use_container_width=True):
        if url_input.strip():
            with st.spinner("Scraping webpage with BeautifulSoup and running NLP classification…"):
                r = run_url_analysis(url_input.strip(), model_id=active_model_id)
                st.session_state["last_url_analysis"] = r
        else:
            st.warning("Please specify a valid URL.")

    if "last_url_analysis" in st.session_state:
        r = st.session_state["last_url_analysis"]
        if r.get("error"):
            st.error(f"URL Error: {r['error']}")
        else:
            vc = r.get("verdict_color", "gray")
            vtext = r.get("verdict", "Unknown")
            conf = r.get("confidence", 0.0)
            st.markdown(
                f'<div class="verdict-banner verdict-{vc}">{vtext} &nbsp;|&nbsp; Calibrated Confidence: {conf*100:.1f}%</div>',
                unsafe_allow_html=True,
            )

            # Article Metadata Cards
            uc1, uc2, uc3 = st.columns(3)
            with uc1:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value" style="font-size:1.1rem">{r.get("domain")}</div>'
                    f'<div class="stat-label">Domain ({ "🔒 HTTPS" if r.get("is_https") else "⚠️ Insecure HTTP" })</div></div>',
                    unsafe_allow_html=True,
                )
            with uc2:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value" style="font-size:1.1rem">{r.get("author", "Not Disclosed")}</div>'
                    f'<div class="stat-label">Byline / Author</div></div>',
                    unsafe_allow_html=True,
                )
            with uc3:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value">{r.get("word_count", 0)}</div>'
                    f'<div class="stat-label">Article Word Count</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(f"**Article Title:** *{r.get('title')}*")

            inds = r.get("manipulation_indicators", [])
            if inds:
                st.markdown("#### 🔍 Verification Indicators")
                for ind in inds:
                    st.markdown(f'<div class="indicator-item">{ind}</div>', unsafe_allow_html=True)

            with st.expander("📄 Parsed Web Article Body Text"):
                st.write(r.get("clean_text", "")[:3000])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MULTI-FILE BATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📦 Heterogeneous Multi-File Ingestion")
    st.caption("Upload multiple files across any mixture of modalities (CSV datasets, text documents, photos, video clips). Each file is routed dynamically to its dedicated analyzer.")

    # Mixed Demo Loader
    if st.button("🧪 Load Multi-Modality Mixed Demo Batch", use_container_width=True):
        demo_items = [
            {
                "bytes": (DATA_DIR / "truthlens_research_corpus.csv").read_bytes(),
                "name": "truthlens_research_corpus.csv",
                "mime": "text/csv",
            },
            {
                "bytes": create_demo_image(tampered=True),
                "name": "spliced_press_release.jpg",
                "mime": "image/jpeg",
            },
            {
                "bytes": "The Federal Reserve held benchmark interest rates steady following its meeting.".encode("utf-8"),
                "name": "federal_reserve_policy.txt",
                "mime": "text/plain",
            },
            {
                "bytes": create_demo_video(tampered=True),
                "name": "deepfake_sample_clip.mp4",
                "mime": "video/mp4",
            },
        ]
        with st.spinner("Analyzing mixed batch with multimodal router…"):
            batch_results = batch_route(demo_items, model_id=active_model_id)
        st.session_state["multi_batch_results"] = (batch_results, demo_items)

    batch_files = st.file_uploader(
        "Upload Multiple Files",
        type=["csv", "txt", "jpg", "jpeg", "png", "mp4", "mov"],
        accept_multiple_files=True,
        key="mm_batch_multi_uploader",
    )

    if batch_files:
        if st.button(f"⚡ Ingest & Analyze {len(batch_files)} Uploaded File(s)", type="primary", key="btn_run_multi", use_container_width=True):
            items = []
            for bf in batch_files:
                items.append({
                    "bytes": bf.read(),
                    "name": bf.name,
                    "mime": bf.type or "",
                })

            with st.spinner("Analyzing files with dynamic router…"):
                batch_results = batch_route(items, model_id=active_model_id)

            st.session_state["multi_batch_results"] = (batch_results, items)

    if "multi_batch_results" in st.session_state:
        batch_results, items = st.session_state["multi_batch_results"]

        st.markdown(f"#### 📋 Ingestion Results ({len(batch_results)} files)")
        
        rows = []
        modality_counts = {}
        for r in batch_results:
            mod = r.get("modality", "?")
            modality_counts[mod] = modality_counts.get(mod, 0) + 1
            icon = {"text": "📝", "image": "🖼️", "video": "🎬", "csv": "📊"}.get(mod, "❓")
            rows.append({
                "File": r.get("filename", ""),
                "Modality": f"{icon} {mod.upper()}",
                "Verdict": r.get("verdict", "—"),
                "Confidence / Score": f"{r.get('confidence', 0)*100:.1f}%",
                "Signals / Indicators": len(r.get("manipulation_indicators", [])),
            })

        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True)

        # Modality Distribution Chart
        col_m_pie, col_m_risk = st.columns(2)
        with col_m_pie:
            fig_m = px.pie(
                names=list(modality_counts.keys()),
                values=list(modality_counts.values()),
                title="Batch Modality Composition",
                hole=0.45,
            )
            fig_m.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5e7eb"),
                height=240,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_m, use_container_width=True)

        with col_m_risk:
            # Overall portfolio risk score
            fake_items = sum(1 for r in batch_results if "FAKE" in str(r.get("verdict", "")).upper() or "MANIPULATED" in str(r.get("verdict", "")).upper())
            risk_pct = round((fake_items / max(len(batch_results), 1)) * 100, 1)
            st.markdown("##### 🚨 Portfolio Risk Index")
            st.markdown(
                f'<div class="stat-card" style="margin-top:1.5rem"><div class="stat-value" style="color:{"#ef4444" if risk_pct > 30 else "#22c55e"}">{risk_pct}%</div>'
                f'<div class="stat-label">Flagged High-Risk Misinformation Ratio ({fake_items} of {len(batch_results)} files)</div></div>',
                unsafe_allow_html=True,
            )

        # Detailed Findings per File
        st.markdown("---")
        st.markdown("#### 🔍 Detailed Findings per File")
        for idx, r in enumerate(batch_results):
            mod = r.get("modality", "")
            fname = r.get("filename", f"Item {idx+1}")
            icon = {"text": "📝", "image": "🖼️", "video": "🎬", "csv": "📊"}.get(mod, "📁")

            with st.expander(f"{icon} {fname} — {r.get('verdict', '')}"):
                if mod == "csv":
                    st.markdown(f"**Total Articles Scored:** `{r.get('scored_rows', 0):,}`")
                    st.markdown(f"**Fake News Ratio:** `{r.get('fake_pct', 0):.1f}%` ({r.get('fake_count', 0):,} articles)")
                    st.markdown(f"**Authentic Ratio:** `{r.get('real_pct', 0):.1f}%` ({r.get('real_count', 0):,} articles)")
                    eval_m = r.get("evaluation_metrics", {})
                    if eval_m.get("has_ground_truth"):
                        st.success(f"Ground-Truth Accuracy: **{eval_m.get('accuracy', 0)*100:.2f}%** | F1: **{eval_m.get('f1', 0)*100:.2f}%**")
                    
                    ann_df = r.get("annotated_df")
                    if ann_df is not None:
                        c_dl, c_act = st.columns(2)
                        with c_dl:
                            csv_b = ann_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                f"⬇️ Download Annotated CSV ({fname})",
                                data=csv_b,
                                file_name=f"truthlens_scored_{fname}",
                                mime="text/csv",
                                key=f"dl_{idx}",
                            )
                        with c_act:
                            if st.button(f"💾 Set {fname} as Active Dataset", key=f"act_{idx}"):
                                st.session_state["active_df"] = ann_df
                                st.session_state["active_df_name"] = fname
                                st.success(f"Activated `{fname}` in workspace!")

                elif mod in ("image", "video"):
                    st.markdown(f"**Overall Suspicion:** `{r.get('overall_suspicion_score', 0)*100:.1f}%`")
                    inds = r.get("manipulation_indicators", [])
                    for ind in inds:
                        st.markdown(f'<div class="indicator-item">⚠️ {ind}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f"**Verdict:** `{r.get('verdict')}`")
                    st.markdown(f"**Confidence:** `{r.get('confidence', 0)*100:.1f}%`")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"🛡️ **{APP_NAME} Multimodal Engine** — Advanced multi-modal & dataset intelligence platform. Developer: **D.venkata gowtham**")
