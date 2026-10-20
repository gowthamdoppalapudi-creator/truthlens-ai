"""
11_Multimodal_Detection.py — TruthLens AI
Dynamic Multimodal & Dataset Fake-News Detection Platform:
  • 📝 Text / Article      — Instant NLP classifier, probability margins, linguistic indicators
  • 🖼️ Image / Photo       — Error Level Analysis (ELA), EXIF metadata, pixel statistics, OCR → NLP
  • 🎬 Video               — Frame extraction, temporal divergence, frame ELA, speech transcript → NLP
  • 📊 Dataset / CSV       — Dynamic dataset batch scoring, auto column detection, evaluation & metrics
  • 🌐 URL                 — Live web article scraping & verification
  • 📦 Multi-File Batch    — Ingest and analyze multiple mixed modalities simultaneously
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

from src.config import APP_NAME, APP_ICON, COLOR_FAKE, COLOR_REAL, COLOR_ACCENT, PLOTLY_LAYOUT
from src.model_registry import list_models
from src.multimodal_router import (
    detect_modality,
    run_text_analysis,
    run_image_analysis,
    run_video_analysis,
    run_csv_analysis,
    batch_route,
    get_default_model_id,
)

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
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown(f"# {APP_ICON} Multimodal & Dataset Fake-News Detection")
st.markdown(
    "Dynamically analyze **tabular datasets (CSV), raw text, photographs, or video files**. "
    "TruthLens AI auto-detects input modalities, applies specialized visual and linguistic inspection pipelines, "
    "and produces calibrated authenticity verdicts."
)
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

    csv_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv", "tsv"],
        key="dataset_tab_uploader",
        help="Upload CSV files containing news articles, statements, or claims.",
    )

    if csv_file:
        file_bytes = csv_file.read()
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
            has_label_candidate = False
            label_options = ["None (Unlabeled Dataset)"] + cols
            for idx, c in enumerate(cols):
                if any(kw in c.lower() for kw in label_candidates):
                    default_l_idx = idx + 1
                    has_label_candidate = True
                    break

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
                with st.spinner(f"Dynamically training and calibrating model on `{csv_file.name}`…"):
                    try:
                        from src.dynamic_engine import quick_adapt_to_dataset
                        full_train_df = pd.read_csv(io.BytesIO(file_bytes))
                        adapt_res = quick_adapt_to_dataset(
                            df=full_train_df,
                            text_col=sel_text_col,
                            label_col=sel_label_col,
                            model_name=f"Dynamic ({csv_file.name[:20]})",
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
                    t0 = time.time()
                    analysis_res = run_csv_analysis(
                        file_bytes=file_bytes,
                        filename=csv_file.name,
                        model_id=active_model_id,
                        text_col=sel_text_col,
                        label_col=sel_label_col,
                        sample_limit=sample_limit,
                    )
                    elapsed = time.time() - t0

                if analysis_res.get("error"):
                    st.error(f"Dataset Analysis Error: {analysis_res['error']}")
                else:
                    # Save to session state so user can interact
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

                # Ground Truth Evaluation KPIs (if labels existed)
                eval_metrics = res.get("evaluation_metrics", {})
                if eval_metrics.get("has_ground_truth"):
                    st.markdown("---")
                    st.markdown("#### 🎯 Ground-Truth Evaluation Benchmark")
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
                        height=300,
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
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=20),
                            xaxis_title="Confidence (0.0 to 1.0)",
                            yaxis_title="Article Count",
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)

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
                            file_name=f"truthlens_predictions_{csv_file.name}",
                            mime="text/csv",
                            use_container_width=True,
                        )

                with col_act2:
                    if st.button("💾 Set as Active Workspace Dataset", use_container_width=True):
                        st.session_state["active_df"] = ann_df
                        st.session_state["active_df_name"] = csv_file.name
                        st.session_state["active_df_meta"] = {
                            "text_col": sel_text_col,
                            "label_col": sel_label_col or "predicted_label",
                            "total_samples": len(ann_df),
                            "class_counts": {
                                "FAKE": res["fake_count"],
                                "REAL": res["real_count"],
                            },
                        }
                        st.success(f"Dataset `{csv_file.name}` activated across all workspace modules!")

                # Preview Table
                with st.expander("🔎 Browse Prediction Output Dataframe (First 50 Rows)"):
                    if ann_df is not None:
                        st.dataframe(ann_df.head(50), use_container_width=True)

        except Exception as exc:
            st.error(f"Error parsing uploaded file: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TEXT / ARTICLE
# ══════════════════════════════════════════════════════════════════════════════
with tab_text:
    st.markdown("### 📝 Single Text / Article Verification")
    txt_input = st.text_area(
        "Paste article headline, statement, or full body:",
        height=200,
        placeholder="Enter text to analyze…",
        key="mm_text_area",
    )

    btn_text = st.button("🔍 Analyze Text", type="primary", key="btn_run_text")

    if btn_text and txt_input.strip():
        with st.spinner("Analyzing text with NLP pipeline…"):
            r = run_text_analysis(txt_input, model_id=active_model_id)

        vc = r.get("verdict_color", "gray")
        vtext = r.get("verdict", "Unknown")
        conf = r.get("confidence", 0.0)
        st.markdown(
            f'<div class="verdict-banner verdict-{vc}">{vtext} &nbsp;|&nbsp; Calibrated Confidence: {conf*100:.1f}%</div>',
            unsafe_allow_html=True,
        )

        tc1, tc2, tc3 = st.columns(3)
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
        with tc3:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{r.get("category", "—")}</div>'
                f'<div class="stat-label">Output Category</div></div>',
                unsafe_allow_html=True,
            )

        indicators = r.get("manipulation_indicators", [])
        if indicators:
            st.markdown("#### 💡 Linguistic Analysis")
            for ind in indicators:
                st.markdown(f'<div class="indicator-item">{ind}</div>', unsafe_allow_html=True)

    elif btn_text:
        st.warning("Please provide article text first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IMAGE / PHOTO
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    st.markdown("### 🖼️ Image & Photo Forensics")
    st.caption("Upload a photograph or screenshot. TruthLens AI evaluates compression artifacts (ELA), inspects EXIF metadata, runs pixel statistics, and extracts OCR text for NLP verification.")

    img_file = st.file_uploader(
        "Upload Image File",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"],
        key="mm_img_uploader",
    )

    if img_file:
        img_bytes = img_file.read()
        col_img_prev, col_img_ela = st.columns(2)
        with col_img_prev:
            st.image(img_bytes, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Run Image Forensics", type="primary", key="btn_run_img"):
            with st.spinner("Analyzing image forensics and OCR…"):
                r = run_image_analysis(img_bytes, img_file.name, model_id=active_model_id)

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
                    st.image(ela_img, caption="Error Level Analysis (Bright areas = Recompressed / Modified)", use_container_width=True)

            inds = r.get("manipulation_indicators", [])
            if inds:
                st.markdown("#### ⚠️ Visual Forensics Signals")
                for ind in inds:
                    st.markdown(f'<div class="indicator-item indicator-fake">⚠️ {ind}</div>', unsafe_allow_html=True)
            else:
                st.success("No abnormal visual manipulation signals detected.")

            # OCR Extract
            ocr_text = r.get("extracted_text", "")
            if ocr_text:
                st.markdown("#### 📄 Extracted Embedded Text (OCR)")
                st.info(ocr_text[:1000])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown("### 🎬 Video Forensics & Deepfake Screening")
    st.caption("Inspect video clips for inter-frame temporal divergence, frame-level compression anomalies, and transcript audio matching.")

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    if not has_cv2:
        st.warning("⚠️ `opencv-python` is required for video decoding. Please run: `pip install opencv-python`.")
    else:
        vid_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "webm"], key="mm_vid_uploader")
        if vid_file:
            vid_bytes = vid_file.read()
            st.caption(f"File: **{vid_file.name}** | Size: **{len(vid_bytes)/(1024*1024):.2f} MB**")

            if st.button("🔍 Run Video Forensics", type="primary", key="btn_run_vid"):
                with st.spinner("Extracting frames and screening for temporal divergence…"):
                    r = run_video_analysis(vid_bytes, vid_file.name, model_id=active_model_id)

                if r.get("error"):
                    st.error(r["error"])
                else:
                    vc = r.get("verdict_color", "gray")
                    vtext = r.get("verdict", "Unknown")
                    susp = r.get("overall_suspicion_score", 0.0)
                    st.markdown(
                        f'<div class="verdict-banner verdict-{vc}">{vtext} &nbsp;|&nbsp; Temporal Suspicion: {susp*100:.1f}%</div>',
                        unsafe_allow_html=True,
                    )

                    thumb = r.get("thumbnail")
                    if thumb is not None:
                        from PIL import Image as PILImage
                        st.image(PILImage.fromarray(thumb), caption="Sample Analyzed Frame", width=400)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WEB URL
# ══════════════════════════════════════════════════════════════════════════════
with tab_url:
    st.markdown("### 🌐 Web URL Article Scraper & Classifier")
    url_input = st.text_input("Enter Web News Article URL", placeholder="https://example.com/news-story")

    if st.button("🔍 Fetch & Verify URL", type="primary", key="btn_run_url"):
        if url_input.strip():
            with st.spinner("Scraping webpage and running classification…"):
                try:
                    import urllib.request
                    import re
                    req = urllib.request.Request(
                        url_input.strip(),
                        headers={"User-Agent": "TruthLensAI/2.0 Research Platform"},
                    )
                    with urllib.request.urlopen(req, timeout=12) as response:
                        html_content = response.read().decode("utf-8", errors="replace")
                    
                    # Clean tags
                    clean_text = re.sub(r"<[^>]+>", " ", html_content)
                    clean_text = re.sub(r"\s+", " ", clean_text).strip()[:8000]

                    r = run_text_analysis(clean_text, model_id=active_model_id)
                    vc = r.get("verdict_color", "gray")
                    vtext = r.get("verdict", "Unknown")
                    conf = r.get("confidence", 0.0)
                    st.markdown(
                        f'<div class="verdict-banner verdict-{vc}">{vtext} &nbsp;|&nbsp; Confidence: {conf*100:.1f}%</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("📄 Parsed Web Article Text"):
                        st.write(clean_text[:1500])
                except Exception as exc:
                    st.error(f"Error fetching URL: {exc}")
        else:
            st.warning("Please specify a URL.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MULTI-FILE BATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📦 Heterogeneous Multi-File Ingestion")
    st.caption("Upload multiple files across any mixture of modalities (CSV datasets, text documents, photos, video clips). Each file is routed dynamically to its dedicated analyzer.")

    batch_files = st.file_uploader(
        "Upload Multiple Files",
        type=["csv", "txt", "jpg", "jpeg", "png", "mp4", "mov"],
        accept_multiple_files=True,
        key="mm_batch_multi_uploader",
    )

    if batch_files:
        if st.button(f"⚡ Ingest & Analyze {len(batch_files)} File(s)", type="primary", key="btn_run_multi"):
            items = []
            for bf in batch_files:
                items.append({
                    "bytes": bf.read(),
                    "name": bf.name,
                    "mime": bf.type or "",
                })

            with st.spinner("Analyzing files with dynamic router…"):
                batch_results = batch_route(items, model_id=active_model_id)

            # Store in session state
            st.session_state["multi_batch_results"] = (batch_results, items)

        if "multi_batch_results" in st.session_state:
            batch_results, items = st.session_state["multi_batch_results"]

            st.markdown(f"#### 📋 Ingestion Results ({len(batch_results)} files)")
            
            rows = []
            for r in batch_results:
                mod = r.get("modality", "?")
                icon = {"text": "📝", "image": "🖼️", "video": "🎬", "csv": "📊"}.get(mod, "❓")
                rows.append({
                    "File": r.get("filename", ""),
                    "Modality": f"{icon} {mod.upper()}",
                    "Verdict": r.get("verdict", "—"),
                    "Confidence / Accuracy": f"{r.get('confidence', 0)*100:.1f}%",
                    "Signals / Indicators": len(r.get("manipulation_indicators", [])),
                })

            summary_df = pd.DataFrame(rows)
            st.dataframe(summary_df, use_container_width=True)

            # Detailed Expanders
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
