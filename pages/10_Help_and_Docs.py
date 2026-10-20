"""
10_Help_and_Docs.py — MultiModal AI Studio Comprehensive Documentation & Guide.
Complete user guide, algorithmic formulations, and technical architecture manual.
"""
from __future__ import annotations
import streamlit as st

from src.config import APP_NAME, APP_SUBTITLE, RESEARCHER_NAME, APP_VERSION
from src.theme import get_tokens
from src.utils import (
    inject_css,
    render_sidebar_brand,
    render_sidebar_profile,
    render_top_header,
    page_header,
    footer,
)

st.set_page_config(
    page_title=f"Help & Docs | {APP_NAME}",
    page_icon="ℹ️",
    layout="wide",
)
inject_css()
render_sidebar_brand()
render_sidebar_profile()
render_top_header()

page_header("Help & Technical Documentation", "Complete operational reference, algorithmic formulations, zero-leakage standards, and user manual.")
tokens = get_tokens()

doc_tab1, doc_tab2, doc_tab3, doc_tab4 = st.tabs([
    "📖 Quick Start Guide", "🔬 Forensics Architecture", "📐 Mathematical Formulations", "🛡️ Responsible AI"
])

# ─── TAB 1: QUICK START ───────────────────────────────────────────────────────
with doc_tab1:
    st.markdown("### 🚀 Platform Workflow Overview")
    st.markdown(
        """
        **TruthLens.ai** is organized into 10 cohesive modules designed for enterprise machine learning research:

        1. **Dashboard:** Live operational telemetry, model performance ROC curves, confusion matrix, and recent analysis runs.
        2. **Data Upload:** Ingest custom CSV, TXT, or JSON datasets with automatic column discovery and data quality audits.
        3. **Text Analysis:** Real-time headline and article verification with calibrated confidence gauges and decision margins.
        4. **Image Analysis:** Error Level Analysis (ELA), RGB pixel histograms, EXIF metadata, and OCR text extraction.
        5. **URL Analysis:** Live web news scraping via BeautifulSoup with domain trust verification and NLP scoring.
        6. **Dataset & Batch:** High-throughput batch inference over multi-row CSVs and mixed heterogeneous media queues.
        7. **Model Evaluation:** Held-out ROC-AUC curves, Precision-Recall curves, confusion matrices, and zero-leakage model training.
        8. **Results History:** Searchable, filterable audit trail of all previous analysis runs with CSV export.
        9. **Settings:** Theme preferences (Light / Dark switch), model defaults, and forensic hyperparameters.
        10. **Help & Documentation:** Complete technical manual and architectural reference.
        """
    )

# ─── TAB 2: FORENSICS ARCHITECTURE ───────────────────────────────────────────
with doc_tab2:
    st.markdown("### 🔬 Multimodal Digital Forensics Framework")
    st.markdown(
        """
        The platform implements a multi-tier forensics architecture:

        - **Error Level Analysis (ELA):** Recompresses images at a standard 90% JPEG quality level and calculates pixel-wise delta matrices. Composited or spliced regions display distinct compression error rates compared to surrounding backgrounds.
        - **Optical Flow & Temporal Divergence:** Uses OpenCV to decode video frames and measure consecutive frame pixel disparity $\Delta_t$. Anomalous spikes expose jump cuts, spliced transitions, or deepfake face-swap blending boundaries.
        - **Tesseract OCR Pipeline:** Extracts text embedded in screenshots and infers veracity using the active NLP classifier.
        - **BeautifulSoup Web Scraper:** Pulls article headlines, author bylines, and clean body text from web URLs while flagging unencrypted HTTP or known satirical domains.
        """
    )

# ─── TAB 3: MATHEMATICAL FORMULATIONS ─────────────────────────────────────────
with doc_tab3:
    st.markdown("### 📐 Mathematical Formulations")
    st.markdown(
        r"""
        #### 1. TF-IDF Sublinear Scaling
        Term Frequency is scaled logarithmically to prevent frequent terms from dominating the sparse feature space:
        $$\text{TF}(t, d) = 1 + \log(f_{t,d}) \quad \text{for } f_{t,d} > 0$$
        $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
        $$\mathbf{x} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

        #### 2. Linear Decision Hyperplane & Margins
        $$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$
        The signed distance from the hyperplane $|f(\mathbf{x})|$ provides a continuous margin metric indicating the certainty of the classification.

        #### 3. Error Level Analysis Disparity
        $$\Delta(x, y) = |I_{\text{orig}}(x, y) - I_{\text{recomp}}(x, y)|$$
        $$\text{Score}_{\text{ELA}} = \min\left(1.0, \frac{\frac{1}{W \times H} \sum_{x,y} \Delta(x, y)}{20.0}\right)$$
        """
    )

# ─── TAB 4: RESPONSIBLE AI ────────────────────────────────────────────────────
with doc_tab4:
    st.markdown("### 🛡️ Methodological Standards & Responsible AI")
    st.markdown(
        """
        1. **Zero Data Leakage:** Stratified train/test splitting is strictly enforced prior to fitting vectorizers or transformers.
        2. **Decision Support Orientation:** TruthLens.ai is an analytics and decision-support system. Predictions represent statistical correlations with trained corpora and should be validated by human subject-matter experts.
        3. **Domain & Temporal Shift:** Language patterns in news evolve rapidly. Regular retraining on contemporary corpora is recommended.
        """
    )

footer()
