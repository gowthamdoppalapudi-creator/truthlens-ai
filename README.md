# TruthLens AI — An Intelligent Fake News Detection and Early Detection Research Platform

![TruthLens AI Logo](assets/truthlens_logo.png)

> **Principal Investigator / Developer:** D.venkata gowtham  
> **Platform Version:** 2.0.0  
> **Environment:** Python 3.10.11 · scikit-learn 1.7.2 · Streamlit · Plotly  

---

## 📌 1. Project Overview

**TruthLens AI** is an advanced, research-oriented natural language processing (NLP) and machine learning platform designed for fake news detection, dataset quality assurance, interactive model training, rigorous diagnostic evaluation, model explainability, and early detection research.

Initially conceived as a static TF-IDF + LinearSVC prototype, TruthLens has been upgraded into an interactive, modular research laboratory. It equips researchers and journalists with the tools to train custom models without data leakage, evaluate held-out generalization using empirical ROC/PR curves, analyze batches of news articles, investigate how early in an article misinformation patterns appear, and export reproducibility reports.

---

## 🚀 2. Core Platform Features

| Module | Location | Core Functionality |
| :--- | :--- | :--- |
| **Executive Dashboard** | `pages/1_Dashboard.py` | Real-time workspace telemetry, active model statistics, dataset overview, and rapid navigation. |
| **Real-Time News Classifier** | `pages/2_Predict_News.py` | Instant headline and body text classification with calibrated confidence indices, linear margin metrics, and benchmark examples. |
| **Dataset Studio** | `pages/3_Dataset_Manager.py` | Ingest custom CSV files, interactive schema and label mapping, data validation checks (missing values, short texts, duplicates, severe imbalance), and non-destructive cleaning. |
| **Exploratory Data Analysis (EDA)** | `pages/4_Data_Exploration.py` | Class balance distributions, document length & token histograms, unigram and bigram collocations, and lexical word clouds. |
| **ML Training Studio** | `pages/5_Model_Training.py` | Train LinearSVC, Logistic Regression, Multinomial Naive Bayes, or Random Forest models with zero leakage, custom TF-IDF n-grams, and stratified K-fold cross-validation. |
| **Diagnostic Evaluation Studio** | `pages/6_Model_Evaluation.py` | Held-out empirical accuracy, precision, recall, F1-score, confusion matrix heatmaps, ROC curves with continuous score AUC, and Precision-Recall curves. |
| **Batch News Analyzer** | `pages/7_Batch_Analyzer.py` | High-throughput batch inference over multi-article CSV files with searchable results and downloadable predictions. |
| **Early Detection Engine** | `pages/8_Early_Detection.py` | Truncation experiments testing model sensitivity across 10% to 100% token availability under **Experiment A** (stage-specific training) and **Experiment B** (distribution shift). |
| **Model Explainability** | `pages/9_Explainability.py` | Global linear coefficients and local single-article token attribution showing which words drive the classification decision. |
| **Research Report Generator** | `pages/10_Research_Reports.py` | Automatic compilation and export of peer-review-ready technical reports in Markdown, JSON metadata, and CSV metrics. |

---

## 🏗️ 3. Repository Architecture

```text
truthlens.ai/
├── app.py                      # Main Streamlit platform entrypoint
├── requirements.txt            # Python dependencies (Python 3.10.11 & scikit-learn 1.7.2)
├── README.md                   # Complete documentation and user manual
├── assets/
│   └── truthlens_logo.png      # Application branding logo
├── src/                        # Modular backend architecture
│   ├── config.py               # Central configuration (branding, paths, defaults)
│   ├── utils.py                # UI styling, custom cards, CSS, and Plotly formatters
│   ├── data_loader.py          # Session dataset persistence and CSV loading
│   ├── data_validation.py      # Quality assurance rules and non-destructive cleaning
│   ├── preprocessing.py        # Tokenization, stopword filtering, lemmatization
│   ├── training.py             # Leakage-free ML pipelines & cross-validation
│   ├── evaluation.py           # Empirical metrics, ROC-AUC, PR curves, and plots
│   ├── model_registry.py       # Checkpoint storage, metadata serialization, loading
│   ├── prediction.py           # Single/batch inference with confidence calibration
│   ├── early_detection.py      # Partial-text truncation research protocols
│   ├── explainability.py       # Global coefficients & local term attribution
│   └── report_generator.py     # Academic Markdown & JSON report generator
├── pages/                      # Streamlit multi-page interface
│   ├── 1_Dashboard.py
│   ├── 2_Predict_News.py
│   ├── 3_Dataset_Manager.py
│   ├── 4_Data_Exploration.py
│   ├── 5_Model_Training.py
│   ├── 6_Model_Evaluation.py
│   ├── 7_Batch_Analyzer.py
│   ├── 8_Early_Detection.py
│   ├── 9_Explainability.py
│   └── 10_Research_Reports.py
├── models/                     # Saved model pipelines and registry.json
├── datasets/                   # Workspace datasets and cleaned exports
├── reports/                    # Archived Markdown and JSON research deliverables
├── tests/                      # Automated test suite
│   └── test_platform.py        # 14 comprehensive unit & integration tests
├── fake-news-detection-nlp-main/ # Preserved original project files & artifacts
│   ├── final_svc_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── metrics.json
│   ├── NLP.ipynb
│   └── README.md
└── backup_original/            # Immutable safety backup of original artifacts
```

---

## 💻 4. Installation & Setup (Windows PowerShell)

### Prerequisites
- Windows 10/11
- Python 3.10.11 installed and added to `PATH`

### Step 1: Open PowerShell in Project Directory
```powershell
cd F:\truthlens.ai
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Automated Test Suite
Ensure all 14 unit and integration tests pass:
```powershell
python -m pytest tests/test_platform.py -v
```

### Step 4: Launch the Web Application
```powershell
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 📖 5. User Guides & Workflows

### How to Upload and Validate a Custom Dataset
1. Navigate to **📁 Dataset Manager** (`pages/3_Dataset_Manager.py`).
2. Upload your CSV file under the **Upload Custom CSV** tab (or click **Load Built-in Research Corpus** to explore immediately).
3. Confirm the **Text Feature Column** and **Label Column**.
4. Map unique label values to `FAKE` and `REAL`.
5. Review the automated **Dataset Quality & Integrity Report** for:
   - Missing text or labels
   - Duplicate articles (potential leakage sources)
   - Excessively short texts (<10 characters)
   - Severe class imbalance
6. Click **Clean Dataset & Set as Active Workspace Data** to activate the verified data.

#### Required CSV Format
Your dataset must contain at least one column for the text and one column for the target label:
```csv
text,label
"NASA scientists confirmed atmospheric water vapor on exoplanet...",REAL
"SHOCKING! Secret underground tunnels found under theme park...",FAKE
```

---

### How to Train a Model (Zero-Leakage Protocol)
1. Navigate to **⚙️ ML Training Studio** (`pages/5_Model_Training.py`).
2. Choose your algorithm:
   - **LinearSVC (recommended):** Fast, highly effective on high-dimensional sparse TF-IDF spaces.
   - **Logistic Regression:** Interpretable linear baseline with direct calibrated probabilities.
   - **Multinomial Naive Bayes:** Classical fast Bayesian text classifier.
   - **Random Forest:** Ensemble of decision trees.
3. Configure TF-IDF parameters (N-gram range, maximum vocabulary size, sublinear term frequency).
4. Select held-out test split percentage (e.g., 20%) and random seed.
5. Provide a checkpoint name and click **Train Model Pipeline**.
6. The model is saved to `models/` and registered in `models/registry.json`.

---

### How ROC and AUC are Calculated
- TruthLens AI strictly calculates ROC curves on **held-out test sets** using **continuous model scores** (calibrated probabilities via `predict_proba()` or distance margins via `decision_function()`).
- Discrete hard predictions (0/1) are never substituted for continuous scores.
- The True Positive Rate ($TPR = \frac{TP}{TP + FN}$) is plotted against the False Positive Rate ($FPR = \frac{FP}{FP + TN}$) across continuous decision thresholds.
- The Area Under the Curve (AUC) is computed using the trapezoidal rule. A diagonal dashed line indicates the random classifier benchmark ($AUC = 0.50$).

---

### Early Detection Research Engine
Early detection investigates whether a classifier can identify deceptive stylistic patterns from only partial article text.

TruthLens AI supports two distinct experimental protocols:
- **Experiment A (Stage-Specific Training):** A separate model is trained and evaluated on each partial text stage (e.g., 10%, 25%, 50%, 75%, 100%). Tests whether early sentences contain sufficient distinct features to learn from scratch.
- **Experiment B (Distribution Shift):** A single model is trained on complete articles, but evaluated on truncated held-out articles. Evaluates degradation when a full-text production model receives partial early input.

> **Crucial Rule:** All stages use the **identical held-out article identities** to ensure methodological validity.

---

### Exporting Research Reports
1. Navigate to **📑 Research Reports** (`pages/10_Research_Reports.py`).
2. Select your experiment checkpoint.
3. Download:
   - **Markdown Report (`.md`):** Complete research paper format with methodology, metrics, and limitations.
   - **Metadata (`.json`):** Machine-readable configuration and reproducibility seeds.
   - **Metrics Table (`.csv`):** Per-class precision, recall, and F1-scores.

---

## ⚠️ 6. Methodological Limitations & Responsible Use

1. **Linguistic Proxy vs. Fact Verification:** TruthLens models recognize statistical writing styles, vocabulary patterns, and editorial conventions characteristic of the training corpus. **The platform does not verify claims against real-world external factual databases.**
2. **Domain & Temporal Shift:** Disinformation strategies evolve. A model trained on political news from one election cycle may show degraded accuracy when applied to health topics or modern generative AI text.
3. **Decision Support Only:** This software is designed as a research platform and decision-support aid. Outputs should never be used as sole or definitive proof of authenticity.

---

## 🏛️ 7. Attribution & Provenance

- **Original Project Author:** Khush Pawar (TruthLens Fake News Detection NLP project)
- **Pretrained Baseline Models:** Trained on combined ISOT Fake/Real News and Kaggle datasets (44,275 articles) and validated on an unseen external set of 45,757 articles.
- **Preserved Artifacts:** All original model binaries (`final_svc_model.pkl`, `tfidf_vectorizer.pkl`), evaluation notes (`metrics.json`), and exploration notebooks (`NLP.ipynb`) have been fully preserved and backed up in `fake-news-detection-nlp-main/` and `backup_original/`.
