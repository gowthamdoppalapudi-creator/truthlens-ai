# Research Experiment Report: TruthLens AI
**An Intelligent Fake News Detection and Early Detection Research Platform**

- **Lead Researcher / Developer:** D.venkata gowtham
- **Experiment Identifier:** `benchmark_pac`
- **Timestamp:** 2026-10-20T13:52:08.173112
- **Execution Platform:** Python 3.10.11 | scikit-learn 1.7.2 | OS: Windows 10

---

## 1. Research Objective
Evaluate supervised linguistic pattern classification performance for fake and real news detection. The model isolates term-frequency inverse-document-frequency (TF-IDF) feature representations to discriminate between authentic news reporting and fabricated articles.

## 2. Dataset Metadata & Quality Assurance
- **Dataset Source/Name:** TruthLens Research Benchmark Corpus
- **Total Samples:** 70
- **Training Samples ($N_{train}$):** 56
- **Held-Out Test Samples ($N_{test}$):** 14
- **Class Distribution:** `{'REAL': 35, 'FAKE': 35}`
- **Selected Text Column:** `text`
- **Selected Target Label Column:** `label`
- **Validation Notes:** Fully split prior to vectorizer fitting to guarantee zero test leakage.

## 3. Modeling Architecture & Preprocessing Protocol
- **Classifier Algorithm:** Passive Aggressive Classifier
- **Class Weighting Strategy:** `balanced`
- **Random Seed:** `42`
- **Test Set Ratio:** `0.2`
- **TF-IDF Vectorizer Hyperparameters:**
  - `ngram_range`: `[1, 2]`
  - `max_features`: `50000`
  - `min_df`: `1`
  - `max_df`: `0.95`
  - `sublinear_tf`: `True`
  - `strip_accents`: `unicode`
  - `analyzer`: `word`

## 4. Empirical Evaluation Results (Held-Out Test Set)
| Metric | Measured Score |
| :--- | :--- |
| **Overall Accuracy** | 92.86% |
| **Macro Precision** | 93.75% |
| **Macro Recall** | 92.86% |
| **Macro F1-Score** | 92.82% |
| **ROC-AUC Score** | 1.0000 |
| **Average Precision (PR-AUC)** | 1.0000 |
| **Training Duration** | 0.00 seconds |
| **Inference Latency** | 0.0000 seconds |

### Confusion Matrix
| Actual \ Predicted | Pred FAKE | Pred REAL |
| :--- | :---: | :---: |
| **FAKE** | 6 | 1 |
| **REAL** | 0 | 7 |

## 6. Methodological Limitations & Responsible Use
1. **Stylistic Proxy Limitation:** The classifier relies on lexical frequencies and syntactic markers characteristic of the training corpus; it cannot perform real-time external fact verification against verified factual repositories.
2. **Domain & Temporal Shift:** Language patterns in political commentary and disinformation campaigns evolve rapidly. A model trained on historical data may exhibit degraded performance when applied to novel rhetorical topics or unseen publishing domains.
3. **Decision Support Only:** This application is intended strictly as a research platform and decision-support tool. Predictions should never be treated as definitive or legal determinations of falsity.

*(Report generated automatically by TruthLens AI Research Engine)*