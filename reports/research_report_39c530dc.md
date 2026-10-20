# Research Experiment Report: TruthLens AI
**An Intelligent Fake News Detection and Early Detection Research Platform**

- **Lead Researcher / Developer:** D.venkata gowtham
- **Experiment Identifier:** `39c530dc`
- **Timestamp:** 2026-10-19T13:56:32.222743
- **Execution Platform:** Python 3.10.11 | scikit-learn 1.7.2 | OS: Windows 10

---

## 1. Research Objective
Evaluate supervised linguistic pattern classification performance for fake and real news detection. The model isolates term-frequency inverse-document-frequency (TF-IDF) feature representations to discriminate between authentic news reporting and fabricated articles.

## 2. Dataset Metadata & Quality Assurance
- **Dataset Source/Name:** shuffled_data.csv
- **Total Samples:** 19,792
- **Training Samples ($N_{train}$):** 15,833
- **Held-Out Test Samples ($N_{test}$):** 3,959
- **Class Distribution:** `{'REAL': 9900, 'FAKE': 9892}`
- **Selected Text Column:** `Data`
- **Selected Target Label Column:** `label`
- **Validation Notes:** Fully split prior to vectorizer fitting to guarantee zero test leakage.

## 3. Modeling Architecture & Preprocessing Protocol
- **Classifier Algorithm:** LinearSVC (recommended)
- **Class Weighting Strategy:** `balanced`
- **Random Seed:** `42`
- **Test Set Ratio:** `0.2`
- **TF-IDF Vectorizer Hyperparameters:**
  - `ngram_range`: `[1, 2]`
  - `max_features`: `50000`
  - `min_df`: `2`
  - `max_df`: `0.95`
  - `sublinear_tf`: `True`
  - `strip_accents`: `unicode`
  - `analyzer`: `word`

## 4. Empirical Evaluation Results (Held-Out Test Set)
| Metric | Measured Score |
| :--- | :--- |
| **Overall Accuracy** | 89.67% |
| **Macro Precision** | 89.68% |
| **Macro Recall** | 89.67% |
| **Macro F1-Score** | 89.67% |
| **ROC-AUC Score** | 0.0367 |
| **Average Precision (PR-AUC)** | 0.9651 |
| **Training Duration** | 0.00 seconds |
| **Inference Latency** | 0.0000 seconds |

### Confusion Matrix
| Actual \ Predicted | Pred FAKE | Pred REAL |
| :--- | :---: | :---: |
| **FAKE** | 1759 | 220 |
| **REAL** | 189 | 1791 |

## 6. Methodological Limitations & Responsible Use
1. **Stylistic Proxy Limitation:** The classifier relies on lexical frequencies and syntactic markers characteristic of the training corpus; it cannot perform real-time external fact verification against verified factual repositories.
2. **Domain & Temporal Shift:** Language patterns in political commentary and disinformation campaigns evolve rapidly. A model trained on historical data may exhibit degraded performance when applied to novel rhetorical topics or unseen publishing domains.
3. **Decision Support Only:** This application is intended strictly as a research platform and decision-support tool. Predictions should never be treated as definitive or legal determinations of falsity.

*(Report generated automatically by TruthLens AI Research Engine)*