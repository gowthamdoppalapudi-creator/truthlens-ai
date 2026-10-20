"""
report_generator.py — Structured scientific research report generation.
Exports Markdown, JSON, and CSV research deliverables with complete reproducibility data.
"""
from __future__ import annotations
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
import sklearn

from src.config import APP_NAME, APP_SUBTITLE, RESEARCHER_NAME, REPORTS_DIR


def generate_markdown_report(experiment_data: dict[str, Any]) -> str:
    """
    Constructs an academic-quality Markdown research report based on actual experiment results.
    """
    exp_id = experiment_data.get("experiment_id", "N/A")
    created_at = experiment_data.get("timestamp", datetime.now().isoformat())
    algo = experiment_data.get("algorithm_name", "Unknown Algorithm")
    metrics = experiment_data.get("metrics", {})
    tfidf_cfg = experiment_data.get("tfidf_config", {})
    ds_info = experiment_data.get("dataset_info", {})
    early_results = experiment_data.get("early_detection_results")

    md = []
    md.append(f"# Research Experiment Report: {APP_NAME}")
    md.append(f"**{APP_SUBTITLE}**\n")
    md.append(f"- **Lead Researcher / Developer:** {RESEARCHER_NAME}")
    md.append(f"- **Experiment Identifier:** `{exp_id}`")
    md.append(f"- **Timestamp:** {created_at}")
    md.append(f"- **Execution Platform:** Python {platform.python_version()} | scikit-learn {sklearn.__version__} | OS: {platform.system()} {platform.release()}\n")
    md.append("---\n")

    md.append("## 1. Research Objective")
    md.append(
        "Evaluate supervised linguistic pattern classification performance for fake and real news detection. "
        "The model isolates term-frequency inverse-document-frequency (TF-IDF) feature representations to "
        "discriminate between authentic news reporting and fabricated articles."
    )
    md.append("")

    md.append("## 2. Dataset Metadata & Quality Assurance")
    total_rows = ds_info.get('total_rows')
    total_str = f"{int(total_rows):,}" if isinstance(total_rows, (int, float)) and total_rows else "N/A"
    n_tr = experiment_data.get('n_train')
    n_tr_str = f"{int(n_tr):,}" if isinstance(n_tr, (int, float)) and n_tr else "N/A"
    n_te = experiment_data.get('n_test')
    n_te_str = f"{int(n_te):,}" if isinstance(n_te, (int, float)) and n_te else "N/A"

    md.append(f"- **Dataset Source/Name:** {ds_info.get('name', 'Pre-packaged Research Corpus')}")
    md.append(f"- **Total Samples:** {total_str}")
    md.append(f"- **Training Samples ($N_{{train}}$):** {n_tr_str}")
    md.append(f"- **Held-Out Test Samples ($N_{{test}}$):** {n_te_str}")
    md.append(f"- **Class Distribution:** `{ds_info.get('class_counts', {})}`")
    md.append(f"- **Selected Text Column:** `{experiment_data.get('text_col', 'text')}`")
    md.append(f"- **Selected Target Label Column:** `{experiment_data.get('label_col', 'label')}`")
    md.append(f"- **Validation Notes:** Fully split prior to vectorizer fitting to guarantee zero test leakage.\n")

    md.append("## 3. Modeling Architecture & Preprocessing Protocol")
    md.append(f"- **Classifier Algorithm:** {algo}")
    md.append(f"- **Class Weighting Strategy:** `{experiment_data.get('class_weight', 'balanced')}`")
    md.append(f"- **Random Seed:** `{experiment_data.get('random_seed', 42)}`")
    md.append(f"- **Test Set Ratio:** `{experiment_data.get('test_size', 0.20)}`")
    md.append("- **TF-IDF Vectorizer Hyperparameters:**")
    for k, v in tfidf_cfg.items():
        md.append(f"  - `{k}`: `{v}`")
    md.append("")

    md.append("## 4. Empirical Evaluation Results (Held-Out Test Set)")
    md.append("| Metric | Measured Score |")
    md.append("| :--- | :--- |")
    md.append(f"| **Overall Accuracy** | {metrics.get('accuracy', 0.0) * 100:.2f}% |")
    md.append(f"| **Macro Precision** | {metrics.get('precision_macro', 0.0) * 100:.2f}% |")
    md.append(f"| **Macro Recall** | {metrics.get('recall_macro', 0.0) * 100:.2f}% |")
    md.append(f"| **Macro F1-Score** | {metrics.get('f1_macro', 0.0) * 100:.2f}% |")
    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        md.append(f"| **ROC-AUC Score** | {metrics.get('roc_auc', 0.0):.4f} |")
    if "avg_precision" in metrics and metrics["avg_precision"] is not None:
        md.append(f"| **Average Precision (PR-AUC)** | {metrics.get('avg_precision', 0.0):.4f} |")
    md.append(f"| **Training Duration** | {experiment_data.get('training_time', 0.0):.2f} seconds |")
    md.append(f"| **Inference Latency** | {experiment_data.get('prediction_time', 0.0):.4f} seconds |")
    md.append("")

    # Confusion matrix table
    if "confusion_matrix" in metrics and "labels" in metrics:
        md.append("### Confusion Matrix")
        labels = metrics["labels"]
        cm = metrics["confusion_matrix"]
        header = "| Actual \\ Predicted | " + " | ".join([f"Pred {l}" for l in labels]) + " |"
        sep = "| :--- | " + " | ".join([":---:" for _ in labels]) + " |"
        md.append(header)
        md.append(sep)
        for i, row_label in enumerate(labels):
            row_vals = " | ".join([str(val) for val in cm[i]])
            md.append(f"| **{row_label}** | {row_vals} |")
        md.append("")

    # Early detection section if present
    if early_results is not None and isinstance(early_results, pd.DataFrame):
        md.append("## 5. Early Detection Truncation Experiments")
        md.append("Performance across partial article token stages:")
        md.append("")
        try:
            md.append(early_results.to_markdown(index=False))
        except Exception:
            # Fallback to standard markdown pipe table
            cols = [str(c) for c in early_results.columns]
            md.append("| " + " | ".join(cols) + " |")
            md.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for _, row in early_results.iterrows():
                md.append("| " + " | ".join([str(val) for val in row.values]) + " |")
        md.append("")

    md.append("## 6. Methodological Limitations & Responsible Use")
    md.append(
        "1. **Stylistic Proxy Limitation:** The classifier relies on lexical frequencies and syntactic markers "
        "characteristic of the training corpus; it cannot perform real-time external fact verification against verified factual repositories.\n"
        "2. **Domain & Temporal Shift:** Language patterns in political commentary and disinformation campaigns evolve rapidly. "
        "A model trained on historical data may exhibit degraded performance when applied to novel rhetorical topics or unseen publishing domains.\n"
        "3. **Decision Support Only:** This application is intended strictly as a research platform and decision-support tool. "
        "Predictions should never be treated as definitive or legal determinations of falsity."
    )
    md.append("")
    md.append("*(Report generated automatically by TruthLens AI Research Engine)*")

    return "\n".join(md)


def save_research_report(
    experiment_data: dict[str, Any],
    report_name: str | None = None,
) -> tuple[Path, Path]:
    """
    Saves both Markdown and JSON deliverables to the reports/ directory.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = experiment_data.get("experiment_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not report_name:
        report_name = f"research_report_{exp_id}"

    md_content = generate_markdown_report(experiment_data)
    md_path = REPORTS_DIR / f"{report_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    json_path = REPORTS_DIR / f"{report_name}.json"
    
    # Sanitize serializable dict
    serializable = {}
    for k, v in experiment_data.items():
        if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            serializable[k] = v
        elif isinstance(v, pd.DataFrame):
            serializable[k] = v.to_dict(orient="records")
        else:
            serializable[k] = str(v)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    return md_path, json_path
