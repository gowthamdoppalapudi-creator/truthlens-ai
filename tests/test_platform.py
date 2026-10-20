"""
tests/test_platform.py — Automated quality assurance suite for TruthLens AI.
Validates all machine learning pipelines, evaluation metrics, dataset validation,
early detection experiments, explainability, and report generation.
"""
from __future__ import annotations
import pytest
import numpy as np
import pandas as pd

from src.data_loader import create_sample_research_dataset
from src.data_validation import validate_dataset, clean_dataset
from src.training import (
    build_tfidf_config,
    train_model,
    ALGORITHM_REGISTRY,
    save_pipeline,
)
from src.evaluation import compute_metrics
from src.model_registry import list_models, get_model, register_model, delete_model
from src.prediction import predict_single_text, predict_batch_dataframe
from src.early_detection import run_early_detection_experiment, truncate_text_to_percentage
from src.explainability import get_global_feature_importance, explain_single_prediction
from src.report_generator import generate_markdown_report, save_research_report


@pytest.fixture
def sample_df():
    """Provides the built-in 40-article benchmark dataset."""
    return create_sample_research_dataset()


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 1: Dataset Validation & Cleaning
# ──────────────────────────────────────────────────────────────────────────────
def test_dataset_validation_valid(sample_df):
    report = validate_dataset(
        df=sample_df,
        text_col="text",
        label_col="label",
        label_mapping={"REAL": "REAL", "FAKE": "FAKE"},
        min_samples_per_class=10,
    )
    assert report.is_valid is True
    assert report.total_rows == 40
    assert report.duplicate_rows == 0
    assert len(report.class_counts) == 2


def test_dataset_validation_missing_and_corrupt():
    corrupt_df = pd.DataFrame({
        "text": ["Valid article text here", "", np.nan, "Too short", "Real text 1234567890"],
        "label": ["REAL", "FAKE", "FAKE", "UNKNOWN", "REAL"],
    })
    report = validate_dataset(
        df=corrupt_df,
        text_col="text",
        label_col="label",
        label_mapping={"REAL": "REAL", "FAKE": "FAKE"},
        min_samples_per_class=1,
    )
    assert report.missing_text == 1
    assert report.empty_text == 1
    assert report.short_text >= 1
    assert report.invalid_labels >= 1


def test_dataset_cleaning(sample_df):
    # Add a duplicate and a short text
    dirty_df = pd.concat([
        sample_df,
        pd.DataFrame([{"text": sample_df["text"].iloc[0], "label": "REAL"},
                      {"text": "tiny", "label": "FAKE"}]),
    ], ignore_index=True)

    cleaned = clean_dataset(
        df=dirty_df,
        text_col="text",
        label_col="label",
        label_mapping={"REAL": "REAL", "FAKE": "FAKE"},
        drop_duplicates=True,
        drop_short=True,
        short_threshold=10,
    )
    assert len(cleaned) == 40  # Duplicates and tiny text safely dropped
    assert cleaned["text"].iloc[0] == sample_df["text"].iloc[0]


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 2: Machine Learning Pipeline Training (Zero-Leakage)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("algo", ["LinearSVC (recommended)", "Logistic Regression", "Multinomial Naive Bayes"])
def test_train_model_algorithms(sample_df, algo):
    tfidf_cfg = build_tfidf_config(max_features=1000)
    res = train_model(
        df=sample_df,
        text_col="text",
        label_col="label",
        algorithm_name=algo,
        tfidf_config=tfidf_cfg,
        test_size=0.25,
        random_seed=42,
    )
    assert "pipeline" in res
    assert len(res["X_train"]) == 30
    assert len(res["X_test"]) == 10
    assert len(res["y_pred"]) == 10
    assert res["training_time"] > 0


def test_train_cross_validation(sample_df):
    tfidf_cfg = build_tfidf_config(max_features=500)
    res = train_model(
        df=sample_df,
        text_col="text",
        label_col="label",
        algorithm_name="Logistic Regression",
        tfidf_config=tfidf_cfg,
        test_size=0.25,
        random_seed=42,
        cv_folds=3,
    )
    assert "cv_results" in res
    assert "cv_accuracy_mean" in res["cv_results"]
    assert 0.0 <= res["cv_results"]["cv_accuracy_mean"] <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 3: Empirical Metrics, ROC-AUC, and Edge Cases
# ──────────────────────────────────────────────────────────────────────────────
def test_compute_metrics_standard(sample_df):
    res = train_model(
        df=sample_df,
        text_col="text",
        label_col="label",
        algorithm_name="Logistic Regression",
        tfidf_config=build_tfidf_config(max_features=500),
        test_size=0.25,
        random_seed=42,
    )
    metrics = compute_metrics(
        y_true=res["y_test"],
        y_pred=res["y_pred"],
        pipeline=res["pipeline"],
        X_test=res["X_test"],
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert "confusion_matrix" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_single_class_test_set_handling():
    """Ensure ROC does not crash when test split has only 1 class."""
    y_true = np.array(["REAL", "REAL", "REAL", "REAL"])
    y_pred = np.array(["REAL", "REAL", "REAL", "REAL"])
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    assert metrics["accuracy"] == 1.0
    # ROC should not fail or raise exception
    assert "roc_auc" not in metrics or metrics.get("roc_auc") is None


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 4: Model Registry, Persistence & Inference
# ──────────────────────────────────────────────────────────────────────────────
def test_model_registration_and_prediction(sample_df):
    res = train_model(
        df=sample_df,
        text_col="text",
        label_col="label",
        algorithm_name="Logistic Regression",
        tfidf_config=build_tfidf_config(max_features=500),
        test_size=0.25,
        random_seed=42,
    )
    pipe_path = save_pipeline(res["pipeline"], {}, "test_checkpoint")
    model_id = register_model(
        name="Test Regression Checkpoint",
        pipeline_path=pipe_path,
        metadata={
            "experiment_id": res["experiment_id"],
            "algorithm_name": "Logistic Regression",
            "classes": res["classes"],
            "metrics": {"accuracy": 0.95},
        },
    )

    models = list_models()
    assert any(m["id"] == model_id for m in models)

    # Test single prediction
    single_res = predict_single_text("NASA scientists discovered a new exoplanet atmosphere.", model_id)
    assert single_res["status"] == "success"
    assert "verdict" in single_res
    assert single_res["confidence"] is not None

    # Test batch prediction
    batch_out = predict_batch_dataframe(sample_df.head(5), text_col="text", model_id=model_id)
    assert len(batch_out) == 5
    assert "prediction_verdict" in batch_out.columns

    # Clean up test artifact
    delete_model(model_id, delete_file=True)


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 5: Early Detection Experiments
# ──────────────────────────────────────────────────────────────────────────────
def test_text_truncation():
    text = "one two three four five six seven eight nine ten"
    truncated_50 = truncate_text_to_percentage(text, 50)
    assert len(truncated_50.split()) == 5
    truncated_10 = truncate_text_to_percentage(text, 10, min_words=3)
    assert len(truncated_10.split()) >= 3


def test_early_detection_experiment_a_and_b(sample_df):
    # Experiment A
    res_a = run_early_detection_experiment(
        df=sample_df,
        text_col="text",
        label_col="label",
        experiment_type="A",
        stages=[25, 100],
        algorithm_name="Logistic Regression",
        test_size=0.25,
        random_seed=42,
    )
    assert len(res_a["summary_df"]) == 2
    assert "accuracy" in res_a["summary_df"].columns

    # Experiment B
    res_b = run_early_detection_experiment(
        df=sample_df,
        text_col="text",
        label_col="label",
        experiment_type="B",
        stages=[25, 100],
        algorithm_name="Logistic Regression",
        test_size=0.25,
        random_seed=42,
    )
    assert len(res_b["summary_df"]) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 6: Explainability & Reports
# ──────────────────────────────────────────────────────────────────────────────
def test_explainability(sample_df):
    res = train_model(
        df=sample_df,
        text_col="text",
        label_col="label",
        algorithm_name="Logistic Regression",
        tfidf_config=build_tfidf_config(max_features=500),
        test_size=0.25,
        random_seed=42,
    )
    pipe_path = save_pipeline(res["pipeline"], {}, "test_expl_checkpoint")
    model_id = register_model(
        name="Test Explain Checkpoint",
        pipeline_path=pipe_path,
        metadata={"experiment_id": res["experiment_id"], "algorithm_name": "Logistic Regression", "classes": res["classes"]},
    )

    global_res = get_global_feature_importance(model_id, top_n=5)
    assert global_res["status"] == "success"
    assert len(global_res["pos_features"]) > 0

    local_res = explain_single_prediction("NASA confirms atmospheric water vapor exoplanet", model_id)
    assert local_res["status"] in ("success", "no_features")

    # Clean up test artifact
    delete_model(model_id, delete_file=True)


def test_report_generation():
    pkg = {
        "experiment_id": "test_exp_001",
        "algorithm_name": "LinearSVC",
        "metrics": {"accuracy": 0.985, "f1_macro": 0.984, "roc_auc": 0.998},
        "tfidf_config": {"ngram_range": (1, 2), "max_features": 50000},
        "dataset_info": {"name": "Synthetic Corpus", "total_rows": 1000},
        "n_train": 800,
        "n_test": 200,
        "text_col": "text",
        "label_col": "label",
    }
    md_content = generate_markdown_report(pkg)
    assert "Research Experiment Report" in md_content
    assert "test_exp_001" in md_content
    assert "98.50%" in md_content

    md_path, json_path = save_research_report(pkg, "test_report")
    assert md_path.exists()
    assert json_path.exists()
