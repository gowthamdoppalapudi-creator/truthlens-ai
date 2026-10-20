"""
explainability.py — Model interpretability and feature importance analysis.
Extracts global top positive/negative coefficients for linear models,
and per-prediction word contributions with scientific disclaimers.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib

from src.model_registry import get_model
from src.config import PLOTLY_LAYOUT, COLOR_REAL, COLOR_FAKE, COLOR_ACCENT
from src.preprocessing import preprocess_text


def get_global_feature_importance(model_id: str, top_n: int = 20) -> dict[str, Any]:
    """
    Extracts the highest-weighted positive and negative vocabulary features
    from compatible linear classifiers (LinearSVC, LogisticRegression, etc.).
    """
    model_entry = get_model(model_id)
    if not model_entry:
        return {"status": "error", "message": f"Model '{model_id}' not found."}

    pipeline_path = model_entry.get("pipeline_path")
    if not pipeline_path and model_entry.get("type") == "legacy":
        # Handle legacy separate model and vectorizer
        try:
            clf = joblib.load(model_entry["model_path"])
            vec = joblib.load(model_entry["vec_path"])
        except Exception as e:
            return {"status": "error", "message": f"Could not load legacy model: {e}"}
    else:
        try:
            bundle = joblib.load(pipeline_path)
            pipeline = bundle["pipeline"]
            vec = pipeline.named_steps["tfidf"]
            clf = pipeline.named_steps["clf"]
            # Handle calibrated classifier
            if hasattr(clf, "calibrated_classifiers_"):
                clf = clf.calibrated_classifiers_[0].estimator
        except Exception as e:
            return {"status": "error", "message": f"Could not extract steps from pipeline: {e}"}

    if hasattr(clf, "coef_"):
        coef = clf.coef_
    elif hasattr(clf, "feature_log_prob_") and len(clf.feature_log_prob_) >= 2:
        coef = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    elif hasattr(clf, "feature_importances_"):
        coef = clf.feature_importances_
    else:
        return {
            "status": "unsupported",
            "message": f"Model type '{type(clf).__name__}' does not expose linear coefficients or feature importance weights.",
        }

    feature_names = np.array(vec.get_feature_names_out())
    if hasattr(coef, "ndim") and coef.ndim > 1 and coef.shape[0] == 1:
        coef = coef[0]
    elif hasattr(coef, "ndim") and coef.ndim > 1:
        # Multiclass: take first class vs rest
        coef = coef[0]

    # Sort indices
    top_pos_idx = np.argsort(coef)[-top_n:][::-1]
    top_neg_idx = np.argsort(coef)[:top_n]

    classes = model_entry.get("classes", ["Class 0", "Class 1"])
    pos_label = classes[-1] if len(classes) >= 2 else "Positive Class"
    neg_label = classes[0] if len(classes) >= 2 else "Negative Class"

    pos_features = pd.DataFrame({
        "feature": feature_names[top_pos_idx],
        "weight": coef[top_pos_idx],
        "favors_class": pos_label,
    })
    neg_features = pd.DataFrame({
        "feature": feature_names[top_neg_idx],
        "weight": coef[top_neg_idx],
        "favors_class": neg_label,
    })

    # Combined bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=pos_features["feature"][::-1],
        x=pos_features["weight"][::-1],
        orientation="h",
        name=f"Favors {pos_label}",
        marker_color=COLOR_REAL if "REAL" in str(pos_label).upper() else COLOR_ACCENT,
    ))
    fig.add_trace(go.Bar(
        y=neg_features["feature"][::-1],
        x=neg_features["weight"][::-1],
        orientation="h",
        name=f"Favors {neg_label}",
        marker_color=COLOR_FAKE if "FAKE" in str(neg_label).upper() else "#f59e0b",
    ))
    fig.update_layout(
        title=f"Top Influential Features ({model_entry.get('name', model_id)})",
        xaxis_title="Linear Coefficient Weight",
        yaxis_title="TF-IDF N-gram Feature",
        barmode="relative",
        **PLOTLY_LAYOUT,
        height=520,
    )

    return {
        "status": "success",
        "figure": fig,
        "pos_features": pos_features,
        "neg_features": neg_features,
        "pos_label": pos_label,
        "neg_label": neg_label,
    }


def explain_single_prediction(text: str, model_id: str, top_k: int = 10) -> dict[str, Any]:
    """
    Examines which words in the user's input matched high-coefficient TF-IDF features.
    """
    model_entry = get_model(model_id)
    if not model_entry:
        return {"status": "error", "message": "Model not found."}

    if model_entry.get("type") == "legacy":
        clf = joblib.load(model_entry["model_path"])
        vec = joblib.load(model_entry["vec_path"])
    else:
        bundle = joblib.load(model_entry["pipeline_path"])
        pipeline = bundle["pipeline"]
        vec = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        if hasattr(clf, "calibrated_classifiers_"):
            clf = clf.calibrated_classifiers_[0].estimator

    if hasattr(clf, "coef_"):
        coef = clf.coef_
    elif hasattr(clf, "feature_log_prob_") and len(clf.feature_log_prob_) >= 2:
        coef = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    elif hasattr(clf, "feature_importances_"):
        coef = clf.feature_importances_
    else:
        return {"status": "unsupported", "message": "Model does not support coefficient inspection."}

    # Vectorize input
    cleaned_text = preprocess_text(text)
    x_vec = vec.transform([cleaned_text])
    
    feature_names = np.array(vec.get_feature_names_out())
    if hasattr(coef, "ndim") and coef.ndim > 1:
        coef = coef[0]
    
    # Non-zero indices for this text
    row_indices = x_vec.indices
    row_data = x_vec.data
    
    if len(row_indices) == 0:
        return {
            "status": "no_features",
            "message": "None of the vocabulary terms from this text matched the model's vocabulary.",
        }

    # Contribution = tfidf_value * coef_weight
    contributions = row_data * coef[row_indices]
    words = feature_names[row_indices]

    df_contrib = pd.DataFrame({
        "term": words,
        "tfidf_score": np.round(row_data, 4),
        "model_weight": np.round(coef[row_indices], 4),
        "contribution": np.round(contributions, 4),
    }).sort_values(by="contribution", ascending=False)

    classes = model_entry.get("classes", ["Class 0", "Class 1"])
    pos_class = classes[-1] if len(classes) >= 2 else "Positive"
    neg_class = classes[0] if len(classes) >= 2 else "Negative"

    df_contrib["direction"] = [
        f"Pushes toward {pos_class}" if c > 0 else f"Pushes toward {neg_class}"
        for c in df_contrib["contribution"]
    ]

    return {
        "status": "success",
        "contributions_df": df_contrib.head(top_k * 2),
        "total_active_features": len(row_indices),
        "cleaned_text": cleaned_text,
    }
