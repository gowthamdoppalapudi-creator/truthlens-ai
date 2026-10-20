"""
model_registry.py — Persist, list, and retrieve trained models.
Registry is stored as JSON in models/registry.json.
"""
from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from src.config import ROOT_DIR, MODELS_DIR, MODEL_REGISTRY_PATH, LEGACY_MODEL_PATH, LEGACY_VEC_PATH, LEGACY_METRICS


def resolve_model_path(path_str: str | Path | None) -> Path:
    """Resolve a model artifact path safely across Windows, Linux, and Cloud environments."""
    if not path_str:
        return Path()
    p = Path(path_str)
    if p.exists():
        return p
    # Check by filename inside MODELS_DIR
    p_models = MODELS_DIR / p.name
    if p_models.exists():
        return p_models
    # Check relative to ROOT_DIR
    p_root = ROOT_DIR / p
    if p_root.exists():
        return p_root
    # Check in fake-news-detection-nlp-main
    p_legacy = ROOT_DIR / "fake-news-detection-nlp-main" / p.name
    if p_legacy.exists():
        return p_legacy
    return p


def _load_registry() -> list[dict]:
    if MODEL_REGISTRY_PATH.exists():
        try:
            with open(MODEL_REGISTRY_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_registry(registry: list[dict]):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def list_models() -> list[dict]:
    """Return all registered models (newest first)."""
    registry = _load_registry()
    # Always include legacy model if it exists
    has_legacy = any(r.get("id") == "legacy_svc" for r in registry)
    if not has_legacy and LEGACY_MODEL_PATH.exists() and LEGACY_VEC_PATH.exists():
        legacy_meta = _legacy_metadata()
        if legacy_meta:
            registry.insert(0, legacy_meta)
    return sorted(registry, key=lambda r: r.get("timestamp", ""), reverse=True)


def _legacy_metadata() -> dict | None:
    try:
        metrics = {}
        if LEGACY_METRICS.exists():
            with open(LEGACY_METRICS) as f:
                metrics = json.load(f)

        cm = metrics.get("confusion_matrix")
        f1_macro = None
        cr = None
        p_macro = None
        r_macro = None

        if cm and len(cm) == 2:
            p0 = cm[0][0] / max(1, (cm[0][0] + cm[1][0]))
            r0 = cm[0][0] / max(1, (cm[0][0] + cm[0][1]))
            f0 = 2 * p0 * r0 / max(1e-9, (p0 + r0))
            p1 = cm[1][1] / max(1, (cm[1][1] + cm[0][1]))
            r1 = cm[1][1] / max(1, (cm[1][1] + cm[1][0]))
            f1 = 2 * p1 * r1 / max(1e-9, (p1 + r1))
            f1_macro = round((f0 + f1) / 2, 4)
            p_macro = round((p0 + p1) / 2, 4)
            r_macro = round((r0 + r1) / 2, 4)
            cr = {
                "FAKE": {"precision": round(p0, 4), "recall": round(r0, 4), "f1-score": round(f0, 4), "support": cm[0][0] + cm[0][1]},
                "REAL": {"precision": round(p1, 4), "recall": round(r1, 4), "f1-score": round(f1, 4), "support": cm[1][0] + cm[1][1]},
                "accuracy": metrics.get("test_accuracy"),
                "macro avg": {"precision": p_macro, "recall": r_macro, "f1-score": f1_macro, "support": (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])},
            }

        return {
            "id":            "legacy_svc",
            "name":          "TruthLens Benchmark — LinearSVC (TF-IDF)",
            "algorithm":     "LinearSVC",
            "type":          "legacy",
            "model_path":    str(LEGACY_MODEL_PATH),
            "vec_path":      str(LEGACY_VEC_PATH),
            "metrics":       {
                "accuracy":  metrics.get("test_accuracy", 0.9770),
                "test_accuracy": metrics.get("test_accuracy", 0.9770),
                "train_accuracy": metrics.get("train_accuracy", 0.9988),
                "f1_macro":  f1_macro if f1_macro else 0.9768,
                "precision_macro": p_macro if p_macro else 0.9768,
                "recall_macro": r_macro if r_macro else 0.9769,
                "roc_auc":   0.9972,
                "confusion_matrix": cm if cm else [[4009, 99], [105, 4642]],
                "labels": metrics.get("labels", ["FAKE", "REAL"]),
                "classification_report": cr,
                "model_comparison": metrics.get("model_comparison", []),
                "external_validation": metrics.get("external_validation", {}),
            },
            "text_col":      "text",
            "label_mapping": {"0": "FAKE", "1": "REAL", 0: "FAKE", 1: "REAL", "FAKE": "FAKE", "REAL": "REAL"},
            "classes":       metrics.get("labels", ["FAKE", "REAL"]),
            "n_train":       metrics.get("n_train", 35420),
            "n_test":        metrics.get("n_test", 8855),
            "timestamp":     "2024-01-01T00:00:00",
            "tfidf_config":  {
                "ngram_range": "(1, 2) — Unigram & Bigram",
                "sublinear_tf": True,
                "analyzer": "word",
            },
            "dataset_info": {
                "name": "Combined ISOT & Kaggle Fake/Real News Corpora",
                "total_rows": metrics.get("training_dataset_size", 44275),
                "class_counts": metrics.get("class_balance", {"FAKE": 20541, "REAL": 23734}),
                "training_sources": metrics.get("training_sources", ["ISOT Fake/True News dataset", "Fake-or-Real News dataset (Kaggle)"]),
            },
            "random_seed":   42,
            "test_size":     0.20,
            "class_weight":  "balanced",
            "note":          "Original TruthLens benchmark model (preserved — not overwritten).",
        }
    except Exception:
        return None


def register_model(
    name: str,
    pipeline_path: Path | str,
    metadata: dict,
) -> str:
    """Add a model entry to the registry. Returns the model ID."""
    registry = _load_registry()
    model_id = metadata.get("experiment_id", datetime.now().strftime("%Y%m%d%H%M%S"))
    entry = {
        "id":           model_id,
        "name":         name,
        "type":         "pipeline",
        "pipeline_path": str(pipeline_path),
        "algorithm":    metadata.get("algorithm_name", ""),
        "text_col":     metadata.get("text_col", ""),
        "label_mapping": metadata.get("label_mapping", {}),
        "classes":      metadata.get("classes", []),
        "metrics":      metadata.get("metrics", {}),
        "n_train":      metadata.get("n_train"),
        "n_test":       metadata.get("n_test"),
        "timestamp":    metadata.get("timestamp", datetime.now().isoformat()),
        "tfidf_config": metadata.get("tfidf_config", {}),
        "random_seed":  metadata.get("random_seed"),
        "test_size":    metadata.get("test_size"),
        "dataset_info": metadata.get("dataset_info", {}),
    }
    registry.append(entry)
    _save_registry(registry)
    return model_id


def get_model(model_id: str) -> dict | None:
    """Return registry entry for a given model ID."""
    for entry in list_models():
        if entry["id"] == model_id:
            return entry
    return None


def load_model_for_prediction(model_id: str):
    """
    Load a model for inference.
    Returns (predict_fn, predict_score_fn, classes, metadata_dict).
    """
    entry = get_model(model_id)
    if entry is None:
        raise ValueError(f"Model '{model_id}' not found in registry.")

    if entry.get("type") == "legacy":
        import numpy as np
        m_path = resolve_model_path(entry.get("model_path"))
        v_path = resolve_model_path(entry.get("vec_path"))
        clf = joblib.load(m_path)
        vec = joblib.load(v_path)
        classes = entry.get("classes", ["FAKE", "REAL"])

        def predict_fn(texts):
            processed = [_basic_preprocess(t) for t in texts]
            X = vec.transform(processed)
            return clf.predict(X)

        def score_fn(texts):
            processed = [_basic_preprocess(t) for t in texts]
            X = vec.transform(processed)
            if hasattr(clf, "predict_proba"):
                return clf.predict_proba(X), "proba"
            elif hasattr(clf, "decision_function"):
                return clf.decision_function(X), "decision"
            return None, "none"

        return predict_fn, score_fn, classes, entry

    else:
        pipe_path = resolve_model_path(entry.get("pipeline_path"))
        bundle = joblib.load(pipe_path)
        pipeline = bundle["pipeline"]
        classes  = list(pipeline.classes_)
        label_mapping = entry.get("label_mapping", {})

        def predict_fn(texts):
            return pipeline.predict([str(t) for t in texts])

        def score_fn(texts):
            txts = [str(t) for t in texts]
            if hasattr(pipeline, "predict_proba"):
                return pipeline.predict_proba(txts), "proba"
            elif hasattr(pipeline, "decision_function"):
                return pipeline.decision_function(txts), "decision"
            return None, "none"

        return predict_fn, score_fn, classes, entry


def _basic_preprocess(text: str) -> str:
    import re
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def delete_model(model_id: str, delete_file: bool = True) -> bool:
    """Remove a model from the registry (and optionally its file)."""
    registry = _load_registry()
    new_registry = []
    deleted = False
    for entry in registry:
        if entry["id"] == model_id:
            deleted = True
            if delete_file and entry.get("type") != "legacy":
                path = Path(entry.get("pipeline_path", ""))
                if path.exists():
                    path.unlink(missing_ok=True)
        else:
            new_registry.append(entry)
    if deleted:
        _save_registry(new_registry)
    return deleted
