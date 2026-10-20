"""
prediction.py — Prediction engine with score calibration, confidence semantics,
disclaimers, and batch processing.
"""
from __future__ import annotations
import time
from typing import Any
import numpy as np
import pandas as pd

from src.model_registry import load_model_for_prediction, list_models, get_model


DISCLAIMER_TEXT = (
    "This prediction is based on patterns learned from the training dataset. "
    "It does not independently verify the factual accuracy of the article. "
    "Verify important claims using reliable sources."
)


def predict_single_text(text: str, model_id: str) -> dict[str, Any]:
    """
    Run inference on a single piece of text.
    Handles empty/short text, score interpretation, and metadata reporting.
    """
    if not text or len(text.strip()) < 5:
        return {
            "status": "error",
            "message": "Input text is too short or empty. Please enter at least a few words.",
        }

    try:
        predict_fn, score_fn, classes, model_meta = load_model_for_prediction(model_id)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load model '{model_id}': {str(e)}",
        }

    t0 = time.time()
    preds = predict_fn([text])
    scores, score_type = score_fn([text])
    latency_ms = (time.time() - t0) * 1000

    raw_label = str(preds[0])
    label_mapping = model_meta.get("label_mapping", {})
    # Map if mapped
    resolved_label = label_mapping.get(raw_label, raw_label)
    if str(raw_label).upper() in ("1", "REAL"):
        verdict = "Likely Real"
        category = "REAL"
    elif str(raw_label).upper() in ("0", "FAKE"):
        verdict = "Likely Fake"
        category = "FAKE"
    else:
        verdict = f"Likely {resolved_label}"
        category = str(resolved_label).upper()

    confidence = None
    decision_margin = None
    interpretation = ""

    if score_type == "proba" and scores is not None:
        proba_arr = scores[0]
        try:
            class_idx = list(classes).index(raw_label)
            confidence = float(proba_arr[class_idx])
            interpretation = (
                f"Calibrated probability estimate: {confidence * 100:.1f}%. "
                "Reflects statistical confidence of stylistic patterns."
            )
        except Exception:
            confidence = float(np.max(proba_arr))
            interpretation = f"Estimated probability: {confidence * 100:.1f}%."

    elif score_type == "decision" and scores is not None:
        dec = scores[0]
        if isinstance(dec, (list, np.ndarray)):
            decision_margin = float(dec[0]) if len(dec) > 0 else 0.0
        else:
            decision_margin = float(dec)
        
        # Sigmoid squash for pseudo-confidence with clear documentation
        pseudo_conf = float(1.0 / (1.0 + np.exp(-abs(decision_margin))))
        confidence = pseudo_conf
        interpretation = (
            f"Linear boundary distance: {decision_margin:+.2f} (squashed confidence ~ {pseudo_conf*100:.1f}%). "
            "Note: Distance from separating hyperplane, not an empirical probability."
        )
    else:
        interpretation = "Categorical decision output without probability estimation."

    return {
        "status": "success",
        "verdict": verdict,
        "category": category,
        "raw_label": raw_label,
        "confidence": confidence,
        "decision_margin": decision_margin,
        "score_type": score_type,
        "interpretation": interpretation,
        "model_id": model_id,
        "model_name": model_meta.get("name", model_id),
        "algorithm": model_meta.get("algorithm", "Unknown"),
        "latency_ms": round(latency_ms, 2),
        "char_count": len(text),
        "word_count": len(text.split()),
        "disclaimer": DISCLAIMER_TEXT,
    }


def predict_batch_dataframe(
    df: pd.DataFrame,
    text_col: str,
    model_id: str,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Process batch prediction on a dataframe, adding predicted labels and scores
    while preserving all original columns.
    """
    predict_fn, score_fn, classes, model_meta = load_model_for_prediction(model_id)
    
    texts = df[text_col].fillna("").astype(str).tolist()
    total = len(texts)
    
    batch_size = 250
    all_preds = []
    all_confidences = []
    all_score_types = []
    
    t0 = time.time()
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        chunk = texts[start_idx:end_idx]
        
        preds_chunk = predict_fn(chunk)
        scores_chunk, stype = score_fn(chunk)
        
        for i, pred in enumerate(preds_chunk):
            all_preds.append(str(pred))
            all_score_types.append(stype)
            
            if stype == "proba" and scores_chunk is not None:
                p_arr = scores_chunk[i]
                try:
                    c_idx = list(classes).index(pred)
                    all_confidences.append(float(p_arr[c_idx]))
                except Exception:
                    all_confidences.append(float(np.max(p_arr)))
            elif stype == "decision" and scores_chunk is not None:
                dec = scores_chunk[i]
                val = float(dec[0]) if isinstance(dec, (list, np.ndarray)) else float(dec)
                p_conf = float(1.0 / (1.0 + np.exp(-abs(val))))
                all_confidences.append(p_conf)
            else:
                all_confidences.append(np.nan)
                
        if progress_callback:
            progress_callback(min(end_idx / total, 1.0))
            
    total_time = time.time() - t0
    
    result_df = df.copy()
    label_mapping = model_meta.get("label_mapping", {})
    
    result_df["predicted_label"] = [
        label_mapping.get(p, p) for p in all_preds
    ]
    result_df["prediction_verdict"] = [
        "Likely Fake" if str(p).upper() in ("0", "FAKE") else
        ("Likely Real" if str(p).upper() in ("1", "REAL") else f"Likely {p}")
        for p in result_df["predicted_label"]
    ]
    result_df["model_confidence"] = all_confidences
    result_df["model_id"] = model_id
    result_df["text_length"] = [len(t) for t in texts]
    result_df["word_count"] = [len(t.split()) for t in texts]
    
    # Store batch execution stats on dataframe attributes
    result_df.attrs["batch_time_seconds"] = round(total_time, 3)
    result_df.attrs["processed_count"] = total
    result_df.attrs["model_name"] = model_meta.get("name", model_id)
    
    return result_df
