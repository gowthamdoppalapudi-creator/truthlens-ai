"""
pages/5_Model_Training.py — Machine Learning Training Studio.
Constructs leakage-free NLP classification pipelines, trains models, executes cross-validation,
and registers trained models with full metadata.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd

from src.config import APP_NAME
from src.utils import inject_css, page_header, stat_card, badge_row, footer
from src.data_loader import get_session_dataset
from src.training import (
    ALGORITHM_REGISTRY,
    build_tfidf_config,
    train_model,
    save_pipeline,
)
from src.evaluation import compute_metrics
from src.model_registry import register_model


st.set_page_config(
    page_title=f"Training Studio | {APP_NAME}",
    page_icon="⚙️",
    layout="wide",
)
inject_css()

page_header(
    "Machine Learning Training Studio",
    "Configure, train, and validate leak-free text classification pipelines on your dataset.",
)
badge_row([
    "Leakage-Free Pipelines",
    "TF-IDF Vectorization",
    "Stratified Splitting",
    "Model Registry Tracking",
])

# Check dataset
df = get_session_dataset()
if df is None:
    st.warning("⚠️ No active dataset found. Please load or upload a dataset in the Dataset Manager.")
    if st.button("📁 Go to Dataset Manager", type="primary"):
        st.switch_page("pages/3_Dataset_Manager.py")
    st.stop()

meta = st.session_state.get("active_df_meta", {})
text_col = meta.get("text_column", "text" if "text" in df.columns else df.columns[0])
label_col = meta.get("label_column", "label" if "label" in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0]))

st.info(f"📂 Training Corpus: **{st.session_state.get('active_df_name', 'dataset.csv')}** ({len(df):,} rows). Features: `{text_col}`, Label: `{label_col}`.")

# ──────────────────────────────────────────────────────────────────────────────
# Training Hyperparameter Configuration Form
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 1. Algorithm Selection")
algo_col1, algo_col2 = st.columns([1, 1])

with algo_col1:
    selected_algo = st.selectbox(
        "Classification Algorithm",
        options=list(ALGORITHM_REGISTRY.keys()),
        index=0,
    )
with algo_col2:
    algo_info = ALGORITHM_REGISTRY[selected_algo]
    st.caption(f"**Description:** {algo_info['description']}")
    st.caption(f"**Supports Native Probabilities:** {'Yes' if algo_info['supports_proba'] else 'No (Calibrated probability estimator applied)'}")

st.markdown("---")
st.markdown("### 2. Feature Extraction (TF-IDF Vectorization)")
tf_c1, tf_c2, tf_c3 = st.columns(3)

with tf_c1:
    ngram_mode = st.selectbox(
        "N-Gram Range",
        options=["(1, 1) — Unigrams only", "(1, 2) — Unigrams & Bigrams", "(1, 3) — Up to Trigrams"],
        index=1,
    )
    ngram_mapping = {
        "(1, 1) — Unigrams only": (1, 1),
        "(1, 2) — Unigrams & Bigrams": (1, 2),
        "(1, 3) — Up to Trigrams": (1, 3),
    }
    selected_ngram = ngram_mapping[ngram_mode]

with tf_c2:
    max_features = st.select_slider(
        "Max Feature Vocabulary Size",
        options=[5_000, 10_000, 25_000, 50_000, 75_000, 100_000],
        value=50_000,
    )

with tf_c3:
    sublinear_tf = st.checkbox("Apply Sublinear TF Scaling (log(TF))", value=True)

st.markdown("---")
st.markdown("### 3. Evaluation Split & Cross-Validation Protocol")
split_c1, split_c2, split_c3, split_c4 = st.columns(4)

with split_c1:
    test_size_pct = st.slider("Held-out Test Split (%)", min_value=10, max_value=40, value=20, step=5)
    test_size = test_size_pct / 100.0

with split_c2:
    random_seed = st.number_input("Random Reproducibility Seed", value=42, step=1)

with split_c3:
    class_weight_choice = st.selectbox("Class Weighting Strategy", ["balanced", "None"], index=0)
    class_weight = "balanced" if class_weight_choice == "balanced" else None

with split_c4:
    cv_folds = st.slider("Cross-Validation Folds (k)", min_value=0, max_value=10, value=0, help="0 disables K-Fold CV for faster single-split training.")

model_save_name = st.text_input("Checkpoint Model Name", value=f"TruthLens_{algo_info['key'].upper()}_V1")

# ──────────────────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🚀 Train Model Pipeline", type="primary", use_container_width=True):
    with st.status("Executing end-to-end training pipeline...", expanded=True) as status:
        st.write("1. Stratifying and partitioning corpus into train/test splits...")
        
        tfidf_cfg = build_tfidf_config(
            ngram_range=selected_ngram,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )

        st.write(f"2. Fitting TF-IDF vectorizer and training {selected_algo} on training partition...")
        results = train_model(
            df=df,
            text_col=text_col,
            label_col=label_col,
            algorithm_name=selected_algo,
            tfidf_config=tfidf_cfg,
            test_size=test_size,
            random_seed=int(random_seed),
            class_weight=class_weight,
            cv_folds=cv_folds,
        )

        st.write("3. Evaluating predictions strictly on held-out test split...")
        metrics = compute_metrics(
            y_true=results["y_test"],
            y_pred=results["y_pred"],
            pipeline=results["pipeline"],
            X_test=results["X_test"],
        )
        results["metrics"] = metrics

        st.write("4. Packaging pipeline bundle and persisting model checkpoint...")
        pipeline_path = save_pipeline(
            pipeline=results["pipeline"],
            metadata={
                "algorithm_name": selected_algo,
                "text_col": text_col,
                "label_col": label_col,
                "classes": results["classes"],
                "metrics": metrics,
                "n_train": results["n_train"],
                "n_test": results["n_test"],
                "random_seed": int(random_seed),
                "test_size": test_size,
                "tfidf_config": tfidf_cfg,
            },
            name=model_save_name,
        )

        model_id = register_model(
            name=model_save_name,
            pipeline_path=pipeline_path,
            metadata={
                "experiment_id": results["experiment_id"],
                "algorithm_name": selected_algo,
                "text_col": text_col,
                "classes": results["classes"],
                "metrics": metrics,
                "n_train": results["n_train"],
                "n_test": results["n_test"],
                "random_seed": int(random_seed),
                "test_size": test_size,
                "tfidf_config": tfidf_cfg,
                "dataset_info": {
                    "name": st.session_state.get("active_df_name", "dataset.csv"),
                    "total_rows": len(df),
                    "class_counts": df[label_col].value_counts().to_dict(),
                },
            },
        )

        st.session_state["active_model_id"] = model_id
        st.session_state["last_experiment"] = results
        status.update(label="✅ Training Complete & Checkpoint Registered!", state="complete")

    # Display Result Highlights
    st.success(f"Model **{model_save_name}** successfully trained and set as active model! (ID: `{model_id}`)")

    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
    with res_c1:
        stat_card(f"{metrics['accuracy']*100:.2f}%", "Held-out Test Accuracy", "stat-green")
    with res_c2:
        stat_card(f"{metrics['f1_macro']*100:.2f}%", "Macro F1-Score", "stat-accent")
    with res_c3:
        auc_str = f"{metrics['roc_auc']:.3f}" if "roc_auc" in metrics else "N/A"
        stat_card(auc_str, "Held-out ROC-AUC", "stat-purple")
    with res_c4:
        stat_card(f"{results['training_time']:.2f}s", "Training Duration", "stat-accent")

    st.markdown("---")
    eval_col, _ = st.columns([1, 2])
    with eval_col:
        if st.button("📊 View Full Metrics & Curves in Evaluation Studio", type="primary", use_container_width=True):
            st.switch_page("pages/6_Model_Evaluation.py")

st.markdown("---")
footer()
