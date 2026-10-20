"""
init_benchmark_models.py — Generates and registers benchmark models:
1. TruthLens — Logistic Regression (TF-IDF)
2. TruthLens — Multinomial Naive Bayes (TF-IDF)
3. TruthLens — Random Forest Classifier (TF-IDF)
4. TruthLens — Passive Aggressive Classifier (TF-IDF)
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.config import MODELS_DIR, DATA_DIR
from src.data_loader import create_sample_research_dataset
from src.training import (
    ALGORITHM_REGISTRY,
    build_tfidf_config,
    train_model,
    save_pipeline,
)
from src.evaluation import compute_metrics
from src.model_registry import register_model, list_models

# Expanded rich dataset to train realistic benchmark models
ADDITIONAL_REAL = [
    ("The European Space Agency confirmed successful deployment of the advanced atmospheric monitoring satellite into sun-synchronous orbit.", "REAL"),
    ("Federal health regulators released a comprehensive comparative review evaluating pediatric immunization safety profiles over twenty years.", "REAL"),
    ("The Ministry of Finance reported a slight reduction in sovereign debt yields following fiscal consolidation measures approved by parliament.", "REAL"),
    ("Researchers at Oxford and MIT published findings demonstrating a 40 percent efficiency boost in tandem perovskite silicon photovoltaic cells.", "REAL"),
    ("The National Oceanic and Atmospheric Administration issued its annual hurricane outlook noting near-average sea surface temperature anomalies.", "REAL"),
    ("City municipal councils ratified new zoning regulations designed to incentivize transit-oriented affordable residential developments.", "REAL"),
    ("The Department of Agriculture announced supplementary funding for rural irrigation modernization initiatives across nine drought-affected states.", "REAL"),
    ("Biopharmaceutical clinical trials showed statistically significant reduction in cardiovascular mortality among high-risk patient cohorts.", "REAL"),
    ("International nuclear inspectors concluded quarterly safeguards assessments with zero undeclared enrichment activity detected.", "REAL"),
    ("The Federal Aviation Administration completed a comprehensive review of autopilot software redundancy architectures.", "REAL"),
    ("The Central Statistics Bureau announced consumer price inflation dropped to 2.4 percent annualized in the previous fiscal quarter.", "REAL"),
    ("Automotive safety testing agencies awarded five-star collision mitigation ratings to multiple new electric vehicle platforms.", "REAL"),
    ("Forestry service biologists documented the natural recovery of native riparian plant communities following controlled seasonal burns.", "REAL"),
    ("Telecommunications consortium partners began trials on open radio access network infrastructure to reduce proprietary hardware dependence.", "REAL"),
    ("The World Bank approved developmental loans focused on regional wastewater treatment facilities and clean drinking water access.", "REAL"),
]

ADDITIONAL_FAKE = [
    ("URGENT ALERT: Secret military order authorizes immediate confiscation of all home vegetable gardens starting midnight tonight!", "FAKE"),
    ("SHOCKING PROOF: Drinking boiled copper coin tea permanently reprograms human DNA to resist government satellite thought rays!", "FAKE"),
    ("LEAKED AUDIO: Famous tech billionaire admits smart television remote controls emit secret frequencies that trigger consumer purchases!", "FAKE"),
    ("THE MEDIA IS TERRIFIED: Banned herbal root eliminates every known virus within three hours but doctors are forbidden to prescribe it!", "FAKE"),
    ("MAINSTREAM LIES: Underground civilization discovered beneath Grand Canyon using antigravity flying saucers to harvest ancient quartz crystals!", "FAKE"),
    ("SCANDAL OF THE CENTURY: Secret global summit votes to eliminate physical currency by Friday and replace all bank accounts with synthetic credits!", "FAKE"),
    ("CONFIDENTIAL PAPERS: Giant hidden magnetic wall constructed in the Antarctic ocean to prevent ships from reaching the secret ice kingdom!", "FAKE"),
    ("YOU MUST SHARE THIS: Smartphone chargers secretly install backdoor retinal scanning microphones directly into household wallpaper!", "FAKE"),
    ("EXPOSED AT LAST: Government admits clouds are artificial synthetic polymers sprayed by secret stealth drones to hide alien motherships!", "FAKE"),
    ("DOCTORS BAFFLED: Grandma cures all joint pain overnight using fermented pickle juice mixed with crushed charcoal and volcanic ash!", "FAKE"),
    ("GLOBAL COVER-UP: Ancient tablet proves dinosaurs were actually robotic vehicles operated by subterranean extraterrestrial explorers!", "FAKE"),
    ("SHOCKING ADMISSION: Secret society caught releasing robotic mosquitoes designed to inject microscopic tracking bar codes during sleep!", "FAKE"),
    ("BREAKING: World leaders sign covert treaty to make daylight illegal for two weeks every November to conduct dark ritual ceremonies!", "FAKE"),
    ("TERRIFYING TRUTH: Microwave tower radiation secretly alters tap water molecules into obedience liquid according to anonymous researcher!", "FAKE"),
    ("UNBELIEVABLE LEAK: Military base holds live dragon discovered frozen in Siberian glacier since 1740 that breathes plasma fire!", "FAKE"),
]


def create_expanded_dataset() -> pd.DataFrame:
    base = create_sample_research_dataset()
    extra_real = pd.DataFrame(ADDITIONAL_REAL, columns=["text", "label"])
    extra_fake = pd.DataFrame(ADDITIONAL_FAKE, columns=["text", "label"])
    combined = pd.concat([base, extra_real, extra_fake], ignore_index=True)
    return combined


def build_models():
    print("Building expanded dataset...")
    df = create_expanded_dataset()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_csv = DATA_DIR / "truthlens_research_corpus.csv"
    df.to_csv(dataset_csv, index=False)
    print(f"Saved expanded research corpus to {dataset_csv} ({len(df)} rows)")

    algorithms = [
        ("TruthLens — Logistic Regression (TF-IDF)", "Logistic Regression", "benchmark_logreg"),
        ("TruthLens — Multinomial Naive Bayes (TF-IDF)", "Multinomial Naive Bayes", "benchmark_mnb"),
        ("TruthLens — Random Forest Classifier (TF-IDF)", "Random Forest", "benchmark_rf"),
        ("TruthLens — Passive Aggressive Classifier (TF-IDF)", "Passive Aggressive Classifier", "benchmark_pac"),
    ]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, algo_name, exp_id in algorithms:
        print(f"\n--- Training {model_name} ({algo_name}) ---")
        tfidf_cfg = build_tfidf_config(ngram_range=(1, 2), min_df=1, max_df=0.95)

        results = train_model(
            df=df,
            text_col="text",
            label_col="label",
            algorithm_name=algo_name,
            tfidf_config=tfidf_cfg,
            test_size=0.20,
            random_seed=42,
            class_weight="balanced",
            cv_folds=3,
        )

        pipeline = results["pipeline"]

        # Predict score for ROC-AUC
        y_score = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(results["X_test"])
            y_score = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        elif hasattr(pipeline, "decision_function"):
            y_score = pipeline.decision_function(results["X_test"])

        metrics = compute_metrics(
            y_true=results["y_test"],
            y_pred=results["y_pred"],
            pipeline=pipeline,
            X_test=results["X_test"],
        )

        pipeline_path = save_pipeline(pipeline, results, model_name)
        print(f"Saved pipeline to: {pipeline_path}")

        metadata = {
            "experiment_id": exp_id,
            "algorithm_name": algo_name,
            "text_col": "text",
            "label_col": "label",
            "label_mapping": {"0": "FAKE", "1": "REAL", "FAKE": "FAKE", "REAL": "REAL"},
            "classes": results["classes"],
            "metrics": metrics,
            "n_train": results["n_train"],
            "n_test": results["n_test"],
            "random_seed": 42,
            "test_size": 0.20,
            "tfidf_config": tfidf_cfg,
            "dataset_info": {
                "name": "TruthLens Research Benchmark Corpus",
                "total_rows": len(df),
                "class_counts": df["label"].value_counts().to_dict(),
                "training_sources": ["TruthLens Curated Fact-Checking & Synthetic Benchmark Corpus"],
            },
        }

        registered_id = register_model(name=model_name, pipeline_path=pipeline_path, metadata=metadata)
        print(f"Registered model: {model_name} (ID: {registered_id})")

    print("\nAll models registered. Listing available models:")
    all_models = list_models()
    for m in all_models:
        print(f" - [{m['id']}] {m['name']} ({m.get('algorithm')}) | Acc: {m.get('metrics', {}).get('accuracy')}")


if __name__ == "__main__":
    build_models()
