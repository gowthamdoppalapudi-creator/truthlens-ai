import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="TruthLens | Fake News Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #0b0d12; }
    h1, h2, h3 { font-family: 'Trebuchet MS', sans-serif; }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6ee7f9, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub { color: #9ca3af; font-size: 1.05rem; margin-top: 0.2rem; }

    .badge-row { display:flex; gap:0.5rem; flex-wrap:wrap; margin: 0.6rem 0 1.2rem 0; }
    .badge {
        background:#161a23; border:1px solid #2a2f3a; color:#9ca3af;
        padding:0.25rem 0.7rem; border-radius:999px; font-size:0.78rem;
    }

    .stat-card {
        background: linear-gradient(160deg, #151925, #10131b);
        border: 1px solid #262b36;
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        text-align: left;
    }
    .stat-value { font-size: 1.9rem; font-weight: 800; color: #e5e7eb; line-height:1.1; }
    .stat-label { color: #8b93a3; font-size: 0.82rem; margin-top: 0.25rem; }
    .stat-accent { color: #6ee7f9; }

    .metric-card {
        background: #161a23;
        border: 1px solid #262b36;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
    }
    .verdict-real {
        background: linear-gradient(135deg, #0f2e21, #123a2a);
        border: 1px solid #1f7a52;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #3a1414, #4a1717);
        border: 1px solid #a13636;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
    }
    .highlight-box {
        background: linear-gradient(160deg, #101b2e, #0d1420);
        border: 1px solid #23405f;
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin: 0.8rem 0 1.4rem 0;
    }
    .highlight-box h4 { margin-top:0; color:#8ecdf7; }

    .footer-note { color: #6b7280; font-size: 0.8rem; text-align: center; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# NLTK setup (downloads once, cached across reruns)
# --------------------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    return stopwords.words("english"), WordNetLemmatizer()


stop_words, lemma = setup_nltk()


def preprocess_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"http\S+|www\S+", "", txt)
    txt = re.sub(r"[^a-zA-Z0-9\s]", "", txt)
    txt = word_tokenize(txt)
    txt = [i for i in txt if i not in stop_words]
    txt = [lemma.lemmatize(i) for i in txt]
    return " ".join(txt)


# --------------------------------------------------------------------------------------
# Load artifacts
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = BASE_DIR / "final_svc_model.pkl"
    vec_path = BASE_DIR / "tfidf_vectorizer.pkl"
    metrics_path = BASE_DIR / "metrics.json"

    model, vectorizer, metrics = None, None, None
    errors = []

    try:
        model = joblib.load(model_path)
    except Exception as e:
        errors.append(f"Model file could not be loaded ({e.__class__.__name__}). "
                       f"It may have been saved/transferred incorrectly (not in pure binary mode).")

    try:
        vectorizer = joblib.load(vec_path)
    except Exception as e:
        errors.append(f"Vectorizer file could not be loaded ({e.__class__.__name__}). "
                       f"It may have been saved/transferred incorrectly (not in pure binary mode).")

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return model, vectorizer, metrics, errors


model, vectorizer, metrics, load_errors = load_artifacts()

# Map numeric label encodings back to the original string labels, if needed
LABEL_MAP = {0: "FAKE", 1: "REAL"}


def resolve_label(raw_label):
    if isinstance(raw_label, (int, np.integer)):
        return LABEL_MAP.get(int(raw_label), str(raw_label))
    label_str = str(raw_label).upper()
    if label_str in ("0", "1"):
        return LABEL_MAP[int(label_str)]
    return label_str


def predict(text: str):
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    raw_pred = model.predict(vec)[0]
    label = resolve_label(raw_pred)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        conf = float(proba[classes.index(raw_pred)])
    else:
        # LinearSVC has no predict_proba - use decision_function instead
        score = model.decision_function(vec)
        conf = float(1 / (1 + np.exp(-abs(score[0]))))
    return label, conf


# --------------------------------------------------------------------------------------
# Real example articles (pulled from the external validation set, news.csv)
# --------------------------------------------------------------------------------------
EXAMPLES = {
    "— pick an example or write your own —": "",
    "📰 Real: Obama & Guantanamo (Reuters)": (
        "Obama administration expects to close Guantanamo prison: White House "
        "WASHINGTON (Reuters) - U.S. President Barack Obama is still committed to closing "
        "the U.S. prison for terrorism suspects at Guantanamo Bay, Cuba, and his "
        "administration expects to complete the task, White House said on Thursday. "
        "\u201cWe\u2019re going to do our best to try to get this closed and it\u2019s our "
        "expectation that that\u2019s what we\u2019ll do,\u201d White House spokesman Josh "
        "Earnest said at a news briefing."
    ),
    "📰 Real: Putin on U.S. relations (Reuters)": (
        "Russia's Putin says Obama administration does not stick to any deals "
        "KRASNAYA POLYANA, Russia (Reuters) - Russian President Vladimir Putin said on "
        "Thursday it was hard for him to work with the current U.S. administration because "
        "it did not stick to any agreements, including on Syria. Putin said he was ready to "
        "engage with a new president however, whoever the American people chose, and to "
        "discuss any problem."
    ),
    "🚩 Fake: Sensational clickbait headline": (
        "HYSTERICAL! TRUMP TO GERMANY\u2019S MERKEL on Obama\u2019s Wiretapping: \u201cAt "
        "least we have something in common\u201d [VIDEO] Trump to Merkel on Obama's "
        "wiretapping \u2013 at least we have something in common \u2013 WikiLeaks shared the "
        "video which has since gone viral across social media, with commenters split on "
        "whether the remark was a joke."
    ),
    "🚩 Fake: Unverified tabloid-style report": (
        "Sweden: Neo-Nazis Clash With Anti-Fascists, Police At Yom Kippur Rally "
        "% of readers think this story is Fact. Add your two cents. (Before It's News) "
        "More than 30 people were arrested on Saturday as both neo-Nazis and anti-fascists "
        "clashed with police during a march by the extreme right-wing Nordic Resistance "
        "Movement (NMR) in the Swedish city of Gothenburg, police said."
    ),
}


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛰️ TruthLens")
    st.caption("NLP-powered fake news classifier")
    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Classifier", "📊 Model Performance", "ℹ️ About the Project"])
    st.markdown("---")
    st.markdown("**Built by** Khush")
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-View%20Code-181717?logo=github)](#)"
    )

if load_errors:
    for e in load_errors:
        st.error(e)
    st.stop()


def stat_card(value, label):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value stat-accent">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Page: Classifier
# --------------------------------------------------------------------------------------
if page == "🔍 Classifier":
    st.markdown('<p class="hero-title">TruthLens</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Paste a news article and let a trained NLP model judge '
        "whether it looks REAL or FAKE, based on patterns learned across three combined "
        "news datasets.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="badge-row">
            <span class="badge">TF-IDF + LinearSVC</span>
            <span class="badge">Trained on 44K+ articles</span>
            <span class="badge">Validated on 45.7K unseen articles</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics:
        ev = metrics.get("external_validation", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stat_card(f"{metrics['test_accuracy']*100:.1f}%", "Held-out test accuracy")
        with c2:
            stat_card(f"{ev.get('accuracy', 0)*100:.1f}%", "External validation accuracy")
        with c3:
            stat_card(f"{metrics.get('training_dataset_size', 0):,}", "Training articles")
        with c4:
            stat_card("45,757", "Unseen articles tested on")

    st.write("")
    col_main, col_side = st.columns([2, 1])

    with col_main:
        choice = st.selectbox("Try a real example", list(EXAMPLES.keys()))
        default_text = EXAMPLES[choice]

        user_text = st.text_area(
            "Article text",
            value=default_text,
            height=220,
            placeholder="Paste any news article or headline here...",
        )

        analyze = st.button("Analyze", type="primary", use_container_width=True)

    with col_side:
        st.markdown("#### How it works")
        st.markdown(
            """
1. Text is cleaned (lowercased, URLs and punctuation removed)
2. Tokenized, stopwords removed, and lemmatized
3. Converted into **TF-IDF** features
4. Fed into a trained **LinearSVC** classifier
5. Model outputs a label + confidence score
            """
        )
        st.caption(
            "The 4 examples above are real articles from the model's independent "
            "external validation set — not synthetic text — so the demo reflects "
            "genuine model behavior."
        )

    if analyze:
        if not user_text or len(user_text.strip()) < 10:
            st.warning("Please enter at least a sentence or two of article text.")
        else:
            with st.spinner("Analyzing..."):
                label, confidence = predict(user_text)

            st.write("")
            if label == "REAL":
                st.markdown(
                    f"""
                    <div class="verdict-real">
                        <h2 style="margin:0;">✅ Predicted: REAL</h2>
                        <p style="color:#a7f3d0; margin-top:0.3rem;">
                            Confidence: <b>{confidence*100:.1f}%</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-fake">
                        <h2 style="margin:0;">🚩 Predicted: FAKE</h2>
                        <p style="color:#fca5a5; margin-top:0.3rem;">
                            Confidence: <b>{confidence*100:.1f}%</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#22c55e" if label == "REAL" else "#ef4444"},
                        "steps": [
                            {"range": [0, 50], "color": "#1f2430"},
                            {"range": [50, 100], "color": "#262b36"},
                        ],
                    },
                    title={"text": "Model Confidence"},
                )
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e5e7eb"})
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "⚠️ This is a statistical text-pattern classifier trained on labeled "
                "news datasets — not a fact-checker. Always verify important claims "
                "against trusted sources."
            )

# --------------------------------------------------------------------------------------
# Page: Model performance
# --------------------------------------------------------------------------------------
elif page == "📊 Model Performance":
    st.markdown('<p class="hero-title">Model Performance</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Evaluated on a held-out test split, then re-checked on a '
        "completely independent dataset the model never trained on.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    if metrics:
        ev = metrics.get("external_validation", {})

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Held-out test accuracy", f"{metrics['test_accuracy']*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Training accuracy", f"{metrics['train_accuracy']*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("External validation accuracy", f"{ev.get('accuracy', 0)*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="highlight-box">
                <h4>🧪 Independent Generalization Test</h4>
                <p style="color:#c7d2e0; margin-bottom:0;">
                    Beyond the usual train/test split, the final model was evaluated on
                    <b>{ev.get('dataset', 'an external dataset')}</b> — {ev.get('description', '')}.
                    It scored <b>{ev.get('accuracy', 0)*100:.2f}%</b> accuracy on those
                    45,757 unseen articles, close to its {metrics['test_accuracy']*100:.2f}%
                    held-out test accuracy, indicating the model generalizes rather than
                    having just memorized its training data.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        cm_col1, cm_col2 = st.columns(2)
        labels = metrics["labels"]

        with cm_col1:
            st.markdown("#### Confusion Matrix — Held-out Test Set")
            cm = np.array(metrics["confusion_matrix"])
            fig = px.imshow(
                cm, text_auto=True,
                x=[f"Predicted {l}" for l in labels],
                y=[f"Actual {l}" for l in labels],
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e5e7eb"})
            st.plotly_chart(fig, use_container_width=True)

        with cm_col2:
            st.markdown("#### Confusion Matrix — External Validation")
            cm_ext = np.array(ev.get("confusion_matrix", [[0, 0], [0, 0]]))
            fig2 = px.imshow(
                cm_ext, text_auto=True,
                x=[f"Predicted {l}" for l in labels],
                y=[f"Actual {l}" for l in labels],
                color_continuous_scale="Purples",
            )
            fig2.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e5e7eb"})
            st.plotly_chart(fig2, use_container_width=True)

        st.write("")
        st.markdown("#### Model Comparison (during experimentation)")
        comp = metrics.get("model_comparison")
        if comp:
            comp_df = pd.DataFrame(comp).rename(columns={
                "model": "Model", "features": "Feature Extraction",
                "train_accuracy": "Train Accuracy", "test_accuracy": "Test Accuracy",
            })
            comp_df["Train Accuracy"] = (comp_df["Train Accuracy"] * 100).round(2).astype(str) + "%"
            comp_df["Test Accuracy"] = (comp_df["Test Accuracy"] * 100).round(2).astype(str) + "%"
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("#### Final Pipeline")
        st.code(
            f"Vectorizer: {metrics.get('vectorizer', 'TF-IDF')}\n"
            f"Model: {metrics.get('model_type', 'LinearSVC (C = 1)')}\n"
            f"Training data: {metrics.get('training_dataset_size', 0):,} articles "
            f"({', '.join(metrics.get('training_sources', []))})",
            language="text",
        )
    else:
        st.info("No metrics file found. Run the training script to generate metrics.json.")

# --------------------------------------------------------------------------------------
# Page: About
# --------------------------------------------------------------------------------------
else:
    st.markdown('<p class="hero-title">About this project</p>', unsafe_allow_html=True)
    st.write("")
    st.markdown(
        """
**TruthLens** is a fake news detection tool trained on three combined, publicly
available news datasets (ISOT Fake/Real News + the Kaggle Fake-or-Real News dataset),
totaling **44,275** cleaned, deduplicated articles.

Unlike a typical single-dataset project, the final model was also stress-tested on a
**fourth, completely independent dataset of 45,757 articles** it never trained on —
to check that it generalizes to genuinely unseen news rather than just memorizing
patterns specific to its training data.

**Pipeline**
- Combine and clean multiple labeled news datasets
- Text cleaning (lowercasing, URL removal, punctuation stripping)
- Tokenization (NLTK), stopword removal, lemmatization (WordNetLemmatizer)
- TF-IDF vectorization
- Compare Logistic Regression, Multinomial Naive Bayes, and LinearSVC across
  Bag-of-Words and TF-IDF
- Hyperparameter tuning (`C`) on the top two candidates
- Final model selected using **both** held-out test performance and independent
  external validation performance
- LinearSVC (C = 1) + TF-IDF selected as the final model

**Tech stack**: Python · scikit-learn · NLTK · pandas · Streamlit · Plotly

**Limitations**
- Judges *linguistic style/patterns*, not factual accuracy — it cannot verify
  claims against real-world facts the way a fact-checker does
- Trained on datasets from specific time periods and news sources — may not
  generalize equally well to entirely new topics, satire, or non-English text
- Should be treated as a decision-support signal, not a ground-truth verdict
        """
    )

st.markdown('<p class="footer-note">Built with Streamlit · scikit-learn · Plotly</p>', unsafe_allow_html=True)
