"""
preprocessing.py — Text cleaning and NLP preprocessing pipeline.
Uses NLTK with graceful fallback if resources are missing.
"""
from __future__ import annotations
import re
import streamlit as st

# NLTK downloads are cached via Streamlit's cache_resource
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


@st.cache_resource(show_spinner=False)
def _setup_nltk():
    if not _NLTK_AVAILABLE:
        return set(), None
    for res in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass
    try:
        sw = set(stopwords.words("english"))
        lem = WordNetLemmatizer()
        return sw, lem
    except Exception:
        return set(), None


def get_nltk_resources():
    return _setup_nltk()


def preprocess_text(text: str, use_nltk: bool = True) -> str:
    """
    Full NLP preprocessing:
      1. Lowercase
      2. Remove URLs
      3. Remove punctuation / special chars
      4. Tokenise
      5. Remove stopwords
      6. Lemmatise
    Falls back to basic regex cleaning if NLTK is unavailable.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not use_nltk or not _NLTK_AVAILABLE:
        return text

    stop_words, lemmatizer = get_nltk_resources()
    if lemmatizer is None:
        return text

    try:
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
        return " ".join(tokens)
    except Exception:
        return text


def preprocess_series(series, use_nltk: bool = True, show_progress: bool = False):
    """Apply preprocessing to a pandas Series."""
    import pandas as pd
    total = len(series)
    results = []
    if show_progress:
        bar = st.progress(0, text="Preprocessing text…")
    for i, text in enumerate(series):
        results.append(preprocess_text(str(text), use_nltk=use_nltk))
        if show_progress and i % max(1, total // 100) == 0:
            bar.progress(min(int((i + 1) / total * 100), 100), text=f"Preprocessing… {i+1}/{total}")
    if show_progress:
        bar.progress(100, text="Done!")
    return pd.Series(results, index=series.index)
