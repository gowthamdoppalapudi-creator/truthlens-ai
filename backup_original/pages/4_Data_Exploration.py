"""
pages/4_Data_Exploration.py — Exploratory Data Analysis (EDA) for news datasets.
"""
from __future__ import annotations
from collections import Counter
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import APP_NAME, PLOTLY_LAYOUT, COLOR_REAL, COLOR_FAKE, COLOR_ACCENT
from src.utils import inject_css, page_header, stat_card, badge_row, footer
from src.data_loader import get_session_dataset
from src.preprocessing import preprocess_text

# Optional WordCloud support
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


st.set_page_config(
    page_title=f"Data Exploration | {APP_NAME}",
    page_icon="📊",
    layout="wide",
)
inject_css()

page_header(
    "Corpus Diagnostics & Text EDA",
    "Inspect class distributions, token statistics, linguistic frequencies, and structural properties of your dataset.",
)
badge_row([
    "Class Balance Analysis",
    "Length & Token Distribution",
    "N-Gram Collocations",
    "Descriptive Linguistics",
])

df = get_session_dataset()
if df is None:
    st.warning("⚠️ No dataset loaded in active workspace. Please load or upload a dataset in the Dataset Manager first.")
    if st.button("📁 Go to Dataset Manager", type="primary"):
        st.switch_page("pages/3_Dataset_Manager.py")
    st.stop()

meta = st.session_state.get("active_df_meta", {})
text_col = meta.get("text_column", "text" if "text" in df.columns else df.columns[0])
label_col = meta.get("label_column", "label" if "label" in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0]))

st.caption(f"Analyzing active dataset: `{st.session_state.get('active_df_name', 'dataset.csv')}` ({len(df):,} records).")

# ──────────────────────────────────────────────────────────────────────────────
# High-Level Metrics
# ──────────────────────────────────────────────────────────────────────────────
char_lens = df[text_col].astype(str).str.len()
word_counts = df[text_col].astype(str).str.split().str.len()

m1, m2, m3, m4 = st.columns(4)
with m1:
    stat_card(f"{len(df):,}", "Total Corpus Records", "stat-accent")
with m2:
    stat_card(f"{int(word_counts.mean()):,}", "Avg Word Count / Doc", "stat-purple")
with m3:
    stat_card(f"{int(word_counts.median()):,}", "Median Word Count", "stat-green")
with m4:
    stat_card(f"{int(char_lens.max()):,}", "Max Character Length", "stat-accent")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Class Distribution & Imbalance
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 1. Class Distribution & Imbalance Ratios")

col_dist1, col_dist2 = st.columns(2)
class_counts = df[label_col].value_counts()

with col_dist1:
    color_map = {
        "REAL": COLOR_REAL,
        "FAKE": COLOR_FAKE,
    }
    pie_colors = [color_map.get(str(c).upper(), COLOR_ACCENT) for c in class_counts.index]
    fig_pie = go.Figure(
        go.Pie(
            labels=class_counts.index,
            values=class_counts.values,
            hole=0.45,
            marker=dict(colors=pie_colors),
        )
    )
    fig_pie.update_layout(title="Class Proportion Breakdown", **PLOTLY_LAYOUT, height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_dist2:
    fig_bar = px.bar(
        x=class_counts.index,
        y=class_counts.values,
        labels={"x": "Class Label", "y": "Article Count"},
        color=class_counts.index,
        color_discrete_map=color_map,
        title="Class Counts",
    )
    fig_bar.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Document Length & Token Distributions
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 2. Document Length & Word Count Distributions")

col_len1, col_len2 = st.columns(2)

with col_len1:
    fig_word = px.histogram(
        word_counts,
        nbins=40,
        title="Word Count Distribution per Article",
        labels={"value": "Word Count", "count": "Frequency"},
        color_discrete_sequence=[COLOR_ACCENT],
    )
    fig_word.update_layout(**PLOTLY_LAYOUT, height=350)
    st.plotly_chart(fig_word, use_container_width=True)

with col_len2:
    fig_char = px.histogram(
        char_lens,
        nbins=40,
        title="Character Length Distribution per Article",
        labels={"value": "Character Length", "count": "Frequency"},
        color_discrete_sequence=[COLOR_REAL],
    )
    fig_char.update_layout(**PLOTLY_LAYOUT, height=350)
    st.plotly_chart(fig_char, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Linguistic N-Gram Frequencies & Word Cloud
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 3. Lexical N-Gram Frequency Analysis")
st.caption("Examine the most prevalent unigram and bigram collocations within the corpus.")

filter_class = st.radio(
    "Filter Analysis by Class:",
    options=["All Articles"] + list(class_counts.index),
    horizontal=True,
)

if filter_class == "All Articles":
    sub_texts = df[text_col]
else:
    sub_texts = df[df[label_col] == filter_class][text_col]

# Sample for speed if large
sample_size = min(len(sub_texts), 500)
sample_texts = sub_texts.sample(n=sample_size, random_state=42) if len(sub_texts) > sample_size else sub_texts

with st.spinner("Extracting cleaned n-gram distributions..."):
    all_words = []
    bigrams = []
    for text in sample_texts:
        cleaned = preprocess_text(str(text))
        tokens = cleaned.split()
        all_words.extend([t for t in tokens if len(t) > 2])
        for i in range(len(tokens) - 1):
            bigrams.append(f"{tokens[i]} {tokens[i+1]}")

col_freq1, col_freq2 = st.columns(2)

with col_freq1:
    top_words = Counter(all_words).most_common(15)
    if top_words:
        df_top_words = pd.DataFrame(top_words, columns=["Word", "Frequency"])
        fig_words = px.bar(
            df_top_words[::-1],
            x="Frequency",
            y="Word",
            orientation="h",
            title=f"Top 15 Unigrams ({filter_class})",
            color_discrete_sequence=[COLOR_ACCENT],
        )
        fig_words.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_words, use_container_width=True)
    else:
        st.info("No tokens found.")

with col_freq2:
    top_bigrams = Counter(bigrams).most_common(15)
    if top_bigrams:
        df_top_bigrams = pd.DataFrame(top_bigrams, columns=["Bigram", "Frequency"])
        fig_bigrams = px.bar(
            df_top_bigrams[::-1],
            x="Frequency",
            y="Bigram",
            orientation="h",
            title=f"Top 15 Bigrams ({filter_class})",
            color_discrete_sequence=["#a78bfa"],
        )
        fig_bigrams.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_bigrams, use_container_width=True)
    else:
        st.info("No bigrams found.")

# Word Cloud
if WORDCLOUD_AVAILABLE and all_words:
    st.markdown("#### ☁️ Lexical Word Cloud")
    text_corpus = " ".join(all_words)
    wc = WordCloud(
        width=900,
        height=380,
        background_color="#0b0d12",
        colormap="Blues",
        max_words=100,
    ).generate(text_corpus)

    fig_wc, ax = plt.subplots(figsize=(10, 4), facecolor="#0b0d12")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)

st.info(
    "💡 **Methodological Note:** Word-frequency patterns are strictly descriptive stylistic characteristics of the dataset. "
    "The presence of frequent terms does not independently establish whether any given news article is true or false."
)

st.markdown("---")
footer()
