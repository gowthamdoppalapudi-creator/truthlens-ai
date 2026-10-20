# Fake News Detection using NLP and Machine Learning

A Natural Language Processing (NLP) and Machine Learning project that classifies news articles as **FAKE** or **REAL** based on their textual content — trained on three combined datasets and validated on a fourth, completely independent dataset.

The project covers the complete NLP/ML workflow: data cleaning, text preprocessing, feature extraction, model comparison, hyperparameter tuning, final model selection, and **out-of-distribution generalization testing**.

---

## 📌 Project Overview

The objective is to build a machine learning model capable of classifying news articles as **FAKE** or **REAL** based on their textual content, and to verify that the model generalizes beyond the dataset it was trained on — not just memorizes it.

The complete workflow includes:

- Combining multiple labeled datasets
- Data cleaning
- Text preprocessing (tokenization, stopword removal, lemmatization)
- Feature extraction using Bag of Words and TF-IDF
- Model training and comparison
- Hyperparameter tuning
- Final model selection
- Model serialization
- **Independent external validation on unseen data**

---

## 📂 Datasets

**Training data** — three datasets combined into one pool of 44,275 cleaned articles:

- `Fake.csv` + `True.csv` (ISOT Fake/Real News dataset)
- `fake_or_real_news.csv` (Kaggle Fake-or-Real News dataset)

| Class | Count |
|-------|-------|
| REAL  | 23,734 |
| FAKE  | 20,541 |

**External validation data** — `news.csv`, a separate 45,757-article dataset that was **not used in training or tuning at all**, used only to check whether the model generalizes to news it has never seen.

All datasets share the same schema: `title`, `text`, `label`, combined into a single `context` field.

---

## 🧹 Data Cleaning

- Checked for duplicate records, missing values, and empty text entries
- Dropped rows with empty article text
- Removed duplicate records
- Verified class balance after cleaning

---

## 🔤 Text Preprocessing

1. Convert text to lowercase
2. Remove URLs
3. Remove special characters and punctuation
4. Tokenize (NLTK `word_tokenize`)
5. Remove English stopwords
6. Lemmatize (`WordNetLemmatizer`)
7. Rejoin tokens into cleaned text

```text
Raw News Article → Lowercase → Remove URLs → Remove Special Characters
                  → Tokenization → Stopword Removal → Lemmatization → Cleaned Text
```

---

## 🔢 Feature Extraction

Two representations were tested:

- **Bag of Words (BoW)** — raw word-frequency counts
- **TF-IDF** — frequency weighted by how distinctive a word is across the corpus

---

## 🤖 Models Compared

| Model | Features | Train Acc. | Test Acc. |
|---|---|---|---|
| **Linear SVM** | **TF-IDF** | 99.88% | **97.69%** |
| Logistic Regression | Bag of Words | 99.98% | 97.34% |
| Linear SVM | Bag of Words | 100.00% | 96.72% |
| Logistic Regression | TF-IDF | 97.65% | 96.43% |
| Multinomial Naive Bayes | Bag of Words | 94.32% | 92.47% |
| Multinomial Naive Bayes | TF-IDF | 90.76% | 89.28% |

---

## ⚙️ Hyperparameter Tuning

`C` was swept over `[0.01, 0.1, 1, 2, 5, 10, 20]` for both Logistic Regression (BoW) and LinearSVC (TF-IDF).

- Logistic Regression + BoW peaked at **C = 0.1** → 97.71% test accuracy
- LinearSVC + TF-IDF peaked at **C = 1** → 97.70% test accuracy

The two were nearly tied on the held-out test set — the external validation step (below) is what decided the final pick.

---

## 🏆 Final Model

**LinearSVC (C = 1) with TF-IDF features**

| Metric | Score |
|---|---|
| Test Accuracy (held-out split) | 97.70% |
| Train Accuracy | 99.88% |

**Confusion Matrix (held-out test set)**
```text
[[4009   99]
 [ 105 4642]]
```

### Why this model over Logistic Regression + BoW?

Both scored almost identically on the held-out test split (97.70% vs 97.71%). The decision came down to performance on data neither model had ever seen — see below.

---

## 🧪 Independent External Validation

To check whether the model actually generalizes — rather than having just learned quirks of its own training data — both final candidates were evaluated on `news.csv`, a **45,757-article dataset that played no role in training or tuning**.

| Model | External Accuracy |
|---|---|
| **LinearSVC + TF-IDF** | **97.41%** |
| Logistic Regression + BoW | 97.35% |

**Confusion Matrix (external validation, LinearSVC + TF-IDF)**
```text
[[22491   366]
 [  821 22079]]
```

LinearSVC + TF-IDF was selected as the final model based on this combination of held-out and independent external performance.

> This measures accuracy on this particular external dataset and should not be interpreted as real-world fact-checking accuracy — the model recognizes linguistic patterns, not verified facts.

---

## 🔮 Prediction Pipeline

```text
Raw News Text → Text Preprocessing → TF-IDF Vectorization → Trained LinearSVC Model → FAKE / REAL Prediction
```

A single reusable `preprocess_text()` function ensures new input goes through the exact same steps used during training.

---

## 💾 Model Saving

```python
import joblib

joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(final_svc_model, "final_svc_model.pkl")
```

```python
loaded_tfidf = joblib.load("tfidf_vectorizer.pkl")
loaded_model = joblib.load("final_svc_model.pkl")
```

---

## 📁 Project Structure

```text
Fake-News-Detection/
│
├── NLP.ipynb
├── final_svc_model.pkl
├── tfidf_vectorizer.pkl
├── metrics.json
├── app.py
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- Joblib
- Streamlit, Plotly
- Jupyter Notebook

---

## 🚀 Future Improvements

- Deploying the model via API (FastAPI)
- Using n-grams for feature extraction
- k-fold cross-validation
- Deep learning models (LSTM)
- Transformer-based models (BERT)

---

## 👤 Author

**Khush Pawar**

Aspiring Data Scientist | Machine Learning Enthusiast

---
## 🚀 Live Demo

👉 [**Try TruthLens Live**](https://fake-news-detection-nlp-4to8gkxq3zhwuh2tddsv77.streamlit.app/)
> **Note:** This model is trained on labeled news datasets and performs text classification based on linguistic patterns learned from that data. It should not be considered a real-world fact-checking system.
