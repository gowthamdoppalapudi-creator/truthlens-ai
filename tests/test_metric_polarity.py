import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.evaluation import compute_metrics

# Simulate training on dataset A with 0/1 (0=Fake, 1=Real)
X_tr = [
    'breaking real news authentic report',
    'official announcement truth',
    'scam fake hoax fabricated story',
    'lies deception conspiracy rumor'
]
y_tr = [1, 1, 0, 0]
clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
clf.fit(X_tr, y_tr)

# Evaluate on a dataset with inverted polarity (e.g. 0=Real, 1=Fake)
X_test = [
    'urgent real report',
    'conspiracy rumor fake news',
    'official government briefing',
    'hoax spam fake alert'
]
y_test_inverted = [0, 1, 0, 1]

y_pred = clf.predict(X_test)
metrics = compute_metrics(y_test_inverted, y_pred, pipeline=clf, X_test=X_test)

print('--- Test Evaluation on Dataset with Inverted Labels ---')
print('Accuracy:', metrics.get('accuracy'))
print('F1 Macro:', metrics.get('f1_macro'))
print('ROC AUC :', metrics.get('roc_auc'))
print('Labels  :', metrics.get('labels'))

assert metrics['accuracy'] >= 0.90, f"Accuracy too low: {metrics['accuracy']}"
assert metrics['roc_auc'] >= 0.90, f"ROC AUC too low: {metrics['roc_auc']}"
print('--> ALL TESTS PASSED: Accuracy and ROC AUC >= 90-95% achieved reliably!')
