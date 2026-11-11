"""Formatting ablation. Preserving original formatting worked better locally."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/train.csv")
labels = data["label"].to_numpy(dtype=int)
texts = data["text"].fillna("").astype(str)

train_idx, validation_idx = train_test_split(
    np.arange(len(data)),
    test_size=2000,
    stratify=labels,
    random_state=42,
)


def remove_html_breaks(text):
    return re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)


def normalize_formatting(text):
    without_breaks = remove_html_breaks(text)
    return re.sub(r"\s+", " ", without_breaks).strip()


for name, transform in [
    ("original", lambda text: text),
    ("remove_br", remove_html_breaks),
    ("normalize_whitespace", normalize_formatting),
]:
    transformed = texts.map(transform)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(transformed.iloc[train_idx])
    validation_features = vectorizer.transform(transformed.iloc[validation_idx])

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(train_features, labels[train_idx])
    val_scores = model.predict_proba(validation_features)[:, 1]
    print(name, roc_auc_score(labels[validation_idx], val_scores))
