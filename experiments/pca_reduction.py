"""Dimensionality-reduction experiment; do not feed reduced features to final model."""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/train.csv")
labels = data["label"].to_numpy(dtype=int)
texts = data["text"].fillna("").astype(str)

train_idx, val_idx = train_test_split(
    np.arange(len(data)),
    test_size=2000,
    stratify=labels,
    random_state=42,
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
)
x_train = vectorizer.fit_transform(texts.iloc[train_idx])
x_val = vectorizer.transform(texts.iloc[val_idx])

for dimensions in [32, 64, 128]:
    reducer = TruncatedSVD(
        n_components=dimensions,
        random_state=42,
    )

    reduced_train = reducer.fit_transform(x_train)
    reduced_val = reducer.transform(x_val)

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(reduced_train, labels[train_idx])

    train_auc = roc_auc_score(
        labels[train_idx],
        model.predict_proba(reduced_train)[:, 1],
    )
    validation_auc = roc_auc_score(
        labels[val_idx],
        model.predict_proba(reduced_val)[:, 1],
    )
    print(dimensions, train_auc, validation_auc)

# SVD is the sparse-matrix analogue used here; it is not identical
# to centering a dense TF-IDF matrix and applying PCA.
