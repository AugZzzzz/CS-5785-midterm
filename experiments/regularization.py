"""Regularization experiment (NOT used in final submission)."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/train.csv")
texts = data["text"].fillna("").astype(str)
labels = data["label"].to_numpy(dtype=int)

train_indices, validation_indices = train_test_split(
    np.arange(len(labels)),
    test_size=2000,
    stratify=labels,
    random_state=42,
)

word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=4000,
    min_df=2,
    sublinear_tf=True,
)
character_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=4096,
    min_df=3,
)

train_word = word_vectorizer.fit_transform(texts.iloc[train_indices])
train_character = character_vectorizer.fit_transform(texts.iloc[train_indices])
validation_word = word_vectorizer.transform(texts.iloc[validation_indices])
validation_character = character_vectorizer.transform(texts.iloc[validation_indices])

train_features = hstack([train_word, train_character], format="csr")
validation_features = hstack([validation_word, validation_character], format="csr")

results = []

for penalty in [None, "l1", "l2", "elasticnet"]:
    model = SGDClassifier(
        loss="log_loss",
        penalty=penalty,
        alpha=0.0001,
        l1_ratio=0.5 if penalty == "elasticnet" else 0.15,
        random_state=42,
        max_iter=1000,
        tol=0.001,
    )
    model.fit(train_features, labels[train_indices])

    train_probability = model.predict_proba(train_features)[:, 1]
    validation_probability = model.predict_proba(validation_features)[:, 1]

    results.append(
        {
            "penalty": str(penalty),
            "train_auc": roc_auc_score(labels[train_indices], train_probability),
            "validation_auc": roc_auc_score(labels[validation_indices], validation_probability),
        }
    )

result_table = pd.DataFrame(results)
print(result_table.to_string(index=False))
result_table.to_csv("regularization_results.csv", index=False)
