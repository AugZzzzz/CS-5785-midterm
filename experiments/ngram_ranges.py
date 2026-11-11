"""Character n-gram screening experiment; reduced feature budget for speed."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/train.csv")
texts = data["text"].fillna("").astype(str)
labels = data["label"].to_numpy(dtype=int)

train_idx, validation_idx = train_test_split(
    np.arange(len(data)),
    test_size=2000,
    random_state=42,
    stratify=labels,
)

for character_range in [(3, 5), (2, 8)]:
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=2500,
    )
    character_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=character_range,
        min_df=3,
        max_features=2500,
    )

    train_features = hstack(
        [
            word_vectorizer.fit_transform(texts.iloc[train_idx]),
            character_vectorizer.fit_transform(texts.iloc[train_idx]),
        ],
        format="csr",
    )
    validation_features = hstack(
        [
            word_vectorizer.transform(texts.iloc[validation_idx]),
            character_vectorizer.transform(texts.iloc[validation_idx]),
        ],
        format="csr",
    )

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(train_features, labels[train_idx])

    scores = model.predict_proba(validation_features)[:, 1]
    print(character_range, roc_auc_score(labels[validation_idx], scores))

# Full 61K-wide (2,8) comparison was inconclusive due to runtime limits.
