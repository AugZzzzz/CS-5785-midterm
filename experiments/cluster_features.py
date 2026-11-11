"""Add text-cluster similarities using training text only; not final model."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
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
    stratify=labels,
    random_state=42,
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
)
train_sparse = vectorizer.fit_transform(texts.iloc[train_idx])
validation_sparse = vectorizer.transform(texts.iloc[validation_idx])

reducer = TruncatedSVD(n_components=50, random_state=42)
train_low_dim = reducer.fit_transform(train_sparse)
validation_low_dim = reducer.transform(validation_sparse)

clusterer = MiniBatchKMeans(
    n_clusters=10,
    random_state=42,
    batch_size=256,
)
clusterer.fit(train_low_dim)

def centroid_similarity(matrix):
    distances = clusterer.transform(matrix)
    return csr_matrix(1.0 / (1.0 + distances))

variants = {
    "tfidf_only": (train_sparse, validation_sparse),
    "tfidf_plus_clusters": (
        hstack([train_sparse, centroid_similarity(train_low_dim)], format="csr"),
        hstack([validation_sparse, centroid_similarity(validation_low_dim)], format="csr"),
    ),
}

for name, pair in variants.items():
    train_features, validation_features = pair
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(train_features, labels[train_idx])
    scores = model.predict_proba(validation_features)[:, 1]
    print(name, roc_auc_score(labels[validation_idx], scores))
