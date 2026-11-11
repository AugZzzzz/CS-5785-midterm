"""Semi-supervised pseudo-labeling experiment; never used in final submission."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

labeled = pd.read_csv("data/train.csv")
unlabeled = pd.read_csv("data/unlabeled.csv")
labels = labeled["label"].to_numpy(dtype=int)

train_idx, validation_idx = train_test_split(
    np.arange(len(labeled)),
    test_size=2000,
    stratify=labels,
    random_state=42,
)

train_texts = labeled["text"].fillna("").astype(str)
unlabeled_texts = unlabeled["text"].fillna("").astype(str)

word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=12000,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32,
)
character_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_features=8000,
    sublinear_tf=True,
    dtype=np.float32,
)

train_word = word_vectorizer.fit_transform(train_texts.iloc[train_idx])
train_char = character_vectorizer.fit_transform(train_texts.iloc[train_idx])
val_word = word_vectorizer.transform(train_texts.iloc[validation_idx])
val_char = character_vectorizer.transform(train_texts.iloc[validation_idx])
unlabeled_word = word_vectorizer.transform(unlabeled_texts)
unlabeled_char = character_vectorizer.transform(unlabeled_texts)

x_train = hstack([train_word, train_char], format="csr")
x_validation = hstack([val_word, val_char], format="csr")
x_unlabeled = hstack([unlabeled_word, unlabeled_char], format="csr")


def fit_classifier(features, targets):
    model = SGDClassifier(
        loss="log_loss",
        penalty=None,
        max_iter=30,
        tol=0.001,
        random_state=42,
    )
    model.fit(features, targets)
    return model


supervised = fit_classifier(x_train, labels[train_idx])
scores = supervised.predict_proba(x_validation)[:, 1]
print("Supervised:", roc_auc_score(labels[validation_idx], scores))

pseudo_probabilities = supervised.predict_proba(x_unlabeled)[:, 1]
for confidence in [0.99, 0.95]:
    selected = np.flatnonzero(
        (pseudo_probabilities >= confidence)
        | (pseudo_probabilities <= 1 - confidence)
    )
    pseudo_labels = (
        pseudo_probabilities[selected] >= confidence
    ).astype(int)

    augmented_features = vstack(
        [x_train, x_unlabeled[selected]],
        format="csr",
    )
    augmented_labels = np.concatenate(
        [labels[train_idx], pseudo_labels]
    )

    candidate = fit_classifier(augmented_features, augmented_labels)
    validation_scores = candidate.predict_proba(x_validation)[:, 1]
    validation_auc = roc_auc_score(
        labels[validation_idx],
        validation_scores,
    )
    print("Confidence:", confidence, "Pseudo-labels:", len(selected), "AUROC:", validation_auc)

# Held-out validation labels are never used for generating pseudo-labels.
