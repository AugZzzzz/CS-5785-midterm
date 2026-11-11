"""Baseline: word/character TF-IDF and unregularized logistic regression."""

import argparse

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--sample", default="data/sample_submission.csv")
    parser.add_argument("--output", default="submission_baseline.csv")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def make_vectorizers():
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=26000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )

    character_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=35000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    return word_vectorizer, character_vectorizer


def combine_features(texts, word_vectorizer, character_vectorizer, fit=False):
    if fit:
        word_features = word_vectorizer.fit_transform(texts)
        character_features = character_vectorizer.fit_transform(texts)
    else:
        word_features = word_vectorizer.transform(texts)
        character_features = character_vectorizer.transform(texts)

    return hstack(
        [word_features, character_features],
        format="csr",
    )


def main():
    args = parse_args()
    train = pd.read_csv(args.train)
    texts = train["text"].fillna("").astype(str)
    labels = train["label"].to_numpy(dtype=int)

    word_vectorizer, character_vectorizer = make_vectorizers()

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=3000,
    )

    if args.validation:
        train_indices, validation_indices = train_test_split(
            np.arange(len(labels)),
            test_size=2000,
            random_state=args.split_seed,
            stratify=labels,
        )

        train_features = combine_features(
            texts.iloc[train_indices],
            word_vectorizer,
            character_vectorizer,
            fit=True,
        )

        validation_features = combine_features(
            texts.iloc[validation_indices],
            word_vectorizer,
            character_vectorizer,
        )

        model.fit(train_features, labels[train_indices])

        train_scores = model.predict_proba(train_features)[:, 1]
        validation_scores = model.predict_proba(validation_features)[:, 1]

        print("Training AUROC:", roc_auc_score(labels[train_indices], train_scores))
        print("Validation AUROC:", roc_auc_score(labels[validation_indices], validation_scores))
        return

    train_features = combine_features(
        texts,
        word_vectorizer,
        character_vectorizer,
        fit=True,
    )
    model.fit(train_features, labels)

    test = pd.read_csv(args.test)
    test_texts = test["text"].fillna("").astype(str)
    probabilities = np.empty(len(test), dtype=float)

    for start in range(0, len(test), 100):
        end = min(start + 100, len(test))
        test_features = combine_features(
            test_texts.iloc[start:end],
            word_vectorizer,
            character_vectorizer,
        )
        probabilities[start:end] = model.predict_proba(test_features)[:, 1]

    submission = pd.DataFrame(
        {
            "id": test["id"],
            "label": probabilities,
        }
    )

    sample = pd.read_csv(args.sample)
    assert submission["id"].equals(sample["id"])
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))

    submission.to_csv(args.output, index=False)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
