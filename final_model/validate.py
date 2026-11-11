"""Validate the final five-epoch SGD stylometric model on a fixed 8K/2K split.

This is separate from the Kaggle submission, which retrains on all 10K rows.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from train_predict import STYLE_WEIGHT, extract_style_features


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--output", default="validation_predictions.csv")
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_arguments()
    data = pd.read_csv(args.train)

    texts = data["text"].fillna("").astype(str)
    labels = data["label"].to_numpy(dtype=int)

    train_idx, validation_idx = train_test_split(
        np.arange(len(data)),
        test_size=2000,
        stratify=labels,
        random_state=args.split_seed,
    )

    train_texts = texts.iloc[train_idx]
    validation_texts = texts.iloc[validation_idx]

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

    train_word = word_vectorizer.fit_transform(train_texts)
    validation_word = word_vectorizer.transform(validation_texts)

    train_character = character_vectorizer.fit_transform(train_texts)
    validation_character = character_vectorizer.transform(validation_texts)

    train_style = np.asarray(
        [extract_style_features(text) for text in train_texts],
        dtype=np.float32,
    )
    validation_style = np.asarray(
        [extract_style_features(text) for text in validation_texts],
        dtype=np.float32,
    )

    scaler = StandardScaler()
    train_style = scaler.fit_transform(train_style)
    validation_style = scaler.transform(validation_style)

    train_style = np.clip(train_style, -5, 5).astype(np.float32)
    validation_style = np.clip(validation_style, -5, 5).astype(np.float32)

    train_features = hstack(
        [
            train_word,
            train_character,
            csr_matrix(train_style * STYLE_WEIGHT),
        ],
        format="csr",
    )
    validation_features = hstack(
        [
            validation_word,
            validation_character,
            csr_matrix(validation_style * STYLE_WEIGHT),
        ],
        format="csr",
    )

    model = SGDClassifier(
        loss="log_loss",
        penalty=None,
        max_iter=5,
        tol=None,
        random_state=42,
        average=False,
    )
    model.fit(train_features, labels[train_idx])

    train_probabilities = model.predict_proba(train_features)[:, 1]
    validation_probabilities = model.predict_proba(
        validation_features
    )[:, 1]

    train_auc = roc_auc_score(labels[train_idx], train_probabilities)
    validation_auc = roc_auc_score(
        labels[validation_idx],
        validation_probabilities,
    )

    print("Training AUROC:", train_auc)
    print("Validation AUROC:", validation_auc)

    if "id" in data.columns:
        ids = data["id"].iloc[validation_idx].to_numpy()
    else:
        ids = validation_idx

    output = pd.DataFrame(
        {
            "id": ids,
            "label": labels[validation_idx],
            "probability_ai": validation_probabilities,
            "word_count": validation_texts.str.findall(
                r"\b[\w']+\b"
            ).str.len().to_numpy(),
        }
    )
    output.to_csv(args.output, index=False)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
