"""Reproduce the stylometric SGD classifier submitted to Kaggle.

Feature representation:
- Up to 26,000 word TF-IDF unigrams and bigrams.
- Up to 35,000 character TF-IDF n-grams (3 to 5).
- Exactly 27 standardized and clipped stylometric variables weighted by 0.1.

Final classifier: unregularized SGD logistic loss, 5 training epochs.
Semi-supervised pseudo-labeling is NOT part of this pipeline.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse
import gc
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


STYLE_WEIGHT = 0.1
BATCH_SIZE = 100
RANDOM_SEED = 42


def extract_style_features(text):
    """Return the 27 stylometric features in the original trained order."""
    words = re.findall(r"\b[\w']+\b", text.lower())
    word_count = len(words)

    if words:
        word_lengths = np.array([len(word) for word in words])
    else:
        word_lengths = np.array([0])

    sentences = [
        sentence
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]

    sentence_count = max(1, len(sentences))
    sentence_lengths = [
        len(re.findall(r"\b[\w']+\b", sentence))
        for sentence in sentences
    ]

    safe_word_count = max(1, word_count)
    safe_character_count = max(1, len(text))
    unique_word_count = len(set(words))

    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))

    paragraphs = [
        paragraph
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    paragraph_count = len(paragraphs)

    # Text length and sentence structure: seven features.
    text_structure = [
        np.log1p(word_count),
        np.log1p(safe_character_count),
        np.mean(word_lengths),
        np.std(word_lengths),
        np.mean(sentence_lengths),
        np.std(sentence_lengths),
        sentence_count / safe_word_count,
    ]

    # Vocabulary diversity: four features.
    vocabulary = [
        unique_word_count / safe_word_count,
        unique_word_count / np.sqrt(safe_word_count),
        sum(len(word) <= 3 for word in words) / safe_word_count,
        sum(len(word) >= 8 for word in words) / safe_word_count,
    ]

    # Repeated phrases: two features.
    repetition = [
        (len(bigrams) - len(set(bigrams))) / max(1, len(bigrams)),
        (len(trigrams) - len(set(trigrams))) / max(1, len(trigrams)),
    ]

    # Punctuation per word: nine features.
    punctuation = [
        text.count(",") / safe_word_count,
        text.count(";") / safe_word_count,
        text.count(":") / safe_word_count,
        text.count("?") / safe_word_count,
        text.count("!") / safe_word_count,
        text.count("(") / safe_word_count,
        text.count('"') / safe_word_count,
        text.count("'") / safe_word_count,
        text.count("-") / safe_word_count,
    ]

    # Paragraph and whitespace structure: two features.
    paragraphs_and_lines = [
        text.count("\n") / safe_word_count,
        paragraph_count / safe_word_count,
    ]

    # Character composition: three features.
    character_composition = [
        sum(character.isdigit() for character in text) / safe_character_count,
        sum(character.isupper() for character in text) / safe_character_count,
        sum(character.isspace() for character in text) / safe_character_count,
    ]

    features = (
        text_structure
        + vocabulary
        + repetition
        + punctuation
        + paragraphs_and_lines
        + character_composition
    )

    assert len(features) == 27
    return features


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate Kaggle probabilities with the final stylometric model"
    )
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--sample", default="data/sample_submission.csv")
    parser.add_argument("--output", default="submission.csv")
    return parser.parse_args()


def main():
    args = parse_arguments()
    start_time = time.time()

    # Step 1: Load the labeled data and Kaggle test examples.
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    train_texts = train["text"].fillna("").astype(str)
    test_texts = test["text"].fillna("").astype(str)
    labels = train["label"].to_numpy(dtype=float)

    # Step 2: Extract and standardize the 27 stylometric features.
    train_style = np.asarray(
        [extract_style_features(text) for text in train_texts],
        dtype=np.float32,
    )
    test_style = np.asarray(
        [extract_style_features(text) for text in test_texts],
        dtype=np.float32,
    )

    scaler = StandardScaler()
    scaler.fit(train_style)

    train_style = np.clip(
        scaler.transform(train_style),
        -5,
        5,
    ).astype(np.float32)

    test_style = np.clip(
        scaler.transform(test_style),
        -5,
        5,
    ).astype(np.float32)

    print("Style features complete", time.time() - start_time, flush=True)

    # Step 3: Word TF-IDF unigrams and bigrams.
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=26000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )

    train_word = word_vectorizer.fit_transform(train_texts)
    test_word = word_vectorizer.transform(test_texts)

    print("Word TF-IDF complete", train_word.shape, flush=True)

    # Step 4: Character TF-IDF n-grams of length three through five.
    character_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=35000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    train_character = character_vectorizer.fit_transform(train_texts)
    print("Character TF-IDF complete", train_character.shape, flush=True)

    # Step 5: Join feature groups without dimensionality reduction.
    train_features = hstack(
        [
            train_word,
            train_character,
            csr_matrix(train_style * STYLE_WEIGHT),
        ],
        format="csr",
    )

    del train_word
    del train_character
    gc.collect()

    print("Combined feature shape", train_features.shape, flush=True)

    # Step 6: Train unregularized logistic regression with SGD.
    model = SGDClassifier(
        loss="log_loss",
        penalty=None,
        max_iter=5,
        tol=None,
        random_state=RANDOM_SEED,
        average=False,
    )

    model.fit(train_features, labels.astype(int))
    print("Model trained", model.n_iter_, flush=True)

    # Step 7: Transform character features and predict in batches.
    probabilities = np.empty(len(test), dtype=float)

    for start in range(0, len(test), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(test))

        batch_character = character_vectorizer.transform(
            test_texts.iloc[start:end]
        )
        batch_features = hstack(
            [
                test_word[start:end],
                batch_character,
                csr_matrix(test_style[start:end] * STYLE_WEIGHT),
            ],
            format="csr",
        )
        probabilities[start:end] = model.predict_proba(
            batch_features
        )[:, 1]

        print("Predicted", end, "passages", flush=True)

    # Step 8: Verify IDs, probabilities, and Kaggle CSV layout.
    submission = pd.DataFrame(
        {
            "id": test["id"],
            "label": probabilities,
        }
    )

    sample_submission = pd.read_csv(args.sample)

    assert len(submission) == len(sample_submission) == 3000
    assert submission["id"].equals(sample_submission["id"])
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    submission.to_csv(
        args.output,
        index=False,
        float_format="%.12g",
    )

    print(submission.head().to_string(index=False), flush=True)
    print("Saved", args.output, time.time() - start_time, flush=True)


if __name__ == "__main__":
    main()
