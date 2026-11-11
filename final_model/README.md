# Final submission: TF-IDF + 27 stylometric features

**Team:** Auguste Zhou, Nishant Ray, Chenghao Guan  
**Course:** CS 5785 Applied Machine Learning — Human or AI?  
**Reported public Kaggle AUROC:** approximately **0.96** (not independently re-submitted here).  
**Private score:** not yet available.

## Model development

The main branch contains the original supervised unregularized logistic regression baseline. The original word/character TF-IDF pipeline scored **0.8781** on the public leaderboard (user-reported). Seven isolated experimental branches document approaches that were tried but **not adopted**:

- `experiment/regularization` — L1/L2/Elastic Net comparisons.
- `experiment/pca-feature-reduction` — reduced dimensionality.
- `experiment/tree-models` — Random Forest and gradient-boosted trees.
- `experiment/feature-ablation` — formatting / HTML / whitespace cleanup.
- `experiment/clustering` — cluster-similarity features.
- `experiment/semi-supervised` — confidence-based pseudo-labels.
- `experiment/ngrams` — alternative character n-gram windows.

Those branches are independent from the baseline and were **not merged** into the final model. Their scripts are representative runnable experiments, and may use smaller feature budgets than the originally reported runs. Do not attribute old results to a newly run script without checking its output.

## What the final model uses

```text
train.csv (10,000 labeled examples)
    |-- Word TF-IDF: unigrams + bigrams; max 26,000
    |-- Character TF-IDF: 3–5 grams; max 35,000
    |-- 27 scaled writing-style variables (weight 0.1)
    +--> sparse feature concatenation
         +--> SGDClassifier(loss="log_loss", penalty=None, max_iter=5)
              +--> P(AI) per test passage
```

The 27 variables measure passage and word lengths, sentence variation, vocabulary diversity, repeated word n-grams, punctuation, newlines/paragraphs, digits, capitalization, and whitespace. The style scaler uses only training examples. The features are clipped to [-5, 5] before multiplying by 0.1.

There is **no PCA, no self-training, no pseudo-labels, and no use of unlabeled.csv** in the final model.

## Dependencies

Python 3.10+ recommended.

```bash
python -m pip install -r requirements.txt
```

The Git repository intentionally excludes the Kaggle competition data. Place these files in `data/` at the repository root:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Generate submission.csv

From the repository root:

```bash
python final_model/train_predict.py \
  --train data/train.csv \
  --test data/test.csv \
  --sample data/sample_submission.csv \
  --output submission.csv
```

The program checks that IDs and ordering match the sample submission and probabilities are finite and inside [0, 1].

## Validate on fixed seed-42 holdout

```bash
python final_model/validate.py --train data/train.csv
```

The validation script uses 8,000 labeled training examples and 2,000 validation examples (`random_state=42`, stratified). It fits all transformations on the training portion only and saves row-level predicted probabilities. Previously reported seed-42 AUROC for the final five-epoch SGD recipe was **0.99009**, though the GitHub upload has only been syntax-reviewed, not rerun against competition data in this environment.

**Important optimizer distinction:** earlier three-seed local validation experiments used unregularized L-BFGS optimization and obtained mean AUROC **0.99239**. The successful Kaggle submission and the validation entrypoint here instead use **five-epoch SGD**; those two sets of metrics must not be mixed.

## Reproducibility and data handling

- Fixed split seed: 42; supplementary robustness-check seeds: 13, 123.
- Final SGD shuffle random seed: 42; no regularization.
- Train TF-IDF vectorizers and style scaler on training-only data for validation.
- Refit on all 10,000 labeled rows only to generate final Kaggle predictions.
- No hidden test labels accessed; no external LLM used to label competition passages.
- Do not commit Kaggle source data, tokens, cookies, or predictions with unapproved labels.

## GitHub / course submission caveat

This repository (`AugZzzzz/CS-5785-midterm`) is **not a fork** of `sekhari/human-or-ai-submissions`. If your instructor requires a pull request into the course repository, create a fork of that repository separately and copy this team's code to `submissions/<your-exact-kaggle-team-name>/` in a course submission branch.

## AI assistance disclosure

ChatGPT helped with model code, debugging, experiment analysis, repository organization and report writing. Confirm the generated outputs and metrics from your own rerun and Kaggle submission before the final course PR.
