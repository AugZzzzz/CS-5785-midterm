# Human or AI? — CS 5785 Midterm

Cornell Tech Applied Machine Learning — binary classification of human-written (0) vs AI-generated (1) passages.

**Team:** Auguste Zhou, Nishant Ray, Chenghao Guan.

## Development structure

The starting point is **unregularized logistic regression with word and character TF-IDF**. Separate `experiment/*` branches record approaches explored but **not included in the final classifier**. The final endpoint is the `submission/stylometric-final` branch, which adds **27 standardized writing-style features** to the original TF-IDF representation.

See the final submission branch for the full reproducible training script, requirements, and results.

| Stage | Method | Reported Kaggle public AUROC |
|---|---|---|
| Baseline | TF-IDF + unregularized logistic regression (L-BFGS) | 0.8781 |
| Experimental branches | PCA, regularization, trees, clustering, pseudo-labeling, n-gram and feature ablation tests | Not selected |
| Final | TF-IDF + 27 stylometric features + unregularized SGD logistic loss | Approximately 0.96 |

The final model **does not use pseudo-labels or unlabeled data**. Its optimizer is `SGDClassifier(loss="log_loss", penalty=None, max_iter=5, tol=None, random_state=42)`. Earlier local stylometric studies used L-BFGS, so those metrics are not exact SGD scores.

## Data

Obtain `train.csv`, `test.csv`, and `sample_submission.csv` from the class Kaggle competition and place them in `data/`. Do not commit the datasets or hidden test labels.

## Course submission

This personal repository is not itself a fork of [the course submission repository](https://github.com/sekhari/human-or-ai-submissions). The course also requires a separate PR to that repository, under `submissions/<kaggle-team-name>/`.

## AI assistance

ChatGPT assisted with code, debugging, experiment design, and report drafting. All experimental metrics should be checked against saved outputs and Kaggle before final submission.
