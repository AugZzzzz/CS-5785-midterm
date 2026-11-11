"""Random Forest and XGBoost diagnostics; neither is used in the final model."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/train.csv")
labels = data["label"].to_numpy(dtype=int)
texts = data["text"].fillna("").astype(str)

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
train_features = vectorizer.fit_transform(texts.iloc[train_idx])
validation_features = vectorizer.transform(texts.iloc[validation_idx])

models = {
    "random_forest": RandomForestClassifier(
        n_estimators=30,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=2,
    )
}

try:
    from xgboost import XGBClassifier

    models["xgboost"] = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
    )
except ImportError:
    print("Install optional package xgboost to run the tree boosting comparison")

for name, model in models.items():
    model.fit(train_features, labels[train_idx])
    train_scores = model.predict_proba(train_features)[:, 1]
    validation_scores = model.predict_proba(validation_features)[:, 1]
    print(
        name,
        roc_auc_score(labels[train_idx], train_scores),
        roc_auc_score(labels[validation_idx], validation_scores),
    )
