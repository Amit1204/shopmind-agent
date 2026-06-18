"""
Fake Review Detector — XGBoost classifier on linguistic + behavioural features.
Labels: 0 = fake/suspicious, 1 = genuine
Run: python scripts/train_models.py --model fake_review
"""
import os
import re
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Extract 15 linguistic + behavioural features from reviews DataFrame."""
    features = []
    for _, row in df.iterrows():
        text = str(row.get("reviewText", ""))
        features.append([
            # Linguistic
            len(text),                                          # char length
            len(text.split()),                                  # word count
            len(set(text.lower().split())),                     # unique words
            text.count("!"),                                    # exclamation marks
            text.count("?"),                                    # question marks
            len(re.findall(r'\b[A-Z]{2,}\b', text)),           # ALL CAPS words
            text.lower().count("product"),                      # generic word usage
            text.lower().count("i "),                           # first person usage
            sum(1 for w in text.split() if len(w) > 8) / (len(text.split()) + 1),  # long word ratio

            # Behavioural / metadata
            int(row.get("verified_purchase", 1)),               # verified purchase
            float(row.get("overall", 3.0)),                     # star rating
            abs(float(row.get("overall", 3.0)) - 3.0),         # rating extremity
            float(row.get("helpful_votes", 0)),                 # helpful votes
            float(row.get("total_votes", 0)),                   # total votes
            float(row.get("reviewer_review_count", 5)),         # reviewer prolificacy
        ])
    return np.array(features, dtype=float)


def create_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Create weak labels from available signals.
    In production: replace with human-annotated labels.
    """
    labels = np.ones(len(df))  # default genuine
    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row.get("reviewText", ""))
        rating = float(row.get("overall", 3.0))
        verified = int(row.get("verified_purchase", 1))
        reviewer_count = float(row.get("reviewer_review_count", 5))

        suspicious = 0
        if len(text) < 20:             suspicious += 1  # too short
        if abs(rating - 3.0) > 1.8:   suspicious += 1  # extreme rating
        if not verified:               suspicious += 1  # unverified
        if reviewer_count > 100:       suspicious += 1  # prolific reviewer
        if text.count("!") > 5:       suspicious += 1  # over-excited

        if suspicious >= 3:
            labels[i] = 0  # fake
    return labels


def train(reviews_csv: str = None, output_path: str = None):
    reviews_csv = reviews_csv or settings.reviews_csv
    output_path = output_path or settings.fake_review_model_path

    logger.info(f"Training fake review classifier from {reviews_csv}")
    df = pd.read_csv(reviews_csv).sample(min(50000, len(pd.read_csv(reviews_csv))), random_state=42)

    X = extract_features(df)
    y = create_labels(df)

    logger.info(f"Dataset: {len(df)} reviews, {y.mean():.1%} genuine")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    with mlflow.start_run(run_name="fake_review_xgb"):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_genuine", report["1.0"]["f1-score"])
        mlflow.log_metric("f1_fake", report["0.0"]["f1-score"])

        logger.info(f"ROC-AUC: {auc:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info(f"Fake review model saved → {output_path}")
    return {"roc_auc": auc, "report": report}


if __name__ == "__main__":
    train()
