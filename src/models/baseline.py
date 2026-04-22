from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.preprocessing import build_preprocessor, split_features_target


RANDOM_STATE = 42


def _encode_target(y: pd.Series) -> pd.Series:
    return y.map({"Rejected": 0, "Approved": 1})


def _evaluate_binary_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def train_baseline_models(df: pd.DataFrame, target_column: str = "loan_status") -> dict[str, Any]:
    """Train Logistic Regression and Random Forest baselines.

    Returns trained models and evaluation metrics for quick comparison.
    """
    X, y_raw = split_features_target(df, target_column=target_column)
    y = _encode_target(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(
        numerical_features=[col for col in X.columns if col not in ["education", "self_employed"]],
        categorical_features=[col for col in ["education", "self_employed"] if col in X.columns],
    )

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    metrics = {
        "logistic_regression": _evaluate_binary_classifier(logistic_model, X_test, y_test),
        "random_forest": _evaluate_binary_classifier(rf_model, X_test, y_test),
    }

    return {
        "models": {
            "logistic_regression": logistic_model,
            "random_forest": rf_model,
        },
        "metrics": metrics,
        "test_size": int(len(X_test)),
    }
