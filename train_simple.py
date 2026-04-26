import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from src.data.preprocesamiento import load_and_clean
from src.features.transformers import build_model_pipeline
from xgboost import XGBClassifier
from config.settings import NUM_COLS, CAT_COLS, TARGET, XGBOOST_PARAMS

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("credit_approval/baseline")

df, le = load_and_clean("data/loan_approval_dataset.csv")

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name="xgboost_simple"):
    model = XGBClassifier(**XGBOOST_PARAMS)
    pipeline = build_model_pipeline(model)
    pipeline.fit(X_train, y_train)

    from sklearn.metrics import roc_auc_score
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model_xgboost",
        registered_model_name="credit_approval_model",
    )
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Modelo registrado en MLflow")