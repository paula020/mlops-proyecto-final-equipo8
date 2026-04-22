import pandas as pd

from src.data.quality import apply_quality_fixes
from src.features.engineering import add_financial_features
from src.models.baseline import train_baseline_models


def test_train_baseline_models_returns_metrics_for_both_models() -> None:
    rows = []
    for i in range(60):
        approved = i % 2 == 0
        cibil = 740 if approved else 540
        rows.append(
            {
                "loan_id": i + 1,
                "no_of_dependents": i % 4,
                "education": "Graduate" if i % 3 else "Not Graduate",
                "self_employed": "Yes" if i % 5 == 0 else "No",
                "income_annum": 900000 + (i * 15000),
                "loan_amount": 300000 + (i * 5000),
                "loan_term": 10 + (i % 6),
                "cibil_score": cibil,
                "residential_assets_value": 400000 + (i * 10000),
                "commercial_assets_value": 200000 + (i * 8000),
                "luxury_assets_value": 100000 + (i * 6000),
                "bank_asset_value": 150000 + (i * 7000),
                "loan_status": "Approved" if approved else "Rejected",
            }
        )

    df = pd.DataFrame(rows)
    df = apply_quality_fixes(df)
    df = add_financial_features(df)

    output = train_baseline_models(df)

    assert "metrics" in output
    assert "logistic_regression" in output["metrics"]
    assert "random_forest" in output["metrics"]

    for model_name in ["logistic_regression", "random_forest"]:
        model_metrics = output["metrics"][model_name]
        assert set(model_metrics.keys()) == {"accuracy", "f1", "recall", "roc_auc"}
        assert model_metrics["accuracy"] >= 0.0
