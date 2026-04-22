import pandas as pd

from src.features.preprocessing import build_preprocessor, split_features_target


def test_split_features_target_drops_target_and_id() -> None:
    df = pd.DataFrame(
        {
            "loan_id": [1, 2],
            "income_annum": [1000, 2000],
            "loan_status": ["Approved", "Rejected"],
        }
    )

    X, y = split_features_target(df)

    assert "loan_id" not in X.columns
    assert "loan_status" not in X.columns
    assert y.tolist() == ["Approved", "Rejected"]


def test_build_preprocessor_can_fit_transform_dataframe() -> None:
    X = pd.DataFrame(
        {
            "no_of_dependents": [0, 1, 2],
            "income_annum": [1000, 2000, 3000],
            "loan_amount": [500, 700, 900],
            "loan_term": [10, 12, 15],
            "cibil_score": [700, 640, 610],
            "residential_assets_value": [100, 200, 300],
            "commercial_assets_value": [50, 60, 70],
            "luxury_assets_value": [20, 30, 40],
            "bank_asset_value": [10, 20, 30],
            "total_assets": [180, 310, 440],
            "loan_to_income_ratio": [0.5, 0.35, 0.3],
            "loan_to_assets_ratio": [2.7, 2.25, 2.04],
            "net_worth_proxy": [-320, -390, -460],
            "education": ["Graduate", "Not Graduate", "Graduate"],
            "self_employed": ["No", "Yes", "No"],
        }
    )

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 0
