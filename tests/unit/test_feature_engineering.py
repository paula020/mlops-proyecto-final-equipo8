import pandas as pd

from src.features.engineering import add_financial_features


def test_add_financial_features_creates_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "income_annum": [100000, 0],
            "loan_amount": [50000, 12000],
            "residential_assets_value": [10000, -10],
            "commercial_assets_value": [5000, 1000],
            "luxury_assets_value": [8000, 500],
            "bank_asset_value": [3000, 2000],
        }
    )

    result = add_financial_features(df)

    assert "total_assets" in result.columns
    assert "loan_to_income_ratio" in result.columns
    assert "loan_to_assets_ratio" in result.columns
    assert "net_worth_proxy" in result.columns

    assert result.loc[0, "total_assets"] == 26000
    assert result.loc[1, "total_assets"] == 3500
    assert result.loc[1, "loan_to_income_ratio"] == 0.0
    assert result.loc[1, "loan_to_assets_ratio"] > 0
