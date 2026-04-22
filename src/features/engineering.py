from __future__ import annotations

import pandas as pd


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ratio and aggregate features from financial columns."""
    featured_df = df.copy()

    income = featured_df["income_annum"].where(featured_df["income_annum"] != 0)

    featured_df["total_assets"] = (
        featured_df["residential_assets_value"].clip(lower=0)
        + featured_df["commercial_assets_value"]
        + featured_df["luxury_assets_value"]
        + featured_df["bank_asset_value"]
    )

    featured_df["loan_to_income_ratio"] = (featured_df["loan_amount"] / income).fillna(0.0)

    featured_df["loan_to_assets_ratio"] = (
        featured_df["loan_amount"] / featured_df["total_assets"].where(featured_df["total_assets"] != 0)
    ).fillna(0.0)

    featured_df["net_worth_proxy"] = featured_df["total_assets"] - featured_df["loan_amount"]

    return featured_df
