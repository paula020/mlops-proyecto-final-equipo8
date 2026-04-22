from __future__ import annotations

import pandas as pd


NEGATIVE_CHECK_COLUMNS = [
    "income_annum",
    "loan_amount",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


def build_data_quality_report(df: pd.DataFrame) -> dict[str, object]:
    """Compute key quality metrics for the input dataframe."""
    null_counts = df.isnull().sum().to_dict()

    negative_counts: dict[str, int] = {}
    for col in NEGATIVE_CHECK_COLUMNS:
        if col in df.columns:
            negative_counts[col] = int((df[col] < 0).sum())

    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_counts": null_counts,
        "negative_counts": negative_counts,
    }
    return report


def apply_quality_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic quality fixes discovered during EDA."""
    fixed_df = df.copy()

    if "residential_assets_value" in fixed_df.columns:
        fixed_df["residential_assets_value"] = fixed_df["residential_assets_value"].clip(lower=0)

    return fixed_df
