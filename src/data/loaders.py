from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_CATEGORICAL_COLUMNS = ["education", "self_employed", "loan_status"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load CSV dataset and normalize column names."""
    dataset_path = Path(path)
    df = pd.read_csv(dataset_path)
    df.columns = [col.strip() for col in df.columns]
    return df


def apply_basic_cleaning(
    df: pd.DataFrame,
    categorical_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Standardize common text columns and trim whitespace values."""
    cleaned_df = df.copy()
    columns = categorical_columns or DEFAULT_CATEGORICAL_COLUMNS

    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    return cleaned_df
