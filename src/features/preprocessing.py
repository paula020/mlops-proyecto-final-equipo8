from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_NUMERICAL_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "total_assets",
    "loan_to_income_ratio",
    "loan_to_assets_ratio",
    "net_worth_proxy",
]

BASE_CATEGORICAL_FEATURES = ["education", "self_employed"]



def split_features_target(
    df: pd.DataFrame,
    target_column: str = "loan_status",
    id_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into model features and target series."""
    ids = id_columns or ["loan_id"]
    columns_to_drop = [target_column, *[col for col in ids if col in df.columns]]

    X = df.drop(columns=columns_to_drop)
    y = df[target_column].copy()
    return X, y


def build_preprocessor(
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Create sklearn preprocessor with imputers, scaling and one-hot encoding."""
    num_features = numerical_features or BASE_NUMERICAL_FEATURES
    cat_features = categorical_features or BASE_CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, [f for f in num_features if f in num_features]),
            ("categorical", categorical_pipeline, [f for f in cat_features if f in cat_features]),
        ],
        remainder="drop",
    )
