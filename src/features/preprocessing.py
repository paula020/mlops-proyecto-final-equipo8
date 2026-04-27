"""Compatibility shim ."""

from src.features.transformers import (  # noqa: F401
    build_preprocessor,
    split_features_target,
)

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
