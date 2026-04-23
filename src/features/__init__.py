"""Feature engineering and preprocessing utilities."""

from .engineering import add_financial_features

__all__ = [
    "add_financial_features",
    "BASE_NUMERICAL_FEATURES",
    "BASE_CATEGORICAL_FEATURES",
    "build_preprocessor",
    "split_features_target",
]
