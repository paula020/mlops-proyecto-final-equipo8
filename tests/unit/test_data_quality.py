import pandas as pd

from src.data.quality import apply_quality_fixes, build_data_quality_report


def test_build_data_quality_report_counts_duplicates_nulls_and_negatives() -> None:
    df = pd.DataFrame(
        {
            "loan_id": [1, 1, 2],
            "income_annum": [1000, 1000, 2000],
            "loan_amount": [500, 500, 1000],
            "residential_assets_value": [-10, -10, -20],
            "commercial_assets_value": [0, 0, 10],
            "luxury_assets_value": [0, 0, 5],
            "bank_asset_value": [0, 0, 1],
            "education": ["Graduate", "Graduate", None],
        }
    )

    report = build_data_quality_report(df)

    assert report["rows"] == 3
    assert report["columns"] == 8
    assert report["duplicate_rows"] == 1
    assert report["null_counts"]["education"] == 1
    assert report["negative_counts"]["residential_assets_value"] == 3


def test_apply_quality_fixes_clips_negative_residential_assets() -> None:
    df = pd.DataFrame({"residential_assets_value": [-100, 0, 50]})

    fixed_df = apply_quality_fixes(df)

    assert fixed_df["residential_assets_value"].tolist() == [0, 0, 50]
