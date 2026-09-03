"""
Automated unit and integration test suite for Phase 4: Automated Data Quality & Cleaning Engine.
Tests all 12 issue detection algorithms, transformation preview generation,
configurable pipeline transformations, raw dataset immutability, processed dataset persistence,
and Flask REST API endpoints.
"""

import io
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from app import create_app
from config.settings import Config
from analysis.cleaning import (
    detect_data_quality_issues,
    generate_cleaning_preview,
    execute_cleaning_pipeline,
)
from utils.file_handler import (
    save_uploaded_file,
    load_dataset,
    save_processed_dataset,
    get_latest_processed_file,
)


@pytest.fixture
def client():
    """Flask test client fixture."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def messy_dataframe():
    """Generates a realistic messy DataFrame containing all 12 data quality defect categories."""
    n = 40
    return pd.DataFrame({
        # 1. Missing values & whitespace & categorical casing
        "gender": [" male ", "Male", "MALE", " female", "Female", "female ", None, "Male"] * 5,
        # 2. Potential numeric column stored as strings ($ and commas)
        "revenue": ["$1,200.50", "$3,400.00", "$500.25", "$9,999.99", "$2,150.00"] * 8,
        # 3. Potential datetime column stored as strings
        "signup_date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-05-12"] * 8,
        # 4. Numerical with missing values, invalid sentinel codes, and extreme outliers
        "age": [25, 30, -999, 45, np.nan, 28, 150, 35] * 5,
        # 5. Domain anomaly (negative price) & infinity
        "price": [10.5, 20.0, -15.0, np.inf, 55.0, 99.9, 12.0, 40.0] * 5,
        # 6. Constant column (0 variance)
        "country": ["USA"] * n,
        # 7. Near-constant column (97.5% dominant)
        "is_verified": [True] * 39 + [False],
        # 8. High missing percentage (>50%)
        "optional_notes": [None] * 25 + ["Note A", "Note B", "Note C"] * 5,
        # 9. Duplicate rows target (row 0 to 4 are repeated)
        "customer_id": [f"CUST_{i % 10}" for i in range(n)],
    })


# =====================================================================
# 1. 12 DATA QUALITY DETECTORS UNIT TESTS
# =====================================================================

def test_detect_duplicate_rows():
    df = pd.DataFrame({
        "a": [1, 2, 1, 2, 3],
        "b": ["x", "y", "x", "y", "z"],
    })
    issues = detect_data_quality_issues(df)
    dup_issues = [i for i in issues if i["issue_type"] == "duplicate_rows"]
    assert len(dup_issues) == 1
    assert dup_issues[0]["column"] == "(Entire Dataset)"
    assert dup_issues[0]["affected_rows"] == 2
    assert dup_issues[0]["percentage_affected"] == 40.0


def test_detect_missing_values_and_high_missing():
    df = pd.DataFrame({
        "moderate_missing": [1, 2, None, 4, None],  # 40% missing
        "critically_missing": [1, None, None, None, None],  # 80% missing
        "clean_col": [10, 20, 30, 40, 50],
    })
    issues = detect_data_quality_issues(df)
    
    crit_issues = [i for i in issues if i["issue_type"] == "high_missing_percentage" and i["column"] == "critically_missing"]
    assert len(crit_issues) == 1
    assert crit_issues[0]["severity"] == "Critical"
    assert crit_issues[0]["percentage_affected"] == 80.0

    mod_issues = [i for i in issues if i["issue_type"] == "missing_values" and i["column"] == "moderate_missing"]
    assert len(mod_issues) == 1
    assert mod_issues[0]["affected_rows"] == 2


def test_detect_constant_and_near_constant():
    df = pd.DataFrame({
        "constant_col": ["Fixed"] * 20,
        "near_constant_col": ["Main"] * 19 + ["Other"],
        "normal_col": [f"val_{i}" for i in range(20)],
    })
    issues = detect_data_quality_issues(df)

    const_issues = [i for i in issues if i["issue_type"] == "constant_column"]
    assert len(const_issues) == 1
    assert const_issues[0]["column"] == "constant_col"
    assert const_issues[0]["severity"] == "High"

    near_const_issues = [i for i in issues if i["issue_type"] == "near_constant_column"]
    assert len(near_const_issues) == 1
    assert near_const_issues[0]["column"] == "near_constant_col"
    assert near_const_issues[0]["percentage_affected"] == 95.0


def test_detect_whitespace_and_inconsistent_categorical():
    df = pd.DataFrame({
        "category": [" male ", "Male", "male", "Female", " female ", "FEMALE"] * 4,
    })
    issues = detect_data_quality_issues(df)

    ws_issues = [i for i in issues if i["issue_type"] == "leading_trailing_whitespace"]
    assert len(ws_issues) == 1
    assert ws_issues[0]["column"] == "category"

    inc_issues = [i for i in issues if i["issue_type"] == "inconsistent_categorical"]
    assert len(inc_issues) == 1
    assert inc_issues[0]["column"] == "category"


def test_detect_potential_numeric_and_datetime_strings():
    df = pd.DataFrame({
        "money": ["$1,000.50", "$2,500.00", "$450.75", "$9,100.20", "$3,200.00"] * 4,
        "dates": ["2024-01-01", "2024-02-15", "2024-03-20", "2024-04-10", "2024-05-05"] * 4,
    })
    issues = detect_data_quality_issues(df)

    num_str_issues = [i for i in issues if i["issue_type"] == "potential_numeric_as_string"]
    assert len(num_str_issues) == 1
    assert num_str_issues[0]["column"] == "money"
    assert num_str_issues[0]["severity"] == "High"

    dt_str_issues = [i for i in issues if i["issue_type"] == "potential_datetime_as_string"]
    assert len(dt_str_issues) == 1
    assert dt_str_issues[0]["column"] == "dates"


def test_detect_invalid_numerical_values():
    df = pd.DataFrame({
        "age": [25, 30, -5, -999, 40, 50, 35, 28, 45, 60],
        "score": [10.0, np.inf, 25.0, 30.0, -np.inf, 50.0, 60.0, 70.0, 80.0, 90.0],
    })
    issues = detect_data_quality_issues(df)

    age_issues = [i for i in issues if i["issue_type"] == "invalid_numerical_values" and i["column"] == "age"]
    assert len(age_issues) == 1
    assert age_issues[0]["details"]["negative_count"] > 0
    assert age_issues[0]["details"]["sentinel_count"] > 0

    score_issues = [i for i in issues if i["issue_type"] == "invalid_numerical_values" and i["column"] == "score"]
    assert len(score_issues) == 1
    assert score_issues[0]["details"]["inf_count"] == 2


def test_detect_extreme_outliers():
    # 30 regular points + 1 massive extreme outlier
    np.random.seed(42)
    vals = list(np.random.normal(50, 5, 30)) + [1000.0]
    df = pd.DataFrame({"metric": vals})
    issues = detect_data_quality_issues(df)

    outlier_issues = [i for i in issues if i["issue_type"] == "extreme_outliers"]
    assert len(outlier_issues) == 1
    assert outlier_issues[0]["column"] == "metric"
    assert outlier_issues[0]["affected_rows"] >= 1


# =====================================================================
# 2. TRANSFORMATION PREVIEW TESTS
# =====================================================================

def test_generate_cleaning_preview(messy_dataframe):
    preview = generate_cleaning_preview(messy_dataframe)
    assert "previews" in preview
    assert preview["total_preview_pairs"] > 0
    
    previews = preview["previews"]
    categories = {p["category"] for p in previews}
    assert "Whitespace Standardization" in categories
    assert "Categorical Normalization" in categories
    assert "Numeric Type Casting" in categories


# =====================================================================
# 3. CLEANING PIPELINE & SUMMARY EXECUTION TESTS
# =====================================================================

def test_execute_cleaning_pipeline_complete(messy_dataframe):
    # Keep copy of original to verify immutability
    original_copy = messy_dataframe.copy(deep=True)

    cleaned_df, summary = execute_cleaning_pipeline(messy_dataframe)

    # 1. Verify original DataFrame was NOT modified
    pd.testing.assert_frame_equal(messy_dataframe, original_copy)

    # 2. Verify summary metrics
    assert summary["rows_before"] == 40
    assert summary["rows_after"] <= 40
    assert summary["duplicates_removed"] >= 0
    assert summary["missing_values_handled"] > 0
    assert summary["columns_converted"] >= 2  # revenue and signup_date
    assert summary["values_standardized"] > 0
    assert len(summary["operations_applied"]) > 0

    # 3. Verify revenue column converted to float
    assert pd.api.types.is_numeric_dtype(cleaned_df["revenue"])
    assert cleaned_df["revenue"].iloc[0] == 1200.50

    # 4. Verify signup_date column converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["signup_date"])

    # 5. Verify whitespace stripped and categories standardized
    assert not cleaned_df["gender"].astype(str).str.contains(r"^\s|\s$").any()
    assert set(cleaned_df["gender"].unique()).issubset({"Male", "Female"})

    # 6. Verify missing values filled
    assert cleaned_df["age"].isna().sum() == 0


def test_execute_cleaning_pipeline_drop_options(messy_dataframe):
    operations = {
        "remove_duplicates": True,
        "fill_numeric_missing": "median",
        "fill_categorical_missing": "mode",
        "standardize_whitespace": True,
        "standardize_categorical": True,
        "convert_numeric_strings": True,
        "convert_datetime_strings": True,
        "handle_invalid_numerical": True,
        "drop_high_missing_columns": True,
        "high_missing_threshold": 0.5,
        "drop_constant_columns": True,
        "cap_outliers": True,
    }

    cleaned_df, summary = execute_cleaning_pipeline(messy_dataframe, operations)

    # Verify constant column 'country' dropped
    assert "country" not in cleaned_df.columns
    # Verify high-missing column 'optional_notes' (>60% missing) dropped
    assert "optional_notes" not in cleaned_df.columns
    assert "country" in summary["columns_dropped"]
    assert "optional_notes" in summary["columns_dropped"]


# =====================================================================
# 4. PERSISTENCE & IMMUTABILITY TESTS (data/processed/)
# =====================================================================

def test_save_processed_dataset(messy_dataframe, tmp_path):
    # Upload raw file
    csv_bytes = io.BytesIO()
    messy_dataframe.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    file_storage = type("MockFileStorage", (), {
        "filename": "dirty_input.csv",
        "seek": csv_bytes.seek,
        "tell": csv_bytes.tell,
        "save": lambda self, path: csv_bytes.seek(0) or open(path, "wb").write(csv_bytes.read()),
    })()

    dataset_id, raw_path, err, meta = save_uploaded_file(file_storage)
    assert dataset_id is not None
    assert raw_path.exists()
    raw_mtime_before = raw_path.stat().st_mtime

    # Clean dataset
    clean_df, summary = execute_cleaning_pipeline(messy_dataframe)

    # Save to data/processed/
    proc_id, proc_path, save_err = save_processed_dataset(clean_df, dataset_id, summary)
    assert save_err is None
    assert proc_id is not None
    assert proc_path.exists()
    assert "data\\processed" in str(proc_path) or "data/processed" in str(proc_path)

    # Verify raw file was NEVER altered
    assert raw_path.stat().st_mtime == raw_mtime_before
    raw_df, load_err = load_dataset(dataset_id, is_processed=False)
    assert load_err is None
    assert len(raw_df) == 40

    # Verify processed file can be loaded
    proc_df, proc_err = load_dataset(proc_id, is_processed=True)
    assert proc_err is None
    assert len(proc_df) == len(clean_df)


# =====================================================================
# 5. REST API ENDPOINT INTEGRATION TESTS
# =====================================================================

def test_api_cleaning_audit_endpoint(client):
    # Load sample ecommerce dataset
    sample_res = client.post("/api/sample/ecommerce")
    assert sample_res.status_code == 201
    dataset_id = sample_res.get_json()["dataset_id"]

    # Run audit
    audit_res = client.get(f"/api/cleaning/audit/{dataset_id}")
    assert audit_res.status_code == 200
    data = audit_res.get_json()
    assert data["success"] is True
    assert "issues" in data
    assert "severity_counts" in data
    assert "preview" in data


def test_api_cleaning_apply_and_download_endpoints(client):
    # Load sample ecommerce dataset
    sample_res = client.post("/api/sample/ecommerce")
    dataset_id = sample_res.get_json()["dataset_id"]

    # Apply cleaning
    apply_payload = {
        "operations": {
            "remove_duplicates": True,
            "fill_numeric_missing": "median",
            "fill_categorical_missing": "mode",
            "standardize_whitespace": True,
            "standardize_categorical": True,
            "convert_numeric_strings": True,
            "convert_datetime_strings": True,
            "handle_invalid_numerical": True,
            "drop_high_missing_columns": False,
            "drop_constant_columns": False,
            "cap_outliers": False,
        }
    }
    apply_res = client.post(f"/api/cleaning/apply/{dataset_id}", json=apply_payload)
    assert apply_res.status_code == 200
    apply_data = apply_res.get_json()
    assert apply_data["success"] is True
    assert "summary" in apply_data
    assert "download_url" in apply_data

    summary = apply_data["summary"]
    assert "rows_before" in summary
    assert "rows_after" in summary
    assert "missing_values_handled" in summary

    # Download cleaned CSV
    download_res = client.get(f"/api/cleaning/download/{dataset_id}")
    assert download_res.status_code == 200
    assert download_res.content_type.startswith("text/csv")
    assert "attachment" in download_res.headers.get("Content-Disposition", "")
    assert len(download_res.data) > 0


def test_api_cleaning_invalid_dataset_id(client):
    res = client.get("/api/cleaning/audit/non_existent_dataset_12345")
    assert res.status_code == 404
    data = res.get_json()
    assert data["success"] is False
