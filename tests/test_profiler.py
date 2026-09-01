"""
Comprehensive Automated Test Suite for Phase 3: Automatic Dataset Profiling.
Tests dataset-level metrics, 5-type classification, numerical/categorical/datetime/boolean stats,
data quality scoring, DataFrame/dict conversions, REST API endpoints, and multi-dataset profiling.
"""

import io
import json
import pytest
import numpy as np
import pandas as pd

from app import create_app
from analysis.profiler import (
    DatasetProfiler,
    profile_dataset,
    profile_column,
    infer_column_type,
    infer_column_types,
    get_dataset_preview,
)


@pytest.fixture
def client():
    """Flask test client fixture."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def comprehensive_dataframe():
    """
    Constructs a rich DataFrame with all 5 column types, missing values,
    and duplicates for profiling verification.
    """
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    
    # Intentionally skewed data (e.g. exponential distribution)
    np.random.seed(42)
    skewed_vals = np.round(np.random.exponential(scale=50, size=n) + 10, 2)
    
    df = pd.DataFrame({
        "order_id": [f"ORD-{1000 + i}" for i in range(n)],
        "order_date": dates,
        "customer_age": np.random.randint(18, 75, size=n).astype(float),
        "product_category": np.random.choice(["Electronics", "Clothing", "Home", "Books", "Toys"], size=n),
        "sales_amount": skewed_vals,
        "is_returned": np.random.choice([True, False], size=n, p=[0.15, 0.85]),
        "customer_notes": [f"Customer order note regarding shipment #{i}" if i % 3 == 0 else "Standard delivery" for i in range(n)],
    })

    # Inject missing values
    df.loc[5:9, "customer_age"] = np.nan  # 5 missing in customer_age
    df.loc[20:22, "product_category"] = np.nan  # 3 missing in category
    
    # Inject 2 duplicate rows
    df.iloc[50] = df.iloc[49].copy()
    df.iloc[60] = df.iloc[59].copy()

    return df


# -------------------------------------------------------------
# 1. Dataset-Level Metrics Tests
# -------------------------------------------------------------
def test_dataset_level_metrics(comprehensive_dataframe):
    profiler = DatasetProfiler(comprehensive_dataframe, dataset_id="comp_test")
    profile = profiler.profile()
    overview = profile["overview"]

    assert overview["total_rows"] == 100
    assert overview["total_columns"] == 7
    assert overview["total_cells"] == 700
    assert overview["total_missing_cells"] == 8  # 5 in age, 3 in category
    assert overview["missing_cells_percentage"] == round(8 / 700 * 100, 2)
    assert overview["duplicate_rows"] == 2
    assert overview["duplicate_rows_percentage"] == 2.0
    assert overview["memory_usage_bytes"] > 0
    assert "KB" in overview["memory_usage_formatted"] or "Bytes" in overview["memory_usage_formatted"]


# -------------------------------------------------------------
# 2. Data Quality & Health Score Tests
# -------------------------------------------------------------
def test_data_quality_scoring(comprehensive_dataframe):
    profiler = DatasetProfiler(comprehensive_dataframe)
    profile = profiler.profile()
    quality = profile["quality"]

    assert 0 <= quality["health_score"] <= 100
    assert quality["health_grade"] in ["A+", "A", "B", "C", "D", "F"]
    assert quality["completeness_score"] == round(100.0 - (8 / 700 * 100), 2)
    assert quality["uniqueness_score"] == 98.0  # 100 - 2% duplicates
    assert quality["passed_checks"] >= 1
    assert len(quality["warnings"]) > 0


# -------------------------------------------------------------
# 3. Five-Category Classification Tests
# -------------------------------------------------------------
def test_five_type_classification():
    # 1. Numerical
    assert infer_column_type(pd.Series([10, 20, 30, 40, 50], name="prices")) == "Numerical"
    assert infer_column_type(pd.Series([1.5, 2.7, 3.9, 4.1], name="ratios")) == "Numerical"

    # 2. Categorical
    assert infer_column_type(pd.Series(["Red", "Blue", "Green", "Red", "Blue"], name="colors")) == "Categorical"
    assert infer_column_type(pd.Series(["Male", "Female", "Female", "Male"], name="gender")) == "Categorical"

    # 3. Datetime
    assert infer_column_type(pd.Series(pd.date_range("2024-01-01", periods=10))) == "Datetime"
    assert infer_column_type(pd.Series(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], name="dates")) == "Datetime"

    # 4. Boolean
    assert infer_column_type(pd.Series([True, False, True, True], name="is_verified")) == "Boolean"
    assert infer_column_type(pd.Series(["yes", "no", "yes", "no"], name="opt_in")) == "Boolean"
    assert infer_column_type(pd.Series([1, 0, 1, 0, 1], name="is_active")) == "Boolean"

    # 5. Other (high-cardinality IDs and text)
    assert infer_column_type(pd.Series([f"UUID_{i}_{np.random.randint(10000, 99999)}" for i in range(50)], name="session_id")) == "Other"


# -------------------------------------------------------------
# 4. Numerical Column Profiling Statistics Tests
# -------------------------------------------------------------
def test_numerical_column_statistics():
    # Known distribution: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    nums = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    series = pd.Series(nums, name="revenue")
    
    col_profile = profile_column(series, "revenue")
    assert col_profile["classified_type"] == "Numerical"
    assert col_profile["non_null_count"] == 10
    assert col_profile["missing_count"] == 0
    assert col_profile["unique_count"] == 10
    assert col_profile["duplicate_count"] == 0

    ns = col_profile["numerical_stats"]
    assert ns is not None
    assert ns["mean"] == 55.0
    assert ns["median"] == 55.0
    assert ns["min"] == 10.0
    assert ns["max"] == 100.0
    assert ns["q1"] == 32.5
    assert ns["q3"] == 77.5
    assert ns["iqr"] == 45.0
    assert ns["std"] == round(float(np.std(nums, ddof=1)), 4)
    assert abs(ns["skewness"]) < 0.01  # Symmetrical distribution has ~0 skew


# -------------------------------------------------------------
# 5. Categorical Column Profiling Statistics Tests
# -------------------------------------------------------------
def test_categorical_column_statistics():
    cats = ["Electronics", "Fashion", "Electronics", "Home", "Electronics", "Fashion"]
    series = pd.Series(cats, name="department")

    col_profile = profile_column(series, "department")
    assert col_profile["classified_type"] == "Categorical"
    assert col_profile["non_null_count"] == 6
    assert col_profile["unique_count"] == 3
    assert col_profile["duplicate_count"] == 3  # 6 non-null - 3 unique = 3 duplicate observations

    cs = col_profile["categorical_stats"]
    assert cs is not None
    assert cs["num_categories"] == 3
    assert cs["most_frequent_category"] == "Electronics"
    assert cs["frequency_most_frequent"] == 3
    assert cs["frequency_percentage"] == 50.0  # 3 / 6 = 50%
    assert len(cs["top_categories"]) == 3


# -------------------------------------------------------------
# 6. Datetime Column Profiling Statistics Tests
# -------------------------------------------------------------
def test_datetime_column_statistics():
    dates = ["2023-01-01", "2023-06-15", "2023-12-31"]
    series = pd.Series(dates, name="timestamp")

    col_profile = profile_column(series, "timestamp")
    assert col_profile["classified_type"] == "Datetime"

    ds = col_profile["datetime_stats"]
    assert ds is not None
    assert "2023-01-01" in ds["min_date"]
    assert "2023-12-31" in ds["max_date"]
    assert "364 days" in ds["date_range"] or "365 days" in ds["date_range"]


# -------------------------------------------------------------
# 7. Boolean Column Profiling Statistics Tests
# -------------------------------------------------------------
def test_boolean_column_statistics():
    bools = [True, True, True, False, True]
    series = pd.Series(bools, name="is_subscriber")

    col_profile = profile_column(series, "is_subscriber")
    assert col_profile["classified_type"] == "Boolean"

    bs = col_profile["boolean_stats"]
    assert bs is not None
    assert bs["true_count"] == 4
    assert bs["false_count"] == 1
    assert bs["true_percentage"] == 80.0
    assert bs["false_percentage"] == 20.0


# -------------------------------------------------------------
# 8. Output Formats (DataFrame and JSON Dictionary)
# -------------------------------------------------------------
def test_profiler_export_methods(comprehensive_dataframe):
    profiler = DatasetProfiler(comprehensive_dataframe, dataset_id="export_test")
    
    # 1. Dictionary export
    profile_dict = profiler.to_dict()
    assert isinstance(profile_dict, dict)
    assert "overview" in profile_dict
    assert "quality" in profile_dict
    assert "columns" in profile_dict
    
    # JSON serializability check (must not raise TypeError)
    json_str = json.dumps(profile_dict)
    assert len(json_str) > 0

    # 2. DataFrame export
    profile_df = profiler.to_dataframe()
    assert isinstance(profile_df, pd.DataFrame)
    assert len(profile_df) == 7
    assert "column_name" in profile_df.columns
    assert "inferred_type" in profile_df.columns
    assert "missing_count" in profile_df.columns
    assert "mean" in profile_df.columns


# -------------------------------------------------------------
# 9. Edge Cases Profiling Tests
# -------------------------------------------------------------
def test_profiler_edge_cases():
    # 1. All-NaN column
    df_nan = pd.DataFrame({"all_nulls": [np.nan, np.nan, np.nan]})
    prof_nan = profile_dataset(df_nan)
    assert prof_nan["overview"]["total_rows"] == 3
    assert prof_nan["columns"][0]["missing_count"] == 3
    assert prof_nan["columns"][0]["missing_percentage"] == 100.0

    # 2. Constant column
    df_const = pd.DataFrame({"constant_col": [42, 42, 42, 42]})
    prof_const = profile_dataset(df_const)
    assert prof_const["columns"][0]["is_constant"] is True
    assert prof_const["columns"][0]["unique_count"] == 1

    # 3. Single-row dataset
    df_single = pd.DataFrame({"age": [30], "city": ["Berlin"], "active": [True]})
    prof_single = profile_dataset(df_single)
    assert prof_single["overview"]["total_rows"] == 1
    assert prof_single["columns"][0]["numerical_stats"]["mean"] == 30.0

    # 4. Empty DataFrame
    df_empty = pd.DataFrame()
    prof_empty = profile_dataset(df_empty)
    assert prof_empty["overview"]["total_rows"] == 0
    assert prof_empty["overview"]["total_columns"] == 0


# -------------------------------------------------------------
# 10. Multi-Dataset Profiling Tests on Diverse Datasets
# -------------------------------------------------------------
def test_profiler_on_multiple_datasets():
    # Dataset 1: Financial Stocks
    stocks_df = pd.DataFrame({
        "Ticker": ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"],
        "Price": [185.50, 175.20, 420.10, 180.00, 120.40],
        "PE_Ratio": [32.5, 26.1, 35.8, 45.2, 58.0],
        "Dividend_Yield": [0.005, 0.0, 0.007, 0.0, 0.002],
        "Is_Tech": [True, True, True, True, True],
    })
    stocks_prof = profile_dataset(stocks_df)
    assert stocks_prof["overview"]["total_rows"] == 5
    assert stocks_prof["type_counts"]["Numerical"] == 3
    assert stocks_prof["type_counts"]["Boolean"] == 1

    # Dataset 2: Students Gradebook with Missing & Semicolon Names
    students_df = pd.DataFrame({
        "Student_ID": [f"STU_{i}" for i in range(1, 21)],
        "Math_Score": np.random.randint(50, 100, size=20).astype(float),
        "Grade": np.random.choice(["A", "B", "C", "D"], size=20),
        "Passed": np.random.choice([True, False], size=20),
    })
    students_df.loc[3, "Math_Score"] = np.nan
    students_prof = profile_dataset(students_df)
    assert students_prof["overview"]["total_rows"] == 20
    assert students_prof["overview"]["total_missing_cells"] == 1
    assert students_prof["type_counts"]["Categorical"] >= 1

    # Dataset 3: Heavy Duplication and Skewness
    skewed_df = pd.DataFrame({
        "metric": [1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 500.0, 1000.0],
        "status": ["OK", "OK", "OK", "OK", "OK", "WARN", "ERROR", "CRITICAL"],
    })
    skewed_prof = profile_dataset(skewed_df)
    assert skewed_prof["columns"][0]["numerical_stats"]["skewness"] > 1.0


# -------------------------------------------------------------
# 11. REST API Profiling Endpoints Tests
# -------------------------------------------------------------
def test_api_profiling_endpoints(client):
    # Ingest ecommerce sample
    upload_res = client.post("/api/sample/ecommerce")
    assert upload_res.status_code == 201
    dataset_id = upload_res.get_json()["dataset_id"]

    # 1. Test /api/profile/<dataset_id>
    prof_res = client.get(f"/api/profile/{dataset_id}")
    assert prof_res.status_code == 200
    prof_data = prof_res.get_json()
    assert prof_data["success"] is True
    assert "overview" in prof_data["profile"]
    assert "quality" in prof_data["profile"]
    assert "columns" in prof_data["profile"]
    assert prof_data["profile"]["overview"]["total_rows"] == 250

    # 2. Test /api/profile/<dataset_id>/columns
    cols_res = client.get(f"/api/profile/{dataset_id}/columns")
    assert cols_res.status_code == 200
    cols_data = cols_res.get_json()
    assert cols_data["success"] is True
    assert cols_data["total_columns"] == 10

    # Test filtering columns by type
    num_cols_res = client.get(f"/api/profile/{dataset_id}/columns?type=Numerical")
    assert num_cols_res.status_code == 200
    num_cols_data = num_cols_res.get_json()
    for col in num_cols_data["columns"]:
        assert col["classified_type"] == "Numerical"

    # 3. Test /api/profile/<dataset_id>/column/<column_name>
    single_col_res = client.get(f"/api/profile/{dataset_id}/column/Sales_Amount")
    assert single_col_res.status_code == 200
    single_col_data = single_col_res.get_json()
    assert single_col_data["success"] is True
    assert single_col_data["column"]["name"] == "Sales_Amount"
    assert single_col_data["column"]["classified_type"] == "Numerical"
    assert "mean" in single_col_data["column"]["numerical_stats"]

    # Test column not found
    bad_col_res = client.get(f"/api/profile/{dataset_id}/column/NonExistentColumn")
    assert bad_col_res.status_code == 404
