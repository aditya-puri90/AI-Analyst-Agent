"""
Unit and Integration Tests for Phase 7: Outlier & Anomaly Detection Engine.
Verifies IQR, Z-Score, and Modified Z-Score detection algorithms, statistical edge cases
(zero standard deviation, zero IQR, small sample sizes), domain error vs potential outlier
classification, non-destructive remediation, Plotly chart specifications, and Flask REST API endpoints.
"""

import io
import pytest
import numpy as np
import pandas as pd

from app import create_app
from analysis.outliers import (
    OutlierDetectionEngine,
    detect_dataset_outliers,
    detect_column_outliers_iqr,
    detect_column_outliers_zscore,
    detect_column_outliers_modified_zscore,
    classify_anomaly_type,
)
from visualization.charts import build_outlier_boxplot_spec, build_outlier_distribution_spec
from utils.file_handler import save_uploaded_file, load_dataset


@pytest.fixture
def test_client():
    """Flask test client fixture."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def synthetic_benchmark_df():
    """Dataset with deterministic outliers, errors, and edge-cases."""
    # 20 regular scores + 2 outliers
    scores = [70.0, 72.0, 68.0, 75.0, 71.0, 69.0, 73.0, 70.0, 74.0, 71.0,
              68.0, 72.0, 70.0, 73.0, 69.0, 71.0, 74.0, 70.0, 72.0, 68.0,
              15.0, 130.0]  # Outliers: 15.0 and 130.0

    # Ages with 2 confirmed data errors (negative & sentinel)
    ages = [25.0, 30.0, 45.0, 50.0, 35.0, 40.0, 28.0, 52.0, 33.0, 48.0,
            29.0, 42.0, 38.0, 55.0, 22.0, 36.0, 44.0, 51.0, 27.0, 39.0,
            -15.0, -999.0]

    # Discounts with 1 confirmed error (>100%)
    discounts = [5.0, 10.0, 15.0, 20.0, 10.0, 5.0, 15.0, 10.0, 20.0, 5.0,
                 10.0, 15.0, 5.0, 20.0, 10.0, 15.0, 5.0, 10.0, 20.0, 15.0,
                 250.0, 5.0]

    # Constant column (zero standard deviation)
    constant_vals = [42.0] * 22

    # Zero IQR column (>50% zeros)
    zero_iqr_vals = [0.0] * 16 + [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    return pd.DataFrame({
        "Exam_Score": scores,
        "Customer_Age": ages,
        "Discount_Pct": discounts,
        "Constant_Col": constant_vals,
        "Zero_IQR_Col": zero_iqr_vals,
        "Category": ["A", "B"] * 11,
    })


class TestIQRMethod:
    """Tests for Interquartile Range (IQR) outlier detection."""

    def test_basic_iqr_detection(self):
        # 10 values around 50 + 2 extreme values (5, 95)
        series = pd.Series([48.0, 49.0, 50.0, 51.0, 52.0, 50.0, 49.0, 51.0, 50.0, 52.0, 5.0, 95.0], name="Metric")
        res = detect_column_outliers_iqr(series, col_name="Metric", multiplier=1.5)

        assert res["method"] == "iqr"
        assert res["total_observations"] == 12
        assert res["valid_observations"] == 12
        assert res["missing_count"] == 0
        assert res["outlier_count"] == 2
        assert res["outlier_percentage"] == round(2 / 12 * 100, 2)
        assert res["lower_threshold"] < 48.0
        assert res["upper_threshold"] > 52.0
        assert len(res["outliers"]) == 2

    def test_iqr_multipliers(self):
        # Extreme vs standard multiplier
        series = pd.Series([10.0, 11.0, 12.0, 10.0, 11.0, 12.0, 10.0, 11.0, 12.0, 11.0, 25.0, 50.0], name="Val")
        res_15 = detect_column_outliers_iqr(series, col_name="Val", multiplier=1.5)
        res_30 = detect_column_outliers_iqr(series, col_name="Val", multiplier=3.0)

        # 3.0 multiplier has wider bounds and fewer or equal outliers
        assert res_30["upper_threshold"] > res_15["upper_threshold"]
        assert res_30["outlier_count"] <= res_15["outlier_count"]

    def test_zero_iqr_warning(self):
        # 16 zeros and 4 positive numbers -> Q1=0, Q3=0, IQR=0
        series = pd.Series([0.0] * 16 + [10.0, 20.0, 30.0, 40.0], name="ZeroInflated")
        res = detect_column_outliers_iqr(series, col_name="ZeroInflated", multiplier=1.5)

        assert res["iqr"] == 0.0
        assert any("Zero IQR warning" in w for w in res["warnings"])

    def test_small_sample_size_warning_iqr(self):
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 100.0], name="Small")  # N = 5 < 10
        res = detect_column_outliers_iqr(series, col_name="Small", multiplier=1.5)

        assert res["valid_observations"] == 5
        assert any("Small sample size warning" in w for w in res["warnings"])


class TestZScoreMethod:
    """Tests for Parametric Z-Score outlier detection."""

    def test_basic_zscore_detection(self):
        # 30 normal numbers + 1 massive 10-sigma outlier
        np.random.seed(42)
        vals = list(np.random.normal(50, 2, 30)) + [150.0]
        series = pd.Series(vals, name="Values")
        res = detect_column_outliers_zscore(series, col_name="Values", threshold=3.0)

        assert res["method"] == "zscore"
        assert res["total_observations"] == 31
        assert res["outlier_count"] >= 1
        assert res["mean"] is not None
        assert res["std"] is not None

    def test_zero_standard_deviation_warning(self):
        # Constant series
        series = pd.Series([42.0] * 20, name="Constant")
        res = detect_column_outliers_zscore(series, col_name="Constant", threshold=3.0)

        assert res["std"] == 0.0
        assert res["outlier_count"] == 0
        assert res["status"] == "Constant Feature"
        assert any("Zero standard deviation warning" in w for w in res["warnings"])

    def test_small_sample_size_warning_zscore(self):
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], name="Tiny")  # N = 5 < 30
        res = detect_column_outliers_zscore(series, col_name="Tiny", threshold=3.0)

        assert any("Small sample size warning" in w for w in res["warnings"])


class TestModifiedZScoreMethod:
    """Tests for Modified Z-Score (MAD) method."""

    def test_modified_zscore_detection(self):
        series = pd.Series([10.0, 11.0, 10.0, 12.0, 11.0, 10.0, 11.0, 12.0, 10.0, 11.0, 100.0], name="Robust")
        res = detect_column_outliers_modified_zscore(series, col_name="Robust", threshold=3.5)

        assert res["method"] == "modified_zscore"
        assert res["outlier_count"] >= 1
        assert res["mad"] is not None
        assert res["mad"] > 0


class TestDataErrorClassification:
    """Tests for domain error heuristics vs potential outlier classification."""

    def test_negative_age_classified_as_error(self):
        is_err, label, rationale = classify_anomaly_type(
            val=-15.0,
            col_name="Customer_Age",
            lower_bound=10.0,
            upper_bound=80.0,
            series_min=-15.0,
            series_max=75.0,
        )
        assert is_err is True
        assert label == "Confirmed Data Error"
        assert "Negative value" in rationale

    def test_sentinel_value_classified_as_error(self):
        is_err, label, rationale = classify_anomaly_type(
            val=-999.0,
            col_name="Monthly_Income",
            lower_bound=1000.0,
            upper_bound=15000.0,
            series_min=-999.0,
            series_max=12000.0,
        )
        assert is_err is True
        assert label == "Confirmed Data Error"
        assert "sentinel" in rationale.lower()

    def test_percentage_overflow_classified_as_error(self):
        is_err, label, rationale = classify_anomaly_type(
            val=350.0,
            col_name="Discount_Pct",
            lower_bound=0.0,
            upper_bound=40.0,
            series_min=5.0,
            series_max=350.0,
        )
        assert is_err is True
        assert label == "Confirmed Data Error"
        assert "exceeds" in rationale.lower()

    def test_legitimate_extreme_value_classified_as_potential_outlier(self):
        is_err, label, rationale = classify_anomaly_type(
            val=45000.0,
            col_name="Sales_Amount",
            lower_bound=100.0,
            upper_bound=5000.0,
            series_min=50.0,
            series_max=45000.0,
        )
        assert is_err is False
        assert label == "Potential Outlier"
        assert "Extreme statistical observation" in rationale


class TestOutlierDetectionEngine:
    """Tests for full dataset outlier analysis coordinator."""

    def test_analyze_full_dataset(self, synthetic_benchmark_df):
        engine = OutlierDetectionEngine(synthetic_benchmark_df, dataset_id="test_bench")
        res = engine.analyze(method="iqr", param=1.5)

        assert res["dataset_id"] == "test_bench"
        assert res["method"] == "iqr"
        assert res["numerical_columns_count"] == 5
        assert res["total_outlier_instances"] > 0
        assert res["total_confirmed_errors"] >= 3
        assert len(res["summary_table"]) == 5
        assert len(res["warnings"]) > 0

    def test_analyze_specific_column(self, synthetic_benchmark_df):
        engine = OutlierDetectionEngine(synthetic_benchmark_df)
        res = engine.analyze(method="iqr", column="Exam_Score")

        assert len(res["summary_table"]) == 1
        assert res["summary_table"][0]["column_name"] == "Exam_Score"
        assert res["summary_table"][0]["outlier_count"] == 2


class TestNonDestructiveRemediation:
    """Tests for non-destructive dataset remediation operations."""

    def test_remove_outliers_operation(self, synthetic_benchmark_df):
        engine = OutlierDetectionEngine(synthetic_benchmark_df)
        original_rows = len(synthetic_benchmark_df)

        clean_df, summary = engine.remediate_outliers(action="remove", method="iqr")

        # Original is untouched
        assert len(synthetic_benchmark_df) == original_rows
        # New DataFrame has fewer rows
        assert len(clean_df) < original_rows
        assert summary["rows_dropped"] == original_rows - len(clean_df)
        assert summary["action"] == "remove"

    def test_remove_errors_only_operation(self, synthetic_benchmark_df):
        engine = OutlierDetectionEngine(synthetic_benchmark_df)
        original_rows = len(synthetic_benchmark_df)

        clean_df, summary = engine.remediate_outliers(action="remove_errors_only", method="iqr")

        assert len(synthetic_benchmark_df) == original_rows
        assert len(clean_df) < original_rows
        assert summary["rows_dropped"] >= 2

    def test_cap_outliers_operation(self, synthetic_benchmark_df):
        engine = OutlierDetectionEngine(synthetic_benchmark_df)
        original_rows = len(synthetic_benchmark_df)

        capped_df, summary = engine.remediate_outliers(action="cap", method="iqr")

        # No rows dropped during capping
        assert len(capped_df) == original_rows
        assert summary["rows_dropped"] == 0
        assert summary["values_capped"] > 0


class TestPlotlyVisualizations:
    """Tests for interactive Plotly specification generators."""

    def test_boxplot_spec_generation(self, synthetic_benchmark_df):
        spec = build_outlier_boxplot_spec(synthetic_benchmark_df, columns=["Exam_Score", "Customer_Age"])

        assert "data" in spec
        assert "layout" in spec
        assert len(spec["data"]) == 2
        assert spec["data"][0]["type"] == "box"
        assert spec["data"][0]["name"] == "Exam_Score"

    def test_distribution_spec_generation(self, synthetic_benchmark_df):
        series = synthetic_benchmark_df["Exam_Score"]
        spec = build_outlier_distribution_spec(
            series=series,
            col_name="Exam_Score",
            lower_thresh=60.0,
            upper_thresh=85.0,
            method_name="IQR",
        )

        assert "data" in spec
        assert "layout" in spec
        assert len(spec["data"]) >= 1
        assert spec["data"][0]["type"] == "histogram"
        assert "shapes" in spec["layout"]
        assert len(spec["layout"]["shapes"]) >= 2  # Lower and upper threshold shapes


class TestOutlierAPIRoutes:
    """Integration tests for Flask REST API endpoints."""

    def test_get_outliers_endpoint(self, test_client):
        # Generate and ingest ecommerce sample dataset
        sample_res = test_client.post("/api/sample/ecommerce")
        assert sample_res.status_code == 201
        sample_data = sample_res.get_json()
        dataset_id = sample_data["dataset_id"]

        # Call GET /api/outliers/<dataset_id>
        res = test_client.get(f"/api/outliers/{dataset_id}?method=iqr&param=1.5")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "outliers" in data
        assert "boxplot_spec" in data
        assert data["outliers"]["numerical_columns_count"] > 0

    def test_get_column_outliers_endpoint(self, test_client):
        sample_res = test_client.post("/api/sample/employee")
        assert sample_res.status_code == 201
        sample_data = sample_res.get_json()
        dataset_id = sample_data["dataset_id"]

        res = test_client.get(f"/api/outliers/{dataset_id}/column/Salary?method=iqr")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["column_name"] == "Salary"
        assert "result" in data
        assert "distribution_spec" in data

    def test_remediate_outliers_endpoint(self, test_client):
        sample_res = test_client.post("/api/sample/ecommerce")
        assert sample_res.status_code == 201
        sample_data = sample_res.get_json()
        dataset_id = sample_data["dataset_id"]

        # Remediate outliers via POST
        payload = {
            "action": "cap",
            "method": "iqr",
            "param": 1.5,
        }
        res = test_client.post(f"/api/outliers/remediate/{dataset_id}", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "processed_id" in data
        assert "download_url" in data
        assert data["summary"]["action"] == "cap"
