"""
Unit and Integration Tests for Phase 5: Statistical Analysis Engine.
Verifies numerical moments, inferential statistics, confidence intervals, percentiles,
categorical distributions, datetime periodicity, edge cases, and REST API endpoints.
"""

import math
import pytest
import numpy as np
import pandas as pd
from app import create_app
from analysis.statistics import (
    StatisticalAnalysisEngine,
    compute_descriptive_statistics,
    analyze_numerical_column,
    analyze_categorical_column,
    analyze_datetime_column,
    generate_statistical_observations,
)


@pytest.fixture
def sample_numeric_series():
    """Standard numerical series with known properties."""
    # Data: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    return pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], name="Score")


@pytest.fixture
def test_client():
    """Flask test client fixture."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestNumericalStatistics:
    """Tests for numerical parametric and non-parametric calculations."""

    def test_basic_moments_and_quartiles(self, sample_numeric_series):
        res = analyze_numerical_column(sample_numeric_series, col_name="Score")

        assert res["name"] == "Score"
        assert res["data_type"] == "Numerical"
        assert res["total_count"] == 10
        assert res["valid_count"] == 10
        assert res["missing_count"] == 0
        assert res["mean"] == 55.0
        assert res["median"] == 55.0
        assert res["min"] == 10.0
        assert res["max"] == 100.0
        assert res["range"] == 90.0
        assert res["q1"] == 32.5
        assert res["q3"] == 77.5
        assert res["iqr"] == 45.0
        assert abs(res["skewness"]) < 0.01  # Symmetric
        assert res["variance"] > 0.0
        assert res["std"] > 0.0

    def test_confidence_intervals_and_percentiles(self, sample_numeric_series):
        res = analyze_numerical_column(sample_numeric_series, col_name="Score", confidence_level=0.95)

        ci = res["confidence_interval"]
        assert ci["level"] == 0.95
        assert ci["lower"] is not None
        assert ci["upper"] is not None
        assert ci["lower"] < res["mean"] < ci["upper"]
        assert ci["margin_of_error"] > 0

        # Percentiles
        p = res["percentiles"]
        assert "p1" in p and "p50" in p and "p99" in p
        assert p["p50"] == res["median"]
        assert p["p1"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p99"]

    def test_outlier_detection_via_iqr(self):
        # Create series with extreme outliers
        normal_data = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        outliers = [150, -80]
        series = pd.Series(normal_data + outliers, name="Values")

        res = analyze_numerical_column(series, col_name="Values")
        outliers_info = res["outliers"]

        assert outliers_info["count"] == 2
        assert outliers_info["percentage"] > 0.0
        assert outliers_info["lower_bound"] is not None
        assert outliers_info["upper_bound"] is not None

    def test_constant_column_handling(self):
        constant_s = pd.Series([42.0, 42.0, 42.0, 42.0, 42.0], name="Const")
        res = analyze_numerical_column(constant_s, col_name="Const")

        assert res["is_constant"] is True
        assert res["mean"] == 42.0
        assert res["median"] == 42.0
        assert res["std"] == 0.0
        assert res["variance"] == 0.0
        assert res["range"] == 0.0
        assert res["iqr"] == 0.0
        assert res["outliers"]["count"] == 0
        assert res["confidence_interval"]["lower"] == 42.0
        assert res["confidence_interval"]["upper"] == 42.0

    def test_small_sample_sizes_and_empty(self):
        # n = 0 (empty)
        empty_s = pd.Series([], dtype=float, name="Empty")
        res_empty = analyze_numerical_column(empty_s, col_name="Empty")
        assert res_empty["is_empty"] is True
        assert res_empty["mean"] is None

        # n = 1
        single_s = pd.Series([99.0], name="Single")
        res_single = analyze_numerical_column(single_s, col_name="Single")
        assert res_single["valid_count"] == 1
        assert res_single["mean"] == 99.0
        assert res_single["std"] is None or res_single["std"] == 0.0

        # n = 2
        pair_s = pd.Series([10.0, 20.0], name="Pair")
        res_pair = analyze_numerical_column(pair_s, col_name="Pair")
        assert res_pair["valid_count"] == 2
        assert res_pair["mean"] == 15.0
        assert res_pair["std"] > 0.0

    def test_missing_values_and_dirty_strings(self):
        dirty_s = pd.Series(["$10.50", "20.00", "None", "$30.25", np.nan, "40.0%"], name="Money")
        res = analyze_numerical_column(dirty_s, col_name="Money")

        assert res["total_count"] == 6
        assert res["valid_count"] >= 4
        assert res["missing_count"] <= 2
        assert res["mean"] > 0.0


class TestCategoricalStatistics:
    """Tests for categorical frequencies, entropy, and distributions."""

    def test_categorical_frequencies_and_entropy(self):
        categories = ["Electronics"] * 50 + ["Fashion"] * 30 + ["Books"] * 15 + ["Toys"] * 5
        cat_series = pd.Series(categories, name="Dept")

        res = analyze_categorical_column(cat_series, col_name="Dept")

        assert res["name"] == "Dept"
        assert res["data_type"] == "Categorical"
        assert res["total_count"] == 100
        assert res["num_categories"] == 4
        assert res["most_frequent_category"] == "Electronics"
        assert res["mode_frequency"] == 50
        assert res["mode_percentage"] == 50.0
        assert res["entropy"] is not None and res["entropy"] > 0.0

        top = res["top_categories"]
        assert len(top) == 4
        assert top[0]["category"] == "Electronics"
        assert top[0]["percentage"] == 50.0
        assert top[-1]["cumulative_percentage"] == 100.0


class TestDatetimeStatistics:
    """Tests for datetime ranges and periodicity cadence detection."""

    def test_datetime_range_and_cadence(self):
        dates = pd.date_range(start="2024-01-01", periods=90, freq="D")
        dt_series = pd.Series(dates, name="EventDate")

        res = analyze_datetime_column(dt_series, col_name="EventDate")

        assert res["name"] == "EventDate"
        assert res["data_type"] == "Datetime"
        assert res["total_count"] == 90
        assert "2024-01-01" in str(res["min_date"])
        assert res["date_range_days"] == 89
        assert res["inferred_frequency"] == "Daily"
        assert len(res["records_by_month"]) > 0
        assert len(res["records_by_day_of_week"]) > 0


class TestStatisticalObservations:
    """Tests for deterministic rule-based statistical observations."""

    def test_generates_skewness_and_outlier_observations(self):
        # Heavily skewed with outliers
        skewed_data = [1.0] * 50 + [100.0, 250.0, 500.0]
        df = pd.DataFrame({
            "SkewedCol": skewed_data,
            "CategoryCol": ["A"] * 40 + ["B"] * 10 + ["C"] * 3,
        })

        engine = StatisticalAnalysisEngine(df, dataset_id="test_obs")
        stats_data = engine.analyze()
        obs = stats_data["statistical_observations"]

        assert len(obs) > 0
        titles = [o["title"] for o in obs]
        assert any("Skewness" in t or "Tail" in t or "Dominant" in t or "Outlier" in t for t in titles)


class TestStatisticalEngineExports:
    """Tests for DataFrame and dictionary exports."""

    def test_exports_dataframes(self):
        df = pd.DataFrame({
            "Age": [25, 30, 35, 40, 45, 50],
            "Salary": [50000, 60000, 75000, 80000, 95000, 110000],
            "Dept": ["HR", "IT", "IT", "Sales", "Sales", "Sales"],
        })

        engine = StatisticalAnalysisEngine(df, dataset_id="export_test")
        num_df = engine.numerical_summary_dataframe()
        cat_df = engine.categorical_summary_dataframe()

        assert isinstance(num_df, pd.DataFrame)
        assert len(num_df) == 2  # Age, Salary
        assert "mean" in num_df.columns
        assert "iqr" in num_df.columns

        assert isinstance(cat_df, pd.DataFrame)
        assert len(cat_df) == 1  # Dept
        assert "entropy" in cat_df.columns


class TestStatisticalAPIEndpoints:
    """Integration tests for Flask backend statistics endpoints."""

    def test_health_check_reports_phase_5(self, test_client):
        res = test_client.get("/api/health")
        assert res.status_code == 200
        data = res.get_json()
        assert "Phase 5" in data["phase"]

    def test_statistics_endpoints_with_sample_dataset(self, test_client):
        # Ingest built-in sample dataset
        upload_res = test_client.post("/api/sample/ecommerce")
        assert upload_res.status_code == 201
        upload_data = upload_res.get_json()
        dataset_id = upload_data["dataset_id"]

        # 1. GET /api/statistics/<dataset_id>
        stats_res = test_client.get(f"/api/statistics/{dataset_id}")
        assert stats_res.status_code == 200
        stats_json = stats_res.get_json()
        assert stats_json["success"] is True
        assert "statistics" in stats_json
        st = stats_json["statistics"]
        assert "numerical_statistics" in st
        assert "categorical_statistics" in st
        assert "datetime_statistics" in st
        assert "statistical_observations" in st

        # 2. GET /api/statistics/<dataset_id>/numerical
        num_res = test_client.get(f"/api/statistics/{dataset_id}/numerical")
        assert num_res.status_code == 200
        num_json = num_res.get_json()
        assert num_json["success"] is True
        assert num_json["count"] > 0
        assert len(num_json["columns"]) > 0

        # 3. GET /api/statistics/<dataset_id>/column/<col>
        first_num_col = num_json["columns"][0]["name"]
        col_res = test_client.get(f"/api/statistics/{dataset_id}/column/{first_num_col}")
        assert col_res.status_code == 200
        col_json = col_res.get_json()
        assert col_json["success"] is True
        assert col_json["column_name"] == first_num_col
        assert "percentiles" in col_json["statistics"]

        # 4. GET /api/statistics/<dataset_id>/observations
        obs_res = test_client.get(f"/api/statistics/{dataset_id}/observations")
        assert obs_res.status_code == 200
        obs_json = obs_res.get_json()
        assert obs_json["success"] is True
        assert isinstance(obs_json["observations"], list)
