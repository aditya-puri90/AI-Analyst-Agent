"""
Unit and Integration Tests for Phase 6: Correlation Analysis Engine.
Tests numerical identification, Pearson correlation matrix, threshold filtering,
strength/direction classification, edge cases (missing values, constant features),
Plotly heatmap generation, and REST API endpoints.
"""

import pytest
import numpy as np
import pandas as pd
from io import BytesIO

from app import create_app
from analysis.correlation import (
    classify_correlation_strength,
    classify_correlation_direction,
    extract_numerical_dataframe,
    calculate_pearson_correlation_matrix,
    compute_ranked_correlation_pairs,
    generate_key_correlations,
    CorrelationAnalysisEngine,
    compute_correlation_analysis,
    DISCLAIMER_TEXT,
)
from visualization.charts import build_correlation_heatmap_spec
from utils.file_handler import save_uploaded_file


@pytest.fixture
def test_client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_numeric_df():
    """Synthetic dataset with known linear relationships."""
    np.random.seed(42)
    n = 100
    x = np.linspace(10, 100, n)
    # Perfect positive
    y_pos = 2.5 * x + 10.0
    # Strong negative
    y_neg = -1.8 * x + np.random.normal(0, 15, n)
    # Moderate positive
    y_mod = 0.8 * x + np.random.normal(0, 30, n)
    # Weak / random noise
    y_noise = np.random.normal(50, 20, n)
    # Constant column
    y_const = np.full(n, 42.0)

    df = pd.DataFrame({
        "Feature_X": x,
        "Perfect_Pos": y_pos,
        "Strong_Neg": y_neg,
        "Moderate_Pos": y_mod,
        "Noise_Var": y_noise,
        "Constant_Col": y_const,
        "Category_Col": ["Cat_A", "Cat_B"] * 50,
    })
    return df


class TestCorrelationClassification:
    """Test correlation magnitude strength and direction categorization."""

    def test_strength_tiers(self):
        assert classify_correlation_strength(1.0) == "Very strong"
        assert classify_correlation_strength(0.85) == "Very strong"
        assert classify_correlation_strength(-0.80) == "Very strong"

        assert classify_correlation_strength(0.79) == "Strong"
        assert classify_correlation_strength(-0.60) == "Strong"

        assert classify_correlation_strength(0.55) == "Moderate"
        assert classify_correlation_strength(-0.40) == "Moderate"

        assert classify_correlation_strength(0.35) == "Weak"
        assert classify_correlation_strength(-0.20) == "Weak"

        assert classify_correlation_strength(0.15) == "Very weak"
        assert classify_correlation_strength(0.0) == "Very weak"
        assert classify_correlation_strength(-0.05) == "Very weak"

        assert classify_correlation_strength(None) == "Undefined"

    def test_direction_classification(self):
        assert classify_correlation_direction(0.75) == "Positive"
        assert classify_correlation_direction(0.001) == "Positive"
        assert classify_correlation_direction(-0.75) == "Negative"
        assert classify_correlation_direction(-0.001) == "Negative"
        assert classify_correlation_direction(0.0) == "Neutral"
        assert classify_correlation_direction(None) == "Neutral"


class TestCorrelationCalculations:
    """Test Pearson correlation calculations and edge case handling."""

    def test_numerical_extraction(self, sample_numeric_df):
        num_df, num_cols, const_cols = extract_numerical_dataframe(sample_numeric_df)
        assert "Category_Col" not in num_cols
        assert "Feature_X" in num_cols
        assert "Perfect_Pos" in num_cols
        assert "Constant_Col" in const_cols
        assert len(num_cols) == 6

    def test_pearson_matrix_and_relationships(self, sample_numeric_df):
        matrix_data = calculate_pearson_correlation_matrix(sample_numeric_df)
        matrix = matrix_data["matrix"]
        
        # Self-correlation is 1.0
        assert matrix["Feature_X"]["Feature_X"] == 1.0
        
        # Perfect positive correlation
        assert matrix["Feature_X"]["Perfect_Pos"] == 1.0
        
        # Strong negative correlation (should be < -0.80)
        neg_r = matrix["Feature_X"]["Strong_Neg"]
        assert neg_r < -0.80
        
        # Constant column safely yields 0.0 without ZeroDivisionError
        assert matrix["Feature_X"]["Constant_Col"] == 0.0

    def test_missing_values_pairwise_handling(self):
        df = pd.DataFrame({
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0],
            "B": [2.0, 4.0, np.nan, 8.0, 10.0, 12.0, 14.0, 16.0],
        })
        matrix_data = calculate_pearson_correlation_matrix(df)
        r = matrix_data["matrix"]["A"]["B"]
        assert r == 1.0
        assert matrix_data["sample_sizes"]["A"]["B"] == 6

    def test_zero_variance_and_small_samples(self):
        # Empty DataFrame
        empty_res = calculate_pearson_correlation_matrix(pd.DataFrame())
        assert empty_res["columns"] == []

        # Single row DataFrame
        single_res = calculate_pearson_correlation_matrix(pd.DataFrame({"A": [1], "B": [2]}))
        assert single_res["matrix"]["A"]["B"] == 0.0


class TestRankedPairsAndKeyCorrelations:
    """Test ranked correlation associations and non-causal key highlight cards."""

    def test_ranked_pairs_sorting_and_threshold(self, sample_numeric_df):
        matrix_data = calculate_pearson_correlation_matrix(sample_numeric_df)
        all_pairs = compute_ranked_correlation_pairs(matrix_data, threshold=0.0)
        
        # Verify sorted descending by absolute correlation
        for k in range(len(all_pairs) - 1):
            assert abs(all_pairs[k]["correlation"]) >= abs(all_pairs[k + 1]["correlation"])

        # Test threshold filter |r| >= 0.5
        filtered_pairs = compute_ranked_correlation_pairs(matrix_data, threshold=0.5)
        for p in filtered_pairs:
            assert abs(p["correlation"]) >= 0.5

    def test_key_correlations_structure(self, sample_numeric_df):
        engine = CorrelationAnalysisEngine(sample_numeric_df, dataset_id="test_ds")
        analysis = engine.analyze()

        key_corr = analysis["key_correlations"]
        assert "strongest_positive" in key_corr
        assert "strongest_negative" in key_corr
        assert "key_highlights" in key_corr

        highlights = key_corr["key_highlights"]
        assert len(highlights) > 0

        # Check required fields and non-causation phrasing
        first_h = highlights[0]
        assert "pair" in first_h
        assert "correlation" in first_h
        assert "direction" in first_h
        assert "strength" in first_h
        assert "Non-causal" in first_h["explanation"]
        assert analysis["disclaimer"] == DISCLAIMER_TEXT


class TestPlotlyHeatmapSpec:
    """Test interactive Plotly Heatmap JSON builder."""

    def test_heatmap_spec_builder(self, sample_numeric_df):
        matrix_data = calculate_pearson_correlation_matrix(sample_numeric_df)
        spec = build_correlation_heatmap_spec(
            corr_matrix=matrix_data["matrix"],
            columns=matrix_data["columns"]
        )

        assert "data" in spec
        assert "layout" in spec
        assert "config" in spec
        
        trace = spec["data"][0]
        assert trace["type"] == "heatmap"
        assert trace["zmin"] == -1.0
        assert trace["zmax"] == 1.0
        assert len(trace["x"]) == len(matrix_data["columns"])


class TestCorrelationAPIEndpoints:
    """Test Flask REST API routes for Correlation Analysis."""

    def test_correlation_endpoints(self, test_client):
        # Create a sample ecommerce dataset
        res_sample = test_client.post("/api/sample/ecommerce")
        assert res_sample.status_code == 201
        ds_id = res_sample.get_json()["dataset_id"]

        # 1. Main correlation endpoint
        res = test_client.get(f"/api/correlation/{ds_id}")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "correlation" in data
        assert "heatmap_spec" in data

        corr = data["correlation"]
        assert corr["total_numerical_columns"] >= 3
        assert len(corr["ranked_pairs"]) > 0
        assert "key_correlations" in corr

        # 2. Pairs endpoint with threshold
        res_pairs = test_client.get(f"/api/correlation/{ds_id}/pairs?threshold=0.1")
        assert res_pairs.status_code == 200
        pairs_data = res_pairs.get_json()
        assert pairs_data["success"] is True
        assert pairs_data["threshold"] == 0.1

        # 3. Non-existent dataset returns 404
        res_404 = test_client.get("/api/correlation/non_existent_dataset_12345")
        assert res_404.status_code == 404
