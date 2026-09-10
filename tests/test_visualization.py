"""
Comprehensive Test Suite for Phase 8: Automatic Visualization Engine.
Tests column schema inspection, rule-based chart generation, recommendation ranking,
custom chart builder, edge cases, and REST API endpoints.
"""

import io
import json
import pytest
import pandas as pd
import numpy as np

from app import create_app
from visualization.charts import (
    inspect_column_types,
    build_histogram_spec,
    build_single_boxplot_spec,
    build_bar_chart_spec,
    build_category_distribution_spec,
    build_scatter_spec,
    build_line_chart_spec,
    build_grouped_bar_spec,
    build_time_category_spec,
    ChartRecommendationEngine,
    recommend_charts,
    build_custom_chart_spec,
    build_correlation_heatmap_spec,
    build_outlier_boxplot_spec,
    build_outlier_distribution_spec,
    generate_summary_charts,
)


@pytest.fixture
def sample_sales_df():
    """Create a realistic benchmark DataFrame with diverse column types."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    categories = ["Electronics", "Fashion", "Home", "Sports"]
    regions = ["North", "South", "East", "West"]
    
    return pd.DataFrame({
        "Order_ID": [f"ORD-{1000 + i}" for i in range(n)],
        "Order_Date": dates,
        "Customer_Age": np.random.randint(18, 70, size=n),
        "Category": np.random.choice(categories, size=n),
        "Region": np.random.choice(regions, size=n),
        "Sales_Amount": np.round(np.random.exponential(100, size=n) + 20, 2),
        "Quantity": np.random.randint(1, 10, size=n),
        "Discount_Pct": np.random.choice([0.0, 0.05, 0.1, 0.2], size=n),
        "Is_Returned": np.random.choice([True, False], size=n, p=[0.1, 0.9]),
    })


@pytest.fixture
def sample_edge_df():
    """Create edge-case DataFrame with missing values, constants, and all-NaN columns."""
    return pd.DataFrame({
        "ID": [f"ID_{i}" for i in range(20)],
        "Constant_Num": [42.0] * 20,
        "Constant_Cat": ["Fixed"] * 20,
        "All_NaN": [np.nan] * 20,
        "Few_Valid": [1.0, 2.0, np.nan, np.nan, 5.0] + [np.nan] * 15,
        "Binary_Flag": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0] * 2,
    })


@pytest.fixture
def test_client():
    """Create a test Flask client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ==============================================================================
# 1. Column Semantic Role & Schema Inspection Tests
# ==============================================================================

class TestColumnTypeInspection:
    def test_inspect_column_types_standard(self, sample_sales_df):
        schema = inspect_column_types(sample_sales_df)
        assert "Order_ID" in schema["identifier"]
        assert "Customer_Age" in schema["numerical"]
        assert "Sales_Amount" in schema["numerical"]
        assert "Category" in schema["categorical"]
        assert "Region" in schema["categorical"]
        assert "Order_Date" in schema["datetime"]
        assert "Is_Returned" in schema["boolean"]

    def test_inspect_column_types_edge_cases(self, sample_edge_df):
        schema = inspect_column_types(sample_edge_df)
        assert "ID" in schema["identifier"]
        assert "Constant_Num" not in schema["numerical"]  # 0 variance
        assert "All_NaN" not in schema["numerical"]
        assert "Binary_Flag" in schema["boolean"]

    def test_inspect_empty_dataframe(self):
        schema = inspect_column_types(pd.DataFrame())
        assert schema["numerical"] == []
        assert schema["categorical"] == []


# ==============================================================================
# 2. Rule-Based Plotly Chart Generator Tests
# ==============================================================================

class TestRuleBasedGenerators:
    def test_histogram_generator(self, sample_sales_df):
        spec = build_histogram_spec(sample_sales_df, "Sales_Amount")
        assert "data" in spec and len(spec["data"]) > 0
        assert spec["data"][0]["type"] == "histogram"
        assert "layout" in spec
        assert "Sales_Amount" in spec["layout"]["xaxis"]["title"]["text"]

    def test_single_boxplot_generator(self, sample_sales_df):
        spec = build_single_boxplot_spec(sample_sales_df, "Customer_Age")
        assert "data" in spec and len(spec["data"]) > 0
        assert spec["data"][0]["type"] == "box"
        assert "Customer_Age" in spec["layout"]["yaxis"]["title"]["text"]

    def test_bar_chart_generator(self, sample_sales_df):
        spec = build_bar_chart_spec(sample_sales_df, "Category")
        assert "data" in spec and len(spec["data"]) > 0
        assert spec["data"][0]["type"] == "bar"
        assert len(spec["data"][0]["x"]) <= 16

    def test_category_distribution_donut_generator(self, sample_sales_df):
        spec = build_category_distribution_spec(sample_sales_df, "Region")
        assert "data" in spec and len(spec["data"]) > 0
        assert spec["data"][0]["type"] == "pie"
        assert spec["data"][0]["hole"] > 0

    def test_scatter_generator(self, sample_sales_df):
        spec = build_scatter_spec(sample_sales_df, "Sales_Amount", "Customer_Age", color_col="Category")
        assert "data" in spec and len(spec["data"]) > 0
        scatter_traces = [t for t in spec["data"] if t["type"] == "scatter"]
        assert len(scatter_traces) >= 1
        assert "Sales_Amount" in spec["layout"]["xaxis"]["title"]["text"]

    def test_line_chart_generator(self, sample_sales_df):
        spec = build_line_chart_spec(sample_sales_df, "Order_Date", "Sales_Amount", agg="sum")
        assert "data" in spec and len(spec["data"]) > 0
        assert spec["data"][0]["type"] == "scatter"
        assert spec["data"][0]["mode"] == "lines+markers"

    def test_grouped_bar_generator(self, sample_sales_df):
        spec = build_grouped_bar_spec(sample_sales_df, "Category", "Sales_Amount", agg="mean")
        assert "data" in spec and len(spec["data"]) > 0
        assert spec["data"][0]["type"] == "bar"
        assert "Sales_Amount" in spec["layout"]["yaxis"]["title"]["text"]

    def test_time_category_generator(self, sample_sales_df):
        spec = build_time_category_spec(sample_sales_df, "Order_Date", "Category")
        assert "data" in spec and len(spec["data"]) > 0
        assert spec["layout"]["barmode"] == "stack"

    def test_nonexistent_column_returns_safe_spec(self, sample_sales_df):
        spec = build_histogram_spec(sample_sales_df, "NonExistentColumn")
        assert "data" in spec
        assert "layout" in spec


# ==============================================================================
# 3. Chart Recommendation Engine Tests
# ==============================================================================

class TestChartRecommendationEngine:
    def test_recommendations_generation(self, sample_sales_df):
        engine = ChartRecommendationEngine(sample_sales_df)
        recs = engine.recommend(limit=8)
        assert len(recs) > 0
        assert len(recs) <= 8

        first = recs[0]
        assert "id" in first
        assert "title" in first
        assert "chart_type" in first
        assert "category" in first
        assert "priority" in first
        assert "score" in first
        assert "rationale" in first
        assert "columns_used" in first
        assert "plotly_spec" in first
        assert "data" in first["plotly_spec"]

    def test_recommendations_category_filtering(self, sample_sales_df):
        engine = ChartRecommendationEngine(sample_sales_df)
        dist_recs = engine.recommend(limit=10, category_filter="Distributions")
        for r in dist_recs:
            assert r["category"] == "Distributions"

        rel_recs = engine.recommend(limit=10, category_filter="Relationships")
        for r in rel_recs:
            assert r["category"] == "Relationships"

    def test_recommendations_ranking_order(self, sample_sales_df):
        recs = recommend_charts(sample_sales_df, limit=10)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommendations_excludes_unique_ids(self, sample_sales_df):
        recs = recommend_charts(sample_sales_df, limit=15)
        for r in recs:
            assert "Order_ID" not in r["columns_used"]


# ==============================================================================
# 4. Custom Chart Builder Tests
# ==============================================================================

class TestCustomChartBuilder:
    def test_custom_bar_chart_grouped(self, sample_sales_df):
        spec = build_custom_chart_spec(
            df=sample_sales_df,
            chart_type="bar",
            x_col="Category",
            y_col="Sales_Amount",
            aggregation="sum",
            title="Custom Total Sales by Category",
        )
        assert "data" in spec and len(spec["data"]) > 0
        assert spec["data"][0]["type"] == "bar"

    def test_custom_scatter_chart(self, sample_sales_df):
        spec = build_custom_chart_spec(
            df=sample_sales_df,
            chart_type="scatter",
            x_col="Sales_Amount",
            y_col="Customer_Age",
            color_col="Category",
        )
        assert "data" in spec and len(spec["data"]) > 0

    def test_custom_pie_chart(self, sample_sales_df):
        spec = build_custom_chart_spec(
            df=sample_sales_df,
            chart_type="pie",
            x_col="Region",
            y_col="Sales_Amount",
            aggregation="mean",
        )
        assert "data" in spec and len(spec["data"]) > 0
        assert spec["data"][0]["type"] == "pie"

    def test_custom_histogram(self, sample_sales_df):
        spec = build_custom_chart_spec(
            df=sample_sales_df,
            chart_type="histogram",
            x_col="Customer_Age",
        )
        assert "data" in spec and len(spec["data"]) > 0
        assert spec["data"][0]["type"] == "histogram"

    def test_custom_chart_empty_dataframe(self):
        spec = build_custom_chart_spec(pd.DataFrame(), "bar", "ColA")
        assert "data" in spec


# ==============================================================================
# 5. Backward Compatibility Tests
# ==============================================================================

class TestBackwardCompatibility:
    def test_correlation_heatmap_spec(self, sample_sales_df):
        corr_matrix = {"Sales_Amount": {"Sales_Amount": 1.0, "Customer_Age": 0.45}, "Customer_Age": {"Sales_Amount": 0.45, "Customer_Age": 1.0}}
        spec = build_correlation_heatmap_spec(corr_matrix, ["Sales_Amount", "Customer_Age"])
        assert "data" in spec and len(spec["data"]) > 0
        assert spec["data"][0]["type"] == "heatmap"

    def test_outlier_boxplot_spec(self, sample_sales_df):
        spec = build_outlier_boxplot_spec(sample_sales_df, ["Sales_Amount", "Customer_Age"])
        assert "data" in spec and len(spec["data"]) == 2

    def test_outlier_distribution_spec(self, sample_sales_df):
        spec = build_outlier_distribution_spec(sample_sales_df["Sales_Amount"], "Sales_Amount", 10.0, 300.0)
        assert "data" in spec and len(spec["data"]) >= 1

    def test_generate_summary_charts(self, sample_sales_df):
        summary_charts = generate_summary_charts(sample_sales_df)
        assert isinstance(summary_charts, list)


# ==============================================================================
# 6. REST API Endpoint Tests
# ==============================================================================

class TestVisualizationAPI:
    def test_recommendations_endpoint(self, test_client):
        # Load sample ecommerce dataset
        sample_res = test_client.post("/api/sample/ecommerce")
        assert sample_res.status_code == 201
        ds_id = sample_res.get_json()["dataset_id"]

        # Call recommendations endpoint
        res = test_client.get(f"/api/visualization/recommendations/{ds_id}?limit=8")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["count"] > 0
        assert len(data["recommendations"]) <= 8
        assert "schema" in data

    def test_custom_chart_endpoint(self, test_client):
        sample_res = test_client.post("/api/sample/employee")
        assert sample_res.status_code == 201
        ds_id = sample_res.get_json()["dataset_id"]

        # Request custom bar chart
        res = test_client.post(
            f"/api/visualization/custom/{ds_id}",
            json={
                "chart_type": "bar",
                "x_col": "Department",
                "y_col": "Salary",
                "aggregation": "mean",
                "title": "Average Salary by Department",
            },
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "spec" in data
        assert "data" in data["spec"]

    def test_schema_endpoint(self, test_client):
        sample_res = test_client.post("/api/sample/ecommerce")
        assert sample_res.status_code == 201
        ds_id = sample_res.get_json()["dataset_id"]

        res = test_client.get(f"/api/visualization/schema/{ds_id}")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "schema" in data
        assert "columns" in data
