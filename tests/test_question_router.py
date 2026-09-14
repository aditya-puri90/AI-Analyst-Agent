"""
Tests for Phase 10: Natural-Language Dataset Q&A Engine.
Verifies all 8+ deterministic analysis tools, LLM & NLP intent routing,
fuzzy column resolution, unsupported column handling, grounded explanations,
interactive Plotly visualization specs, session memory, and Flask REST API routes.
"""

import io
import json
import pytest
import pandas as pd
import numpy as np

from app import create_app
from agent.question_router import (
    dataset_summary,
    column_summary,
    groupby_analysis,
    aggregation_analysis,
    correlation_analysis,
    outlier_analysis,
    time_series_analysis,
    distribution_analysis,
    investigate_further,
    route_query_deterministic,
    synthesize_tool_explanation,
    build_qa_visualization_spec,
    resolve_column_name,
    QuestionRouter,
    route_user_question,
    session_manager,
)


@pytest.fixture
def sample_sales_df():
    """Create a standardized realistic e-commerce sales dataset for tests."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    categories = np.random.choice(["Electronics", "Fashion", "Home & Kitchen", "Books", "Sports"], size=n)
    regions = np.random.choice(["North", "South", "East", "West", "Central"], size=n)
    products = np.random.choice(["Laptop Pro", "Wireless Mouse", "Running Shoes", "Desk Lamp", "Novel Book"], size=n)
    sales = np.random.uniform(20.0, 500.0, size=n).round(2)
    # Inject 3 extreme outliers
    sales[10] = 5500.0
    sales[25] = 4800.0
    sales[50] = 6200.0

    quantities = np.random.randint(1, 10, size=n)
    discounts = np.random.choice([0.0, 0.05, 0.1, 0.15, 0.25], size=n)
    ages = np.random.randint(18, 70, size=n).astype(float)
    ages[5] = np.nan

    return pd.DataFrame({
        "Order_Date": dates,
        "Category": categories,
        "Region": regions,
        "Product": products,
        "Sales_Amount": sales,
        "Quantity": quantities,
        "Discount_Pct": discounts,
        "Customer_Age": ages,
    })


@pytest.fixture
def client(sample_sales_df):
    """Create Flask test client with an ingested test dataset."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        # Upload sample dataset to ensure active dataset exists in storage
        csv_buffer = io.BytesIO()
        sample_sales_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        upload_res = client.post(
            "/api/upload",
            data={"file": (csv_buffer, "test_qa_orders.csv")},
            content_type="multipart/form-data",
        )
        assert upload_res.status_code == 201
        dataset_id = upload_res.get_json()["dataset_id"]
        yield client, dataset_id


# ==============================================================================
# 1. INDIVIDUAL SAFE ANALYSIS TOOL TESTS
# ==============================================================================

class TestSafeAnalysisTools:
    """Test individual deterministic analysis tools for calculation accuracy."""

    def test_dataset_summary(self, sample_sales_df):
        res = dataset_summary(sample_sales_df)
        assert res["status"] == "success"
        assert res["total_rows"] == 100
        assert res["total_columns"] == 8
        assert res["memory_usage_mb"] > 0
        assert "columns" in res
        assert len(res["columns"]) == 8

    def test_column_summary_numerical(self, sample_sales_df):
        res = column_summary(sample_sales_df, "Sales_Amount")
        assert res["status"] == "success"
        assert res["column_type"] == "numerical"
        assert res["total_rows"] == 100
        assert res["mean"] > 0
        assert res["min"] > 0
        assert res["max"] >= 6200.0
        assert res["std"] > 0

    def test_column_summary_categorical(self, sample_sales_df):
        res = column_summary(sample_sales_df, "Category")
        assert res["status"] == "success"
        assert res["column_type"] == "categorical"
        assert res["unique_count"] == 5
        assert len(res["top_categories"]) > 0

    def test_column_summary_datetime(self, sample_sales_df):
        res = column_summary(sample_sales_df, "Order_Date")
        assert res["status"] == "success"
        assert res["column_type"] == "datetime"
        assert "earliest_date" in res
        assert "latest_date" in res
        assert res["total_days_span"] == 99

    def test_column_summary_missing_column(self, sample_sales_df):
        res = column_summary(sample_sales_df, "Non_Existent_Col")
        assert res["status"] == "unsupported"
        assert "available_columns" in res

    def test_groupby_analysis_highest_category_sales(self, sample_sales_df):
        res = groupby_analysis(sample_sales_df, group_by_col="Category", target_col="Sales_Amount", agg_func="sum")
        assert res["status"] == "success"
        assert res["group_by_column"] == "Category"
        assert res["target_column"] == "Sales_Amount"
        assert res["top_performer"] is not None
        assert res["top_performer"]["rank"] == 1
        assert res["top_performer"]["value"] > 0
        assert res["top_performer"]["percentage_of_total"] > 0
        assert len(res["groups"]) == 5

    def test_groupby_analysis_best_region(self, sample_sales_df):
        res = groupby_analysis(sample_sales_df, group_by_col="Region", target_col="Sales_Amount", agg_func="sum")
        assert res["status"] == "success"
        assert res["group_by_column"] == "Region"
        assert res["top_performer"] is not None
        assert len(res["groups"]) == 5

    def test_aggregation_analysis_average(self, sample_sales_df):
        res = aggregation_analysis(sample_sales_df, target_col="Sales_Amount", agg_func="mean")
        assert res["status"] == "success"
        assert res["column_name"] == "Sales_Amount"
        assert res["aggregation_function"] == "mean"
        assert res["primary_value"] == round(float(sample_sales_df["Sales_Amount"].mean()), 4)
        assert "$" in res["formatted_value"]

    def test_aggregation_analysis_unsupported_column(self, sample_sales_df):
        res = aggregation_analysis(sample_sales_df, target_col="profit", agg_func="mean")
        assert res["status"] == "unsupported"
        assert "profit" in res["reason"].lower()
        assert "suggested_columns" in res

    def test_correlation_analysis_global(self, sample_sales_df):
        res = correlation_analysis(sample_sales_df)
        assert res["status"] == "success"
        assert res["analysis_type"] == "global_ranking"
        assert "non_causation_notice" in res
        assert "top_correlated_pairs" in res

    def test_correlation_analysis_pairwise(self, sample_sales_df):
        res = correlation_analysis(sample_sales_df, col1="Sales_Amount", col2="Discount_Pct")
        assert res["status"] == "success"
        assert res["analysis_type"] == "pairwise"
        assert "correlation_coefficient" in res
        assert "strength" in res
        assert "non_causation_notice" in res

    def test_outlier_analysis_single_col(self, sample_sales_df):
        res = outlier_analysis(sample_sales_df, column_name="Sales_Amount")
        assert res["status"] == "success"
        assert res["outlier_count"] >= 3
        assert res["outlier_percentage"] > 0
        assert res["upper_threshold"] < 4800.0

    def test_outlier_analysis_dataset_wide(self, sample_sales_df):
        res = outlier_analysis(sample_sales_df)
        assert res["status"] == "success"
        assert res["analysis_type"] == "dataset_wide"
        assert res["total_outliers_count"] >= 3

    def test_time_series_analysis(self, sample_sales_df):
        res = time_series_analysis(sample_sales_df, date_col="Order_Date", value_col="Sales_Amount")
        assert res["status"] == "success"
        assert res["time_span_days"] == 99
        assert len(res["timeline"]) > 0
        assert "peak_period" in res
        assert "trough_period" in res
        assert "trend_direction" in res

    def test_time_series_analysis_unsupported_without_date(self):
        no_date_df = pd.DataFrame({"Category": ["A", "B"], "Sales": [100, 200]})
        res = time_series_analysis(no_date_df)
        assert res["status"] == "unsupported"
        assert "datetime" in res["reason"].lower()

    def test_distribution_analysis_numerical(self, sample_sales_df):
        res = distribution_analysis(sample_sales_df, column_name="Sales_Amount", bins=10)
        assert res["status"] == "success"
        assert res["column_type"] == "numerical"
        assert len(res["bins"]) > 0
        assert "skewness" in res
        assert "dominant_bin" in res

    def test_investigate_further(self, sample_sales_df):
        res = investigate_further(sample_sales_df)
        assert res["status"] == "success"
        assert res["total_recommendations"] > 0
        assert len(res["recommendations"]) > 0
        assert "suggested_query" in res["recommendations"][0]


# ==============================================================================
# 2. INTENT ROUTING & PROMPT QUESTIONS TEST SUITE
# ==============================================================================

class TestQuestionRoutingQueries:
    """Verify routing and execution for the 8 specific queries in requirements."""

    def test_query_1_highest_category_sales(self, sample_sales_df):
        """Which category has the highest sales?"""
        router = QuestionRouter()
        res = router.route_and_execute(sample_sales_df, "Which category has the highest sales?")
        assert res["success"] is True
        assert res["tool_executed"] == "groupby_analysis"
        assert res["tool_result"]["status"] == "success"
        assert res["tool_result"]["group_by_column"] == "Category"
        assert res["tool_result"]["top_performer"] is not None
        assert "highest performing group is" in res["explanation"].lower() or "leading segment" in res["explanation"].lower()
        assert res["visualization"] is not None
        assert res["visualization"]["chart_type"] == "bar"

    def test_query_2_average_profit_missing_column(self, sample_sales_df):
        """What is the average profit? (Dataset has no profit column -> tests missing column handling)"""
        router = QuestionRouter()
        res = router.route_and_execute(sample_sales_df, "What is the average profit?")
        assert res["success"] is True
        assert res["tool_result"]["status"] == "unsupported"
        assert "profit" in res["explanation"].lower()
        assert "available" in res["explanation"].lower()
        assert len(res["followups"]) > 0

    def test_query_3_best_region(self, sample_sales_df):
        """Which region performs best?"""
        router = QuestionRouter()
        res = router.route_and_execute(sample_sales_df, "Which region performs best?")
        assert res["success"] is True
        assert res["tool_executed"] == "groupby_analysis"
        assert res["tool_result"]["group_by_column"] == "Region"
        assert res["tool_result"]["top_performer"] is not None
        assert res["visualization"]["chart_type"] == "bar"

    def test_query_4_strongest_correlations(self, sample_sales_df):
        """What are the strongest correlations?"""
        router = QuestionRouter()
        res = router.route_and_execute(sample_sales_df, "What are the strongest correlations?")
        assert res["success"] is True
        assert res["tool_executed"] == "correlation_analysis"
        assert "correlation" in res["explanation"].lower()
        assert "causal" in res["explanation"].lower() or "association" in res["explanation"].lower()

    def test_query_5_unusual_values(self, sample_sales_df):
        """Are there unusual values?"""
        router = QuestionRouter()
        res = router.route_and_execute(sample_sales_df, "Are there unusual values?")
        assert res["success"] is True
        assert res["tool_executed"] == "outlier_analysis"
        assert res["tool_result"]["total_outliers_count"] >= 3

    def test_query_6_sales_over_time(self, sample_sales_df):
        """How has sales changed over time?"""
        router = QuestionRouter()
        res = router.route_and_execute(sample_sales_df, "How has sales changed over time?")
        assert res["success"] is True
        assert res["tool_executed"] == "time_series_analysis"
        assert res["tool_result"]["time_span_days"] == 99
        assert res["visualization"]["chart_type"] == "line"

    def test_query_7_products_highest_revenue(self, sample_sales_df):
        """Which products have the highest revenue?"""
        router = QuestionRouter()
        res = router.route_and_execute(sample_sales_df, "Which products have the highest revenue?")
        assert res["success"] is True
        assert res["tool_executed"] == "groupby_analysis"
        assert res["tool_result"]["group_by_column"] == "Product"
        assert res["tool_result"]["top_performer"] is not None

    def test_query_8_investigate_further(self, sample_sales_df):
        """What should I investigate further?"""
        router = QuestionRouter()
        res = router.route_and_execute(sample_sales_df, "What should I investigate further?")
        assert res["success"] is True
        assert res["tool_executed"] == "investigate_further"
        assert len(res["tool_result"]["recommendations"]) > 0
        assert "recommendations" in res["explanation"].lower() or "investigation" in res["explanation"].lower()


# ==============================================================================
# 3. SESSION HISTORY & MEMORY TESTS
# ==============================================================================

class TestConversationSessionMemory:
    """Test multi-turn session persistence and history manipulation."""

    def test_session_lifecycle(self, sample_sales_df):
        dataset_id = "test_session_ds"
        session_manager.clear_history(dataset_id)
        assert len(session_manager.get_history(dataset_id)) == 0

        # Ask question 1
        res1 = route_user_question("Which category has the highest sales?", dataset_id=dataset_id, df=sample_sales_df)
        assert res1["success"] is True
        history = session_manager.get_history(dataset_id)
        assert len(history) == 1
        assert history[0]["user_query"] == "Which category has the highest sales?"

        # Ask question 2
        res2 = route_user_question("Are there unusual values?", dataset_id=dataset_id, df=sample_sales_df)
        assert res2["success"] is True
        history = session_manager.get_history(dataset_id)
        assert len(history) == 2

        # Clear history
        session_manager.clear_history(dataset_id)
        assert len(session_manager.get_history(dataset_id)) == 0

    def test_dynamic_suggested_questions(self, sample_sales_df):
        suggestions = session_manager.get_suggested_questions(sample_sales_df, "test_ds")
        assert len(suggestions) >= 4
        assert any("Category" in s or "category" in s for s in suggestions)
        assert any("Sales" in s or "sales" in s or "average" in s for s in suggestions)


# ==============================================================================
# 4. REST API ENDPOINT TESTS
# ==============================================================================

class TestChatRestApi:
    """Test Flask REST API routes for Phase 10."""

    def test_api_chat_ask(self, client):
        c, dataset_id = client
        payload = {
            "dataset_id": dataset_id,
            "query": "Which category has the highest sales?",
        }
        res = c.post("/api/chat/ask", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["tool_executed"] == "groupby_analysis"
        assert "explanation" in data
        assert "visualization" in data
        assert len(data["followups"]) > 0

    def test_api_chat_ask_missing_params(self, client):
        c, dataset_id = client
        res = c.post("/api/chat/ask", json={})
        assert res.status_code == 400
        assert "dataset_id" in res.get_json()["error"]

        res2 = c.post("/api/chat/ask", json={"dataset_id": dataset_id, "query": ""})
        assert res2.status_code == 400
        assert "query" in res2.get_json()["error"]

    def test_api_chat_history_and_clear(self, client):
        c, dataset_id = client
        # Ask a question
        c.post("/api/chat/ask", json={"dataset_id": dataset_id, "query": "What is the average Sales_Amount?"})

        # Get history
        res_hist = c.get(f"/api/chat/history/{dataset_id}")
        assert res_hist.status_code == 200
        hist_data = res_hist.get_json()
        assert hist_data["success"] is True
        assert hist_data["turns_count"] >= 1

        # Clear history
        res_clear = c.post(f"/api/chat/clear/{dataset_id}")
        assert res_clear.status_code == 200
        assert res_clear.get_json()["success"] is True

        # Verify history is now empty
        res_hist_after = c.get(f"/api/chat/history/{dataset_id}")
        assert res_hist_after.get_json()["turns_count"] == 0

    def test_api_chat_suggested_questions(self, client):
        c, dataset_id = client
        res = c.get(f"/api/chat/suggested-questions/{dataset_id}")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert len(data["suggestions"]) >= 4

    def test_api_chat_tools_list(self, client):
        c, _ = client
        res = c.get("/api/chat/tools")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["total_tools"] >= 8
        tool_names = [t["name"] for t in data["tools"]]
        assert "dataset_summary" in tool_names
        assert "groupby_analysis" in tool_names
        assert "time_series_analysis" in tool_names
        assert "correlation_analysis" in tool_names
