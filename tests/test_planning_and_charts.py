"""
Automated Test Suite for Phase 11: Intelligent Analysis Planning and Chart Generation.

Verifies:
1. Controlled Python Tool Registry security & registration.
2. Strict prohibition of arbitrary Python code execution.
3. Analysis Planner step generation and visualization decision for all 3 reference examples:
   - Example 1: "How have sales changed over time?" -> Time aggregation, line chart, trend explanation
   - Example 2: "Is advertising spend related to revenue?" -> Pearson correlation, scatter plot, non-causation
   - Example 3: "Which category has the highest revenue?" -> Groupby, aggregation, descending sort, bar chart, result explanation
4. Visualization requirement evaluation (requires_visualization True/False).
5. Chart specification generation (line, scatter, bar, box, histogram, heatmap).
6. Flask REST API routes (/api/chat/ask, /api/chat/plan, /api/chat/tools, /api/health).
"""

import io
import json
import pytest
import pandas as pd
import numpy as np

from app import create_app
from analysis.tool_registry import registry, ToolRegistry, RegisteredTool, ToolExecutionError
from agent.planner import AnalysisPlanner, AnalysisPlan
from agent.question_router import QuestionRouter, route_user_question, session_manager


@pytest.fixture
def marketing_df():
    """Create a standardized realistic marketing and e-commerce dataset for Phase 11 tests."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    categories = np.random.choice(["Electronics", "Fashion", "Home & Kitchen", "Books", "Sports"], size=n)
    regions = np.random.choice(["North", "South", "East", "West"], size=n)

    ad_spend = np.random.uniform(500.0, 5000.0, size=n).round(2)
    # Revenue is positively correlated with ad spend plus organic noise
    revenue = (ad_spend * 3.2 + np.random.normal(1500.0, 500.0, size=n)).round(2)
    sales = (revenue * 0.9).round(2)
    profit = (revenue - ad_spend - 500.0).round(2)

    return pd.DataFrame({
        "Order_Date": dates,
        "Category": categories,
        "Region": regions,
        "Advertising_Spend": ad_spend,
        "Revenue": revenue,
        "Sales_Amount": sales,
        "Profit": profit,
    })


@pytest.fixture
def client(marketing_df):
    """Create Flask test client with an uploaded dataset."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        csv_buffer = io.BytesIO()
        marketing_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        upload_res = client.post(
            "/api/upload",
            data={"file": (csv_buffer, "test_planning_marketing.csv")},
            content_type="multipart/form-data",
        )
        assert upload_res.status_code == 201
        dataset_id = upload_res.get_json()["dataset_id"]
        yield client, dataset_id


# ==============================================================================
# 1. CONTROLLED TOOL REGISTRY & SECURITY TESTS
# ==============================================================================

class TestToolRegistrySecurity:
    """Verify tool registration, schema discovery, and strict security against arbitrary code execution."""

    def test_registered_tools_count(self):
        """Verify all core deterministic analysis tools are registered."""
        tools = registry.get_tool_names()
        assert "dataset_summary" in tools
        assert "column_summary" in tools
        assert "groupby_analysis" in tools
        assert "aggregation_analysis" in tools
        assert "correlation_analysis" in tools
        assert "outlier_analysis" in tools
        assert "time_series_analysis" in tools
        assert "distribution_analysis" in tools
        assert "investigate_further" in tools

    def test_unregistered_tool_execution_blocked(self, marketing_df):
        """Verify executing an unauthorized function raises ToolExecutionError."""
        with pytest.raises(ToolExecutionError) as exc_info:
            registry.execute("execute_arbitrary_python", marketing_df, script="import os; os.system('calc')")
        assert "not an authorized registered analysis function" in str(exc_info.value)
        assert "Arbitrary code execution is strictly prohibited" in str(exc_info.value)

    def test_eval_exec_blocked(self, marketing_df):
        """Verify eval/exec cannot be called via registry."""
        with pytest.raises(ToolExecutionError):
            registry.execute("eval", marketing_df, expression="df.sum()")

        with pytest.raises(ToolExecutionError):
            registry.execute("exec", marketing_df, code="print(1)")

    def test_tool_schema_generation(self):
        """Verify JSON-serializable parameter schemas for registered tools."""
        schemas = registry.list_tools()
        assert len(schemas) >= 9
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert "requires_visualization" in schema


# ==============================================================================
# 2. INTELLIGENT ANALYSIS PLANNING TESTS (3 USER REFERENCE EXAMPLES)
# ==============================================================================

class TestIntelligentAnalysisPlanning:
    """Test multi-step analysis plan generation and chart decisions on reference workflows."""

    def test_example_1_sales_over_time(self, marketing_df):
        """
        Example 1: "How have sales changed over time?"
        Expected Plan:
        1. Detect date column
        2. Detect sales column
        3. Perform time aggregation
        4. Generate line chart
        5. Explain trend
        """
        router = QuestionRouter(provider="deterministic_fallback")
        res = router.route_and_execute(marketing_df, "How have sales changed over time?")

        assert res["success"] is True
        assert res["tool_executed"] == "time_series_analysis"
        assert res["requires_visualization"] is True
        assert res["recommended_chart_type"] == "line"

        plan = res["analysis_plan"]
        assert len(plan) == 5
        assert "Detect date column" in plan[0]
        assert "Detect sales" in plan[1] or "target metric" in plan[1]
        assert "time aggregation" in plan[2]
        assert "Generate line chart" in plan[3]
        assert "Explain trend" in plan[4]

        # Verify visualization spec
        viz = res["visualization"]
        assert viz is not None
        assert viz["chart_type"] == "line"
        assert len(viz["data"]) > 0
        assert viz["data"][0]["type"] == "scatter"

    def test_example_2_ad_spend_related_to_revenue(self, marketing_df):
        """
        Example 2: "Is advertising spend related to revenue?"
        Expected Plan:
        1. Identify advertising spend
        2. Identify revenue
        3. Calculate Pearson correlation
        4. Generate scatter plot
        5. Explain relationship without claiming causation
        """
        router = QuestionRouter(provider="deterministic_fallback")
        res = router.route_and_execute(marketing_df, "Is advertising spend related to revenue?")

        assert res["success"] is True
        assert res["tool_executed"] == "correlation_analysis"
        assert res["requires_visualization"] is True
        assert res["recommended_chart_type"] == "scatter"

        plan = res["analysis_plan"]
        assert len(plan) == 5
        assert "Identify first variable" in plan[0] or "Advertising_Spend" in plan[0]
        assert "Identify second variable" in plan[1] or "Revenue" in plan[1]
        assert "Pearson correlation" in plan[2]
        assert "Generate scatter plot" in plan[3]
        assert "without claiming causation" in plan[4]

        # Verify non-causation explanation
        explanation = res["explanation"]
        assert "caus" in explanation.lower() or "association" in explanation.lower()
        assert "r =" in explanation or "Correlation" in explanation

        # Verify scatter plot
        viz = res["visualization"]
        assert viz is not None
        assert viz["chart_type"] == "scatter"
        assert len(viz["data"]) > 0

    def test_example_3_highest_category_revenue(self, marketing_df):
        """
        Example 3: "Which category has the highest revenue?"
        Expected Plan:
        1. Identify category column
        2. Identify revenue column
        3. Group by category
        4. Aggregate revenue
        5. Sort descending
        6. Generate bar chart
        7. Explain result
        """
        router = QuestionRouter(provider="deterministic_fallback")
        res = router.route_and_execute(marketing_df, "Which category has the highest revenue?")

        assert res["success"] is True
        assert res["tool_executed"] == "groupby_analysis"
        assert res["requires_visualization"] is True
        assert res["recommended_chart_type"] == "bar"

        plan = res["analysis_plan"]
        assert len(plan) == 7
        assert "Identify category" in plan[0] or "Category" in plan[0]
        assert "Identify revenue" in plan[1] or "Revenue" in plan[1]
        assert "Group by" in plan[2]
        assert "Aggregate" in plan[3]
        assert "Sort descending" in plan[4]
        assert "Generate bar chart" in plan[5]
        assert "Explain result" in plan[6]

        # Verify bar chart
        viz = res["visualization"]
        assert viz is not None
        assert viz["chart_type"] == "bar"
        assert len(viz["data"]) > 0

    def test_scalar_aggregation_no_chart_required(self, marketing_df):
        """
        Scalar queries like 'What is the average profit?' should set requires_visualization: False.
        """
        router = QuestionRouter(provider="deterministic_fallback")
        res = router.route_and_execute(marketing_df, "What is the average profit?")

        assert res["success"] is True
        assert res["tool_executed"] == "aggregation_analysis"
        assert res["requires_visualization"] is False
        assert res["recommended_chart_type"] == "none"
        assert res["visualization"] is None
        assert "Average" in res["explanation"] or "mean" in res["explanation"].lower()


# ==============================================================================
# 3. DIRECT ANALYSIS PLANNER METHOD TESTS
# ==============================================================================

class TestAnalysisPlannerDirect:
    """Test direct invocation of AnalysisPlanner.plan()."""

    def test_plan_time_series(self, marketing_df):
        plan = AnalysisPlanner.plan(
            "How have sales changed over time?",
            marketing_df,
            "time_series_analysis",
            {"date_col": "Order_Date", "value_col": "Sales_Amount"},
        )
        assert isinstance(plan, AnalysisPlan)
        assert plan.requires_visualization is True
        assert plan.recommended_chart_type == "line"
        assert len(plan.steps) == 5

    def test_plan_bivariate_correlation(self, marketing_df):
        plan = AnalysisPlanner.plan(
            "Is advertising spend related to revenue?",
            marketing_df,
            "correlation_analysis",
            {"col1": "Advertising_Spend", "col2": "Revenue"},
        )
        assert isinstance(plan, AnalysisPlan)
        assert plan.requires_visualization is True
        assert plan.recommended_chart_type == "scatter"
        assert len(plan.steps) == 5

    def test_plan_groupby_ranking(self, marketing_df):
        plan = AnalysisPlanner.plan(
            "Which category has the highest revenue?",
            marketing_df,
            "groupby_analysis",
            {"group_by_col": "Category", "target_col": "Revenue", "agg_func": "sum", "sort_desc": True},
        )
        assert isinstance(plan, AnalysisPlan)
        assert plan.requires_visualization is True
        assert plan.recommended_chart_type == "bar"
        assert len(plan.steps) == 7

    def test_plan_outlier_detection(self, marketing_df):
        plan = AnalysisPlanner.plan(
            "Are there unusual values in revenue?",
            marketing_df,
            "outlier_analysis",
            {"column_name": "Revenue"},
        )
        assert plan.requires_visualization is True
        assert plan.recommended_chart_type == "box"
        assert len(plan.steps) == 5

    def test_plan_distribution_histogram(self, marketing_df):
        plan = AnalysisPlanner.plan(
            "What is the distribution of Advertising_Spend?",
            marketing_df,
            "distribution_analysis",
            {"column_name": "Advertising_Spend"},
        )
        assert plan.requires_visualization is True
        assert plan.recommended_chart_type == "histogram"
        assert len(plan.steps) == 5


# ==============================================================================
# 4. REST API ENDPOINT TESTS
# ==============================================================================

class TestPlanningAndChartsApiRoutes:
    """Test Flask REST API routes for Phase 11."""

    def test_health_check_phase_11(self, client):
        client_app, _ = client
        res = client_app.get("/api/health")
        assert res.status_code == 200
        data = res.get_json()
        assert "Phase 11" in data["phase"] or "Phase 12" in data["phase"]
        assert "intelligent_analysis_planning" in data["capabilities"]
        assert "controlled_python_tool_registry" in data["capabilities"]

    def test_chat_ask_returns_analysis_plan(self, client):
        client_app, dataset_id = client
        res = client_app.post(
            "/api/chat/ask",
            json={
                "dataset_id": dataset_id,
                "query": "How have sales changed over time?",
            },
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "analysis_plan" in data
        assert len(data["analysis_plan"]) == 5
        assert data["requires_visualization"] is True
        assert data["recommended_chart_type"] == "line"
        assert data["visualization"] is not None

    def test_chat_plan_preview_endpoint(self, client):
        client_app, dataset_id = client
        res = client_app.post(
            "/api/chat/plan",
            json={
                "dataset_id": dataset_id,
                "query": "Which category has the highest revenue?",
            },
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "plan" in data
        plan = data["plan"]
        assert plan["requires_visualization"] is True
        assert plan["recommended_chart_type"] == "bar"
        assert len(plan["steps"]) == 7

    def test_chat_tools_from_registry(self, client):
        client_app, _ = client
        res = client_app.get("/api/chat/tools")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["total_tools"] >= 9
        assert "Controlled" in data["execution_policy"]
