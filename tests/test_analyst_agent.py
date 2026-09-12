"""
Comprehensive Test Suite for Phase 9: AI Insight Engine.
Tests structured context aggregation, prompt engineering, strict grounding,
section parsing, multi-provider dispatching, deterministic fallback synthesis,
and REST API endpoints.
"""

import io
import json
import pytest
import pandas as pd
import numpy as np

from app import create_app
from config.settings import Config
from agent.prompts import (
    SYSTEM_ANALYST_PROMPT,
    build_analyst_user_prompt,
    SECTION_KEYS,
)
from agent.analyst_agent import (
    AnalystAgent,
    build_analysis_context,
    parse_insights_sections,
    synthesize_deterministic_insights,
)
from utils.file_handler import save_uploaded_file


@pytest.fixture
def sample_sales_df():
    """Create a realistic benchmark DataFrame with diverse column types."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    categories = ["Electronics", "Fashion", "Home", "Sports"]
    regions = ["North", "South", "East", "West"]

    df = pd.DataFrame({
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
    # Add a few missing cells
    df.loc[5, "Customer_Age"] = np.nan
    df.loc[10, "Sales_Amount"] = np.nan
    return df


@pytest.fixture
def app_client(sample_sales_df):
    """Create Flask test client and pre-seed a test dataset."""
    app = create_app()
    app.config["TESTING"] = True

    # Ingest a benchmark dataset
    csv_buf = io.BytesIO()
    sample_sales_df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    mock_storage = type("MockStorage", (), {
        "filename": "test_insights_sales.csv",
        "seek": csv_buf.seek,
        "tell": csv_buf.tell,
        "save": lambda self, path: csv_buf.seek(0) or open(path, "wb").write(csv_buf.read()),
    })()

    dataset_id, _, _, _ = save_uploaded_file(mock_storage)
    app.test_dataset_id = dataset_id

    with app.test_client() as client:
        yield client


# ==============================================================================
# 1. STRUCTURED CONTEXT EXTRACTION TESTS
# ==============================================================================

class TestStructuredContextExtraction:
    """Verify that Python analysis engines compile complete, accurate context."""

    def test_context_contains_all_9_structured_areas(self, sample_sales_df):
        context = build_analysis_context(sample_sales_df, dataset_id="test_sales")

        required_keys = [
            "dataset_overview",
            "data_quality_results",
            "cleaning_summary",
            "statistical_summaries",
            "correlation_results",
            "outlier_results",
            "categorical_distributions",
            "time_trends",
            "visualization_metadata",
        ]
        for key in required_keys:
            assert key in context, f"Missing required context key: {key}"

    def test_overview_metrics_accuracy(self, sample_sales_df):
        context = build_analysis_context(sample_sales_df, dataset_id="test_sales")
        overview = context["dataset_overview"]

        assert overview["total_rows"] == len(sample_sales_df)
        assert overview["total_columns"] == len(sample_sales_df.columns)
        assert overview["total_missing_cells"] == int(sample_sales_df.isna().sum().sum())
        assert overview["duplicate_rows_count"] == int(sample_sales_df.duplicated().sum())

    def test_statistical_moments_present(self, sample_sales_df):
        context = build_analysis_context(sample_sales_df, dataset_id="test_sales")
        stats = context["statistical_summaries"]

        assert "numerical_columns" in stats
        assert "Sales_Amount" in stats["numerical_columns"]
        sales_stats = stats["numerical_columns"]["Sales_Amount"]
        assert "mean" in sales_stats
        assert "median" in sales_stats
        assert "std" in sales_stats
        assert "skewness" in sales_stats
        assert "kurtosis" in sales_stats

    def test_correlation_and_outlier_results_present(self, sample_sales_df):
        context = build_analysis_context(sample_sales_df, dataset_id="test_sales")
        corrs = context["correlation_results"]
        outliers = context["outlier_results"]

        assert "ranked_pairs_top" in corrs
        assert "non_causation_disclaimer" in corrs
        assert "total_outliers_detected" in outliers
        assert "detection_method" in outliers

    def test_categorical_and_time_trends_present(self, sample_sales_df):
        context = build_analysis_context(sample_sales_df, dataset_id="test_sales")
        cats = context["categorical_distributions"]
        times = context["time_trends"]

        assert "Category" in cats
        assert len(cats["Category"]["top_categories"]) > 0
        assert "Order_Date" in times
        assert "start_date" in times["Order_Date"]
        assert "end_date" in times["Order_Date"]


# ==============================================================================
# 2. PROMPT & PARSER TESTS
# ==============================================================================

class TestPromptAndParser:
    """Verify system prompts, user prompts, and 8-section parsing."""

    def test_system_prompt_contains_7_grounding_rules(self):
        assert "NEVER INVENT NUMBERS" in SYSTEM_ANALYST_PROMPT
        assert "NEVER INVENT COLUMNS" in SYSTEM_ANALYST_PROMPT
        assert "NEVER CLAIM UNRECORDED CORRELATIONS" in SYSTEM_ANALYST_PROMPT
        assert "NEVER CLAIM CAUSATION FROM CORRELATION" in SYSTEM_ANALYST_PROMPT
        assert "EXPLICIT INSUFFICIENCY OF EVIDENCE" in SYSTEM_ANALYST_PROMPT
        assert "GROUND EVERY NUMERICAL STATEMENT" in SYSTEM_ANALYST_PROMPT
        assert "DISTINGUISH OBSERVATIONS FROM RECOMMENDATIONS" in SYSTEM_ANALYST_PROMPT

    def test_user_prompt_construction(self, sample_sales_df):
        context = build_analysis_context(sample_sales_df, dataset_id="test_sales")
        prompt = build_analyst_user_prompt(context)

        assert "GROUND TRUTH CONTEXT" in prompt
        assert "## 1. Executive Summary" in prompt
        assert "## 2. Key Findings" in prompt
        assert "## 8. Suggested Follow-up Analysis" in prompt

    def test_parse_insights_sections(self):
        sample_markdown = """## 1. Executive Summary
- Strategic overview of 100 rows and 9 columns. Overall health grade is A.

## 2. Key Findings
- Mean sales is 120.5 with std 85.2.

## 3. Important Trends
- Distribution exhibits right skewness of 1.45.

## 4. Important Relationships
- Sales Amount and Quantity show positive association (r = 0.45).

## 5. Data Quality Concerns
- Missing 2 cells in Customer Age (2.0%).

## 6. Potential Outlier Findings
- Identified 3 outliers in Sales Amount exceeding upper bound 340.5.

## 7. Business/Data Recommendations
- Deploy non-destructive winsorization for extreme sales outliers.

## 8. Suggested Follow-up Analysis
- Conduct subgroup analysis across product categories.
"""
        sections = parse_insights_sections(sample_markdown)

        assert "100 rows" in sections["executive_summary"]
        assert "Mean sales" in sections["key_findings"]
        assert "right skewness" in sections["important_trends"]
        assert "positive association" in sections["important_relationships"]
        assert "Missing 2 cells" in sections["data_quality_concerns"]
        assert "3 outliers" in sections["potential_outlier_findings"]
        assert "winsorization" in sections["business_recommendations"]
        assert "subgroup analysis" in sections["suggested_follow_up"]


# ==============================================================================
# 3. DETERMINISTIC SYNTHESIZER TESTS
# ==============================================================================

class TestDeterministicSynthesizer:
    """Verify deterministic fallback engine generates grounded 8 sections."""

    def test_synthesizer_generates_all_8_sections(self, sample_sales_df):
        context = build_analysis_context(sample_sales_df, dataset_id="test_sales")
        raw_md, sections = synthesize_deterministic_insights(context)

        assert len(raw_md) > 500
        for key, _, _ in SECTION_KEYS:
            assert key in sections
            assert len(sections[key]) > 0, f"Empty section: {key}"

    def test_synthesizer_contains_exact_ground_truth_numbers(self, sample_sales_df):
        context = build_analysis_context(sample_sales_df, dataset_id="test_sales")
        _, sections = synthesize_deterministic_insights(context)

        total_rows = sample_sales_df.shape[0]
        assert str(total_rows) in sections["executive_summary"]
        assert "Grade" in sections["executive_summary"]


# ==============================================================================
# 4. ANALYST AGENT CLASS TESTS
# ==============================================================================

class TestAnalystAgent:
    """Verify AnalystAgent initialization, provider checking, and insight generation."""

    def test_agent_initialization_defaults(self):
        agent = AnalystAgent()
        assert agent.provider in ["gemini", "openai", "anthropic", "deterministic_fallback"]
        assert agent.is_provider_configured("deterministic_fallback") is True

    def test_agent_generate_insights_deterministic(self, sample_sales_df):
        agent = AnalystAgent(provider="deterministic_fallback")
        context = agent.build_analysis_context(sample_sales_df, dataset_id="test_sales")
        result = agent.generate_insights(context)

        assert result["success"] is True
        assert "raw_markdown" in result
        assert "sections" in result
        assert len(result["sections"]) == 8
        assert result["metadata"]["generation_mode"] in ["deterministic", "fallback"]
        assert "Strictly Grounded" in result["metadata"]["grounding_status"]

    def test_agent_generate_for_dataset(self, sample_sales_df):
        agent = AnalystAgent()
        result = agent.generate_insights_for_dataset("sales_test", df=sample_sales_df)

        assert result["success"] is True
        assert result["context_summary"]["total_rows"] == len(sample_sales_df)


# ==============================================================================
# 5. REST API ENDPOINT TESTS
# ==============================================================================

class TestInsightsAPIEndpoints:
    """Verify Flask REST API endpoints for Phase 9."""

    def test_get_providers_endpoint(self, app_client):
        res = app_client.get("/api/insights/providers")
        assert res.status_code == 200
        data = res.get_json()

        assert data["success"] is True
        assert "providers" in data
        assert "deterministic_fallback" in data["providers"]
        assert data["providers"]["deterministic_fallback"]["configured"] is True

    def test_get_insights_endpoint(self, app_client):
        dataset_id = app_client.application.test_dataset_id
        res = app_client.get(f"/api/insights/{dataset_id}")
        assert res.status_code == 200
        data = res.get_json()

        assert data["success"] is True
        assert "insights" in data
        assert "sections" in data["insights"]
        assert len(data["insights"]["sections"]) == 8

    def test_post_generate_insights_endpoint(self, app_client):
        dataset_id = app_client.application.test_dataset_id
        res = app_client.post(
            f"/api/insights/generate/{dataset_id}",
            json={"provider": "deterministic_fallback"},
        )
        assert res.status_code == 200
        data = res.get_json()

        assert data["success"] is True
        assert data["insights"]["metadata"]["provider"] == "deterministic_fallback"

    def test_get_insights_context_endpoint(self, app_client):
        dataset_id = app_client.application.test_dataset_id
        res = app_client.get(f"/api/insights/context/{dataset_id}")
        assert res.status_code == 200
        data = res.get_json()

        assert data["success"] is True
        assert "analysis_context" in data
        assert "dataset_overview" in data["analysis_context"]
        assert "statistical_summaries" in data["analysis_context"]
