"""
Test Suite for Phase 13: Automated Analysis Report.
Validates:
1. 9-section report generation (Overview, Quality, Cleaning, Stats, Corrs, Outliers, Viz, Insights, Recommendations)
2. Strict distinction of 4 taxonomy categories (Observed results, Potential issues, AI interpretation, Recommendations)
3. Zero-hallucination factual grounding against DataFrame calculations
4. Markdown, HTML, and JSON format exports
5. Flask REST API routes for report generation and downloads
6. Resilience against edge-case datasets (small samples, pure categorical, 100% clean data)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from app import create_app
from reports.report_generator import (
    ReportGenerator,
    CATEGORY_OBSERVED,
    CATEGORY_ISSUES,
    CATEGORY_AI,
    CATEGORY_RECOMMENDATIONS,
    generate_markdown_report,
)


@pytest.fixture
def realistic_df():
    """Create a realistic sample test DataFrame with continuous, categorical, missing, and outlier values."""
    np.random.seed(42)
    n = 80
    return pd.DataFrame({
        "Customer_ID": [f"CUST-{2000 + i}" for i in range(n)],
        "Age": np.concatenate([np.random.randint(18, 65, size=n - 4), [115, -4, np.nan, np.nan]]),
        "Annual_Income": np.concatenate([np.random.randint(30000, 120000, size=n - 2), [1500000, np.nan]]),
        "Spending_Score": np.random.randint(1, 100, size=n),
        "Department": np.random.choice(["Electronics", "Fashion", "Grocery", "Home"], size=n),
        "Is_Loyalty_Member": np.random.choice([True, False], size=n),
    })


@pytest.fixture
def clean_df():
    """Create a perfectly clean DataFrame without nulls or outliers."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "ID": list(range(1, n + 1)),
        "X": np.linspace(10, 50, n),
        "Y": np.linspace(100, 200, n),
        "Group": ["A", "B"] * (n // 2),
    })


@pytest.fixture
def non_numeric_df():
    """Create a DataFrame with only categorical string columns."""
    return pd.DataFrame({
        "Color": ["Red", "Green", "Blue", "Red", "Blue"],
        "Size": ["S", "M", "L", "XL", "M"],
        "City": ["Tokyo", "Paris", "London", "New York", "Tokyo"],
    })


@pytest.fixture
def app_client():
    """Create Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestReportGeneratorCoreEngine:
    """Test unit capabilities and 9 sections of ReportGenerator."""

    def test_all_nine_sections_generated(self, realistic_df):
        generator = ReportGenerator(realistic_df, dataset_id="test_ds_phase13")
        report = generator.generate_structured_report()

        # Check metadata
        assert "metadata" in report
        assert "report_title" in report["metadata"]
        assert "taxonomy_categories" in report["metadata"]

        # Check all 9 required sections
        expected_sections = [
            "section_1_dataset_overview",
            "section_2_data_quality_assessment",
            "section_3_cleaning_summary",
            "section_4_statistical_analysis",
            "section_5_correlation_analysis",
            "section_6_outlier_analysis",
            "section_7_important_visualizations",
            "section_8_ai_insights",
            "section_9_recommendations",
        ]
        for sec in expected_sections:
            assert sec in report, f"Missing section: {sec}"
            assert isinstance(report[sec], dict)

        # Check section numbers
        assert report["section_1_dataset_overview"]["section_number"] == 1
        assert report["section_2_data_quality_assessment"]["section_number"] == 2
        assert report["section_3_cleaning_summary"]["section_number"] == 3
        assert report["section_4_statistical_analysis"]["section_number"] == 4
        assert report["section_5_correlation_analysis"]["section_number"] == 5
        assert report["section_6_outlier_analysis"]["section_number"] == 6
        assert report["section_7_important_visualizations"]["section_number"] == 7
        assert report["section_8_ai_insights"]["section_number"] == 8
        assert report["section_9_recommendations"]["section_number"] == 9

    def test_four_taxonomy_categories_present(self, realistic_df):
        generator = ReportGenerator(realistic_df, dataset_id="test_ds_phase13")
        report = generator.generate_structured_report()

        # Section 1 has Observed results
        assert report["section_1_dataset_overview"]["category"] == CATEGORY_OBSERVED

        # Section 2 has both Observed results and Potential issues
        assert report["section_2_data_quality_assessment"]["observed_metrics"]["category"] == CATEGORY_OBSERVED
        assert report["section_2_data_quality_assessment"]["potential_issues"]["category"] == CATEGORY_ISSUES

        # Section 3 has Cleaning Recommendations
        assert report["section_3_cleaning_summary"]["recommendations"]["category"] == CATEGORY_RECOMMENDATIONS

        # Section 8 has AI interpretation
        assert report["section_8_ai_insights"]["category"] == CATEGORY_AI

        # Section 9 has Recommendations
        assert report["section_9_recommendations"]["category"] == CATEGORY_RECOMMENDATIONS

    def test_zero_hallucination_factual_grounding(self, realistic_df):
        generator = ReportGenerator(realistic_df, dataset_id="test_ds_phase13")
        report = generator.generate_structured_report()

        # Verify exact row & column count match
        s1_sum = report["section_1_dataset_overview"]["summary"]
        assert s1_sum["total_rows"] == len(realistic_df)
        assert s1_sum["total_columns"] == len(realistic_df.columns)
        assert s1_sum["missing_cells_count"] == int(realistic_df.isna().sum().sum())
        assert s1_sum["duplicate_rows_count"] == int(realistic_df.duplicated().sum())

        # Verify numerical moments match pandas
        s4_stats = report["section_4_statistical_analysis"]["observed_results"]["numerical_moments"]
        if "Spending_Score" in s4_stats:
            score_stats = s4_stats["Spending_Score"]
            expected_mean = float(realistic_df["Spending_Score"].mean())
            assert abs(score_stats["mean"] - expected_mean) < 0.01

    def test_markdown_report_formatting(self, realistic_df):
        generator = ReportGenerator(realistic_df, dataset_id="test_ds_phase13")
        md = generator.generate_markdown_report()

        assert "# Automated Executive Analysis Report" in md or "# Executive Data Analysis" in md
        assert "## 1. Dataset Overview" in md
        assert "## 2. Data Quality Assessment" in md
        assert "## 3. Cleaning Summary" in md
        assert "## 4. Statistical Analysis" in md
        assert "## 5. Correlation Analysis" in md
        assert "## 6. Outlier Analysis" in md
        assert "## 7. Important Visualizations" in md
        assert "## 8. AI-Generated Insights" in md
        assert "## 9. Recommendations" in md

        # Verify taxonomy icons / keys
        assert "Observed results" in md
        assert "Potential issues" in md
        assert "AI interpretation" in md
        assert "Recommendations" in md

    def test_html_report_formatting(self, realistic_df):
        generator = ReportGenerator(realistic_df, dataset_id="test_ds_phase13")
        html = generator.generate_html_report()

        assert "<!DOCTYPE html>" in html
        assert "report-container" in html
        assert "kpi-grid" in html
        assert "1. Dataset Overview" in html
        assert "9. Recommendations" in html
        assert "badge-observed" in html
        assert "badge-issues" in html
        assert "badge-ai" in html
        assert "badge-recs" in html
        assert "Plotly.newPlot" in html or "chart-card" in html

    def test_export_report_to_disk(self, realistic_df, tmp_path):
        generator = ReportGenerator(realistic_df, dataset_id="test_ds_phase13")
        
        md_file = tmp_path / "test_report.md"
        html_file = tmp_path / "test_report.html"
        json_file = tmp_path / "test_report.json"

        p_md = generator.export_report(md_file, report_format="md")
        p_html = generator.export_report(html_file, report_format="html")
        p_json = generator.export_report(json_file, report_format="json")

        assert p_md.exists() and p_md.stat().st_size > 500
        assert p_html.exists() and p_html.stat().st_size > 1000
        assert p_json.exists() and p_json.stat().st_size > 500

    def test_clean_dataset_resilience(self, clean_df):
        generator = ReportGenerator(clean_df, dataset_id="clean_ds")
        report = generator.generate_structured_report()
        assert report["section_1_dataset_overview"]["summary"]["missing_cells_count"] == 0
        assert report["section_1_dataset_overview"]["summary"]["duplicate_rows_count"] == 0
        assert report["section_2_data_quality_assessment"]["observed_metrics"]["health_score"] >= 90

    def test_non_numeric_dataset_resilience(self, non_numeric_df):
        generator = ReportGenerator(non_numeric_df, dataset_id="pure_cat_ds")
        report = generator.generate_structured_report()
        md = generator.generate_markdown_report()
        html = generator.generate_html_report()

        assert report["section_4_statistical_analysis"]["observed_results"]["numerical_columns_count"] == 0
        assert "## 1. Dataset Overview" in md
        assert "<!DOCTYPE html>" in html


class TestReportApiRoutes:
    """Test Flask REST API routes for report generation and downloads."""

    def test_generate_report_api_lifecycle(self, app_client):
        # 1. Ingest sample dataset
        sample_res = app_client.post("/api/sample/ecommerce")
        assert sample_res.status_code == 201
        dataset_id = sample_res.get_json()["dataset_id"]

        # 2. Generate report
        res = app_client.get(f"/api/report/generate/{dataset_id}")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "report" in data
        assert "markdown" in data
        assert "html" in data

        rep = data["report"]
        assert "section_1_dataset_overview" in rep
        assert "section_9_recommendations" in rep

    def test_download_report_formats(self, app_client):
        sample_res = app_client.post("/api/sample/ecommerce")
        assert sample_res.status_code == 201
        dataset_id = sample_res.get_json()["dataset_id"]

        # Download HTML
        dl_html = app_client.get(f"/api/report/download/{dataset_id}?format=html")
        assert dl_html.status_code == 200
        assert dl_html.mimetype == "text/html"
        assert b"<!DOCTYPE html>" in dl_html.data

        # Download Markdown
        dl_md = app_client.get(f"/api/report/download/{dataset_id}?format=md")
        assert dl_md.status_code == 200
        assert dl_md.mimetype == "text/markdown"
        assert b"## 1. Dataset Overview" in dl_md.data

        # Download JSON
        dl_json = app_client.get(f"/api/report/download/{dataset_id}?format=json")
        assert dl_json.status_code == 200
        assert dl_json.mimetype == "application/json"
        assert b"section_1_dataset_overview" in dl_json.data

    def test_generate_report_non_existent_dataset(self, app_client):
        res = app_client.get("/api/report/generate/non_existent_dataset_9999")
        assert res.status_code == 404
        assert res.get_json()["success"] is False
