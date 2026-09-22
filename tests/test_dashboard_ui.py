"""
Test Suite for Phase 12: Professional Analytics Dashboard UI & Executive Reports.
Validates report generator, API endpoints, structured report schema, Markdown/HTML formatting,
and integration across all analysis engines.
"""

import pytest
import pandas as pd
import numpy as np

from app import create_app
from reports.report_generator import ReportGenerator, generate_markdown_report


@pytest.fixture
def test_df():
    """Create a realistic sample test DataFrame with various types, missing values, and outliers."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "ID": [f"REC-{1000 + i}" for i in range(n)],
        "Age": np.concatenate([np.random.randint(20, 65, size=n - 3), [120, -5, np.nan]]),
        "Salary": np.concatenate([np.random.randint(30000, 120000, size=n - 2), [950000, np.nan]]),
        "Department": np.random.choice(["Sales", "Engineering", "Marketing", "HR"], size=n),
        "Is_Active": np.random.choice([True, False], size=n),
        "Join_Date": pd.date_range("2022-01-01", periods=n, freq="W"),
    })


@pytest.fixture
def app_client():
    """Create Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestReportGeneratorEngine:
    """Test unit capabilities of ReportGenerator."""

    def test_structured_report_generation(self, test_df):
        generator = ReportGenerator(test_df, dataset_id="test_dataset_12")
        report = generator.generate_structured_report()

        assert "title" in report
        assert "overview" in report
        assert "quality" in report
        assert "statistics" in report
        assert "correlations" in report
        assert "outliers" in report
        assert "ai_insights" in report

        # Validate overview metrics
        overview = report["overview"]
        assert overview.get("total_rows") == len(test_df)
        assert overview.get("total_columns") == len(test_df.columns)

        # Validate quality
        quality = report["quality"]
        assert "health_score" in quality
        assert "health_grade" in quality

    def test_markdown_report_formatting(self, test_df):
        generator = ReportGenerator(test_df, dataset_id="test_dataset_12")
        md = generator.generate_markdown_report()

        assert "# Executive Data Analysis & Quality Report" in md
        assert "## 1. Dataset Overview" in md
        assert "## 2. Data Quality Assessment" in md
        assert "## 3. Cleaning Summary" in md
        assert "## 4. Statistical Analysis" in md
        assert "## 5. Correlation Analysis" in md
        assert "## 6. Outlier Analysis" in md
        assert "## 7. Important Visualizations" in md
        assert "## 8. AI-Generated Insights" in md
        assert "## 9. Recommendations" in md

    def test_html_report_formatting(self, test_df):
        generator = ReportGenerator(test_df, dataset_id="test_dataset_12")
        html = generator.generate_html_report()

        assert "<!DOCTYPE html>" in html
        assert "Executive Data Analysis Report" in html
        assert "kpi-grid" in html
        assert "report-container" in html
        assert "Grade:" in html


class TestDashboardRestApi:
    """Test REST API endpoints for Phase 12."""

    def test_health_check_reports_phase_12(self, app_client):
        res = app_client.get("/api/health")
        assert res.status_code == 200
        data = res.get_json()
        assert "Phase 12" in data.get("phase", "")
        assert "executive_reports" in data.get("capabilities", [])
        assert "professional_dashboard_ui" in data.get("capabilities", [])

    def test_sample_and_report_api_lifecycle(self, app_client):
        # 1. Ingest sample dataset
        sample_res = app_client.post("/api/sample/ecommerce")
        assert sample_res.status_code == 201
        sample_data = sample_res.get_json()
        dataset_id = sample_data["dataset_id"]

        # 2. Generate Report API
        rep_res = app_client.get(f"/api/report/generate/{dataset_id}")
        assert rep_res.status_code == 200
        rep_data = rep_res.get_json()
        assert rep_data["success"] is True
        assert "report" in rep_data
        assert "markdown" in rep_data
        assert "html" in rep_data

        # 3. Download HTML Report
        dl_html = app_client.get(f"/api/report/download/{dataset_id}?format=html")
        assert dl_html.status_code == 200
        assert dl_html.mimetype == "text/html"
        assert b"<!DOCTYPE html>" in dl_html.data

        # 4. Download Markdown Report
        dl_md = app_client.get(f"/api/report/download/{dataset_id}?format=md")
        assert dl_md.status_code == 200
        assert dl_md.mimetype == "text/markdown"
        assert b"# Executive Data Analysis" in dl_md.data

    def test_dashboard_view_route(self, app_client):
        res = app_client.get("/dashboard")
        assert res.status_code == 200
        assert b"AI Data Analyst" in res.data
        assert b"id=\"dashboard-view\"" in res.data
        assert b"id=\"preview-view\"" in res.data
        assert b"id=\"quality-view\"" in res.data
        assert b"id=\"cleaning-view\"" in res.data
        assert b"id=\"statistics-view\"" in res.data
        assert b"id=\"correlation-view\"" in res.data
        assert b"id=\"outliers-view\"" in res.data
        assert b"id=\"visualization-view\"" in res.data
        assert b"id=\"insights-view\"" in res.data
        assert b"id=\"chat-view\"" in res.data
        assert b"id=\"reports-view\"" in res.data
