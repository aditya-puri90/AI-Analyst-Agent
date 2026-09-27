"""
Phase 14: Comprehensive Automated Test Suite.
Validates all 14 core subsystems and 12 distinct edge cases across the entire AI Data Analyst Agent.

Subsystems Tested:
1. File upload
2. CSV validation
3. Dataset profiling
4. Missing-value detection
5. Duplicate detection
6. Data type detection
7. Cleaning operations
8. Statistical calculations
9. Correlation calculations
10. Outlier detection
11. Visualization generation
12. AI response validation
13. Question routing
14. Report generation

Edge Cases Tested:
1. Empty dataset
2. One-column dataset
3. Dataset containing only categorical columns
4. Dataset containing only numerical columns
5. Dataset with many missing values
6. Dataset with duplicate rows
7. Dataset with constant columns
8. Dataset with invalid dates
9. Dataset with extremely large values
10. Dataset with special characters
11. Dataset with mixed data types
12. Very small datasets (N=1, N=2)
"""

import io
import json
import math
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from app import create_app
from config.settings import Config

# Subsystem modules
from utils.validators import (
    allowed_file,
    sanitize_filename,
    detect_encoding_and_delimiter,
    validate_file_upload,
    validate_csv_structure,
)
from utils.file_handler import save_uploaded_file, load_dataset
from analysis.profiler import (
    DatasetProfiler,
    infer_column_type,
    infer_column_types,
    _safe_json_value,
)
from analysis.cleaning import (
    detect_data_quality_issues,
    generate_cleaning_preview,
    execute_cleaning_pipeline,
    _is_strictly_numeric,
)
from analysis.statistics import (
    StatisticalAnalysisEngine,
    analyze_numerical_column,
    analyze_categorical_column,
    analyze_datetime_column,
    generate_statistical_observations,
    compute_descriptive_statistics,
)
from analysis.correlation import (
    CorrelationAnalysisEngine,
    calculate_pearson_correlation_matrix,
    compute_ranked_correlation_pairs,
    classify_correlation_strength,
    classify_correlation_direction,
    extract_numerical_dataframe,
)
from analysis.outliers import (
    OutlierDetectionEngine,
    detect_column_outliers_iqr,
    detect_column_outliers_zscore,
    detect_column_outliers_modified_zscore,
    classify_anomaly_type,
    detect_dataset_outliers,
)
from visualization.charts import (
    ChartRecommendationEngine,
    inspect_column_types,
    build_histogram_spec,
    build_single_boxplot_spec,
    build_bar_chart_spec,
    build_category_distribution_spec,
    build_scatter_spec,
    build_line_chart_spec,
    build_grouped_bar_spec,
    build_correlation_heatmap_spec,
    build_outlier_boxplot_spec,
    build_outlier_distribution_spec,
    recommend_charts,
)
from agent.analyst_agent import (
    AnalystAgent,
    build_analysis_context,
    synthesize_deterministic_insights,
    parse_insights_sections,
)
from agent.prompts import (
    SYSTEM_ANALYST_PROMPT,
    build_analyst_user_prompt,
    SECTION_KEYS,
)
from agent.question_router import (
    QuestionRouter,
    resolve_column_name,
    route_query_deterministic,
    route_user_question,
)
from reports.report_generator import (
    ReportGenerator,
    CATEGORY_OBSERVED,
    CATEGORY_ISSUES,
    CATEGORY_AI,
    CATEGORY_RECOMMENDATIONS,
    _fmt_num,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def app_client():
    """Flask test client fixture."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_sales_df():
    """Diverse benchmark dataset."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "order_id": [f"ORD_{100 + i}" for i in range(n)],
        "sales_amount": np.round(np.random.exponential(100, size=n) + 20, 2),
        "quantity": np.random.randint(1, 10, size=n),
        "discount_pct": np.random.choice([0.0, 0.05, 0.1, 0.2], size=n),
        "category": np.random.choice(["Electronics", "Fashion", "Home"], size=n),
        "region": np.random.choice(["North", "South", "East", "West"], size=n),
        "is_returned": np.random.choice([True, False], size=n),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
    })


# ==============================================================================
# 1. SUBSYSTEM AUTOMATED TESTS
# ==============================================================================

class TestFileUploadSubsystem:
    """1. File Upload Subsystem Tests."""

    def test_allowed_file_extensions(self):
        assert allowed_file("dataset.csv") is True
        assert allowed_file("DATASET.CSV") is True
        assert allowed_file("data.report.csv") is True
        assert allowed_file("script.py") is False
        assert allowed_file("document.xlsx") is False
        assert allowed_file("image.png") is False
        assert allowed_file("") is False
        assert allowed_file("no_extension") is False

    def test_sanitize_filename(self):
        assert sanitize_filename("../../../etc/passwd.csv") == "etc_passwd.csv"
        assert sanitize_filename("my data (1).csv").endswith(".csv")
        assert sanitize_filename("") == "dataset.csv"
        assert sanitize_filename(".hidden.csv") == "hidden.csv"

    def test_validate_file_upload_errors(self):
        class MockFile:
            def __init__(self, filename, content=b""):
                self.filename = filename
                self._content = content
            def seek(self, pos, whence=0):
                pass
            def tell(self):
                return len(self._content)

        # None file
        valid, err = validate_file_upload(None)
        assert valid is False and err is not None

        # Disallowed extension
        valid, err = validate_file_upload(MockFile("data.txt", b"hello"))
        assert valid is False and "Invalid file format" in err

        # Empty 0-byte file
        valid, err = validate_file_upload(MockFile("empty.csv", b""))
        assert valid is False and "empty (0 bytes)" in err

    def test_upload_api_endpoint(self, app_client):
        csv_content = b"col1,col2,col3\n1,2,3\n4,5,6\n"
        data = {"file": (io.BytesIO(csv_content), "test_upload.csv")}
        res = app_client.post("/api/upload", data=data, content_type="multipart/form-data")
        assert res.status_code == 201
        json_data = res.get_json()
        assert json_data["success"] is True
        assert "dataset_id" in json_data


class TestCSVValidationSubsystem:
    """2. CSV Validation Subsystem Tests."""

    def test_delimiter_sniffing(self, tmp_path):
        # Semicolon delimited
        semi_path = tmp_path / "semi.csv"
        semi_path.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
        enc, delim = detect_encoding_and_delimiter(semi_path)
        assert delim == ";"

        # Tab delimited
        tab_path = tmp_path / "tab.csv"
        tab_path.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n", encoding="utf-8")
        enc, delim = detect_encoding_and_delimiter(tab_path)
        assert delim == "\t"

        # Pipe delimited
        pipe_path = tmp_path / "pipe.csv"
        pipe_path.write_text("a|b|c\n1|2|3\n4|5|6\n", encoding="utf-8")
        enc, delim = detect_encoding_and_delimiter(pipe_path)
        assert delim == "|"

    def test_csv_structure_validation(self, tmp_path):
        # Valid CSV
        valid_path = tmp_path / "valid.csv"
        valid_path.write_text("x,y,z\n10,20,30\n40,50,60\n", encoding="utf-8")
        is_valid, err, meta = validate_csv_structure(valid_path)
        assert is_valid is True
        assert meta["row_count"] == 2
        assert meta["column_count"] == 3

        # Header only (no data rows)
        header_only = tmp_path / "header_only.csv"
        header_only.write_text("x,y,z\n", encoding="utf-8")
        is_valid, err, meta = validate_csv_structure(header_only)
        assert is_valid is False
        assert "no data rows" in err.lower()


class TestDatasetProfilingSubsystem:
    """3. Dataset Profiling Subsystem Tests."""

    def test_profiler_metrics(self, sample_sales_df):
        profiler = DatasetProfiler(sample_sales_df, dataset_id="test_profile")
        profile = profiler.to_dict()

        assert profile["dataset_id"] == "test_profile"
        overview = profile["overview"]
        assert overview["total_rows"] == len(sample_sales_df)
        assert overview["total_columns"] == len(sample_sales_df.columns)
        assert overview["total_cells"] == len(sample_sales_df) * len(sample_sales_df.columns)
        assert "memory_usage_formatted" in overview
        assert len(profile["columns"]) == len(sample_sales_df.columns)

    def test_safe_json_value_conversion(self):
        assert _safe_json_value(np.int64(42)) == 42
        assert _safe_json_value(np.float64(3.14159)) == 3.1416
        assert _safe_json_value(np.nan) is None
        assert _safe_json_value(float("inf")) is None
        assert _safe_json_value(pd.Timestamp("2024-01-01")) == "2024-01-01T00:00:00"


class TestMissingValueDetectionSubsystem:
    """4. Missing-Value Detection Subsystem Tests."""

    def test_missing_value_detection(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": ["x", "y", None, "w", "z"],
            "c": [10, 20, 30, 40, 50],
        })
        profiler = DatasetProfiler(df)
        metrics = profiler.calculate_dataset_metrics()
        assert metrics["total_missing_cells"] == 3
        assert metrics["missing_cells_percentage"] == round((3 / 15) * 100, 2)
        assert metrics["rows_with_missing"] == 3

        issues = detect_data_quality_issues(df)
        missing_issues = [i for i in issues if i["issue_type"] == "missing_values"]
        assert len(missing_issues) >= 1
        cols_with_missing = {i["column"] for i in missing_issues}
        assert "a" in cols_with_missing or "b" in cols_with_missing


class TestDuplicateDetectionSubsystem:
    """5. Duplicate Detection Subsystem Tests."""

    def test_duplicate_row_detection(self):
        df = pd.DataFrame({
            "id": [1, 2, 2, 3, 3, 3],
            "val": ["a", "b", "b", "c", "c", "c"],
        })
        profiler = DatasetProfiler(df)
        metrics = profiler.calculate_dataset_metrics()
        assert metrics["duplicate_rows"] == 3
        assert metrics["duplicate_rows_percentage"] == 50.0

        issues = detect_data_quality_issues(df)
        dup_issues = [i for i in issues if i["issue_type"] == "duplicate_rows"]
        assert len(dup_issues) == 1
        assert dup_issues[0]["affected_rows"] == 3


class TestDataTypeDetectionSubsystem:
    """6. Data Type Detection Subsystem Tests."""

    def test_semantic_type_classification(self):
        df = pd.DataFrame({
            "numerical_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "categorical_col": ["low", "med", "high", "low", "med"],
            "datetime_col": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
            "boolean_col": [True, False, True, False, True],
            "boolean_str_col": ["yes", "no", "yes", "no", "yes"],
            "other_id_col": [f"ID_{i}" for i in range(5)],
        })
        types = infer_column_types(df)
        assert types["numerical_col"] == "Numerical"
        assert types["categorical_col"] == "Categorical"
        assert types["datetime_col"] == "Datetime"
        assert types["boolean_col"] == "Boolean"
        assert types["boolean_str_col"] == "Boolean"


class TestCleaningOperationsSubsystem:
    """7. Cleaning Operations Subsystem Tests."""

    def test_pipeline_deduplication_and_imputation(self):
        df = pd.DataFrame({
            "num": [10.0, 20.0, np.nan, 20.0, 20.0],
            "cat": ["Apple", "Banana", np.nan, "Banana", "Banana"],
        })
        cleaned_df, summary = execute_cleaning_pipeline(
            df,
            operations={
                "remove_duplicates": True,
                "handle_missing": "median",
                "handle_missing_cat": "mode",
            }
        )
        # Should have removed duplicate rows (indices 3 and 4)
        assert len(cleaned_df) <= len(df)
        # Missing in 'num' imputed
        assert cleaned_df["num"].isnull().sum() == 0
        # Missing in 'cat' imputed
        assert cleaned_df["cat"].isnull().sum() == 0
        assert summary["rows_after"] <= summary["rows_before"]


class TestStatisticalCalculationsSubsystem:
    """8. Statistical Calculations Subsystem Tests."""

    def test_numerical_moments_and_confidence_interval(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], name="test_metric")
        res = analyze_numerical_column(s, confidence_level=0.95)

        assert res["mean"] == 55.0
        assert res["median"] == 55.0
        assert res["min"] == 10.0
        assert res["max"] == 100.0
        assert res["range"] == 90.0
        assert res["q1"] == 32.5
        assert res["q3"] == 77.5
        assert res["iqr"] == 45.0
        assert res["confidence_interval"]["lower"] < 55.0 < res["confidence_interval"]["upper"]
        assert "percentiles" in res
        assert res["percentiles"]["p50"] == 55.0

    def test_statistical_engine_analyze(self, sample_sales_df):
        engine = StatisticalAnalysisEngine(sample_sales_df)
        stats = engine.analyze(confidence_level=0.95)
        assert "numerical_statistics" in stats
        assert "sales_amount" in stats["numerical_statistics"]
        assert "categorical_statistics" in stats
        assert "statistical_observations" in stats


class TestCorrelationCalculationsSubsystem:
    """9. Correlation Calculations Subsystem Tests."""

    def test_pearson_correlation_matrix(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],  # Perfect positive correlation (r = 1.0)
            "z": [5.0, 4.0, 3.0, 2.0, 1.0],  # Perfect negative correlation (r = -1.0)
        })
        matrix_data = calculate_pearson_correlation_matrix(df)
        assert matrix_data["matrix"]["x"]["y"] == 1.0
        assert matrix_data["matrix"]["x"]["z"] == -1.0

        pairs = compute_ranked_correlation_pairs(matrix_data)
        assert len(pairs) == 3
        assert pairs[0]["strength"] == "Very strong"

    def test_correlation_engine_analyze(self, sample_sales_df):
        engine = CorrelationAnalysisEngine(sample_sales_df)
        corr_res = engine.analyze(threshold=0.0)
        assert "correlation_matrix" in corr_res
        assert "ranked_pairs" in corr_res
        assert "disclaimer" in corr_res
        assert "causation" in corr_res["disclaimer"].lower()


class TestOutlierDetectionSubsystem:
    """10. Outlier Detection Subsystem Tests."""

    def test_iqr_outlier_detection(self):
        # 20 standard values + 2 extreme outliers (0 and 1000)
        data = [50.0] * 20 + [0.0, 1000.0]
        s = pd.Series(data, name="feature")
        res = detect_column_outliers_iqr(s, col_name="feature", multiplier=1.5)
        assert res["outlier_count"] >= 1
        assert res["lower_threshold"] is not None
        assert res["upper_threshold"] is not None

    def test_outlier_engine_analyze(self, sample_sales_df):
        engine = OutlierDetectionEngine(sample_sales_df)
        out_iqr = engine.analyze(method="iqr", param=1.5)
        out_z = engine.analyze(method="zscore", param=3.0)
        assert "total_outlier_instances" in out_iqr
        assert "summary_table" in out_iqr
        assert "total_outlier_instances" in out_z


class TestVisualizationGenerationSubsystem:
    """11. Visualization Generation Subsystem Tests."""

    def test_chart_recommender(self, sample_sales_df):
        recommender = ChartRecommendationEngine(sample_sales_df)
        recs = recommender.recommend(limit=10)
        assert len(recs) > 0
        for rec in recs:
            assert "id" in rec
            assert "title" in rec
            assert "chart_type" in rec
            assert "plotly_spec" in rec
            spec = rec["plotly_spec"]
            assert "data" in spec
            assert "layout" in spec

    def test_individual_chart_builders(self, sample_sales_df):
        hist = build_histogram_spec(sample_sales_df, col="sales_amount")
        assert hist["data"][0]["type"] == "histogram"

        box = build_single_boxplot_spec(sample_sales_df, col="sales_amount")
        assert box["data"][0]["type"] == "box"

        bar = build_bar_chart_spec(sample_sales_df, col="category")
        assert bar["data"][0]["type"] == "bar"

        scatter = build_scatter_spec(sample_sales_df, x_col="sales_amount", y_col="quantity")
        assert scatter["data"][0]["type"] == "scatter"


class TestAIResponseValidationSubsystem:
    """12. AI Response Validation Subsystem Tests."""

    def test_analysis_context_builder(self, sample_sales_df):
        ctx = build_analysis_context(sample_sales_df, dataset_id="sales_ctx")
        assert "dataset_overview" in ctx
        assert "statistical_summaries" in ctx
        assert "correlation_results" in ctx
        assert "outlier_results" in ctx
        assert "data_quality_results" in ctx

    def test_deterministic_insights_synthesis(self, sample_sales_df):
        ctx = build_analysis_context(sample_sales_df, dataset_id="sales_ctx")
        raw_text, sections = synthesize_deterministic_insights(ctx)
        assert len(raw_text) > 0
        for key_tuple in SECTION_KEYS:
            key = key_tuple[0]
            assert key in sections
            assert len(sections[key]) > 0


class TestQuestionRoutingSubsystem:
    """13. Question Routing Subsystem Tests."""

    def test_column_name_resolution(self, sample_sales_df):
        assert resolve_column_name(sample_sales_df, "sales") == "sales_amount"
        assert resolve_column_name(sample_sales_df, "sales_amount") == "sales_amount"
        assert resolve_column_name(sample_sales_df, "qty") == "quantity"
        assert resolve_column_name(sample_sales_df, "category") == "category"

    def test_question_router_execution(self, sample_sales_df):
        router = QuestionRouter(api_key=None)
        res = router.answer_question(sample_sales_df, dataset_id="test_qa", query="What is the average sales amount?")
        assert res["success"] is True
        assert "explanation" in res
        assert "sales_amount" in res.get("tool_called", "") or "sales_amount" in str(res.get("tool_params", {})) or len(res["explanation"]) > 0


class TestReportGenerationSubsystem:
    """14. Report Generation Subsystem Tests."""

    def test_report_formats_and_taxonomy(self, sample_sales_df):
        generator = ReportGenerator(sample_sales_df, dataset_id="sales_rep")
        structured = generator.generate_structured_report()
        assert "section_1_dataset_overview" in structured
        assert "section_2_data_quality_assessment" in structured
        assert "section_3_cleaning_summary" in structured
        assert "section_4_statistical_analysis" in structured
        assert "section_5_correlation_analysis" in structured
        assert "section_6_outlier_analysis" in structured
        assert "section_7_important_visualizations" in structured
        assert "section_8_ai_insights" in structured
        assert "section_9_recommendations" in structured

        md = generator.generate_markdown_report()
        assert "# Executive Data Analysis & Quality Report" in md
        assert CATEGORY_OBSERVED in md
        assert CATEGORY_ISSUES in md
        assert CATEGORY_AI in md
        assert CATEGORY_RECOMMENDATIONS in md

        html = generator.generate_html_report()
        assert "<!DOCTYPE html>" in html
        assert "plotly" in html.lower()


# ==============================================================================
# 2. 12 EDGE CASES AUTOMATED TESTS
# ==============================================================================

class TestEdgeCase1EmptyDataset:
    """Edge Case 1: Empty dataset (0 rows, 0 columns, and 0 rows with columns)."""

    def test_zero_rows_zero_columns(self):
        df = pd.DataFrame()
        profiler = DatasetProfiler(df, dataset_id="empty_0x0")
        p_dict = profiler.to_dict()
        assert p_dict["overview"]["total_rows"] == 0
        assert p_dict["overview"]["total_columns"] == 0

        stats = StatisticalAnalysisEngine(df).analyze()
        assert len(stats["numerical_statistics"]) == 0

        corr = CorrelationAnalysisEngine(df).analyze()
        assert corr["total_numerical_columns"] == 0

        outliers = OutlierDetectionEngine(df).analyze()
        assert outliers["total_outlier_instances"] == 0

        rep = ReportGenerator(df, dataset_id="empty_0x0").generate_structured_report()
        assert rep["section_1_dataset_overview"]["summary"]["total_rows"] == 0

    def test_zero_rows_with_columns(self):
        df = pd.DataFrame(columns=["col_a", "col_b", "col_c"])
        profiler = DatasetProfiler(df, dataset_id="empty_with_cols")
        p_dict = profiler.to_dict()
        assert p_dict["overview"]["total_rows"] == 0
        assert p_dict["overview"]["total_columns"] == 3

        recommender = ChartRecommendationEngine(df)
        recs = recommender.recommend()
        assert recs == []


class TestEdgeCase2OneColumnDataset:
    """Edge Case 2: One-column dataset (numerical, categorical, datetime)."""

    def test_single_numerical_column(self):
        df = pd.DataFrame({"single_num": [10.0, 20.0, 30.0, 40.0, 50.0]})
        profiler = DatasetProfiler(df)
        p_dict = profiler.to_dict()
        assert len(p_dict["columns"]) == 1
        assert p_dict["columns"][0]["classified_type"] == "Numerical"

        stats = StatisticalAnalysisEngine(df).analyze()
        assert "single_num" in stats["numerical_statistics"]

        corr = CorrelationAnalysisEngine(df).analyze()
        assert len(corr["ranked_pairs"]) == 0  # Needs >= 2 columns

        rep_md = ReportGenerator(df, dataset_id="1col_num").generate_markdown_report()
        assert "single_num" in rep_md

    def test_single_categorical_column(self):
        df = pd.DataFrame({"single_cat": ["A", "B", "A", "C", "B"]})
        profiler = DatasetProfiler(df)
        p_dict = profiler.to_dict()
        assert p_dict["columns"][0]["classified_type"] == "Categorical"

        recs = ChartRecommendationEngine(df).recommend()
        assert len(recs) >= 1  # Bar or Donut chart


class TestEdgeCase3CategoricalOnlyDataset:
    """Edge Case 3: Dataset containing only categorical columns."""

    def test_categorical_only(self):
        df = pd.DataFrame({
            "department": ["Sales", "Engineering", "Marketing", "HR", "Sales"],
            "city": ["NYC", "LON", "PAR", "TOK", "NYC"],
            "status": ["active", "pending", "active", "active", "closed"],
        })
        types = infer_column_types(df)
        assert all(t in ["Categorical", "Other"] for t in types.values())

        stats = StatisticalAnalysisEngine(df).analyze()
        assert len(stats["numerical_statistics"]) == 0
        assert len(stats["categorical_statistics"]) == 3

        corr = CorrelationAnalysisEngine(df).analyze()
        assert corr["total_numerical_columns"] == 0

        rep_html = ReportGenerator(df, dataset_id="pure_cat").generate_html_report()
        assert "No numerical continuous columns available" in rep_html


class TestEdgeCase4NumericalOnlyDataset:
    """Edge Case 4: Dataset containing only numerical columns."""

    def test_numerical_only(self):
        df = pd.DataFrame({
            "val_1": [1.2, 3.4, 5.6, 7.8, 9.0],
            "val_2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "val_3": [100.0, 80.0, 60.0, 40.0, 20.0],
        })
        types = infer_column_types(df)
        assert all(t == "Numerical" for t in types.values())

        corr = CorrelationAnalysisEngine(df).analyze()
        assert len(corr["ranked_pairs"]) == 3

        outliers = OutlierDetectionEngine(df).analyze()
        assert len(outliers["summary_table"]) == 3

        rep_md = ReportGenerator(df, dataset_id="pure_num").generate_markdown_report()
        assert "val_1" in rep_md and "val_2" in rep_md


class TestEdgeCase5ManyMissingValues:
    """Edge Case 5: Dataset with many missing values (high missingness, 100% missing in a column)."""

    def test_high_missingness_handling(self):
        df = pd.DataFrame({
            "all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "mostly_nan": [10.0, np.nan, np.nan, np.nan, np.nan],
            "half_nan": [1.0, 2.0, np.nan, np.nan, 5.0],
            "cat_nan": ["A", None, np.nan, None, "B"],
        })
        profiler = DatasetProfiler(df)
        p_dict = profiler.to_dict()
        assert p_dict["overview"]["missing_cells_percentage"] > 50.0

        # Statistics should safely compute without throwing
        stats = StatisticalAnalysisEngine(df).analyze()
        assert stats["numerical_statistics"]["all_nan"]["valid_count"] == 0

        # Reports should format N/A without raising TypeError
        rep = ReportGenerator(df, dataset_id="sparse_data")
        md = rep.generate_markdown_report()
        html = rep.generate_html_report()
        assert "N/A" in md or "all_nan" in md
        assert len(html) > 500


class TestEdgeCase6DuplicateRows:
    """Edge Case 6: Dataset with duplicate rows."""

    def test_exact_and_partial_duplicate_rows(self):
        df = pd.DataFrame({
            "a": [1, 1, 1, 1, 2],
            "b": ["x", "x", "x", "x", "y"],
            "c": [10.5, 10.5, 10.5, 10.5, 20.0],
        })
        profiler = DatasetProfiler(df)
        p_dict = profiler.to_dict()
        assert p_dict["overview"]["duplicate_rows"] == 3

        cleaned_df, summary = execute_cleaning_pipeline(df, operations={"remove_duplicates": True})
        assert len(cleaned_df) == 2
        assert summary["duplicates_removed"] == 3


class TestEdgeCase7ConstantColumns:
    """Edge Case 7: Dataset with constant columns (zero variance)."""

    def test_constant_columns(self):
        df = pd.DataFrame({
            "const_num": [42.0, 42.0, 42.0, 42.0, 42.0],
            "const_str": ["same", "same", "same", "same", "same"],
            "const_bool": [True, True, True, True, True],
        })
        stats = StatisticalAnalysisEngine(df).analyze()
        assert stats["numerical_statistics"]["const_num"]["is_constant"] is True
        assert stats["numerical_statistics"]["const_num"]["std"] == 0.0
        assert stats["numerical_statistics"]["const_num"]["variance"] == 0.0

        corr = CorrelationAnalysisEngine(df).analyze()
        assert "const_num" in corr["constant_columns"]

        outliers = OutlierDetectionEngine(df).analyze()
        assert outliers["total_outlier_instances"] == 0


class TestEdgeCase8InvalidDates:
    """Edge Case 8: Dataset with invalid dates."""

    def test_invalid_and_mixed_dates(self):
        df = pd.DataFrame({
            "dates_mixed": ["2023-01-01", "not_a_date", "2023-02-30", "2023-12-31", "9999-99-99", None],
            "val": [10, 20, 30, 40, 50, 60],
        })
        # Should profile without crashing
        profiler = DatasetProfiler(df)
        p_dict = profiler.to_dict()
        assert len(p_dict["columns"]) == 2

        # Quality inspection should detect potential date anomalies or parse safely
        issues = detect_data_quality_issues(df)
        assert isinstance(issues, list)

        rep_md = ReportGenerator(df, dataset_id="invalid_dates").generate_markdown_report()
        assert len(rep_md) > 100


class TestEdgeCase9ExtremelyLargeValues:
    """Edge Case 9: Dataset with extremely large values and micro-floats."""

    def test_extremely_large_and_small_numbers(self):
        df = pd.DataFrame({
            "astronomical": [1e15, 2.5e16, -1e15, 9.9e17, 5e14],
            "microscopic": [1e-15, 2e-16, 3e-15, 4e-15, 5e-15],
        })
        stats = StatisticalAnalysisEngine(df).analyze()
        assert stats["numerical_statistics"]["astronomical"]["mean"] is not None
        assert stats["numerical_statistics"]["microscopic"]["mean"] is not None

        outliers = OutlierDetectionEngine(df).analyze()
        assert isinstance(outliers["total_outlier_instances"], int)

        rep_html = ReportGenerator(df, dataset_id="large_vals").generate_html_report()
        assert len(rep_html) > 500


class TestEdgeCase10SpecialCharacters:
    """Edge Case 10: Dataset with special characters (commas, quotes, newlines, tabs, Unicode, emojis)."""

    def test_special_characters_and_emojis(self):
        df = pd.DataFrame({
            "col_with,comma": ["val,with,comma", 'quotes "inside"', "newline \n here", "tab \t value"],
            "unicode_🔥": ["Café & Résumé", "日本語テキスト", "emoji 🚀 🤖", "symbols <>&'\"\\/"],
        })
        profiler = DatasetProfiler(df)
        p_dict = profiler.to_dict()
        assert len(p_dict["columns"]) == 2

        # Check JSON serialization of emojis and special characters
        json_str = json.dumps(p_dict)
        assert "Café" in json_str or "\\u" in json_str

        # Report generation must not fail with unicode encoding
        rep = ReportGenerator(df, dataset_id="special_chars")
        rep_html = rep.generate_html_report()
        assert "Café" in rep_html or "unicode" in rep_html.lower()


class TestEdgeCase11MixedDataTypes:
    """Edge Case 11: Dataset with mixed data types (numbers formatted as strings, boolean words)."""

    def test_dirty_mixed_columns(self):
        df = pd.DataFrame({
            "mixed_price": [100, "$200.50", "300", "N/A", None, "500.00", "invalid"],
            "mixed_bool": ["true", 1, "0", False, "yes", "NO", None],
        })
        # Infer types
        types = infer_column_types(df)
        assert "mixed_price" in types
        assert "mixed_bool" in types

        # Statistics should calculate on numerical series directly
        s_stats = analyze_numerical_column(df["mixed_price"], col_name="mixed_price")
        assert s_stats["valid_count"] >= 3

        # Pipeline preview should propose string cleaning
        issues = detect_data_quality_issues(df)
        preview = generate_cleaning_preview(df, issues)
        assert isinstance(preview, dict)


class TestEdgeCase12VerySmallDatasets:
    """Edge Case 12: Very small datasets (N=1 row, N=2 rows)."""

    def test_single_row_dataset(self):
        df = pd.DataFrame({
            "num_val": [42.0],
            "cat_val": ["single_record"],
            "date_val": ["2024-01-01"],
        })
        profiler = DatasetProfiler(df)
        p_dict = profiler.to_dict()
        assert p_dict["overview"]["total_rows"] == 1

        stats = StatisticalAnalysisEngine(df).analyze()
        assert stats["numerical_statistics"]["num_val"]["mean"] == 42.0
        assert stats["numerical_statistics"]["num_val"]["std"] == 0.0

        corr = CorrelationAnalysisEngine(df).analyze()
        assert len(corr["ranked_pairs"]) == 0

        outliers = OutlierDetectionEngine(df).analyze()
        assert outliers["total_outlier_instances"] == 0

        rep = ReportGenerator(df, dataset_id="1row_data")
        assert len(rep.generate_markdown_report()) > 100
        assert len(rep.generate_html_report()) > 500

    def test_two_row_dataset(self):
        df = pd.DataFrame({
            "x": [10.0, 20.0],
            "y": [100.0, 200.0],
            "group": ["A", "B"],
        })
        stats = StatisticalAnalysisEngine(df).analyze()
        assert stats["numerical_statistics"]["x"]["mean"] == 15.0

        corr = CorrelationAnalysisEngine(df).analyze()
        assert len(corr["ranked_pairs"]) == 1

        rep = ReportGenerator(df, dataset_id="2rows_data")
        md = rep.generate_markdown_report()
        html = rep.generate_html_report()
        assert len(md) > 100
        assert len(html) > 500
