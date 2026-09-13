"""
Main Flask Application for AI Data Analyst Agent.
Provides REST API endpoints and web views for dataset uploading, ingestion, profiling, and exploration.
"""

import io
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np

from config.settings import Config
from utils.validators import validate_file_upload
from utils.file_handler import (
    save_uploaded_file,
    load_dataset,
    get_dataset_summary,
    list_uploaded_datasets,
    get_file_path,
    save_processed_dataset,
    get_latest_processed_file,
)
from analysis.profiler import (
    DatasetProfiler,
    profile_dataset,
    profile_column,
    infer_column_type,
    get_dataset_preview,
    _safe_json_value,
)
from analysis.cleaning import (
    detect_data_quality_issues,
    generate_cleaning_preview,
    execute_cleaning_pipeline,
)
from analysis.statistics import (
    StatisticalAnalysisEngine,
    compute_descriptive_statistics,
    analyze_numerical_column,
    analyze_categorical_column,
    analyze_datetime_column,
    generate_statistical_observations,
)
from analysis.correlation import (
    CorrelationAnalysisEngine,
    compute_correlation_analysis,
    classify_correlation_strength,
    classify_correlation_direction,
)
from analysis.outliers import (
    OutlierDetectionEngine,
    detect_dataset_outliers,
    detect_column_outliers_iqr,
    detect_column_outliers_zscore,
    detect_column_outliers_modified_zscore,
    classify_anomaly_type,
)
from visualization.charts import (
    build_correlation_heatmap_spec,
    build_outlier_boxplot_spec,
    build_outlier_distribution_spec,
    ChartRecommendationEngine,
    recommend_charts,
    build_custom_chart_spec,
    inspect_column_types,
)
from agent.analyst_agent import (
    AnalystAgent,
    build_analysis_context,
    synthesize_deterministic_insights,
    parse_insights_sections,
)
from agent.question_router import (
    QuestionRouter,
    route_user_question,
    session_manager,
    dataset_summary,
    column_summary,
    groupby_analysis,
    aggregation_analysis,
    correlation_analysis,
    outlier_analysis,
    time_series_analysis,
    distribution_analysis,
    investigate_further,
)



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application Factory for AI Data Analyst Agent."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Ensure required directories exist
    Config.init_app()

    # -------------------------------------------------------------
    # Error Handlers
    # -------------------------------------------------------------
    @app.errorhandler(413)
    def request_entity_too_large(error):
        max_mb = Config.MAX_CONTENT_LENGTH / (1024 * 1024)
        return jsonify({
            "success": False,
            "error": f"File size exceeds the maximum allowable limit of {max_mb:.0f} MB.",
        }), 413

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Requested resource was not found."}), 404
        return render_template("index.html"), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        logger.error("Internal Server Error: %s", error, exc_info=True)
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500

    # -------------------------------------------------------------
    # Page / View Routes
    # -------------------------------------------------------------
    @app.route("/")
    def index():
        """Render home landing & dataset upload page."""
        return render_template("index.html")

    @app.route("/dashboard")
    def dashboard():
        """Render analytics dashboard page."""
        dataset_id = request.args.get("dataset_id", "")
        return render_template("dashboard.html", dataset_id=dataset_id)

    @app.route("/chat")
    def chat():
        """Render AI Analyst conversational interface."""
        dataset_id = request.args.get("dataset_id", "")
        return render_template("chat.html", dataset_id=dataset_id)

    # -------------------------------------------------------------
    # REST API Routes
    # -------------------------------------------------------------
    @app.route("/api/health", methods=["GET"])
    def health_check():
        """System health and status check."""
        return jsonify({
            "status": "healthy",
            "service": "AI Data Analyst Agent",
            "phase": "Phase 5 - Statistical Analysis & Phase 6 - Correlation & Phase 7 - Outliers & Phase 8 - Visualization & Phase 9 - AI Insights & Phase 10 - Natural-Language Dataset Q&A ('Ask Your Dataset')",
            "max_upload_mb": Config.MAX_CONTENT_LENGTH / (1024 * 1024),
            "allowed_extensions": list(Config.ALLOWED_EXTENSIONS),
        })

    @app.route("/api/upload", methods=["POST"])
    def upload_file():
        """
        Handle secure CSV file upload, validation, ingestion, and extraction of summary metadata and profile.
        """
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file part in the request. Please choose a CSV file."}), 400

        file_storage = request.files["file"]
        if not file_storage or not file_storage.filename:
            return jsonify({"success": False, "error": "No file was selected for upload."}), 400

        dataset_id, saved_path, error, meta = save_uploaded_file(file_storage)
        if error:
            return jsonify({"success": False, "error": error}), 400

        # Extract complete ingestion summary
        summary, sum_err = get_dataset_summary(dataset_id)
        if sum_err:
            return jsonify({"success": False, "error": sum_err}), 400

        # Full Phase 3 profile
        df, load_err = load_dataset(dataset_id)
        profile = profile_dataset(df, dataset_id=dataset_id) if not load_err else {}

        return jsonify({
            "success": True,
            "message": f"Dataset '{file_storage.filename}' uploaded and profiled successfully.",
            "dataset_id": dataset_id,
            "original_filename": file_storage.filename,
            "summary": summary,
            "profile": profile,
        }), 201

    @app.route("/api/summary/<dataset_id>", methods=["GET"])
    def get_summary(dataset_id: str):
        """Retrieve the ingestion summary and first 10 rows for a dataset."""
        summary, error = get_dataset_summary(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404
        return jsonify({"success": True, "summary": summary})

    @app.route("/api/datasets", methods=["GET"])
    def get_datasets():
        """List all uploaded datasets with metadata."""
        datasets = list_uploaded_datasets()
        return jsonify({"success": True, "count": len(datasets), "datasets": datasets})

    @app.route("/api/profile/<dataset_id>", methods=["GET"])
    def get_profile(dataset_id: str):
        """Retrieve the automated full data profile for a dataset."""
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        profiler = DatasetProfiler(df, dataset_id=dataset_id)
        profile = profiler.to_dict()
        return jsonify({"success": True, "profile": profile})

    @app.route("/api/profile/<dataset_id>/columns", methods=["GET"])
    def get_column_profiles(dataset_id: str):
        """Retrieve column-level profiling metrics for a dataset."""
        col_type_filter = request.args.get("type", "").strip()

        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        profiler = DatasetProfiler(df, dataset_id=dataset_id)
        profile = profiler.to_dict()
        columns = profile.get("columns", [])

        if col_type_filter:
            columns = [c for c in columns if c.get("classified_type", "").lower() == col_type_filter.lower()]

        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "total_columns": len(columns),
            "columns": columns,
        })

    @app.route("/api/profile/<dataset_id>/column/<column_name>", methods=["GET"])
    def get_single_column_profile(dataset_id: str, column_name: str):
        """Retrieve detailed profiling metrics for an individual column."""
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        if column_name not in df.columns:
            return jsonify({
                "success": False,
                "error": f"Column '{column_name}' not found in dataset '{dataset_id}'. Available columns: {list(df.columns)}",
            }), 404

        profiler = DatasetProfiler(df, dataset_id=dataset_id)
        col_profile = profiler.profile_column(column_name)
        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "column": col_profile,
        })

    @app.route("/api/preview/<dataset_id>", methods=["GET"])
    def get_preview(dataset_id: str):
        """Retrieve paginated tabular preview records for a dataset."""
        page = request.args.get("page", 1, type=int)
        page_size = request.args.get("page_size", 20, type=int)

        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        preview_data = get_dataset_preview(df, page=page, page_size=page_size)
        return jsonify({"success": True, **preview_data})

    @app.route("/api/statistics/<dataset_id>", methods=["GET"])
    def get_statistics(dataset_id: str):
        """
        Retrieve comprehensive Phase 5 statistical analysis:
        - Numerical descriptive & inferential moments (mean, std, var, min, max, range, IQR, skewness, kurtosis, CI, percentiles)
        - Categorical distributions and entropy
        - Datetime periodicity and temporal breakdown
        - Rule-based deterministic statistical observations
        """
        confidence_level = request.args.get("ci", 0.95, type=float)
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        engine = StatisticalAnalysisEngine(df, dataset_id=dataset_id)
        stats_data = engine.analyze(confidence_level=confidence_level)
        return jsonify({"success": True, "statistics": stats_data})

    @app.route("/api/statistics/<dataset_id>/numerical", methods=["GET"])
    def get_numerical_statistics(dataset_id: str):
        """Retrieve tabular summary of numerical descriptive statistics."""
        confidence_level = request.args.get("ci", 0.95, type=float)
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        engine = StatisticalAnalysisEngine(df, dataset_id=dataset_id)
        stats_data = engine.analyze(confidence_level=confidence_level)
        num_stats = stats_data.get("numerical_statistics", {})
        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "count": len(num_stats),
            "columns": list(num_stats.values()),
        })

    @app.route("/api/statistics/<dataset_id>/column/<column_name>", methods=["GET"])
    def get_column_statistics(dataset_id: str, column_name: str):
        """Retrieve deep statistical profile for an individual column."""
        confidence_level = request.args.get("ci", 0.95, type=float)
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        if column_name not in df.columns:
            return jsonify({
                "success": False,
                "error": f"Column '{column_name}' not found in dataset '{dataset_id}'. Available columns: {list(df.columns)}",
            }), 404

        series = df[column_name]
        col_type = infer_column_type(series)

        if col_type == "Numerical":
            col_stat = analyze_numerical_column(series, col_name=column_name, confidence_level=confidence_level)
        elif col_type == "Categorical" or col_type == "Boolean":
            col_stat = analyze_categorical_column(series, col_name=column_name)
        elif col_type == "Datetime":
            col_stat = analyze_datetime_column(series, col_name=column_name)
        else:
            if pd.api.types.is_numeric_dtype(series):
                col_stat = analyze_numerical_column(series, col_name=column_name, confidence_level=confidence_level)
            else:
                col_stat = analyze_categorical_column(series, col_name=column_name)

        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "column_name": column_name,
            "statistics": col_stat,
        })

    @app.route("/api/statistics/<dataset_id>/observations", methods=["GET"])
    def get_statistical_observations_route(dataset_id: str):
        """Retrieve rule-based statistical observations and data patterns."""
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        engine = StatisticalAnalysisEngine(df, dataset_id=dataset_id)
        stats_data = engine.analyze()
        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "observations": stats_data.get("statistical_observations", []),
        })

    # -------------------------------------------------------------
    # Phase 6: Correlation Analysis API Routes
    # -------------------------------------------------------------
    @app.route("/api/correlation/<dataset_id>", methods=["GET"])
    def get_correlation_analysis_route(dataset_id: str):
        """
        Retrieve comprehensive Phase 6 Correlation Analysis:
        - Numerical features identification
        - Pearson correlation matrix
        - Ranked correlation pairs sorted by strength with direction & classification
        - Key correlations (strongest positive & negative associations)
        - Interactive Plotly Heatmap JSON specification
        - Non-causation guidance disclaimer
        """
        threshold = request.args.get("threshold", 0.0, type=float)
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        engine = CorrelationAnalysisEngine(df, dataset_id=dataset_id)
        corr_data = engine.analyze(threshold=threshold)

        # Generate Plotly heatmap spec
        heatmap_spec = build_correlation_heatmap_spec(
            corr_matrix=corr_data.get("correlation_matrix", {}),
            columns=corr_data.get("numerical_columns", []),
        )

        return jsonify({
            "success": True,
            "correlation": corr_data,
            "heatmap_spec": heatmap_spec,
        })

    @app.route("/api/correlation/<dataset_id>/pairs", methods=["GET"])
    def get_correlation_pairs_route(dataset_id: str):
        """Retrieve ranked correlation pairs filtered by threshold."""
        threshold = request.args.get("threshold", 0.0, type=float)
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        engine = CorrelationAnalysisEngine(df, dataset_id=dataset_id)
        corr_data = engine.analyze(threshold=threshold)

        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "threshold": threshold,
            "count": len(corr_data.get("ranked_pairs", [])),
            "pairs": corr_data.get("ranked_pairs", []),
        })

    # -------------------------------------------------------------
    # Phase 7: Outlier & Anomaly Detection API Routes
    # -------------------------------------------------------------
    @app.route("/api/outliers/<dataset_id>", methods=["GET"])
    def get_dataset_outliers_route(dataset_id: str):
        """
        Retrieve comprehensive Phase 7 Outlier Detection analysis:
        - Method selection (IQR standard 1.5x / extreme 3.0x, Z-Score, Modified Z-Score)
        - Column metrics (Total N, Valid N, Outlier count, Outlier percentage, Lower/Upper bounds)
        - Edge-case statistical warnings (Zero std, Zero IQR, Small sample size)
        - Confirmed data errors vs potential statistical outliers breakdown
        - Multi-column Box Plot Plotly specification
        """
        method = request.args.get("method", "iqr").strip().lower()
        param_val = request.args.get("param", None, type=float)
        if param_val is None:
            param_val = request.args.get("multiplier", None, type=float)
            if param_val is None:
                param_val = request.args.get("threshold", None, type=float)

        col_filter = request.args.get("column", "").strip() or None

        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        engine = OutlierDetectionEngine(df, dataset_id=dataset_id)
        outlier_data = engine.analyze(method=method, param=param_val, column=col_filter)

        # Generate multi-feature Box Plot specification
        num_cols = engine.get_numerical_columns()
        boxplot_spec = build_outlier_boxplot_spec(df, columns=num_cols[:10])

        return jsonify({
            "success": True,
            "outliers": outlier_data,
            "boxplot_spec": boxplot_spec,
        })

    @app.route("/api/outliers/<dataset_id>/column/<column_name>", methods=["GET"])
    def get_column_outliers_route(dataset_id: str, column_name: str):
        """
        Retrieve detailed outlier analysis and interactive charts for an individual column:
        - Specific threshold bounds & moments
        - Row-level outlier instances with Confirmed Data Error vs Potential Outlier classification
        - Plotly Distribution Histogram & Boundary chart specification
        - Plotly Single-Column Box Plot specification
        """
        method = request.args.get("method", "iqr").strip().lower()
        param_val = request.args.get("param", None, type=float)
        if param_val is None:
            param_val = request.args.get("multiplier", None, type=float)
            if param_val is None:
                param_val = request.args.get("threshold", None, type=float)

        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        if column_name not in df.columns:
            return jsonify({
                "success": False,
                "error": f"Column '{column_name}' not found in dataset '{dataset_id}'. Available columns: {list(df.columns)}",
            }), 404

        series = df[column_name]
        if method == "zscore":
            col_res = detect_column_outliers_zscore(series, col_name=column_name, threshold=param_val or 3.0)
        elif method == "modified_zscore":
            col_res = detect_column_outliers_modified_zscore(series, col_name=column_name, threshold=param_val or 3.5)
        else:
            col_res = detect_column_outliers_iqr(series, col_name=column_name, multiplier=param_val or 1.5)

        # Generate distribution and boxplot specs
        dist_spec = build_outlier_distribution_spec(
            series=series,
            col_name=column_name,
            lower_thresh=col_res.get("lower_threshold"),
            upper_thresh=col_res.get("upper_threshold"),
            method_name=col_res.get("method", "IQR"),
        )
        box_spec = build_outlier_boxplot_spec(df, columns=[column_name])

        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "column_name": column_name,
            "result": col_res,
            "distribution_spec": dist_spec,
            "boxplot_spec": box_spec,
        })

    @app.route("/api/outliers/<dataset_id>/chart/boxplot", methods=["GET"])
    def get_outlier_boxplot_chart_route(dataset_id: str):
        """Generate custom multi-feature Box Plot specification."""
        cols_param = request.args.get("columns", "").strip()
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        columns = [c.strip() for c in cols_param.split(",") if c.strip()] if cols_param else None
        boxplot_spec = build_outlier_boxplot_spec(df, columns=columns)

        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "boxplot_spec": boxplot_spec,
        })

    @app.route("/api/outliers/remediate/<dataset_id>", methods=["POST"])
    def remediate_dataset_outliers_route(dataset_id: str):
        """
        Execute non-destructive outlier remediation on a raw dataset:
        - Actions: 'remove' (drop rows with outliers), 'cap' (winsorize bounds), 'remove_errors_only'
        - Creates a new distinct processed dataset in data/processed/ with lineage tracking
        - Original raw upload in data/uploads/ is NEVER modified
        """
        df, error = load_dataset(dataset_id, is_processed=False)
        if error:
            return jsonify({"success": False, "error": error}), 404

        req_json = request.get_json() or {}
        action = req_json.get("action", "remove")
        columns = req_json.get("columns", None)
        method = req_json.get("method", "iqr")
        param = req_json.get("param", None)

        engine = OutlierDetectionEngine(df, dataset_id=dataset_id)
        clean_df, summary = engine.remediate_outliers(
            action=action,
            columns=columns,
            method=method,
            param=param,
        )

        # Save to data/processed/ non-destructively
        proc_id, saved_path, save_err = save_processed_dataset(
            clean_df,
            original_dataset_id=dataset_id,
            cleaning_summary={"outlier_remediation": summary},
        )
        if save_err:
            return jsonify({"success": False, "error": save_err}), 500

        head_records = clean_df.head(10).replace({np.nan: None}).to_dict(orient="records")
        clean_head = [{k: _safe_json_value(v) for k, v in row.items()} for row in head_records]

        return jsonify({
            "success": True,
            "message": f"Outlier remediation '{action}' applied successfully. Saved as new processed dataset.",
            "dataset_id": dataset_id,
            "processed_id": proc_id,
            "summary": summary,
            "preview_head": clean_head,
            "download_url": f"/api/cleaning/download/{dataset_id}",
        }), 200

    @app.route("/api/sample/<sample_type>", methods=["POST"])
    def load_sample_dataset(sample_type: str):
        """
        Provide a built-in realistic sample dataset for immediate testing without local CSV.
        """
        np.random.seed(42)
        n = 250

        if sample_type == "ecommerce":
            categories = ["Electronics", "Fashion", "Home & Kitchen", "Books", "Sports", "Health"]
            regions = ["North", "South", "East", "West", "Central"]
            payment_methods = ["Credit Card", "PayPal", "Debit Card", "Bank Transfer"]
            dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
            
            data = {
                "Order_ID": [f"ORD-{10000 + i}" for i in range(n)],
                "Order_Date": np.random.choice(dates, size=n),
                "Customer_Age": np.random.randint(18, 70, size=n),
                "Category": np.random.choice(categories, size=n),
                "Sales_Amount": np.round(np.random.exponential(scale=120, size=n) + 15, 2),
                "Quantity": np.random.randint(1, 8, size=n),
                "Discount_Pct": np.random.choice([0.0, 0.05, 0.10, 0.15, 0.25], size=n, p=[0.4, 0.2, 0.2, 0.15, 0.05]),
                "Region": np.random.choice(regions, size=n),
                "Payment_Method": np.random.choice(payment_methods, size=n),
                "Is_Returned": np.random.choice([True, False], size=n, p=[0.08, 0.92]),
            }
            df = pd.DataFrame(data)
            df.loc[np.random.choice(df.index, 8, replace=False), "Customer_Age"] = np.nan
            df.loc[np.random.choice(df.index, 5, replace=False), "Discount_Pct"] = np.nan
            filename = "sample_ecommerce_orders.csv"

        elif sample_type == "employee":
            departments = ["Engineering", "Product", "Sales", "Marketing", "HR", "Finance"]
            education = ["Bachelor's", "Master's", "PhD", "High School"]
            ratings = [1, 2, 3, 4, 5]
            
            data = {
                "Employee_ID": [f"EMP-{5000 + i}" for i in range(n)],
                "Department": np.random.choice(departments, size=n),
                "Experience_Years": np.random.randint(0, 25, size=n),
                "Salary": np.random.randint(45000, 185000, size=n),
                "Performance_Score": np.random.choice(ratings, size=n, p=[0.05, 0.15, 0.50, 0.20, 0.10]),
                "Education_Level": np.random.choice(education, size=n),
                "Remote_Work": np.random.choice([True, False], size=n, p=[0.65, 0.35]),
                "Projects_Completed": np.random.randint(1, 20, size=n),
            }
            df = pd.DataFrame(data)
            df.loc[np.random.choice(df.index, 6, replace=False), "Salary"] = np.nan
            filename = "sample_employee_attrition.csv"

        else:
            return jsonify({"success": False, "error": f"Unknown sample dataset type: '{sample_type}'."}), 400

        # Save sample DataFrame to disk via standard upload pipeline
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        file_storage = type("MockFileStorage", (), {
            "filename": filename,
            "seek": csv_buffer.seek,
            "tell": csv_buffer.tell,
            "save": lambda self, path: csv_buffer.seek(0) or open(path, "wb").write(csv_buffer.read()),
        })()

        dataset_id, saved_path, error, meta = save_uploaded_file(file_storage)
        if error:
            return jsonify({"success": False, "error": error}), 500

        summary, sum_err = get_dataset_summary(dataset_id)
        profile = profile_dataset(df, dataset_id=dataset_id)

        return jsonify({
            "success": True,
            "message": f"Sample dataset '{filename}' ingested and profiled successfully.",
            "dataset_id": dataset_id,
            "original_filename": filename,
            "summary": summary,
            "profile": profile,
        }), 201

    # -------------------------------------------------------------
    # Phase 4: Automated Data Quality & Cleaning API Routes
    # -------------------------------------------------------------
    @app.route("/api/cleaning/audit/<dataset_id>", methods=["GET"])
    def audit_dataset_quality(dataset_id: str):
        """
        Run all 12 deterministic data quality issue detection algorithms on the dataset.
        Returns categorized issues, severity breakdown, and transformation previews.
        """
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        issues = detect_data_quality_issues(df)
        preview_data = generate_cleaning_preview(df, issues)

        severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for issue in issues:
            sev = issue.get("severity", "Low")
            if sev in severity_counts:
                severity_counts[sev] += 1

        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "total_issues": len(issues),
            "severity_counts": severity_counts,
            "issues": issues,
            "preview": preview_data,
        })

    @app.route("/api/cleaning/preview/<dataset_id>", methods=["GET"])
    def get_cleaning_transformation_preview(dataset_id: str):
        """
        Generate illustrative before-and-after transformation preview pairs (Original -> Cleaned).
        """
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        preview_data = generate_cleaning_preview(df)
        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            **preview_data,
        })

    @app.route("/api/cleaning/apply/<dataset_id>", methods=["POST"])
    def apply_cleaning_pipeline(dataset_id: str):
        """
        Execute the configurable cleaning pipeline on a raw dataset.
        Never modifies raw files; saves output to data/processed/ and returns comprehensive summary metrics.
        """
        df, error = load_dataset(dataset_id, is_processed=False)
        if error:
            return jsonify({"success": False, "error": error}), 404

        # Parse requested cleaning configuration
        req_json = request.get_json() or {}
        operations = req_json.get("operations") if "operations" in req_json else req_json

        # Execute transformation pipeline non-destructively
        clean_df, summary = execute_cleaning_pipeline(df, operations)

        # Save to data/processed/
        proc_id, saved_path, save_err = save_processed_dataset(
            clean_df,
            original_dataset_id=dataset_id,
            cleaning_summary=summary,
        )
        if save_err:
            return jsonify({"success": False, "error": save_err}), 500

        # Extract first 10 preview rows of cleaned data
        head_records = clean_df.head(10).replace({np.nan: None}).to_dict(orient="records")
        clean_head = [{k: _safe_json_value(v) for k, v in row.items()} for row in head_records]

        return jsonify({
            "success": True,
            "message": "Dataset cleaned and saved to data/processed/ successfully.",
            "dataset_id": dataset_id,
            "processed_id": proc_id,
            "summary": summary,
            "preview_head": clean_head,
            "download_url": f"/api/cleaning/download/{dataset_id}",
        }), 200

    @app.route("/api/cleaning/download/<dataset_id>", methods=["GET"])
    def download_cleaned_dataset(dataset_id: str):
        """
        Download the cleaned CSV dataset from data/processed/.
        If no processed version exists yet, executes default cleaning and serves the file.
        """
        proc_path = get_latest_processed_file(dataset_id)
        
        if not proc_path or not proc_path.exists():
            # Automatically apply standard cleaning pipeline and save
            df, error = load_dataset(dataset_id, is_processed=False)
            if error:
                return jsonify({"success": False, "error": error}), 404

            clean_df, summary = execute_cleaning_pipeline(df)
            proc_id, proc_path, save_err = save_processed_dataset(
                clean_df,
                original_dataset_id=dataset_id,
                cleaning_summary=summary,
            )
            if save_err or not proc_path or not proc_path.exists():
                return jsonify({"success": False, "error": "Could not generate cleaned dataset for download."}), 500

        download_filename = proc_path.name
        return send_file(
            str(proc_path),
            as_attachment=True,
            download_name=download_filename,
            mimetype="text/csv",
        )

    # -------------------------------------------------------------
    # Phase 8: Automatic Visualization Engine API Routes
    # -------------------------------------------------------------
    @app.route("/api/visualization/recommendations/<dataset_id>", methods=["GET"])
    def get_visualization_recommendations_route(dataset_id: str):
        """
        Retrieve automated, ranked, non-redundant chart recommendations:
        - Evaluates single numerical (histograms, box plots), single categorical (bars, donuts),
          bivariate numerical (scatter with trend & r), datetime + numerical (time series),
          categorical + numerical (aggregated bars), and datetime + categorical (temporal distributions).
        - Prunes useless combinations (IDs, 0-variance).
        - Returns Plotly dark-theme JSON specs and analytical rationales.
        """
        limit = request.args.get("limit", 12, type=int)
        category_filter = request.args.get("category", "").strip() or None

        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        engine = ChartRecommendationEngine(df, dataset_id=dataset_id)
        recommendations = engine.recommend(limit=limit, category_filter=category_filter)

        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "count": len(recommendations),
            "schema": engine.schema,
            "recommendations": recommendations,
        })

    @app.route("/api/visualization/custom/<dataset_id>", methods=["POST"])
    def generate_custom_chart_route(dataset_id: str):
        """
        Generate an interactive Plotly chart specification from user-selected parameters:
        - Supports Chart Types: bar, line, scatter, histogram, box, pie/donut, area
        - Supports Aggregations: none, mean, sum, median, count, min, max, std
        - Supports X, Y, Color grouping, and custom titles.
        """
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        req_json = request.get_json() or {}
        chart_type = req_json.get("chart_type", "bar")
        x_col = req_json.get("x_col", "")
        y_col = req_json.get("y_col", None)
        color_col = req_json.get("color_col", None)
        aggregation = req_json.get("aggregation", "none")
        title = req_json.get("title", None)

        if not x_col:
            return jsonify({"success": False, "error": "Parameter 'x_col' is required."}), 400

        spec = build_custom_chart_spec(
            df=df,
            chart_type=chart_type,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            aggregation=aggregation,
            title=title,
        )

        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "chart_type": chart_type,
            "spec": spec,
        })

    @app.route("/api/visualization/schema/<dataset_id>", methods=["GET"])
    def get_visualization_schema_route(dataset_id: str):
        """
        Retrieve column role classification and available features for chart building.
        """
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        schema = inspect_column_types(df)
        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "columns": list(df.columns),
            "schema": schema,
        })

    # -------------------------------------------------------------
    # Phase 9: AI Insight Engine API Routes
    # -------------------------------------------------------------
    # In-memory insights cache: { dataset_id: insight_payload }
    _insights_cache = {}

    @app.route("/api/insights/providers", methods=["GET"])
    def get_ai_providers_route():
        """
        Retrieve available AI providers, configuration status, and model options.
        Never leaks sensitive secret API keys.
        """
        providers = Config.get_available_providers()
        active_provider = Config.AI_PROVIDER
        return jsonify({
            "success": True,
            "active_provider": active_provider,
            "providers": providers,
        })

    @app.route("/api/insights/<dataset_id>", methods=["GET"])
    def get_dataset_insights_route(dataset_id: str):
        """
        Retrieve or compute AI Insights for a dataset.
        If previously generated in cache, returns cached insights immediately.
        """
        force_refresh = request.args.get("refresh", "false").lower() in ("true", "1", "yes")
        provider_override = request.args.get("provider", "").strip() or None
        model_override = request.args.get("model", "").strip() or None

        cache_key = f"{dataset_id}_{provider_override or 'default'}"
        if not force_refresh and cache_key in _insights_cache:
            return jsonify({
                "success": True,
                "cached": True,
                "insights": _insights_cache[cache_key],
            })

        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        agent = AnalystAgent(provider=provider_override, model=model_override)
        context = agent.build_analysis_context(df, dataset_id=dataset_id)
        result = agent.generate_insights(context, provider=provider_override, model=model_override)

        _insights_cache[cache_key] = result

        return jsonify({
            "success": True,
            "cached": False,
            "insights": result,
        })

    @app.route("/api/insights/generate/<dataset_id>", methods=["POST"])
    def generate_dataset_insights_route(dataset_id: str):
        """
        Explicitly trigger AI Insights generation with optional provider or model parameters.
        Assembles structured analysis from Python engines and invokes LLM / deterministic synthesizer.
        """
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        req_json = request.get_json() or {}
        provider = req_json.get("provider", None)
        model = req_json.get("model", None)

        agent = AnalystAgent(provider=provider, model=model)
        context = agent.build_analysis_context(df, dataset_id=dataset_id)
        result = agent.generate_insights(context, provider=provider, model=model)

        cache_key = f"{dataset_id}_{provider or 'default'}"
        _insights_cache[cache_key] = result

        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "insights": result,
        })

    @app.route("/api/insights/context/<dataset_id>", methods=["GET"])
    def get_insights_context_route(dataset_id: str):
        """
        Retrieve the exact structured JSON context generated by the Python analysis engines
        and supplied to the LLM. Provides 100% transparency into fact grounding.
        """
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        context = build_analysis_context(df, dataset_id=dataset_id)
        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "analysis_context": context,
        })

    # -------------------------------------------------------------
    # Phase 10: Natural-Language Dataset Q&A ("Ask Your Dataset") Routes
    # -------------------------------------------------------------
    @app.route("/api/chat/ask", methods=["POST"])
    def chat_ask_route():
        """
        Process user natural language query for a dataset.
        Routes to deterministic Python analysis tool, executes calculation,
        synthesizes grounded explanation, generates Plotly visualization spec,
        and records to session history.
        """
        req_json = request.get_json() or {}
        dataset_id = req_json.get("dataset_id", "").strip()
        query = req_json.get("query", "").strip()
        provider = req_json.get("provider", None)
        model = req_json.get("model", None)

        if not dataset_id:
            return jsonify({"success": False, "error": "Parameter 'dataset_id' is required."}), 400

        if not query:
            return jsonify({"success": False, "error": "Parameter 'query' cannot be empty."}), 400

        df, error = load_dataset(dataset_id)
        if error or df is None:
            return jsonify({"success": False, "error": f"Failed to load dataset '{dataset_id}': {error}"}), 404

        router = QuestionRouter(provider=provider, model=model)
        result = router.route_and_execute(
            df=df,
            query=query,
            dataset_id=dataset_id,
            provider=provider,
            model=model,
        )

        return jsonify(result)

    @app.route("/api/chat/history/<dataset_id>", methods=["GET"])
    def get_chat_history_route(dataset_id: str):
        """
        Retrieve conversation history for the current dataset session.
        """
        history = session_manager.get_history(dataset_id)
        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "turns_count": len(history),
            "history": history,
        })

    @app.route("/api/chat/clear/<dataset_id>", methods=["POST"])
    def clear_chat_history_route(dataset_id: str):
        """
        Clear conversation history for the current dataset session.
        """
        cleared = session_manager.clear_history(dataset_id)
        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "message": "Conversation history successfully cleared.",
            "cleared": cleared,
        })

    @app.route("/api/chat/suggested-questions/<dataset_id>", methods=["GET"])
    def get_chat_suggestions_route(dataset_id: str):
        """
        Retrieve dynamically generated question suggestions tailored to the dataset's columns.
        """
        df, error = load_dataset(dataset_id)
        if error or df is None:
            return jsonify({"success": False, "error": error}), 404

        suggestions = session_manager.get_suggested_questions(df, dataset_id=dataset_id)
        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "suggestions": suggestions,
        })

    @app.route("/api/chat/tools", methods=["GET"])
    def get_chat_tools_route():
        """
        List all available deterministic analysis tools and their descriptions.
        """
        tools_info = [
            {
                "name": "dataset_summary",
                "description": "Comprehensive dataset dimensions, memory, missingness, duplicate rates, and column types.",
                "parameters": {},
            },
            {
                "name": "column_summary",
                "description": "Deep statistical, frequency, or temporal summary of a single column.",
                "parameters": {"column_name": "string (required)"},
            },
            {
                "name": "groupby_analysis",
                "description": "Group a continuous target feature by a categorical dimension with ranking, sorting, and percentage of total.",
                "parameters": {
                    "group_by_col": "string (required)",
                    "target_col": "string (optional)",
                    "agg_func": "string (sum, mean, median, count, min, max, std)",
                    "sort_desc": "boolean (default: true)",
                    "limit": "integer (default: 10)",
                },
            },
            {
                "name": "aggregation_analysis",
                "description": "Compute single or multi-moment scalar statistics across a continuous numerical column.",
                "parameters": {
                    "target_col": "string (required)",
                    "agg_func": "string (mean, sum, median, min, max, std, count)",
                },
            },
            {
                "name": "correlation_analysis",
                "description": "Pairwise Pearson/Spearman linear correlation or global strongest positive/negative association ranking.",
                "parameters": {
                    "col1": "string (optional)",
                    "col2": "string (optional)",
                    "threshold": "float (default: 0.0)",
                    "method": "string (pearson, spearman)",
                },
            },
            {
                "name": "outlier_analysis",
                "description": "Detect statistical anomalies and extreme tail values using IQR (1.5x) or Z-score methods.",
                "parameters": {
                    "column_name": "string (optional)",
                    "method": "string (iqr, zscore)",
                    "threshold": "float (default: 1.5)",
                },
            },
            {
                "name": "time_series_analysis",
                "description": "Chronological progression, growth percentage, peak and trough periods along datetime features.",
                "parameters": {
                    "date_col": "string (optional)",
                    "value_col": "string (optional)",
                    "freq": "string (auto, D, M, Y)",
                    "agg_func": "string (sum, mean)",
                },
            },
            {
                "name": "distribution_analysis",
                "description": "Histogram bins, skewness, kurtosis, spread, and normality evaluation.",
                "parameters": {
                    "column_name": "string (required)",
                    "bins": "integer (default: 10)",
                },
            },
            {
                "name": "investigate_further",
                "description": "Prioritized exploration leads derived from anomalies, strong correlations, and missingness signals.",
                "parameters": {},
            },
        ]
        return jsonify({
            "success": True,
            "total_tools": len(tools_info),
            "tools": tools_info,
        })

    return app



# Create application instance
app = create_app()

if __name__ == "__main__":
    logger.info("Starting AI Data Analyst Agent on http://%s:%s", Config.HOST, Config.PORT)
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
