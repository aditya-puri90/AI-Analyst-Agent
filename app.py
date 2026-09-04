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
from visualization.charts import build_correlation_heatmap_spec

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
            "phase": "Phase 5 - Statistical Analysis & Phase 6 - Correlation Analysis Engine",
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

    return app


# Create application instance
app = create_app()

if __name__ == "__main__":
    logger.info("Starting AI Data Analyst Agent on http://%s:%s", Config.HOST, Config.PORT)
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
