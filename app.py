"""
Main Flask Application for AI Data Analyst Agent.
Provides REST API endpoints and web views for dataset uploading, ingestion, profiling, and exploration.
"""

import io
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template
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
)
from analysis.profiler import profile_dataset, get_dataset_preview

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
            "phase": "Phase 2 - CSV Upload & Ingestion",
            "max_upload_mb": Config.MAX_CONTENT_LENGTH / (1024 * 1024),
            "allowed_extensions": list(Config.ALLOWED_EXTENSIONS),
        })

    @app.route("/api/upload", methods=["POST"])
    def upload_file():
        """
        Handle secure CSV file upload, validation, ingestion, and extraction of summary metadata.
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

        # Full profile
        df, load_err = load_dataset(dataset_id)
        profile = profile_dataset(df, dataset_id=dataset_id) if not load_err else {}

        return jsonify({
            "success": True,
            "message": f"Dataset '{file_storage.filename}' uploaded and ingested successfully.",
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
        """Retrieve the automated data profile for a dataset."""
        df, error = load_dataset(dataset_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        profile = profile_dataset(df, dataset_id=dataset_id)
        return jsonify({"success": True, "profile": profile})

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
            "message": f"Sample dataset '{filename}' ingested successfully.",
            "dataset_id": dataset_id,
            "original_filename": filename,
            "summary": summary,
            "profile": profile,
        }), 201

    return app


# Create application instance
app = create_app()

if __name__ == "__main__":
    logger.info("Starting AI Data Analyst Agent on http://%s:%s", Config.HOST, Config.PORT)
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
