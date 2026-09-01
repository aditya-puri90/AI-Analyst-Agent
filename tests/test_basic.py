"""
Automated unit and integration tests for Phase 1: Foundation, Profiler, Validators, and REST API.
"""

import io
import pytest
import pandas as pd
import numpy as np

from app import create_app
from config.settings import Config
from utils.validators import (
    allowed_file,
    sanitize_filename,
    detect_encoding_and_delimiter,
    validate_csv_structure,
)
from utils.file_handler import save_uploaded_file, load_dataset, list_uploaded_datasets
from analysis.profiler import profile_dataset, infer_column_types, get_dataset_preview


@pytest.fixture
def client():
    """Flask test client fixture."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_dataframe():
    """Generates a representative mixed-type DataFrame."""
    return pd.DataFrame({
        "customer_id": [f"CUST_{i}" for i in range(1, 21)],
        "age": [25, 30, 35, 40, np.nan, 28, 52, 45, 60, 31, 29, 42, 38, 22, 50, 48, 33, 27, 39, 36],
        "signup_date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "tier": ["Gold", "Silver", "Bronze", "Gold", "Silver"] * 4,
        "is_active": [True, False, True, True, False] * 4,
        "spend": [120.50, 89.0, 45.20, 310.0, 99.99, 15.0, 500.0, 240.0, 110.0, 75.50,
                  180.0, 95.0, 40.0, 300.0, 85.0, 20.0, 450.0, 220.0, 105.0, 70.0],
    })


# -------------------------------------------------------------
# 1. Validation & File Security Tests
# -------------------------------------------------------------
def test_allowed_file():
    assert allowed_file("data.csv") is True
    assert allowed_file("my_dataset.CSV") is True
    assert allowed_file("report.pdf") is False
    assert allowed_file("malicious.exe") is False
    assert allowed_file("no_extension") is False
    assert allowed_file("") is False


def test_sanitize_filename():
    assert sanitize_filename("../../../etc/passwd.csv") == "etc_passwd.csv"
    assert sanitize_filename("my data file (1).csv") == "my_data_file_1.csv"
    assert sanitize_filename("") == "dataset.csv"


def test_delimiter_and_encoding_detection(tmp_path):
    # Test semicolon-delimited file with Latin-1 characters
    file_path = tmp_path / "test_semicolon.csv"
    content = "Name;City;Income;Café\nAlice;München;50000;Yes\nBob;Köln;62000;No\n"
    file_path.write_bytes(content.encode("latin-1"))

    encoding, delimiter = detect_encoding_and_delimiter(file_path)
    assert delimiter == ";"
    assert encoding in ["latin-1", "cp1252", "iso-8859-1"]

    is_valid, error, meta = validate_csv_structure(file_path)
    assert is_valid is True
    assert meta["column_count"] == 4
    assert meta["approx_row_count"] == 2


# -------------------------------------------------------------
# 2. Profiling & Type Inference Tests
# -------------------------------------------------------------
def test_infer_column_types(sample_dataframe):
    types = infer_column_types(sample_dataframe)
    assert types["customer_id"] in ["Other", "Categorical"]
    assert types["age"] == "Numerical"
    assert types["signup_date"] == "Datetime"
    assert types["tier"] == "Categorical"
    assert types["is_active"] == "Boolean"
    assert types["spend"] == "Numerical"


def test_profile_dataset(sample_dataframe):
    profile = profile_dataset(sample_dataframe, dataset_id="test_ds")
    overview = profile["overview"]

    assert overview["total_rows"] == 20
    assert overview["total_columns"] == 6
    assert overview["total_missing_cells"] == 1  # age has 1 NaN
    assert overview["duplicate_rows"] == 0
    assert overview["memory_usage_bytes"] > 0

    assert len(profile["columns"]) == 6
    assert len(profile["preview_head"]) == 10


def test_get_dataset_preview(sample_dataframe):
    preview = get_dataset_preview(sample_dataframe, page=1, page_size=5)
    assert len(preview["rows"]) == 5
    assert preview["pagination"]["total_rows"] == 20
    assert preview["pagination"]["total_pages"] == 4
    assert preview["pagination"]["has_next"] is True
    assert preview["pagination"]["has_prev"] is False


# -------------------------------------------------------------
# 3. REST API Endpoint Integration Tests
# -------------------------------------------------------------
def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "AI Data Analyst Agent" in data["service"]


def test_page_views(client):
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert b"AI Data Analyst" in res_index.data

    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200

    res_chat = client.get("/chat")
    assert res_chat.status_code == 200


def test_api_sample_ecommerce(client):
    res = client.post("/api/sample/ecommerce")
    assert res.status_code == 201
    data = res.get_json()
    assert data["success"] is True
    assert "dataset_id" in data
    assert data["profile"]["overview"]["total_rows"] == 250

    # Verify profile endpoint
    dataset_id = data["dataset_id"]
    prof_res = client.get(f"/api/profile/{dataset_id}")
    assert prof_res.status_code == 200
    prof_data = prof_res.get_json()
    assert prof_data["profile"]["overview"]["total_columns"] == 10

    # Verify preview endpoint
    prev_res = client.get(f"/api/preview/{dataset_id}?page=1&page_size=10")
    assert prev_res.status_code == 200
    prev_data = prev_res.get_json()
    assert len(prev_data["rows"]) == 10


def test_api_upload_valid_csv(client):
    csv_content = b"Product,Category,Price,InStock\nLaptop,Tech,1200,True\nChair,Home,150,False\n"
    data = {
        "file": (io.BytesIO(csv_content), "inventory.csv")
    }
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    res_data = res.get_json()
    assert res_data["success"] is True
    assert res_data["profile"]["overview"]["total_rows"] == 2
    assert res_data["profile"]["overview"]["total_columns"] == 4


def test_api_upload_invalid_extension(client):
    bad_content = b"Some random text file"
    data = {
        "file": (io.BytesIO(bad_content), "document.txt")
    }
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    res_data = res.get_json()
    assert res_data["success"] is False
    assert "Invalid file format" in res_data["error"]
