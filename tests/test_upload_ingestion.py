"""
Comprehensive Automated Test Suite for Phase 2: CSV Upload & Dataset Ingestion.
Tests normal CSVs, empty files, header-only files, malformed syntax, missing values,
custom encodings, alternative delimiters, security boundaries, and file immutability.
"""

import io
import pytest
import pandas as pd
import numpy as np

from app import create_app
from config.settings import Config
from utils.file_handler import load_dataset, get_dataset_summary, list_uploaded_datasets


@pytest.fixture
def client():
    """Flask test client fixture."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# -------------------------------------------------------------
# 1. Normal CSV Ingestion
# -------------------------------------------------------------
def test_normal_csv_upload(client):
    csv_text = """Employee_ID,Name,Department,Salary,Join_Date,Is_FullTime
E101,John Doe,Engineering,95000,2022-03-15,True
E102,Jane Smith,Product,105000,2021-06-01,True
E103,Bob Johnson,Sales,72000,2023-01-10,False
E104,Alice Brown,HR,68000,2020-11-20,True
E105,Charlie White,Engineering,98000,2022-08-05,True
"""
    data = {
        "file": (io.BytesIO(csv_text.encode("utf-8")), "employees_q1.csv")
    }
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    json_data = res.get_json()

    assert json_data["success"] is True
    assert "dataset_id" in json_data
    assert json_data["original_filename"] == "employees_q1.csv"

    summary = json_data["summary"]
    assert summary["total_rows"] == 5
    assert summary["total_columns"] == 6
    assert summary["memory_usage_bytes"] > 0
    assert "KB" in summary["memory_usage_formatted"] or "Bytes" in summary["memory_usage_formatted"]
    assert summary["column_names"] == ["Employee_ID", "Name", "Department", "Salary", "Join_Date", "Is_FullTime"]
    
    # Verify column metadata
    col_dict = {col["name"]: col for col in summary["columns"]}
    assert col_dict["Salary"]["pandas_dtype"] in ["int64", "float64", "int32"]
    assert col_dict["Salary"]["inferred_type"] == "numerical"
    assert col_dict["Department"]["inferred_type"] == "categorical"
    assert col_dict["Is_FullTime"]["inferred_type"] == "boolean"

    # Verify first 10 rows preview
    preview = summary["preview_first_10_rows"]
    assert len(preview) == 5
    assert preview[0]["Name"] == "John Doe"
    assert preview[0]["#"] == 1


# -------------------------------------------------------------
# 2. Empty CSV (0 Bytes)
# -------------------------------------------------------------
def test_empty_csv_upload(client):
    data = {
        "file": (io.BytesIO(b""), "empty_dataset.csv")
    }
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    json_data = res.get_json()
    assert json_data["success"] is False
    assert "empty" in json_data["error"].lower()


# -------------------------------------------------------------
# 3. Header-Only CSV (0 Data Rows)
# -------------------------------------------------------------
def test_header_only_csv_upload(client):
    csv_text = "Name,Age,Department,Salary\n"
    data = {
        "file": (io.BytesIO(csv_text.encode("utf-8")), "header_only.csv")
    }
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    json_data = res.get_json()
    assert json_data["success"] is False
    assert "no data rows" in json_data["error"].lower() or "at least 1 row" in json_data["error"].lower()


# -------------------------------------------------------------
# 4. Blank Header CSV
# -------------------------------------------------------------
def test_blank_headers_csv_upload(client):
    csv_text = ",,\n1,2,3\n4,5,6\n"
    data = {
        "file": (io.BytesIO(csv_text.encode("utf-8")), "blank_headers.csv")
    }
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    json_data = res.get_json()
    assert json_data["success"] is False
    assert "header" in json_data["error"].lower()


# -------------------------------------------------------------
# 5. CSV Containing Missing Values (NaNs and Nulls)
# -------------------------------------------------------------
def test_csv_with_missing_values(client):
    csv_text = """Customer,Age,City,Annual_Spend
Alice,29,New York,1200.50
Bob,,London,
Charlie,42,,3400.00
David,35,Tokyo,
,50,Paris,800.00
"""
    data = {
        "file": (io.BytesIO(csv_text.encode("utf-8")), "missing_data.csv")
    }
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    json_data = res.get_json()
    summary = json_data["summary"]

    assert summary["total_rows"] == 5
    assert summary["total_columns"] == 4

    col_dict = {col["name"]: col for col in summary["columns"]}
    assert col_dict["Customer"]["null_count"] == 1
    assert col_dict["Age"]["null_count"] == 1
    assert col_dict["City"]["null_count"] == 1
    assert col_dict["Annual_Spend"]["null_count"] == 2

    # Verify preview contains null values cleanly
    preview = summary["preview_first_10_rows"]
    assert preview[1]["Age"] is None
    assert preview[1]["Annual_Spend"] is None
    assert preview[4]["Customer"] is None


# -------------------------------------------------------------
# 6. Non-UTF-8 Encoding (Latin-1 / CP1252 with Diacritics)
# -------------------------------------------------------------
def test_latin1_encoding_upload(client):
    csv_text = "City,Country,Café_Rating,Specialty\nMünchen,Germany,4.8,Pretzel\nSão Paulo,Brazil,4.9,Coffee\nZürich,Switzerland,4.7,Chocolate\n"
    data = {
        "file": (io.BytesIO(csv_text.encode("latin-1")), "latin1_cafes.csv")
    }
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    json_data = res.get_json()
    summary = json_data["summary"]

    assert summary["total_rows"] == 3
    assert summary["total_columns"] == 4
    assert summary["encoding"] in ["latin-1", "cp1252", "iso-8859-1"]
    assert summary["preview_first_10_rows"][0]["City"] == "München"
    assert summary["preview_first_10_rows"][1]["City"] == "São Paulo"


# -------------------------------------------------------------
# 7. Semicolon-Delimited CSV Auto-Sniffing
# -------------------------------------------------------------
def test_semicolon_delimited_csv_upload(client):
    csv_text = "ProductID;Title;Price;Stock\nP100;Keyboard;45.99;120\nP200;Monitor;249.50;35\nP300;Mouse;19.99;200\n"
    data = {
        "file": (io.BytesIO(csv_text.encode("utf-8")), "catalog_semicolon.csv")
    }
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 201
    json_data = res.get_json()
    summary = json_data["summary"]

    assert summary["delimiter"] == ";"
    assert summary["total_rows"] == 3
    assert summary["total_columns"] == 4
    assert summary["column_names"] == ["ProductID", "Title", "Price", "Stock"]


# -------------------------------------------------------------
# 8. Disallowed Extensions
# -------------------------------------------------------------
def test_disallowed_file_extensions(client):
    for bad_name in ["report.txt", "data.json", "script.py", "malicious.exe", "document.pdf"]:
        data = {
            "file": (io.BytesIO(b"Some text data"), bad_name)
        }
        res = client.post("/api/upload", data=data, content_type="multipart/form-data")
        assert res.status_code == 400
        json_data = res.get_json()
        assert json_data["success"] is False
        assert "Invalid file format" in json_data["error"]


# -------------------------------------------------------------
# 9. File Immutability & Duplicate Upload Collision Avoidance
# -------------------------------------------------------------
def test_file_immutability_and_unique_ids(client):
    csv_content = b"ID,Name,Score\n1,Alpha,90\n2,Beta,85\n"
    
    # Upload 1
    res1 = client.post("/api/upload", data={"file": (io.BytesIO(csv_content), "scorecard.csv")}, content_type="multipart/form-data")
    assert res1.status_code == 201
    id1 = res1.get_json()["dataset_id"]

    # Upload 2 with identical filename
    res2 = client.post("/api/upload", data={"file": (io.BytesIO(csv_content), "scorecard.csv")}, content_type="multipart/form-data")
    assert res2.status_code == 201
    id2 = res2.get_json()["dataset_id"]

    # Ensure distinct dataset IDs were generated
    assert id1 != id2

    # Verify both exist and can be loaded independently
    sum1, err1 = get_dataset_summary(id1)
    sum2, err2 = get_dataset_summary(id2)
    assert err1 is None and err2 is None
    assert sum1["total_rows"] == 2
    assert sum2["total_rows"] == 2
