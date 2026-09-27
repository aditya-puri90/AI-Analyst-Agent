"""
End-to-End Dashboard Flow Integration Test.
Validates the complete lifecycle: Ingestion -> Profiling -> Quality -> Statistics -> Correlation -> Outliers -> Viz -> AI -> Reports -> UI Routes.
Compatible with both pytest (in-memory test client) and standalone script execution.
"""

import json
import sys
from app import create_app


def test_full_pipeline(client=None):
    if client is None:
        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()

    print("Testing Complete Pipeline using Flask test client...")

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.get_data(as_text=True)}"
    print("[PASS] 1. Health check passed")

    # 2. Ingest Sample Dataset
    res = client.post("/api/sample/ecommerce")
    assert res.status_code in (200, 201), f"Sample ingestion failed: {res.get_data(as_text=True)}"
    sample_data = res.get_json()
    assert sample_data.get("success"), f"Sample ingestion reported error: {sample_data}"
    dataset_id = sample_data["dataset_id"]
    print(f"[PASS] 2. Ingested dataset successfully: dataset_id = {dataset_id}")

    # 3. List Datasets
    res = client.get("/api/datasets")
    assert res.status_code == 200
    datasets = res.get_json().get("datasets", [])
    assert len(datasets) > 0, "No datasets returned in /api/datasets"
    found = any((d.get("id") == dataset_id or d.get("dataset_id") == dataset_id) for d in datasets)
    assert found, f"Dataset {dataset_id} not found in /api/datasets"
    print(f"[PASS] 3. Dataset listed in /api/datasets with {len(datasets)} total available datasets")

    # 4. Profile API
    res = client.get(f"/api/profile/{dataset_id}")
    assert res.status_code == 200
    prof = res.get_json().get("profile", {})
    assert prof.get("dataset_id") == dataset_id
    assert len(prof.get("columns", [])) > 0
    print(f"[PASS] 4. Profile computed successfully: {len(prof.get('columns', []))} columns profiled")

    # 5. Quality Audit
    res = client.get(f"/api/cleaning/audit/{dataset_id}")
    assert res.status_code == 200
    print("[PASS] 5. Data quality audit computed successfully")

    # 6. Statistics
    res = client.get(f"/api/statistics/{dataset_id}")
    assert res.status_code == 200
    print("[PASS] 6. Descriptive statistics computed successfully")

    # 7. Correlation
    res = client.get(f"/api/correlation/{dataset_id}")
    assert res.status_code == 200
    print("[PASS] 7. Correlation analysis computed successfully")

    # 8. Outliers
    res = client.get(f"/api/outliers/{dataset_id}?method=iqr")
    assert res.status_code == 200
    print("[PASS] 8. Outlier detection computed successfully")

    # 9. Visualizations
    res = client.get(f"/api/visualization/recommendations/{dataset_id}")
    assert res.status_code == 200
    print("[PASS] 9. Chart recommendations computed successfully")

    # 10. AI Insights
    res = client.get(f"/api/insights/{dataset_id}")
    assert res.status_code == 200
    print("[PASS] 10. AI insights synthesized successfully")

    # 11. Reports Generation & Download
    res = client.post(f"/api/reports/generate/{dataset_id}")
    assert res.status_code == 200
    rep_data = res.get_json()
    assert rep_data.get("success"), f"Report generation failed: {rep_data}"
    print("[PASS] 11. Report generated successfully")

    for fmt in ["markdown", "html", "json"]:
        res = client.get(f"/api/reports/download/{dataset_id}/{fmt}")
        assert res.status_code == 200, f"Report download {fmt} failed: {res.status_code}"
        assert len(res.data) > 100, f"Report content too small for {fmt}"
        print(f"[PASS] 12. Report download ({fmt}) verified ({len(res.data)} bytes)")

    # 13. Dashboard Page Route
    res = client.get(f"/dashboard?dataset_id={dataset_id}")
    assert res.status_code == 200
    text_data = res.get_data(as_text=True)
    assert f'window.INITIAL_DATASET_ID = "{dataset_id}"' in text_data
    print("[PASS] 13. Dashboard page rendered with INITIAL_DATASET_ID set")

    # 14. Chat Page Route
    res = client.get(f"/chat?dataset_id={dataset_id}")
    assert res.status_code == 200
    print("[PASS] 14. Chat page rendered with dataset_id")

    print("\n========================================================")
    print("ALL 14 END-TO-END PIPELINE & COMPUTATION TESTS PASSED!")
    print("========================================================")


if __name__ == "__main__":
    try:
        test_full_pipeline()
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)
