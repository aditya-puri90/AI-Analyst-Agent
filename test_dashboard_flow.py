import requests
import json
import sys

BASE_URL = "http://127.0.0.1:5000"

def test_full_pipeline():
    print("Testing Complete Pipeline against running server at:", BASE_URL)
    
    # 1. Health check
    res = requests.get(f"{BASE_URL}/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("[PASS] 1. Health check passed")

    # 2. Ingest Sample Dataset
    res = requests.post(f"{BASE_URL}/api/sample/ecommerce")
    assert res.status_code in (200, 201), f"Sample ingestion failed: {res.text}"
    sample_data = res.json()
    assert sample_data.get("success"), f"Sample ingestion reported error: {sample_data}"
    dataset_id = sample_data["dataset_id"]
    print(f"[PASS] 2. Ingested dataset successfully: dataset_id = {dataset_id}")

    # 3. List Datasets
    res = requests.get(f"{BASE_URL}/api/datasets")
    assert res.status_code == 200
    datasets = res.json().get("datasets", [])
    assert len(datasets) > 0, "No datasets returned in /api/datasets"
    found = any((d.get("id") == dataset_id or d.get("dataset_id") == dataset_id) for d in datasets)
    assert found, f"Dataset {dataset_id} not found in /api/datasets"
    print(f"[PASS] 3. Dataset listed in /api/datasets with {len(datasets)} total available datasets")

    # 4. Profile API
    res = requests.get(f"{BASE_URL}/api/profile/{dataset_id}")
    assert res.status_code == 200
    prof = res.json().get("profile", {})
    assert prof.get("dataset_id") == dataset_id
    assert len(prof.get("columns", [])) > 0
    print(f"[PASS] 4. Profile computed successfully: {len(prof.get('columns', []))} columns profiled")

    # 5. Quality Audit
    res = requests.get(f"{BASE_URL}/api/cleaning/audit/{dataset_id}")
    assert res.status_code == 200
    print(f"[PASS] 5. Data quality audit computed successfully")

    # 6. Statistics
    res = requests.get(f"{BASE_URL}/api/statistics/{dataset_id}")
    assert res.status_code == 200
    print(f"[PASS] 6. Descriptive statistics computed successfully")

    # 7. Correlation
    res = requests.get(f"{BASE_URL}/api/correlation/{dataset_id}")
    assert res.status_code == 200
    print(f"[PASS] 7. Correlation analysis computed successfully")

    # 8. Outliers
    res = requests.get(f"{BASE_URL}/api/outliers/{dataset_id}?method=iqr")
    assert res.status_code == 200
    print(f"[PASS] 8. Outlier detection computed successfully")

    # 9. Visualizations
    res = requests.get(f"{BASE_URL}/api/visualization/recommendations/{dataset_id}")
    assert res.status_code == 200
    print(f"[PASS] 9. Chart recommendations computed successfully")

    # 10. AI Insights
    res = requests.get(f"{BASE_URL}/api/insights/{dataset_id}")
    assert res.status_code == 200
    print(f"[PASS] 10. AI insights synthesized successfully")

    # 11. Reports Generation & Download
    res = requests.post(f"{BASE_URL}/api/reports/generate/{dataset_id}")
    assert res.status_code == 200
    rep_data = res.json()
    assert rep_data.get("success"), f"Report generation failed: {rep_data}"
    print(f"[PASS] 11. Report generated successfully")

    for fmt in ["markdown", "html", "json"]:
        res = requests.get(f"{BASE_URL}/api/reports/download/{dataset_id}/{fmt}")
        assert res.status_code == 200, f"Report download {fmt} failed: {res.status_code}"
        assert len(res.content) > 100, f"Report content too small for {fmt}"
        print(f"[PASS] 12. Report download ({fmt}) verified ({len(res.content)} bytes)")

    # 13. Dashboard Page Route
    res = requests.get(f"{BASE_URL}/dashboard?dataset_id={dataset_id}")
    assert res.status_code == 200
    assert f'window.INITIAL_DATASET_ID = "{dataset_id}"' in res.text
    print(f"[PASS] 13. Dashboard page rendered with INITIAL_DATASET_ID set")

    # 14. Chat Page Route
    res = requests.get(f"{BASE_URL}/chat?dataset_id={dataset_id}")
    assert res.status_code == 200
    print(f"[PASS] 14. Chat page rendered with dataset_id")

    print("\n========================================================")
    print("ALL 14 END-TO-END PIPELINE & COMPUTATION TESTS PASSED!")
    print("========================================================")

if __name__ == "__main__":
    try:
        test_full_pipeline()
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)
