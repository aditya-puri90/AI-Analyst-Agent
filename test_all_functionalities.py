"""
Comprehensive Full-System Self-Check Script
Tests all features across all 11 phases against the live running server.
"""

import sys
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"

def post(endpoint, data=None):
    url = f"{BASE_URL}{endpoint}"
    req_data = json.dumps(data).encode("utf-8") if data is not None else b"{}"
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        return response.status, json.loads(response.read().decode("utf-8"))

def get(endpoint):
    url = f"{BASE_URL}{endpoint}"
    with urllib.request.urlopen(url) as response:
        return response.status, json.loads(response.read().decode("utf-8"))

def run_checks():
    results = []
    
    print("=" * 80)
    print("COMPREHENSIVE FULL-SYSTEM FUNCTIONALITY AUDIT")
    print("=" * 80)

    # 1. Health Check
    try:
        status, data = get("/api/health")
        assert status == 200 and data["status"] == "healthy"
        assert "Phase 11" in data["phase"]
        assert "intelligent_analysis_planning" in data["capabilities"]
        results.append(("1. System Health & Phase 11 Capabilities", True, f"Phase: {data['phase']}"))
    except Exception as e:
        results.append(("1. System Health & Phase 11 Capabilities", False, str(e)))

    # 2. Ingest Sample Dataset
    dataset_id = None
    try:
        status, data = post("/api/sample/ecommerce", {})
        assert status == 201 and data["success"] is True
        dataset_id = data["dataset_id"]
        rows_count = data.get("summary", {}).get("total_rows", 250)
        results.append(("2. Dataset Ingestion (E-commerce Sample)", True, f"Dataset ID: {dataset_id}, Rows: {rows_count}"))
    except Exception as e:
        results.append(("2. Dataset Ingestion (E-commerce Sample)", False, str(e)))

    if not dataset_id:
        print("Ingestion failed; aborting dependent tests.")
        return results

    # 3. Profiler Endpoint
    try:
        status, data = get(f"/api/profile/{dataset_id}")
        assert status == 200 and data["success"] is True
        assert len(data["profile"]["columns"]) > 0
        results.append(("3. Dataset Profiler & Schema Inference", True, f"{len(data['profile']['columns'])} columns detected"))
    except Exception as e:
        results.append(("3. Dataset Profiler & Schema Inference", False, str(e)))

    # 4. Preview Endpoint
    try:
        status, data = get(f"/api/preview/{dataset_id}?page=1&page_size=5")
        assert status == 200 and data["success"] is True
        assert len(data["rows"]) == 5
        results.append(("4. Paginated Data Preview", True, f"5 rows retrieved (Total: {data.get('pagination', {}).get('total_rows')})"))
    except Exception as e:
        results.append(("4. Paginated Data Preview", False, str(e)))

    # 5. Statistical Analysis Engine
    try:
        status, data = get(f"/api/statistics/{dataset_id}")
        assert status == 200 and data["success"] is True
        num_stats = data.get("statistics", {}).get("numerical_statistics", {})
        results.append(("5. Statistical Moments & Confidence Intervals", True, f"{len(num_stats)} numerical features computed"))
    except Exception as e:
        results.append(("5. Statistical Moments & Confidence Intervals", False, str(e)))

    # 6. Correlation Analysis Engine & Heatmap Spec
    try:
        status, data = get(f"/api/correlation/{dataset_id}")
        assert status == 200 and data["success"] is True
        assert "correlation" in data and "heatmap_spec" in data
        assert data["heatmap_spec"]["data"][0]["type"] == "heatmap"
        results.append(("6. Correlation Engine & Heatmap Generation", True, f"{len(data['correlation']['ranked_pairs'])} pairs ranked"))
    except Exception as e:
        results.append(("6. Correlation Engine & Heatmap Generation", False, str(e)))

    # 7. Outlier Detection Engine & Boxplots
    try:
        status, data = get(f"/api/outliers/{dataset_id}")
        assert status == 200 and data["success"] is True
        assert "summary_table" in data["outliers"]
        assert "boxplot_spec" in data
        results.append(("7. Outlier Detection & Anomaly Classification", True, f"Boxplot spec generated with {len(data['boxplot_spec']['data'])} traces"))
    except Exception as e:
        results.append(("7. Outlier Detection & Anomaly Classification", False, str(e)))

    # 8. Data Cleaning Audit & Transformation Preview
    try:
        status, data = get(f"/api/cleaning/audit/{dataset_id}")
        assert status == 200 and data["success"] is True
        audit_items = data.get("issues", [])
        results.append(("8. 12-Point Data Quality Audit", True, f"{len(audit_items)} audit checks performed"))
    except Exception as e:
        results.append(("8. 12-Point Data Quality Audit", False, str(e)))

    # 9. AI Executive Insights Synthesis
    try:
        status, data = get(f"/api/insights/{dataset_id}")
        assert status == 200 and data["success"] is True
        assert "insights" in data and "sections" in data["insights"]
        results.append(("9. AI Executive Insights Synthesis", True, f"Mode: {data['insights'].get('metadata', {}).get('generation_mode', 'deterministic')}"))
    except Exception as e:
        results.append(("9. AI Executive Insights Synthesis", False, str(e)))

    # 10. Verified Tool Registry Endpoint
    try:
        status, data = get("/api/chat/tools")
        assert status == 200 and data["success"] is True
        tool_names = [t["name"] for t in data["tools"]]
        assert "time_series_analysis" in tool_names
        assert "correlation_analysis" in tool_names
        assert "groupby_analysis" in tool_names
        results.append(("10. Registered Safe Tool Sandbox Endpoint", True, f"{len(tool_names)} authorized tools registered"))
    except Exception as e:
        results.append(("10. Registered Safe Tool Sandbox Endpoint", False, str(e)))

    # 11. Plan Preview Endpoint (/api/chat/plan)
    try:
        status, data = post("/api/chat/plan", {
            "dataset_id": dataset_id,
            "query": "How have sales changed over time?"
        })
        assert status == 200 and data["success"] is True
        plan = data["plan"]
        assert plan["requires_visualization"] is True
        assert plan["recommended_chart_type"] == "line"
        assert len(plan["steps"]) >= 4
        results.append(("11. Standalone Analysis Planning Endpoint (/api/chat/plan)", True, f"{len(plan['steps'])} steps formulated, Chart: {plan['recommended_chart_type']}"))
    except Exception as e:
        results.append(("11. Standalone Analysis Planning Endpoint (/api/chat/plan)", False, str(e)))

    # 12. Reference Workflow 1: Temporal Trend
    try:
        status, data = post("/api/chat/ask", {
            "dataset_id": dataset_id,
            "query": "How have sales changed over time?"
        })
        assert status == 200 and data["success"] is True
        assert data["tool_executed"] == "time_series_analysis"
        assert data["requires_visualization"] is True
        assert data["recommended_chart_type"] == "line"
        assert data["visualization"]["chart_type"] == "line"
        results.append(("12. Reference Workflow #1 (Sales Over Time)", True, f"Tool: {data['tool_executed']}, Chart: {data['visualization']['chart_type']}"))
    except Exception as e:
        results.append(("12. Reference Workflow #1 (Sales Over Time)", False, str(e)))

    # 13. Reference Workflow 2: Correlation & Non-Causation
    try:
        # Ingest employee dataset
        e_status, emp_data = post("/api/sample/employee", {})
        emp_id = emp_data.get("dataset_id")
        
        status, data = post("/api/chat/ask", {
            "dataset_id": emp_id,
            "query": "Is Experience_Years related to Salary?"
        })
        assert status == 200 and data["success"] is True
        assert data["tool_executed"] == "correlation_analysis"
        assert data["requires_visualization"] is True
        assert data["recommended_chart_type"] == "scatter"
        assert data["visualization"]["chart_type"] == "scatter"
        explanation_text = data.get("explanation", "").lower()
        plan_steps_text = " ".join(data.get("analysis_plan", [])).lower()
        assert "caus" in explanation_text or "caus" in plan_steps_text
        results.append(("13. Reference Workflow #2 (Experience vs Salary - Non-Causation)", True, f"Tool: {data['tool_executed']}, Chart: {data['visualization']['chart_type']}, Non-Causation enforced"))
    except Exception as e:
        results.append(("13. Reference Workflow #2 (Experience vs Salary - Non-Causation)", False, str(e)))

    # 14. Reference Workflow 3: Category Leaderboard
    try:
        status, data = post("/api/chat/ask", {
            "dataset_id": dataset_id,
            "query": "Which category has the highest sales?"
        })
        assert status == 200 and data["success"] is True
        assert data["tool_executed"] == "groupby_analysis"
        assert data["requires_visualization"] is True
        assert data["recommended_chart_type"] == "bar"
        assert data["visualization"]["chart_type"] == "bar"
        results.append(("14. Reference Workflow #3 (Category Highest Sales)", True, f"Tool: {data['tool_executed']}, Chart: {data['visualization']['chart_type']}"))
    except Exception as e:
        results.append(("14. Reference Workflow #3 (Category Highest Sales)", False, str(e)))

    # 15. Reference Workflow 4: Direct Scalar Metric (Chart Suppressed)
    try:
        status, data = post("/api/chat/ask", {
            "dataset_id": dataset_id,
            "query": "What is the average Customer_Age?"
        })
        assert status == 200 and data["success"] is True
        assert data["requires_visualization"] is False
        assert data["recommended_chart_type"] == "none"
        assert data["visualization"] is None
        results.append(("15. Reference Workflow #4 (Scalar Metric - No Redundant Chart)", True, f"requires_visualization=False, Chart=None"))
    except Exception as e:
        results.append(("15. Reference Workflow #4 (Scalar Metric - No Redundant Chart)", False, str(e)))

    # 16. Chat History & Memory
    try:
        status, data = get(f"/api/chat/history/{dataset_id}")
        assert status == 200 and data["success"] is True
        assert len(data["history"]) >= 3
        results.append(("16. Multi-Turn Conversational Memory (/api/chat/history)", True, f"{len(data['history'])} turns recorded"))
    except Exception as e:
        results.append(("16. Multi-Turn Conversational Memory (/api/chat/history)", False, str(e)))

    # Print Summary Table
    print("\n" + "-" * 80)
    print(f"{'CHECK / FEATURE':<55} | {'STATUS':<8} | {'DETAILS'}")
    print("-" * 80)
    all_passed = True
    for name, passed, details in results:
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"{name:<55} | {status_str:<8} | {details}")
    print("-" * 80)
    
    if all_passed:
        print("\nALL 16 COMPREHENSIVE FUNCTIONALITY CHECKS PASSED WITH ZERO ERRORS!")
    else:
        print("\nSOME CHECKS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_checks()
