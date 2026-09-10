"""
Standalone CLI demonstration script for Phase 8: Automatic Visualization Engine.
Inspects columns, executes rule-based chart recommendations, displays ranking and analytical rationales,
and demonstrates custom chart specification generation.
"""

import sys
import os
import pandas as pd
from utils.file_handler import list_uploaded_datasets, load_dataset
from visualization.charts import (
    ChartRecommendationEngine,
    inspect_column_types,
    build_custom_chart_spec,
)


def run_demo(target_dataset_query: str = "", limit: int = 8):
    print("=" * 85)
    print("   AI DATA ANALYST AGENT - PHASE 8 AUTOMATIC VISUALIZATION ENGINE")
    print("=" * 85)

    datasets = list_uploaded_datasets()
    if not datasets:
        print("[!] No uploaded datasets found. Please upload or ingest a CSV first.")
        return

    # Select dataset
    target_id = None
    if target_dataset_query:
        for ds in datasets:
            if target_dataset_query.lower() in ds["id"].lower() or target_dataset_query.lower() in ds["original_name"].lower():
                target_id = ds["id"]
                break

    if not target_id:
        target_id = datasets[0]["id"]

    df, err = load_dataset(target_id)
    if err:
        print(f"[!] Error loading dataset {target_id}: {err}")
        return

    print(f"\nAnalyzing Dataset ID: {target_id}")
    print(f"Shape: {len(df)} rows x {len(df.columns)} columns\n")

    # 1. Column Semantic Role Classification
    print("-" * 85)
    print(" [1] COLUMN SEMANTIC ROLE CLASSIFICATION")
    print("-" * 85)
    schema = inspect_column_types(df)
    for role, cols in schema.items():
        print(f"  * {role.capitalize():<14}: ({len(cols)}) {', '.join(cols) if cols else 'None'}")

    # 2. Automated Chart Recommendations
    print("\n" + "-" * 85)
    print(f" [2] AUTOMATIC CHART RECOMMENDATIONS (Ranked Top {limit})")
    print("-" * 85)
    engine = ChartRecommendationEngine(df, dataset_id=target_id)
    recommendations = engine.recommend(limit=limit)

    for idx, rec in enumerate(recommendations):
        print(f"\n  #{idx+1} [{rec['priority']} Priority | Score: {rec['score']}] - {rec['title']}")
        print(f"     Type:        {rec['chart_type'].upper()} ({rec['category']})")
        print(f"     Features:    {', '.join(rec['columns_used'])}")
        print(f"     Rationale:   {rec['rationale']}")
        data_traces = len(rec['plotly_spec'].get('data', []))
        print(f"     Plotly Spec: Valid ({data_traces} trace{'s' if data_traces != 1 else ''})")

    # 3. Custom Chart Builder Demonstration
    print("\n" + "-" * 85)
    print(" [3] BUILD YOUR OWN CHART (Custom Specification Builder)")
    print("-" * 85)
    num_cols = schema.get("numerical", [])
    cat_cols = schema.get("categorical", [])

    if num_cols and cat_cols:
        x_col = cat_cols[0]
        y_col = num_cols[0]
        print(f"  Building custom Mean Bar Chart: X='{x_col}', Y='{y_col}', Agg='mean'")
        custom_spec = build_custom_chart_spec(
            df=df,
            chart_type="bar",
            x_col=x_col,
            y_col=y_col,
            aggregation="mean",
            title=f"Mean {y_col} across {x_col}",
        )
        print(f"  Custom Spec: Title='{custom_spec['layout']['title']['text']}'")
        print(f"  Traces Count: {len(custom_spec['data'])}")

    print("\n" + "=" * 85)
    print(" [OK] Phase 8 Visualization Engine execution completed successfully.")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    run_demo(query)
