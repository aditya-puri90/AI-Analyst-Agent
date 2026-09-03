"""
Standalone CLI demonstration script for Phase 5 Statistical Analysis Engine.
Executes pure Python calculations across numerical, categorical, and datetime columns.
"""

import sys
import os
import pandas as pd
from utils.file_handler import list_uploaded_datasets, load_dataset
from analysis.statistics import StatisticalAnalysisEngine, compute_descriptive_statistics


def run_demo(target_dataset_query: str = ""):
    print("=" * 80)
    print("   AI DATA ANALYST AGENT - PHASE 5 STATISTICAL ANALYSIS ENGINE")
    print("=" * 80)

    datasets = list_uploaded_datasets()
    if not datasets:
        print("[!] No uploaded datasets found. Please upload a CSV first.")
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

    engine = StatisticalAnalysisEngine(df, dataset_id=target_id)
    stats_data = engine.analyze()

    num_stats = stats_data["numerical_statistics"]
    cat_stats = stats_data["categorical_statistics"]
    dt_stats = stats_data["datetime_statistics"]
    obs = stats_data["statistical_observations"]

    # 1. Numerical Moments Table
    print("-" * 80)
    print(f" [1] NUMERICAL SUMMARY STATISTICS ({len(num_stats)} Features)")
    print("-" * 80)
    print(f"{'Feature':<18} {'Mean':<10} {'Std Dev':<10} {'Median':<10} {'IQR':<10} {'Skewness':<10} {'Kurtosis':<10} {'95% CI':<16}")
    print("-" * 80)
    for col, n in num_stats.items():
        ci = n.get("confidence_interval", {})
        ci_str = f"[{ci.get('lower')}, {ci.get('upper')}]" if ci.get("lower") is not None else "--"
        print(f"{col:<18} {str(n.get('mean')):<10} {str(n.get('std')):<10} {str(n.get('median')):<10} {str(n.get('iqr')):<10} {str(n.get('skewness')):<10} {str(n.get('kurtosis')):<10} {ci_str:<16}")

    # 2. Categorical Distributions
    if cat_stats:
        print("\n" + "-" * 80)
        print(f" [2] CATEGORICAL DISTRIBUTIONS ({len(cat_stats)} Features)")
        print("-" * 80)
        for col, c in cat_stats.items():
            top = c.get("top_categories", [])
            top_str = ", ".join([f"{item['category']}: {item['percentage']}%" for item in top[:3]])
            print(f"* {col:<18} (Cats: {c.get('num_categories')}, Mode: '{c.get('most_frequent_category')}' ({c.get('mode_percentage')}%), Entropy: {c.get('entropy')})")
            print(f"  Top: {top_str}")

    # 3. Datetime Spans
    if dt_stats:
        print("\n" + "-" * 80)
        print(f" [3] DATETIME TEMPORAL ANALYSIS ({len(dt_stats)} Features)")
        print("-" * 80)
        for col, d in dt_stats.items():
            print(f"* {col:<18} Span: {d.get('date_range_formatted')} (Min: {d.get('min_date')}, Max: {d.get('max_date')}, Cadence: {d.get('inferred_frequency')})")

    # 4. Statistical Observations
    print("\n" + "-" * 80)
    print(f" [4] IMPORTANT STATISTICAL OBSERVATIONS ({len(obs)} Findings)")
    print("-" * 80)
    for o in obs:
        sev_tag = f"[{o.get('severity', '').upper()}]"
        print(f"* {sev_tag:<10} {o.get('title')}:")
        print(f"  {o.get('message')}\n")

    print("=" * 80)
    print("Statistical analysis completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    run_demo(query)
