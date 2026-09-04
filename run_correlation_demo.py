"""
Standalone CLI demonstration script for Phase 6 Correlation Analysis Engine.
Executes Pearson correlation matrix calculations, classifies strength and direction,
ranks variable associations, and generates key non-causal correlation insights.
"""

import sys
import os
import pandas as pd
from utils.file_handler import list_uploaded_datasets, load_dataset
from analysis.correlation import CorrelationAnalysisEngine, compute_correlation_analysis


def run_demo(target_dataset_query: str = "", threshold: float = 0.0):
    print("=" * 85)
    print("   AI DATA ANALYST AGENT - PHASE 6 CORRELATION ANALYSIS ENGINE")
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

    engine = CorrelationAnalysisEngine(df, dataset_id=target_id)
    corr_data = engine.analyze(threshold=threshold)

    cols = corr_data["numerical_columns"]
    matrix = corr_data["correlation_matrix"]
    pairs = corr_data["ranked_pairs"]
    key_corrs = corr_data["key_correlations"]

    # 1. Pearson Correlation Matrix Table
    print("-" * 85)
    print(f" [1] PEARSON CORRELATION MATRIX ({len(cols)} Numerical Features)")
    print("-" * 85)
    if cols:
        header_str = f"{'Feature':<20}" + "".join([f"{c[:10]:>12}" for c in cols])
        print(header_str)
        print("-" * len(header_str))
        for row_c in cols:
            row_str = f"{row_c:<20}"
            for col_c in cols:
                val = matrix.get(row_c, {}).get(col_c)
                val_str = f"{val:.4f}" if val is not None else "--"
                row_str += f"{val_str:>12}"
            print(row_str)
    else:
        print("No numerical columns found.")

    # 2. Key Correlations Highlights
    print("\n" + "-" * 85)
    print(" [2] KEY CORRELATIONS (Prominent Linear Associations)")
    print("-" * 85)
    highlights = key_corrs.get("key_highlights", [])
    if highlights:
        for h in highlights:
            pair_display = h['pair'].replace('\u2194', '<->')
            print(f"* {pair_display}")
            print(f"  Correlation: {h['correlation']:.4f}")
            print(f"  Direction:   {h['direction']}")
            print(f"  Strength:    {h['strength']}")
            print(f"  Insight:     {h['explanation'].replace(chr(8596), '<->')}\n")
    else:
        print("  No strong or moderate linear associations detected.")

    # 3. Ranked Correlations Association Table
    print("-" * 85)
    print(f" [3] RANKED CORRELATIONS TABLE ({len(pairs)} Pairs, Threshold |r| >= {threshold:.2f})")
    print("-" * 85)
    print(f"{'#':<4} {'Variable A':<20} {'Variable B':<20} {'Correlation':<14} {'Strength':<15} {'Direction':<12}")
    print("-" * 85)
    for idx, p in enumerate(pairs):
        r_str = f"{p['correlation']:.4f}" if p['correlation'] is not None else "--"
        print(f"{idx+1:<4} {p['variable_a']:<20} {p['variable_b']:<20} {r_str:<14} {p['strength']:<15} {p['direction']:<12}")

    # 4. Non-Causation Disclaimer
    print("\n" + "=" * 85)
    print(" SCIENTIFIC & METHODOLOGICAL NOTICE:")
    print(f" {corr_data['disclaimer']}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    run_demo(query, threshold=thresh)
