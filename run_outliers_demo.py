"""
Standalone CLI demonstration script for Phase 7 Outlier & Anomaly Detection Engine.
Executes IQR, Z-Score, and Modified Z-Score methods, handles statistical edge cases
(zero std, zero IQR, small sample sizes), classifies potential outliers vs confirmed data errors,
and showcases non-destructive remediation.
"""

import sys
import os
import numpy as np
import pandas as pd
from utils.file_handler import list_uploaded_datasets, load_dataset
from analysis.outliers import (
    OutlierDetectionEngine,
    detect_column_outliers_iqr,
    detect_column_outliers_zscore,
    detect_column_outliers_modified_zscore,
)


def create_demo_dataset() -> pd.DataFrame:
    """Create a rich synthetic dataset with intentional statistical outliers, data errors, and edge cases."""
    np.random.seed(42)
    n = 120

    # 1. Normal feature with a few extreme statistical outliers
    scores = np.random.normal(70, 8, n)
    scores[5] = 115.0   # Extreme high
    scores[18] = 22.0   # Extreme low

    # 2. Right-skewed feature (e.g., Transaction Amount) with valid high outliers
    sales = np.random.exponential(scale=100, size=n) + 15
    sales[12] = 1850.0  # Legitimate mega-transaction
    sales[45] = 2400.0  # Legitimate mega-transaction

    # 3. Feature with Confirmed Data Errors (e.g., Negative age, sentinel code, 250 years)
    ages = np.random.randint(18, 65, size=n).astype(float)
    ages[3] = -999.0    # Sentinel missing code (Data Error)
    ages[22] = -14.0    # Negative age (Data Error)
    ages[77] = 240.0    # Improbable age (Data Error)

    # 4. Feature with Bounded Percentage errors
    discounts = np.random.choice([5.0, 10.0, 15.0, 20.0], size=n)
    discounts[10] = 350.0  # Impossible discount % (Data Error)
    discounts[50] = -25.0  # Negative discount % (Data Error)

    # 5. Edge Case: Constant feature (Zero Standard Deviation)
    constant_col = np.full(n, 50.0)

    # 6. Edge Case: Zero IQR feature (>60% identical zeros)
    zero_iqr_col = np.zeros(n)
    zero_iqr_col[:20] = np.random.randint(5, 50, size=20)

    df = pd.DataFrame({
        "Exam_Score": np.round(scores, 1),
        "Sales_Amount": np.round(sales, 2),
        "Customer_Age": ages,
        "Discount_Pct": discounts,
        "Constant_Feature": constant_col,
        "Zero_Inflated_Calls": zero_iqr_col,
    })
    return df


def run_demo(target_dataset_query: str = "", method: str = "iqr"):
    print("=" * 95)
    print("   AI DATA ANALYST AGENT - PHASE 7 OUTLIER & ANOMALY DETECTION ENGINE")
    print("=" * 95)

    df = None
    target_id = "Synthetic Demo Dataset"

    if target_dataset_query:
        datasets = list_uploaded_datasets()
        for ds in datasets:
            if target_dataset_query.lower() in ds["id"].lower() or target_dataset_query.lower() in ds["original_name"].lower():
                target_id = ds["id"]
                df, err = load_dataset(target_id)
                if err:
                    print(f"[!] Error loading dataset {target_id}: {err}")
                    return
                break

    if df is None:
        print("[i] Using synthetic multi-pattern benchmark dataset with intentional anomalies & edge cases.")
        df = create_demo_dataset()

    print(f"\nAnalyzing Dataset: {target_id}")
    print(f"Shape: {len(df)} rows x {len(df.columns)} columns\n")

    # Run Outlier Detection Engine
    engine = OutlierDetectionEngine(df, dataset_id=target_id)
    outliers_data = engine.analyze(method=method)

    # 1. Executive Summary KPIs
    print("-" * 95)
    print(" [1] EXECUTIVE OUTLIER METRICS & OVERVIEW")
    print("-" * 95)
    print(f" Detection Method:           {outliers_data['method'].upper()} (Param: {outliers_data['parameter']})")
    print(f" Numerical Features:         {outliers_data['numerical_columns_count']}")
    print(f" Columns with Outliers:      {outliers_data['columns_with_outliers_count']}")
    print(f" Clean Features (0 Anom):    {outliers_data['clean_columns_count']}")
    print(f" Total Outlier Instances:    {outliers_data['total_outlier_instances']}")
    print(f"   -> Confirmed Data Errors: {outliers_data['total_confirmed_errors']} (Requires correction/removal)")
    print(f"   -> Potential Outliers:    {outliers_data['total_potential_outliers']} (Legitimate extreme values)")
    print(f" Rows Affected:              {outliers_data['affected_rows_count']} / {outliers_data['total_rows']} ({outliers_data['affected_rows_percentage']}%)")

    # 2. Warnings Banner
    if outliers_data.get("warnings"):
        print("\n" + "-" * 95)
        print(f" [2] STATISTICAL EDGE CASES & WARNINGS ({len(outliers_data['warnings'])} Detected)")
        print("-" * 95)
        for w in outliers_data["warnings"]:
            print(f" [!] {w}")

    # 3. Column Outlier Summary Table
    print("\n" + "-" * 95)
    print(" [3] COLUMN OUTLIER SUMMARY TABLE")
    print("-" * 95)
    header = f"{'#':<3} {'Feature':<22} {'Valid N':<9} {'Outliers':<10} {'Outlier %':<11} {'Lower Bound':<14} {'Upper Bound':<14} {'Diagnosis':<16}"
    print(header)
    print("-" * len(header))

    for idx, row in enumerate(outliers_data["summary_table"]):
        col_name = row["column_name"]
        n_val = str(row["valid_observations"])
        out_cnt = f"{row['outlier_count']}"
        out_pct = f"{row['outlier_percentage']:.2f}%"
        lower_str = f"{row['lower_threshold']:.2f}" if row['lower_threshold'] is not None else "--"
        upper_str = f"{row['upper_threshold']:.2f}" if row['upper_threshold'] is not None else "--"
        diag = row["status"]
        print(f"{idx+1:<3} {col_name:<22} {n_val:<9} {out_cnt:<10} {out_pct:<11} {lower_str:<14} {upper_str:<14} {diag:<16}")

    # 4. Drill-Down: Flagged Records & Error Classification
    print("\n" + "-" * 95)
    print(" [4] DETAILED ANOMALY INSPECTION (Sample Flagged Instances)")
    print("-" * 95)
    for col, details in outliers_data["column_details"].items():
        if details.get("outlier_count", 0) > 0:
            print(f"\n * Feature '{col}' ({details['outlier_count']} outliers, bounds: [{details['lower_threshold']}, {details['upper_threshold']}]):")
            for o in details["outliers"][:4]:  # Show top 4 per column
                tag = f"[{o['classification'].upper()}]"
                print(f"   - Row {o['row_index']:<4} | Value: {o['value']:<8} | {tag:<25} | {o['rationale']}")
            if len(details["outliers"]) > 4:
                print(f"     ... and {len(details['outliers']) - 4} more flagged records.")

    # 5. Method Comparison: IQR vs Z-Score
    print("\n" + "-" * 95)
    print(" [5] METHOD COMPARISON: IQR (1.5x) vs. Z-SCORE (3.0 std)")
    print("-" * 95)
    z_engine = engine.analyze(method="zscore", param=3.0)
    print(f"{'Feature':<22} {'IQR Outliers':<16} {'IQR Bounds':<25} {'Z-Score Outliers':<18} {'Z-Score Bounds':<25}")
    print("-" * 95)
    for col in outliers_data["column_details"]:
        iqr_res = outliers_data["column_details"][col]
        z_res = z_engine["column_details"].get(col, {})
        iqr_b = f"[{iqr_res.get('lower_threshold')}, {iqr_res.get('upper_threshold')}]"
        z_b = f"[{z_res.get('lower_threshold')}, {z_res.get('upper_threshold')}]"
        print(f"{col:<22} {iqr_res.get('outlier_count', 0):<16} {iqr_b:<25} {z_res.get('outlier_count', 0):<18} {z_b:<25}")

    # 6. Non-Destructive Remediation Demonstration
    print("\n" + "-" * 95)
    print(" [6] NON-DESTRUCTIVE REMEDIATION SIMULATION")
    print("-" * 95)
    
    # Operation A: Remove data errors only
    df_err_cleaned, sum_err = engine.remediate_outliers(action="remove_errors_only", method="iqr")
    print(f" [A] Remove Data Errors Only: {sum_err['rows_dropped']} rows removed. Rows remaining: {len(df_err_cleaned)} (Raw immutable: {len(df)})")
    
    # Operation B: Cap extreme values (Winsorization)
    df_capped, sum_cap = engine.remediate_outliers(action="cap", method="iqr")
    print(f" [B] Cap Outliers to Bounds:  {sum_cap['values_capped']} values winsorized. Rows remaining: {len(df_capped)} (Raw immutable: {len(df)})")

    # Operation C: Remove all outliers
    df_all_cleaned, sum_all = engine.remediate_outliers(action="remove", method="iqr")
    print(f" [C] Remove All Outliers:     {sum_all['rows_dropped']} rows removed. Rows remaining: {len(df_all_cleaned)} (Raw immutable: {len(df)})")

    print("\n" + "=" * 95)
    print(" [OK] Phase 7 Outlier Detection Demonstration Complete.")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else ""
    m = sys.argv[2] if len(sys.argv) > 2 else "iqr"
    run_demo(q, method=m)
