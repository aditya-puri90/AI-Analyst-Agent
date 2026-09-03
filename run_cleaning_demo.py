"""
Interactive Showcase & Verification Demo for Phase 4: Automated Data Quality and Cleaning Engine.
"""

import sys
import io
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from analysis.cleaning import (
    detect_data_quality_issues,
    generate_cleaning_preview,
    execute_cleaning_pipeline,
)
from utils.file_handler import save_uploaded_file, save_processed_dataset, load_dataset


def run_demo():
    print("=" * 75)
    print("AI DATA ANALYST AGENT - PHASE 4 CLEANING ENGINE SHOWCASE")
    print("=" * 75)

    # 1. Create a representative messy dataset with all 12 defect categories
    data = {
        "Full_Name": [" John Doe ", "Jane Smith", "JOHN DOE", "alice brown ", "Bob Martin", "Jane Smith"],
        "Customer_Age": [28, 34, -999, 45, np.nan, 34],
        "Annual_Salary": ["$65,000.00", "$92,500.50", "$48,000.00", "$120,000.00", "$85,000.00", "$92,500.50"],
        "Join_Date": ["2022-01-10", "2021-05-15", "2023-09-01", "2020-11-20", "2022-08-30", "2021-05-15"],
        "Department": [" engineering ", "Sales", "sales", "Engineering", "Marketing", "Sales"],
        "Account_Balance": [1500.0, 3200.0, -50.0, 95000.0, 4200.0, 3200.0],
        "Company_Branch": ["HQ-North"] * 6,  # Constant (0 variance)
        "Survey_Feedback": [None, None, "Great", None, None, None],  # High missing (83.3%)
        "Is_Active": [True, True, True, True, True, False],  # Near constant (83.3%)
    }
    df = pd.DataFrame(data)

    print("\n[1] RAW DATASET (Shape: {} rows, {} columns):".format(*df.shape))
    print(df)

    # 2. Run the 12 Data Quality Detectors
    print("\n[2] RUNNING 12 DATA QUALITY ISSUE DETECTORS:")
    issues = detect_data_quality_issues(df)
    print(f"Total Issues Detected: {len(issues)}")
    print("-" * 75)
    for idx, issue in enumerate(issues, 1):
        print(f"[{idx:02d}] {issue['severity'].upper():<8} | Col: {issue['column']:<16} | Type: {issue['issue_type']:<28}")
        print(f"     Affected: {issue['affected_rows']} rows ({issue['percentage_affected']}%)")
        print(f"     Action:   {issue['recommended_action']}")

    # 3. Generate Transformation Preview
    print("\n[3] TRANSFORMATION PREVIEWS (Original -> Proposed Cleaned Value):")
    preview = generate_cleaning_preview(df, issues)
    for p in preview["previews"]:
        print(f"  * [{p['category']}] {p['column']}: '{p['original_value']}' -> '{p['cleaned_value']}' ({p['rule']})")

    # 4. Execute Non-Destructive Cleaning Pipeline
    print("\n[4] EXECUTING CONFIGURABLE CLEANING PIPELINE:")
    operations = {
        "remove_duplicates": True,
        "fill_numeric_missing": "median",
        "fill_categorical_missing": "mode",
        "standardize_whitespace": True,
        "standardize_categorical": True,
        "convert_numeric_strings": True,
        "convert_datetime_strings": True,
        "handle_invalid_numerical": True,
        "drop_high_missing_columns": True,
        "high_missing_threshold": 0.5,
        "drop_constant_columns": True,
        "cap_outliers": True,
    }
    cleaned_df, summary = execute_cleaning_pipeline(df, operations)

    print("\n[5] CLEANING EXECUTION SUMMARY:")
    for k, v in summary.items():
        if k != "column_transformations":
            print(f"  * {k:<25}: {v}")

    print("\nColumn Transformations Detail:")
    for col, trans in summary["column_transformations"].items():
        print(f"  * {col}: {', '.join(trans)}")

    print("\n[6] CLEANED DATASET (Shape: {} rows, {} columns):".format(*cleaned_df.shape))
    print(cleaned_df)
    print("\nCleaned Column Dtypes:")
    for col in cleaned_df.columns:
        print(f"  * {col:<20}: {cleaned_df[col].dtype}")

    print("\n" + "=" * 75)
    print("SUCCESS: PHASE 4 AUTOMATED DATA QUALITY AND CLEANING ENGINE VERIFIED!")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
