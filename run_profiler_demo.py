"""
Interactive Phase 3 Dataset Profiler Runner and Terminal Demo.
Supports choosing datasets via CLI argument, e.g. `python run_profiler_demo.py ecommerce`
"""

import sys
import pandas as pd
from analysis.profiler import DatasetProfiler, profile_dataset
from utils.file_handler import list_uploaded_datasets, load_dataset


def main():
    print("==================================================================")
    print("   AI DATA ANALYST AGENT - PHASE 3 DATASET PROFILING ENGINE      ")
    print("==================================================================")
    
    datasets = list_uploaded_datasets()
    print(f"Total uploaded datasets in system: {len(datasets)}")
    
    if not datasets:
        print("No datasets found in data/uploads/. Run the web app to upload CSVs.")
        return

    # Check for CLI search term
    search_term = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    target = None

    if search_term:
        for ds in datasets:
            if search_term in ds["original_name"].lower() or search_term in ds["id"].lower():
                target = ds
                break

    if not target:
        # Default to ecommerce or employee or first dataset
        for ds in datasets:
            if "ecommerce" in ds["filename"].lower():
                target = ds
                break
        if not target:
            target = datasets[0]

    print(f"\nAnalyzing dataset: '{target['original_name']}' (ID: {target['id']})")
    
    df, err = load_dataset(target["id"])
    if err:
        print(f"Error loading dataset: {err}")
        return

    profiler = DatasetProfiler(df, dataset_id=target["id"])
    profile = profiler.profile()
    ov = profile["overview"]
    q = profile["quality"]

    print("\n--- [1] DATASET-LEVEL METRICS ---")
    print(f"  Total Rows:             {ov['total_rows']:,}")
    print(f"  Total Columns:          {ov['total_columns']:,}")
    print(f"  Total Cells:            {ov['total_cells']:,}")
    print(f"  Memory Usage:           {ov['memory_usage_formatted']} ({ov['memory_usage_bytes']:,} bytes)")
    print(f"  Missing Cells:          {ov['total_missing_cells']:,} ({ov['missing_cells_percentage']}%)")
    print(f"  Rows With Missing:      {ov['rows_with_missing']:,} ({ov['rows_with_missing_percentage']}%)")
    print(f"  Duplicate Rows:         {ov['duplicate_rows']:,} ({ov['duplicate_rows_percentage']}%)")

    print("\n--- [2] DATA QUALITY SUMMARY ---")
    print(f"  Composite Health Score: {q['health_score']}/100")
    print(f"  Health Grade:           {q['health_grade']} ({q['quality_status']})")
    print(f"  Completeness Score:     {q['completeness_score']}%")
    print(f"  Uniqueness Score:       {q['uniqueness_score']}%")
    print(f"  Checks Passed:          {q['passed_checks']} of {q['total_checks']}")
    print("  Quality Insights / Warnings:")
    for w in q["warnings"]:
        print(f"    * {w}")

    print("\n--- [3] COLUMN SEMANTIC CLASSIFICATION ---")
    print(f"  Numerical:    {profile['type_counts'].get('Numerical', 0)}")
    print(f"  Categorical:  {profile['type_counts'].get('Categorical', 0)}")
    print(f"  Datetime:     {profile['type_counts'].get('Datetime', 0)}")
    print(f"  Boolean:      {profile['type_counts'].get('Boolean', 0)}")
    print(f"  Other:        {profile['type_counts'].get('Other', 0)}")

    print("\n--- [4] COLUMN PROFILING BREAKDOWN ---")
    print(f"{'#':<3} {'Column Name':<20} {'Type':<12} {'Missing':<14} {'Unique/Dups':<18} {'Summary / Moments'}")
    print("-" * 110)
    
    for i, col in enumerate(profile["columns"], 1):
        col_type = col["classified_type"]
        missing_str = f"{col['missing_count']} ({col['missing_percentage']}%)"
        unique_str = f"{col['unique_count']} u / {col['duplicate_count']} d"
        
        stats_str = ""
        if col_type == "Numerical" and col["numerical_stats"]:
            ns = col["numerical_stats"]
            stats_str = f"Mean: {ns['mean']}, Med: {ns['median']}, Std: {ns['std']}, IQR: {ns['iqr']}, Skew: {ns['skewness']}"
        elif col_type == "Categorical" and col["categorical_stats"]:
            cs = col["categorical_stats"]
            stats_str = f"Cats: {cs['num_categories']}, Top: '{cs['most_frequent_category']}' ({cs['frequency_percentage']}%)"
        elif col_type == "Datetime" and col["datetime_stats"]:
            ds = col["datetime_stats"]
            stats_str = f"Span: {ds['date_range']}"
        elif col_type == "Boolean" and col["boolean_stats"]:
            bs = col["boolean_stats"]
            stats_str = f"True: {bs['true_percentage']}%, False: {bs['false_percentage']}%"
        elif col["other_stats"]:
            os = col["other_stats"]
            stats_str = f"Avg Len: {os['avg_length']} chars"
        
        print(f"{i:<3} {col['name']:<20} {col_type:<12} {missing_str:<14} {unique_str:<18} {stats_str}")

    print("\n--- [5] TABULAR DATAFRAME SUMMARY EXPORT (to_dataframe) ---")
    summary_df = profiler.to_dataframe()
    cols_to_show = [c for c in ["column_name", "inferred_type", "non_null_count", "missing_count", "unique_count", "mean", "median"] if c in summary_df.columns]
    print(summary_df[cols_to_show].to_string(index=False))

    print("\n==================================================================")
    print("Web Dashboard is LIVE at: http://127.0.0.1:5000/dashboard")
    print("==================================================================")


if __name__ == "__main__":
    main()
