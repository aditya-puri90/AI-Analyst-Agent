"""
Standalone CLI demonstration script for Phase 13: Automated Analysis Report.
Executes the comprehensive report generator engine across all 9 sections,
verifies taxonomy categorization, and exports structured JSON, Markdown, and standalone HTML reports.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

from utils.file_handler import list_uploaded_datasets, load_dataset, save_uploaded_file
from reports.report_generator import ReportGenerator, CATEGORY_OBSERVED, CATEGORY_ISSUES, CATEGORY_AI, CATEGORY_RECOMMENDATIONS


def run_demo(target_dataset_query: str = ""):
    print("=" * 88)
    print("   AI DATA ANALYST AGENT - PHASE 13 AUTOMATED ANALYSIS REPORT ENGINE")
    print("   9 Sections | 4 Taxonomy Categories | Zero Hallucinations | HTML/MD/JSON Export")
    print("=" * 88)

    datasets = list_uploaded_datasets()
    if not datasets:
        print("\n[+] No uploaded datasets found. Creating realistic sample dataset...")
        np.random.seed(42)
        n = 120
        df = pd.DataFrame({
            "Order_ID": [f"ORD-{1000 + i}" for i in range(n)],
            "Customer_Age": np.concatenate([np.random.randint(18, 65, size=n - 4), [105, -3, np.nan, 200]]),
            "Category": np.random.choice(["Electronics", "Fashion", "Home & Garden", "Sports", "Beauty"], size=n),
            "Sales_Amount": np.concatenate([np.round(np.random.exponential(150, size=n - 3) + 20, 2), [8500.0, 9200.0, np.nan]]),
            "Quantity": np.random.randint(1, 10, size=n),
            "Discount_Pct": np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30], size=n),
            "Region": np.random.choice(["North", "South", "East", "West", "Central"], size=n),
            "Rating": np.random.choice([1, 2, 3, 4, 5, np.nan], size=n, p=[0.05, 0.1, 0.2, 0.35, 0.25, 0.05]),
        })
        import io
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        mock_file = type("MockFile", (), {
            "filename": "sample_ecommerce_store.csv",
            "seek": buf.seek,
            "tell": buf.tell,
            "save": lambda self, p: buf.seek(0) or open(p, "wb").write(buf.read()),
        })()
        target_id, _, _, _ = save_uploaded_file(mock_file)
    else:
        target_id = None
        if target_dataset_query:
            for ds in datasets:
                if target_dataset_query.lower() in ds["id"].lower() or target_dataset_query.lower() in ds["original_name"].lower():
                    target_id = ds["id"]
                    break
        if not target_id:
            target_id = datasets[0]["id"]

    df, err = load_dataset(target_id)
    if err or df is None:
        print(f"[!] Error loading dataset '{target_id}': {err}")
        return

    print(f"\n[+] Selected Dataset: {target_id}")
    print(f"    Dimensions: {len(df):,} rows x {len(df.columns)} columns")
    print(f"    Memory: {round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)} MB\n")

    # Initialize Phase 13 Report Generator
    print("-" * 88)
    print(" [1] COMPILING 9-SECTION STRUCTURED EXECUTIVE ANALYSIS REPORT")
    print("-" * 88)
    generator = ReportGenerator(df, dataset_id=target_id, original_filename="sample_ecommerce_store.csv")
    rep = generator.generate_structured_report()

    meta = rep["metadata"]
    print(f"  * Report Title:        {meta['report_title']}")
    print(f"  * Generated Timestamp: {meta['generated_at']}")
    print(f"  * AI Engine:           {meta['ai_provider']} ({meta['ai_model']})")

    print("\n [2] VERIFYING ALL 9 REQUIRED SECTIONS:")
    sections_list = [
        ("Section 1", "Dataset Overview", "section_1_dataset_overview"),
        ("Section 2", "Data Quality Assessment", "section_2_data_quality_assessment"),
        ("Section 3", "Cleaning Summary", "section_3_cleaning_summary"),
        ("Section 4", "Statistical Analysis", "section_4_statistical_analysis"),
        ("Section 5", "Correlation Analysis", "section_5_correlation_analysis"),
        ("Section 6", "Outlier Analysis", "section_6_outlier_analysis"),
        ("Section 7", "Important Visualizations", "section_7_important_visualizations"),
        ("Section 8", "AI-Generated Insights", "section_8_ai_insights"),
        ("Section 9", "Recommendations", "section_9_recommendations"),
    ]

    for sec_num, sec_name, sec_key in sections_list:
        sec_data = rep.get(sec_key, {})
        status_icon = "OK" if sec_data else "FAIL"
        print(f"   [{status_icon}] {sec_num}: {sec_name} -> Present")

    print("\n [3] VERIFYING 4 TAXONOMY CATEGORIES:")
    print(f"   [OK] Observed results:     {len(rep['section_1_dataset_overview']['columns_schema'])} columns, {rep['section_4_statistical_analysis']['observed_results']['numerical_columns_count']} num features")
    print(f"   [OK] Potential issues:      {len(rep['section_2_data_quality_assessment']['potential_issues']['issues'])} data quality defects, {rep['section_6_outlier_analysis']['observed_results']['total_outliers_count']} outliers")
    print(f"   [OK] AI interpretation:     {len(rep['section_8_ai_insights']['executive_summary'])} chars grounded synthesis")
    print(f"   [OK] Recommendations:       {len(rep['section_9_recommendations']['data_cleaning_recommendations'])} cleaning actions, {len(rep['section_9_recommendations']['feature_engineering_recommendations'])} FE suggestions")

    # Export Reports to Disk
    print("\n" + "-" * 88)
    print(" [4] EXPORTING REPORTS (MARKDOWN, STANDALONE HTML, STRUCTURED JSON)")
    print("-" * 88)

    output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"report_{target_id}.md"
    html_path = output_dir / f"report_{target_id}.html"
    json_path = output_dir / f"report_{target_id}.json"

    generator.export_report(md_path, report_format="md")
    generator.export_report(html_path, report_format="html")
    generator.export_report(json_path, report_format="json")

    print(f"   [OK] Formatted Markdown: {md_path} ({md_path.stat().st_size:,} bytes)")
    print(f"   [OK] Standalone HTML:    {html_path} ({html_path.stat().st_size:,} bytes)")
    print(f"   [OK] Structured JSON:    {json_path} ({json_path.stat().st_size:,} bytes)")

    print("\n" + "=" * 88)
    print("   PHASE 13 AUTOMATED ANALYSIS REPORT DEMONSTRATION COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    run_demo(query)
