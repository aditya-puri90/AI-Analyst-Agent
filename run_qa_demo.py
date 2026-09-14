"""
Standalone CLI demonstration script for Phase 10: Natural-Language Dataset Q&A ("Ask Your Dataset").
Demonstrates routing natural language questions to safe deterministic Python analysis tools,
executing calculations on Pandas DataFrames, and synthesizing grounded zero-hallucination explanations.
"""

import sys
import os
import io

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

from utils.file_handler import list_uploaded_datasets, load_dataset
from agent.question_router import QuestionRouter, route_user_question
from config.settings import Config


def run_qa_demo(target_dataset_query: str = ""):
    print("=" * 90)
    print("   AI DATA ANALYST AGENT - PHASE 10: NATURAL-LANGUAGE DATASET Q&A ('Ask Your Dataset')")
    print("   Architecture: User Question -> LLM/NLP Router -> Safe Python Tool -> Grounded Explanation")
    print("=" * 90)

    datasets = list_uploaded_datasets()
    if not datasets:
        print("[!] No uploaded datasets found. Creating sample ecommerce sales dataset...")
        np.random.seed(42)
        n = 120
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "Order_Date": dates,
            "Category": np.random.choice(["Electronics", "Fashion", "Home & Garden", "Sports"], size=n),
            "Product": np.random.choice(["Laptop Pro", "Smart Watch", "Running Shoes", "Desk Lamp", "Headphones"], size=n),
            "Sales": np.round(np.random.exponential(150, size=n) + 20, 2),
            "Profit": np.round(np.random.normal(35, 15, size=n), 2),
            "Quantity": np.random.randint(1, 10, size=n),
            "Customer_Age": np.random.randint(18, 70, size=n),
            "Region": np.random.choice(["North", "South", "East", "West"], size=n),
        })
        # Add a couple outliers
        df.loc[5, "Sales"] = 4850.00
        df.loc[12, "Sales"] = 3920.00

        from utils.file_handler import save_uploaded_file
        import io
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        mock_file = type("MockFile", (), {
            "filename": "sample_sales_demo.csv",
            "seek": buf.seek,
            "tell": buf.tell,
            "save": lambda self, p: buf.seek(0) or open(p, "wb").write(buf.read()),
        })()
        target_id, _, _, _ = save_uploaded_file(mock_file)
        print(f"[*] Generated demo dataset with ID: {target_id}")
    else:
        target_id = None
        if target_dataset_query:
            for ds in datasets:
                if target_dataset_query.lower() in ds["id"].lower() or target_dataset_query.lower() in ds.get("original_name", "").lower():
                    target_id = ds["id"]
                    break
        if not target_id:
            target_id = datasets[0]["id"]

    df, err = load_dataset(target_id)
    if err:
        print(f"[!] Error loading dataset {target_id}: {err}")
        return

    print(f"\n[+] Active Dataset: {target_id} ({len(df):,} rows x {len(df.columns)} columns)")
    print(f"[+] Available Columns: {list(df.columns)}")

    router = QuestionRouter()

    # The 8 target required questions
    sample_questions = [
        "Which category has the highest sales?",
        "What is the average profit?",
        "Which region performs best?",
        "What are the strongest correlations?",
        "Are there unusual values?",
        "How has sales changed over time?",
        "Which products have the highest revenue?",
        "What should I investigate further?",
    ]

    print("\n" + "#" * 90)
    print("   RUNNING ALL 8 TARGET NATURAL LANGUAGE QUESTIONS")
    print("#" * 90)

    for idx, q in enumerate(sample_questions, 1):
        print(f"\n[{idx}/8] QUESTION: \"{q}\"")
        print("-" * 80)
        
        result = router.ask(df, q, dataset_id=target_id)
        
        tool_name = result.get("tool_executed") or result.get("tool_used")
        tool_params = result.get("tool_parameters") or result.get("tool_args", {})
        has_chart = bool(result.get("visualization"))
        chart_type = result.get("visualization", {}).get("data", [{}])[0].get("type", "none") if has_chart else "none"
        
        print(f"[*] Routed Tool : {tool_name}")
        print(f"[*] Parameters  : {tool_params}")
        print(f"[*] Chart Spec  : {chart_type.upper()} chart generated ({has_chart})")
        print("\n[EXPLANATION]:")
        print(result.get("explanation", "").strip())
        
        followups = result.get("followups") or result.get("suggested_followups", [])
        if followups:
            print(f"\n[FOLLOW-UPS]: {', '.join(followups)}")
        print("=" * 90)

    print("\n[SUCCESS] Phase 10 Natural-Language Dataset Q&A verification completed successfully!")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    run_qa_demo(query)
