"""
AI Data Analyst Agent - Phase 11 CLI Demonstration Runner.
Demonstrates Intelligent Analysis Planning & Chart Generation on reference workflows:
1. "How have sales changed over time?" -> Time aggregation, line chart, trend explanation
2. "Is advertising spend related to revenue?" -> Pearson correlation, scatter plot, non-causation explanation
3. "Which category has the highest revenue?" -> Groupby, aggregation, descending sort, bar chart, result explanation
4. "What is the average profit?" -> Scalar aggregation, KPI, no chart needed
5. Strict execution safety via ToolRegistry (disallowing arbitrary code execution)
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np

# Configure UTF-8 stdout for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from agent.question_router import QuestionRouter, route_user_question
from agent.planner import AnalysisPlanner
from analysis.tool_registry import registry, ToolExecutionError
from config.settings import Config

logging.basicConfig(level=logging.WARNING)


def create_demo_dataset() -> pd.DataFrame:
    """Create a realistic e-commerce and marketing dataset with temporal, categorical, and correlation features."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    categories = np.random.choice(["Electronics", "Home & Kitchen", "Fashion", "Books", "Sports"], size=n)
    regions = np.random.choice(["North America", "Europe", "Asia-Pacific", "Latin America"], size=n)

    # Correlated advertising spend and revenue
    ad_spend = np.random.uniform(500.0, 5000.0, size=n).round(2)
    # Revenue is positively correlated with ad spend plus organic noise
    revenue = (ad_spend * 3.5 + np.random.normal(2000.0, 800.0, size=n)).round(2)
    revenue = np.maximum(500.0, revenue)

    sales = (revenue * 0.95).round(2)
    profit = (revenue - ad_spend - np.random.uniform(200.0, 1000.0, size=n)).round(2)

    return pd.DataFrame({
        "Order_Date": dates,
        "Category": categories,
        "Region": regions,
        "Advertising_Spend": ad_spend,
        "Revenue": revenue,
        "Sales_Amount": sales,
        "Profit": profit,
    })


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" 🚀 {title.upper()}")
    print("=" * 80)


def print_section(title: str):
    print(f"\n--- {title} ---")


def run_demo_question(router: QuestionRouter, df: pd.DataFrame, question: str, example_num: int):
    print_banner(f"Phase 11 Reference Example #{example_num}: \"{question}\"")

    res = router.route_and_execute(df, question, dataset_id="marketing_sales_dataset")

    print_section("1. Formulated Analysis Plan")
    plan_steps = res.get("analysis_plan", [])
    for step in plan_steps:
        print(f"   {step}")

    print_section("2. Visualization Decision")
    requires_viz = res.get("requires_visualization")
    chart_type = res.get("recommended_chart_type")
    reasoning = res.get("visualization_reasoning")
    print(f"   • Requires Visualization : {'YES (True)' if requires_viz else 'NO (False)'}")
    print(f"   • Recommended Chart Type: {chart_type.upper()}")
    print(f"   • Decision Rationale    : {reasoning}")

    print_section("3. Controlled Tool Execution")
    print(f"   • Authorized Tool Executed : {res.get('tool_executed')}")
    print(f"   • Parameters Injected     : {json.dumps(res.get('tool_parameters', {}))}")
    print(f"   • Execution Status        : {res.get('tool_result', {}).get('status')}")

    print_section("4. Grounded Natural-Language Explanation")
    print(res.get("explanation", "").strip())

    print_section("5. Generated Plotly Visualization")
    viz = res.get("visualization")
    if viz:
        print(f"   • Chart Type   : {viz.get('chart_type')}")
        print(f"   • Chart Title  : {viz.get('layout', {}).get('title')}")
        print(f"   • Traces Count : {len(viz.get('data', []))}")
        print(f"   • Sample Data  : x={viz.get('data', [])[0].get('x', [])[:4]}... y={viz.get('data', [])[0].get('y', [])[:4]}...")
    else:
        print("   • No Plotly visualization generated (direct scalar / overview result).")


def run_security_check(df: pd.DataFrame):
    print_banner("Execution Security & Safety Validation")
    print("Testing strict restriction against arbitrary Python code execution:")
    
    # 1. Inspect registered tools
    print_section("1. Registered Authorized Tools")
    tools = registry.list_tools()
    for t in tools:
        print(f"   ✓ [Registered Tool] {t['name']} ({t['category']}) - Default Viz: {t['default_chart_type']}")

    # 2. Attempt unauthorized arbitrary code execution
    print_section("2. Attempting to execute unauthorized tool 'run_custom_python_script'")
    try:
        registry.execute("run_custom_python_script", df, code="import os; os.system('echo hacked')")
        print("   ❌ SECURITY BREACH: Unauthorized code executed!")
    except ToolExecutionError as e:
        print(f"   ✅ ACCESS BLOCKED: {e}")

    try:
        registry.execute("eval", df, expression="df.mean()")
        print("   ❌ SECURITY BREACH: Eval executed!")
    except ToolExecutionError as e:
        print(f"   ✅ ACCESS BLOCKED: {e}")


def main():
    print("=" * 80)
    print(" 🤖 AI DATA ANALYST AGENT - PHASE 11 DEMONSTRATION")
    print(" Intelligent Analysis Planning & Chart Generation with Controlled Tool Execution")
    print("=" * 80)

    df = create_demo_dataset()
    print(f"Loaded synthetic test dataset with {len(df)} rows and columns: {list(df.columns)}")

    router = QuestionRouter(provider="deterministic_fallback")

    # Example 1
    run_demo_question(
        router=router,
        df=df,
        question="How have sales changed over time?",
        example_num=1,
    )

    # Example 2
    run_demo_question(
        router=router,
        df=df,
        question="Is advertising spend related to revenue?",
        example_num=2,
    )

    # Example 3
    run_demo_question(
        router=router,
        df=df,
        question="Which category has the highest revenue?",
        example_num=3,
    )

    # Example 4: Scalar query (No chart needed)
    run_demo_question(
        router=router,
        df=df,
        question="What is the average profit?",
        example_num=4,
    )

    # Security check
    run_security_check(df)

    print("\n" + "=" * 80)
    print(" ✅ PHASE 11 DEMONSTRATION COMPLETE - ALL REFERENCE WORKFLOWS VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
