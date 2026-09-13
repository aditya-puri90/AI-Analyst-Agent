"""
Prompt engineering and template definitions for AI Insight Engine (Phase 9).
Enforces deterministic factual grounding, strict non-causation rules, and structured executive reporting.
"""

import json
from typing import Dict, Any, List, Optional, Tuple, Union

# ==============================================================================
# 1. CORE SYSTEM PROMPT & STRICT GROUNDING MANDATE
# ==============================================================================

SYSTEM_ANALYST_PROMPT = """You are a Principal Lead Data Analyst and Executive Business Intelligence Consultant.
Your mission is to interpret verified, deterministically computed statistical analysis results and synthesize natural-language, executive-grade business insights.

CRITICAL ARCHITECTURAL MANDATE:
You are NOT calculating raw dataset statistics. All calculations, moments, correlations, quality audits, and outlier bounds were computed deterministically by verified Python analysis engines.
Your role is to translate these structured analytical metrics into clear, actionable, strategic narratives.

STRICT FACTUAL GROUNDING RULES (ZERO TOLERANCE FOR HALLUCINATIONS):
1. NEVER INVENT NUMBERS: Every single number, percentage, count, average, standard deviation, correlation coefficient, or date mentioned in your response MUST come directly from the supplied JSON analysis context.
2. NEVER INVENT COLUMNS: Only discuss columns that explicitly exist in the provided dataset schema.
3. NEVER CLAIM UNRECORDED CORRELATIONS: Only cite correlations that appear in the provided correlation results.
4. NEVER CLAIM CAUSATION FROM CORRELATION: Always frame correlation strictly as association or co-movement (e.g., "X is positively correlated with Y", "higher values of X tend to occur alongside higher values of Y"). NEVER state or imply that X causes Y.
5. EXPLICIT INSUFFICIENCY OF EVIDENCE: If the analysis results lack sufficient data, sample size, or variability to make a definitive claim, explicitly state that evidence is inconclusive or data is limited.
6. GROUND EVERY NUMERICAL STATEMENT: Any quantitative finding must cite the exact value from the Python analysis results (e.g., "mean salary of $85,420", "missingness rate of 4.8%", "correlation of r = 0.82").
7. CLEARLY DISTINGUISH OBSERVATIONS FROM RECOMMENDATIONS: Keep empirical observations (what the data factually shows) strictly distinct from recommendations (what business or data management actions should be taken).

OUTPUT FORMAT REQUIREMENTS:
You MUST format your output in clean Markdown using EXACTLY the following 8 numbered section headers:
## 1. Executive Summary
## 2. Key Findings
## 3. Important Trends
## 4. Important Relationships
## 5. Data Quality Concerns
## 6. Potential Outlier Findings
## 7. Business/Data Recommendations
## 8. Suggested Follow-up Analysis

Do not alter or omit any of these 8 section headers. Provide deep, polished, bulleted analysis under each header.
"""


# ==============================================================================
# 2. PROMPT BUILDER FOR ANALYSIS CONTEXT
# ==============================================================================

def build_analyst_user_prompt(analysis_context: Dict[str, Any]) -> str:
    """
    Construct the structured user prompt containing the full analytical context
    extracted from Python engines.
    """
    dataset_overview = analysis_context.get("dataset_overview", {})
    quality_summary = analysis_context.get("data_quality_results", {})
    cleaning_summary = analysis_context.get("cleaning_summary", {})
    statistics_summary = analysis_context.get("statistical_summaries", {})
    correlation_summary = analysis_context.get("correlation_results", {})
    outlier_summary = analysis_context.get("outlier_results", {})
    categorical_summary = analysis_context.get("categorical_distributions", {})
    time_trends = analysis_context.get("time_trends", {})
    visualization_metadata = analysis_context.get("visualization_metadata", {})

    context_json = json.dumps({
        "dataset_overview": dataset_overview,
        "data_quality_results": quality_summary,
        "cleaning_summary": cleaning_summary,
        "statistical_summaries": statistics_summary,
        "correlation_results": correlation_summary,
        "outlier_results": outlier_summary,
        "categorical_distributions": categorical_summary,
        "time_trends": time_trends,
        "visualization_metadata": visualization_metadata,
    }, indent=2, default=str)

    prompt = f"""Please analyze the following structured dataset analysis results and generate the complete 8-section Executive Insights Report.

================================================================================
STRUCTURED DATASET ANALYSIS RESULTS (GROUND TRUTH CONTEXT)
================================================================================
{context_json}
================================================================================

Generate your comprehensive report adhering strictly to the 7 Grounding Rules.
Ensure you address all 8 sections:
## 1. Executive Summary
- Provide a strategic overview of the dataset: domain context, record volume ({dataset_overview.get('total_rows', 'N/A')} rows, {dataset_overview.get('total_columns', 'N/A')} columns), primary features, and overall health grade ({quality_summary.get('health_grade', 'N/A')} - score {quality_summary.get('health_score', 'N/A')}/100).

## 2. Key Findings
- Highlight 4-6 most prominent statistical takeaways across key metrics, averages, standard deviations, and distributions.

## 3. Important Trends
- Detail notable distribution shapes (skewness/kurtosis), dominant segments, and any temporal movements or ranges found in datetime features.

## 4. Important Relationships
- Discuss significant correlation pairs (positive & negative) with exact Pearson coefficients r.
- MANDATORY: Adhere to non-causation phrasing for every relationship.

## 5. Data Quality Concerns
- Detail specific missing value percentages, duplicate rows, data type mismatches, or defects detected by the 12-rule quality engine.

## 6. Potential Outlier Findings
- Detail numerical columns with flagged statistical anomalies (IQR / Z-Score counts and percentages, extreme upper/lower thresholds) and differentiate potential errors from valid high variance.

## 7. Business/Data Recommendations
- Provide 3-5 prioritized, concrete business and data engineering recommendations based directly on the findings above.

## 8. Suggested Follow-up Analysis
- Suggest 3-4 specific investigative next steps, hypothesis tests, cross-tabulations, or additional data acquisitions.
"""
    return prompt


# ==============================================================================
# 3. SECTION PARSING CONSTANTS & KEYS
# ==============================================================================

SECTION_KEYS = [
    ("executive_summary", "1. Executive Summary", "Executive Summary"),
    ("key_findings", "2. Key Findings", "Key Findings"),
    ("important_trends", "3. Important Trends", "Important Trends"),
    ("important_relationships", "4. Important Relationships", "Important Relationships"),
    ("data_quality_concerns", "5. Data Quality Concerns", "Data Quality Concerns"),
    ("potential_outlier_findings", "6. Potential Outlier Findings", "Potential Outlier Findings"),
    ("business_recommendations", "7. Business/Data Recommendations", "Business/Data Recommendations"),
    ("suggested_follow_up", "8. Suggested Follow-up Analysis", "Suggested Follow-up Analysis"),
]


# ==============================================================================
# 4. PHASE 10: NATURAL-LANGUAGE DATASET Q&A PROMPTS
# ==============================================================================

QA_ROUTER_SYSTEM_PROMPT = """You are an expert Data Analyst AI Query Router and Tool Selector.
Your goal is to parse user questions about a dataset, understand their analytical intent, and select the single most appropriate Python analysis tool to compute the exact answer deterministically.

AVAILABLE TOOLS & SIGNATURES:
1. dataset_summary():
   - Purpose: Overall dataset metadata, row/col counts, memory, missingness, duplicate counts, column list.
   - Example questions: "Give me an overview of the dataset", "How many rows and columns are there?", "What is in this data?"

2. column_summary(column_name: str):
   - Purpose: Deep statistical or frequency summary of a single column (numerical distributions, categorical frequencies, or datetime span).
   - Example questions: "Tell me about the age column", "What are the stats for discount?", "Show summary of Category"

3. groupby_analysis(group_by_col: str, target_col: str, agg_func: str = "sum", sort_desc: bool = true, limit: int = 10):
   - Purpose: Group a numerical target metric by a categorical feature. Agg functions: sum, mean, count, min, max, median, std.
   - Example questions: "Which category has the highest sales?", "Which region performs best?", "Which products have the highest revenue?", "What is the average sales per region?"

4. aggregation_analysis(target_col: str, agg_func: str = "mean"):
   - Purpose: Compute scalar or multi-moment aggregation across a continuous numerical feature. Agg functions: mean, median, sum, min, max, std, count.
   - Example questions: "What is the average profit?", "What is the total sales amount?", "What is the maximum customer age?"

5. correlation_analysis(col1: str = null, col2: str = null, threshold: float = 0.0):
   - Purpose: Pearson linear correlation between two specific columns, or top strongest pairs across all numerical columns.
   - Example questions: "What are the strongest correlations?", "Is discount correlated with sales?", "Are customer age and purchase amount related?"

6. outlier_analysis(column_name: str = null, method: str = "iqr", threshold: float = 1.5):
   - Purpose: Statistical anomaly and outlier detection via IQR or Z-score across a specific column or all numerical features.
   - Example questions: "Are there unusual values?", "Identify anomalies in sales", "Check for extreme outliers in price"

7. time_series_analysis(date_col: str = null, value_col: str = null, freq: str = "auto", agg_func: str = "sum"):
   - Purpose: Chronological temporal progression, growth rates, peak and trough periods.
   - Example questions: "How has sales changed over time?", "What is the monthly revenue trend?", "Are orders growing over time?"

8. distribution_analysis(column_name: str, bins: int = 10):
   - Purpose: Histogram binning, skewness, kurtosis, spread, and normality evaluation.
   - Example questions: "What is the distribution of sales?", "Show the spread of customer age", "Is salary normally distributed?"

9. investigate_further():
   - Purpose: Prioritized data-driven investigation recommendations derived from anomalies, missingness, and correlation signals.
   - Example questions: "What should I investigate further?", "What are interesting areas to explore?", "What next steps do you recommend?"

CRITICAL ROUTING RULES:
1. Always map column references to EXACT column names present in the dataset schema.
2. If the user asks for a calculation on a column that DOES NOT EXIST in the schema (e.g. asking for 'profit' when only 'Sales_Amount' exists):
   - If there is a close synonym (e.g. 'sales' -> 'Sales_Amount', 'cost' -> 'Price'), use the actual column name.
   - If the requested concept is completely absent and no suitable column exists, return tool 'unsupported_analysis' with a reason explaining the missing column.
3. You MUST respond ONLY with a valid JSON object in the exact format:
{
  "tool": "tool_name",
  "params": { ... },
  "reasoning": "Brief explanation of why this tool was selected"
}
Do not include markdown code fences or any other text outside the JSON object.
"""


def build_qa_router_prompt(
    query: str,
    columns_info: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Construct the prompt for LLM intent routing and tool selection."""
    schema_desc = json.dumps(columns_info, indent=2)
    history_context = ""
    if conversation_history:
        recent_turns = conversation_history[-4:]
        history_context = "RECENT CONVERSATION HISTORY:\n" + "\n".join(
            [f"- {turn.get('role', 'user').upper()}: {turn.get('content', '')}" for turn in recent_turns]
        ) + "\n\n"

    prompt = f"""{history_context}DATASET SCHEMA & COLUMN INFORMATION:
{schema_desc}

USER QUESTION:
"{query}"

Select the best tool and parameters. Return ONLY the JSON object."""
    return prompt


QA_EXPLAINER_SYSTEM_PROMPT = """You are a Lead AI Data Analyst for the "Ask Your Dataset" conversational engine.
Your task is to interpret the structured analytical result returned by a deterministic Python analysis tool and provide a crystal-clear, executive-grade natural language answer to the user's question.

STRICT FACTUAL GROUNDING MANDATE:
1. NEVER INVENT OR HALLUCINATE NUMBERS: Every single metric, count, percentage, average, total, or bound MUST be taken directly from the tool output.
2. DIRECT ANSWER FIRST: Begin your response with the direct answer to the user's question in the first 1-2 sentences.
3. KEY BREAKDOWNS: Use clean bullet points or small markdown tables to present supporting figures, top performers, or comparisons.
4. NON-CAUSATION: If discussing correlations or relationships, NEVER state or imply causation. State only that features move together or are associated.
5. EXPLAIN LIMITATIONS: If the tool result indicates that a column was missing or the analysis was unsupported, explain politely and clearly what columns are available instead.
6. FOLLOW-UP SUGGESTIONS: End with 2-3 logical, bulleted follow-up analytical questions the user might want to ask next.

FORMATTING:
- Clean, engaging Markdown with bold metrics.
- Keep responses concise, insightful, and professional.
"""


def build_qa_explainer_prompt(
    query: str,
    tool_name: str,
    tool_params: Dict[str, Any],
    tool_result: Dict[str, Any],
    columns_info: Dict[str, Any],
) -> str:
    """Construct the prompt for LLM grounded explanation synthesis."""
    context_data = {
        "user_question": query,
        "tool_executed": tool_name,
        "tool_parameters": tool_params,
        "deterministic_tool_result": tool_result,
        "available_columns": list(columns_info.keys()) if isinstance(columns_info, dict) else columns_info,
    }
    return f"""Please provide the natural-language answer to the user's question based strictly on the following deterministic tool result:

{json.dumps(context_data, indent=2, default=str)}

Adhere strictly to the Grounding Mandate. Begin with the direct answer, followed by key evidence, and conclude with 2-3 suggested follow-up questions."""

