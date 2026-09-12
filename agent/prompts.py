"""
Prompt engineering and template definitions for AI Insight Engine (Phase 9).
Enforces deterministic factual grounding, strict non-causation rules, and structured executive reporting.
"""

import json
from typing import Dict, Any

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
