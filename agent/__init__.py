"""
AI Agent package for AI Data Analyst Agent.
Handles natural language dataset Q&A, business insights, and deterministic tool dispatching.
"""

from .analyst_agent import (
    AnalystAgent,
    build_analysis_context,
    synthesize_deterministic_insights,
    parse_insights_sections,
)
from .question_router import (
    QuestionRouter,
    route_user_question,
    session_manager,
    dataset_summary,
    column_summary,
    groupby_analysis,
    aggregation_analysis,
    correlation_analysis,
    outlier_analysis,
    time_series_analysis,
    distribution_analysis,
    investigate_further,
    route_query_deterministic,
    synthesize_tool_explanation,
    build_qa_visualization_spec,
)
from .prompts import (
    SYSTEM_ANALYST_PROMPT,
    build_analyst_user_prompt,
    SECTION_KEYS,
    QA_ROUTER_SYSTEM_PROMPT,
    QA_EXPLAINER_SYSTEM_PROMPT,
    build_qa_router_prompt,
    build_qa_explainer_prompt,
)

__all__ = [
    "AnalystAgent",
    "build_analysis_context",
    "synthesize_deterministic_insights",
    "parse_insights_sections",
    "QuestionRouter",
    "route_user_question",
    "session_manager",
    "dataset_summary",
    "column_summary",
    "groupby_analysis",
    "aggregation_analysis",
    "correlation_analysis",
    "outlier_analysis",
    "time_series_analysis",
    "distribution_analysis",
    "investigate_further",
    "route_query_deterministic",
    "synthesize_tool_explanation",
    "build_qa_visualization_spec",
    "SYSTEM_ANALYST_PROMPT",
    "build_analyst_user_prompt",
    "SECTION_KEYS",
    "QA_ROUTER_SYSTEM_PROMPT",
    "QA_EXPLAINER_SYSTEM_PROMPT",
    "build_qa_router_prompt",
    "build_qa_explainer_prompt",
]

