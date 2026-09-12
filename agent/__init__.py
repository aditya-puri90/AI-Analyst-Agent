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
from .question_router import route_user_question
from .prompts import (
    SYSTEM_ANALYST_PROMPT,
    build_analyst_user_prompt,
    SECTION_KEYS,
)

__all__ = [
    "AnalystAgent",
    "build_analysis_context",
    "synthesize_deterministic_insights",
    "parse_insights_sections",
    "route_user_question",
    "SYSTEM_ANALYST_PROMPT",
    "build_analyst_user_prompt",
    "SECTION_KEYS",
]

