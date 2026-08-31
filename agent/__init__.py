"""
AI Agent package for AI Data Analyst Agent.
Handles natural language dataset Q&A, business insights, and deterministic tool dispatching.
"""

from .analyst_agent import AnalystAgent
from .question_router import route_user_question

__all__ = ["AnalystAgent", "route_user_question"]
