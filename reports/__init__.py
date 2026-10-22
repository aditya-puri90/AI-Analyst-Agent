"""
Reports package for AI Data Analyst Agent.
Automated executive reports (HTML, MD, JSON).
"""

from .report_generator import ReportGenerator, generate_markdown_report

__all__ = ["ReportGenerator", "generate_markdown_report"]
