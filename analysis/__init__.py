"""
Analysis package for AI Data Analyst Agent.
Houses deterministic data analysis, statistics, correlations, outliers, and cleaning modules.
"""

from .profiler import profile_dataset, infer_column_types, get_dataset_preview

__all__ = [
    "profile_dataset",
    "infer_column_types",
    "get_dataset_preview",
]
