"""
Analysis package for AI Data Analyst Agent.
Houses deterministic data analysis, statistics, correlations, outliers, and cleaning modules.
"""

from .profiler import (
    DatasetProfiler,
    profile_dataset,
    profile_column,
    infer_column_type,
    infer_column_types,
    get_dataset_preview,
)

__all__ = [
    "DatasetProfiler",
    "profile_dataset",
    "profile_column",
    "infer_column_type",
    "infer_column_types",
    "get_dataset_preview",
]
