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

from .statistics import (
    StatisticalAnalysisEngine,
    compute_descriptive_statistics,
    analyze_numerical_column,
    analyze_categorical_column,
    analyze_datetime_column,
    generate_statistical_observations,
)

__all__ = [
    "DatasetProfiler",
    "profile_dataset",
    "profile_column",
    "infer_column_type",
    "infer_column_types",
    "get_dataset_preview",
    "StatisticalAnalysisEngine",
    "compute_descriptive_statistics",
    "analyze_numerical_column",
    "analyze_categorical_column",
    "analyze_datetime_column",
    "generate_statistical_observations",
]
