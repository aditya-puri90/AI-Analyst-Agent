"""
Visualization package for AI Data Analyst Agent.
Handles interactive Plotly chart specifications and dashboard rendering.
"""

from .charts import (
    generate_summary_charts,
    build_correlation_heatmap_spec,
    build_outlier_boxplot_spec,
    build_outlier_distribution_spec,
)

__all__ = [
    "generate_summary_charts",
    "build_correlation_heatmap_spec",
    "build_outlier_boxplot_spec",
    "build_outlier_distribution_spec",
]
