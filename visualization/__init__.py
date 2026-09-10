"""
Visualization package for AI Data Analyst Agent.
Handles interactive Plotly chart specifications and dashboard rendering.
"""

from .charts import (
    generate_summary_charts,
    recommend_charts,
    build_custom_chart_spec,
    inspect_column_types,
    ChartRecommendationEngine,
    build_histogram_spec,
    build_single_boxplot_spec,
    build_bar_chart_spec,
    build_category_distribution_spec,
    build_scatter_spec,
    build_line_chart_spec,
    build_grouped_bar_spec,
    build_time_category_spec,
    build_correlation_heatmap_spec,
    build_outlier_boxplot_spec,
    build_outlier_distribution_spec,
)

__all__ = [
    "generate_summary_charts",
    "recommend_charts",
    "build_custom_chart_spec",
    "inspect_column_types",
    "ChartRecommendationEngine",
    "build_histogram_spec",
    "build_single_boxplot_spec",
    "build_bar_chart_spec",
    "build_category_distribution_spec",
    "build_scatter_spec",
    "build_line_chart_spec",
    "build_grouped_bar_spec",
    "build_time_category_spec",
    "build_correlation_heatmap_spec",
    "build_outlier_boxplot_spec",
    "build_outlier_distribution_spec",
]
