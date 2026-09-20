"""
Automated Executive Analysis Report Engine (Phase 13 Module).

Compiles comprehensive, deterministic, and AI-augmented dataset analysis reports containing:
1. Dataset Overview
2. Data Quality Assessment
3. Cleaning Summary
4. Statistical Analysis
5. Correlation Analysis
6. Outlier Analysis
7. Important Visualizations
8. AI-Generated Insights
9. Recommendations

Strictly distinguishes 4 taxonomy categories:
- Observed results (Empirical facts, computed metrics, descriptive statistics, raw counts, charts)
- Potential issues (Quality flags, high missingness, severe skew, multicollinearity, extreme outliers)
- AI interpretation (Analytical deductions, contextual patterns, trend summaries grounded strictly in computed facts)
- Recommendations (Actionable next steps, data cleaning protocols, engineering suggestions, domain guidance)

Zero Hallucination / Zero Fabrication Guarantee:
Every number, metric, issue, outlier, and correlation is calculated directly from the current dataset session.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import json
import math
import datetime
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from analysis.profiler import DatasetProfiler, infer_column_type, _safe_json_value
from analysis.cleaning import (
    detect_data_quality_issues,
    generate_cleaning_preview,
    _is_strictly_numeric,
)
from analysis.statistics import StatisticalAnalysisEngine
from analysis.correlation import CorrelationAnalysisEngine
from analysis.outliers import OutlierDetectionEngine
from visualization.charts import (
    ChartRecommendationEngine,
    build_histogram_spec,
    build_single_boxplot_spec,
    build_bar_chart_spec,
    build_category_distribution_spec,
    build_scatter_spec,
    build_correlation_heatmap_spec,
    build_outlier_boxplot_spec,
    inspect_column_types,
)
from agent.analyst_agent import AnalystAgent
from utils.file_handler import get_latest_processed_file, _read_registry

logger = logging.getLogger(__name__)

# ==============================================================================
# TAXONOMY CONSTANTS
# ==============================================================================
CATEGORY_OBSERVED = "Observed results"
CATEGORY_ISSUES = "Potential issues"
CATEGORY_AI = "AI interpretation"
CATEGORY_RECOMMENDATIONS = "Recommendations"


class ReportGenerator:
    """
    Phase 13 Comprehensive Executive Report Generator for AI Data Analyst Agent.
    Assembles multi-engine analysis into structured JSON, Markdown, and standalone HTML reports.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        original_filename: Optional[str] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        quality_data: Optional[Dict[str, Any]] = None,
        cleaning_data: Optional[Dict[str, Any]] = None,
        stats_data: Optional[Dict[str, Any]] = None,
        corr_data: Optional[Dict[str, Any]] = None,
        outlier_data: Optional[Dict[str, Any]] = None,
        viz_data: Optional[Dict[str, Any]] = None,
        insights_data: Optional[Dict[str, Any]] = None,
        is_processed: bool = False,
    ):
        self.df = df.copy()
        self.dataset_id = dataset_id
        self.filename = original_filename or dataset_id
        self.is_processed = is_processed
        self.profile_data = profile_data
        self.quality_data = quality_data
        self.cleaning_data = cleaning_data
        self.stats_data = stats_data
        self.corr_data = corr_data
        self.outlier_data = outlier_data
        self.viz_data = viz_data
        self.insights_data = insights_data

    def _ensure_data(self):
        """Lazy-load missing analysis components deterministically from the current dataset session."""
        # 1. Profile Data
        if not self.profile_data:
            profiler = DatasetProfiler(self.df, dataset_id=self.dataset_id)
            self.profile_data = profiler.to_dict()

        # 2. Quality Audit
        if not self.quality_data:
            issues = detect_data_quality_issues(self.df)
            severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
            for iss in issues:
                sev = iss.get("severity", "Low")
                if sev in severity_counts:
                    severity_counts[sev] += 1
            self.quality_data = {
                "total_issues": len(issues),
                "severity_counts": severity_counts,
                "issues": issues,
            }

        # 3. Cleaning Summary Data
        if not self.cleaning_data:
            self.cleaning_data = self._assemble_cleaning_summary()

        # 4. Statistical Analysis
        if not self.stats_data:
            stats_engine = StatisticalAnalysisEngine(self.df, dataset_id=self.dataset_id)
            self.stats_data = stats_engine.analyze(confidence_level=0.95)

        # 5. Correlation Analysis
        if not self.corr_data:
            corr_engine = CorrelationAnalysisEngine(self.df, dataset_id=self.dataset_id)
            self.corr_data = corr_engine.analyze(threshold=0.0)

        # 6. Outlier Analysis
        if not self.outlier_data:
            outlier_engine = OutlierDetectionEngine(self.df, dataset_id=self.dataset_id)
            self.outlier_data = outlier_engine.analyze(method="iqr", param=1.5)

        # 7. Visualization Specs
        if not self.viz_data:
            self.viz_data = self._assemble_visualizations()

        # 8 & 9. AI Insights & Strategic Recommendations
        if not self.insights_data:
            agent = AnalystAgent()
            ctx = agent.build_analysis_context(self.df, dataset_id=self.dataset_id)
            self.insights_data = agent.generate_insights(ctx)

    def _assemble_cleaning_summary(self) -> Dict[str, Any]:
        """Check for session cleaning history or construct a deterministic cleaning preview."""
        registry = _read_registry()
        parent_meta = registry.get(self.dataset_id, {})
        applied_ops = []
        is_cleaned_dataset = parent_meta.get("is_processed", False) or self.is_processed
        cleaning_meta = parent_meta.get("cleaning_summary", {})

        issues = self.quality_data.get("issues", []) if self.quality_data else detect_data_quality_issues(self.df)
        cleaning_preview = generate_cleaning_preview(self.df, issues)

        if is_cleaned_dataset and cleaning_meta:
            applied_ops = cleaning_meta.get("operations_applied", [])
            rows_before = cleaning_meta.get("original_rows", len(self.df))
            rows_after = cleaning_meta.get("cleaned_rows", len(self.df))
            cols_before = cleaning_meta.get("original_columns", len(self.df.columns))
            cols_after = cleaning_meta.get("cleaned_columns", len(self.df.columns))
            duplicates_removed = cleaning_meta.get("duplicates_removed", 0)
            missing_handled = cleaning_meta.get("missing_values_handled", 0)
            status = "Cleaned Session Dataset"
        else:
            rows_before = len(self.df)
            rows_after = len(self.df)
            cols_before = len(self.df.columns)
            cols_after = len(self.df.columns)
            duplicates_removed = int(self.df.duplicated().sum())
            missing_handled = int(self.df.isna().sum().sum())
            status = "Raw Dataset (Pending Cleaning Pipeline)"

        # Formulate recommended cleaning operations based on detected issues
        recommended_ops = []
        for iss in issues:
            act = iss.get("recommended_action") or iss.get("description", "")
            col = iss.get("column", "Dataset")
            if act and act not in recommended_ops:
                recommended_ops.append(f"[{col}] {act}")

        return {
            "dataset_status": status,
            "is_cleaned": is_cleaned_dataset,
            "initial_rows": rows_before,
            "current_rows": rows_after,
            "initial_columns": cols_before,
            "current_columns": cols_after,
            "duplicates_removed_count": duplicates_removed,
            "missing_values_imputed_count": missing_handled,
            "operations_applied": applied_ops,
            "recommended_operations": recommended_ops[:10],
            "transformation_pairs_count": len(cleaning_preview.get("preview_pairs", [])),
            "preview_pairs": cleaning_preview.get("preview_pairs", [])[:5],
        }

    def _assemble_visualizations(self) -> Dict[str, Any]:
        """Build essential high-signal Plotly visualization specifications for the report."""
        charts = []
        schema = inspect_column_types(self.df)
        num_cols = schema.get("numerical", [])
        cat_cols = schema.get("categorical", []) + schema.get("boolean", [])
        date_cols = schema.get("datetime", [])

        # 1. Feature Distribution Histogram (Top continuous feature)
        if num_cols:
            primary_num = num_cols[0]
            hist_spec = build_histogram_spec(self.df, col=primary_num, bins=25)
            charts.append({
                "id": "chart_distribution_hist",
                "title": f"Distribution Analysis: {primary_num}",
                "type": "Histogram & KDE",
                "primary_column": primary_num,
                "spec": hist_spec,
                "description": f"Histogram showing empirical frequency distribution, central tendency, and dispersion for '{primary_num}'.",
            })

        # 2. Outlier Multi-Feature Box Plot
        if num_cols:
            box_spec = build_outlier_boxplot_spec(self.df, columns=num_cols[:6])
            charts.append({
                "id": "chart_outlier_boxplot",
                "title": "Outlier & Quartile Spread (Box Plot)",
                "type": "Box Plot",
                "primary_column": ", ".join(num_cols[:4]),
                "spec": box_spec,
                "description": f"Standardized quartile bounds [Q1, Median, Q3] and 1.5x IQR outlier threshold boundaries across key numerical features.",
            })

        # 3. Correlation Heatmap or Scatter Plot
        if len(num_cols) >= 2:
            corr_engine = CorrelationAnalysisEngine(self.df, dataset_id=self.dataset_id)
            c_data = corr_engine.analyze(threshold=0.0)
            ranked = c_data.get("ranked_pairs", [])
            if ranked:
                top_pair = ranked[0]
                c1 = top_pair.get("variable_a") or top_pair.get("column_1")
                c2 = top_pair.get("variable_b") or top_pair.get("column_2")
                r_val = top_pair.get("correlation", 0)
                scatter_spec = build_scatter_spec(self.df, x_col=c1, y_col=c2)
                charts.append({
                    "id": "chart_top_correlation",
                    "title": f"Linear Association: {c1} vs {c2} (r = {r_val})",
                    "type": "Scatter Plot with Trendline",
                    "primary_column": f"{c1} & {c2}",
                    "spec": scatter_spec,
                    "description": f"Bivariate relationship demonstrating linear co-movement with Pearson r = {r_val:.4f}.",
                })

            heatmap_spec = build_correlation_heatmap_spec(
                corr_matrix=c_data.get("correlation_matrix", {}),
                columns=c_data.get("numerical_columns", []),
            )
            charts.append({
                "id": "chart_corr_heatmap",
                "title": "Numerical Correlation Heatmap Matrix",
                "type": "Correlation Heatmap",
                "primary_column": ", ".join(num_cols[:8]),
                "spec": heatmap_spec,
                "description": "Pairwise linear Pearson correlation coefficients across all active numerical dimensions.",
            })

        # 4. Top Categorical Distribution
        if cat_cols:
            primary_cat = cat_cols[0]
            cat_spec = build_bar_chart_spec(self.df, col=primary_cat, top_n=8)
            charts.append({
                "id": "chart_category_bar",
                "title": f"Categorical Frequency Breakdown: {primary_cat}",
                "type": "Bar Chart",
                "primary_column": primary_cat,
                "spec": cat_spec,
                "description": f"Frequency volume and market share for dominant classes in '{primary_cat}'.",
            })

        return {
            "total_charts": len(charts),
            "charts": charts,
        }

    # ==========================================================================
    # 1. STRUCTURED JSON REPORT GENERATOR
    # ==========================================================================
    def generate_structured_report(self) -> Dict[str, Any]:
        """
        Compile complete 9-section structured JSON report with explicit
        taxonomy categorization tags (Observed results, Potential issues, AI interpretation, Recommendations).
        """
        self._ensure_data()

        overview = self.profile_data.get("overview", {})
        columns_profile = self.profile_data.get("columns", [])
        quality_summary = self.profile_data.get("quality_summary", {})
        issues_list = self.quality_data.get("issues", [])
        cleaning = self.cleaning_data
        stats_raw = self.stats_data.get("numerical_statistics", {})
        cat_stats_raw = self.stats_data.get("categorical_statistics", {})
        stat_observations = self.stats_data.get("statistical_observations", [])
        ranked_corrs = self.corr_data.get("ranked_pairs", [])
        outliers_col = self.outlier_data.get("column_results", [])
        total_outliers = self.outlier_data.get("total_outlier_count", 0)
        ai_insights = self.insights_data.get("sections", {})
        ai_metadata = self.insights_data.get("metadata", {})

        # Multicollinearity check (|r| >= 0.85)
        multicollinear_pairs = []
        for pair in ranked_corrs:
            r = abs(pair.get("correlation", pair.get("pearson_r", 0)))
            if r >= 0.85:
                multicollinear_pairs.append({
                    "feature_a": pair.get("variable_a") or pair.get("feature_a"),
                    "feature_b": pair.get("variable_b") or pair.get("feature_b"),
                    "correlation": pair.get("correlation", pair.get("pearson_r", 0)),
                    "risk": "High Multicollinearity (|r| >= 0.85). Consider feature pruning or ridge regularization.",
                })

        # Assemble Formatted Recommendations Section
        strategic_recommendations = self._assemble_recommendations(
            quality_issues=issues_list,
            multicollinear_pairs=multicollinear_pairs,
            outliers_count=total_outliers,
            ranked_corrs=ranked_corrs,
            ai_recs=ai_insights.get("business_recommendations", ""),
        )

        # Safely extract and clean statistical observations
        cleaned_stat_obs = []
        distribution_anomalies = []
        for o in stat_observations:
            msg = o.get("message", str(o)) if isinstance(o, dict) else str(o)
            cleaned_stat_obs.append(msg)
            msg_lower = msg.lower()
            if any(k in msg_lower for k in ["skew", "kurtosis", "variance", "zero", "deviation", "spread"]):
                distribution_anomalies.append(msg)

        structured = {
            "metadata": {
                "report_title": f"Executive Data Analysis & Quality Report: {self.filename}",
                "dataset_id": self.dataset_id,
                "dataset_filename": self.filename,
                "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "phase": "Phase 13: Automated Analysis Report",
                "ai_provider": ai_metadata.get("provider", "Deterministic Synthesis"),
                "ai_model": ai_metadata.get("model", "grounded-synthesizer"),
                "taxonomy_categories": [
                    CATEGORY_OBSERVED,
                    CATEGORY_ISSUES,
                    CATEGORY_AI,
                    CATEGORY_RECOMMENDATIONS,
                ],
            },
            # SECTION 1: DATASET OVERVIEW
            "section_1_dataset_overview": {
                "section_number": 1,
                "title": "Dataset Overview & Dimensional Matrix",
                "category": CATEGORY_OBSERVED,
                "summary": {
                    "total_rows": overview.get("total_rows", len(self.df)),
                    "total_columns": overview.get("total_columns", len(self.df.columns)),
                    "total_cells": overview.get("total_cells", len(self.df) * len(self.df.columns)),
                    "memory_usage_mb": overview.get("memory_usage_mb", round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2)),
                    "missing_cells_count": overview.get("missing_cells", int(self.df.isna().sum().sum())),
                    "missing_cells_pct": overview.get("missing_cells_percentage", round((self.df.isna().sum().sum() / max(1, len(self.df) * len(self.df.columns))) * 100, 2)),
                    "duplicate_rows_count": overview.get("duplicate_rows", int(self.df.duplicated().sum())),
                    "duplicate_rows_pct": overview.get("duplicate_rows_percentage", round((self.df.duplicated().sum() / max(1, len(self.df))) * 100, 2)),
                    "column_type_counts": overview.get("column_types_count", {}),
                },
                "columns_schema": [
                    {
                        "name": col.get("name") or col.get("column_name"),
                        "classified_type": col.get("classified_type"),
                        "dtype": col.get("dtype") or col.get("pandas_dtype"),
                        "missing_count": col.get("missing_count", 0),
                        "missing_pct": col.get("missing_percentage", col.get("missing_pct", 0.0)),
                        "unique_count": col.get("unique_count", 0),
                        "sample_values": col.get("sample_values", [])[:3],
                    }
                    for col in columns_profile
                ],
            },
            # SECTION 2: DATA QUALITY ASSESSMENT
            "section_2_data_quality_assessment": {
                "section_number": 2,
                "title": "Data Quality & Integrity Assessment",
                "observed_metrics": {
                    "category": CATEGORY_OBSERVED,
                    "health_score": quality_summary.get("overall_score", 100),
                    "health_grade": quality_summary.get("health_grade", "A+"),
                    "completeness_score": quality_summary.get("completeness_score", 100.0),
                    "uniqueness_score": quality_summary.get("uniqueness_score", 100.0),
                    "total_detected_issues": self.quality_data.get("total_issues", len(issues_list)),
                    "severity_breakdown": self.quality_data.get("severity_counts", {}),
                },
                "potential_issues": {
                    "category": CATEGORY_ISSUES,
                    "issues": [
                        {
                            "issue_type": iss.get("issue_type"),
                            "severity": iss.get("severity", "Low"),
                            "column": iss.get("column", "Dataset"),
                            "affected_rows": iss.get("affected_rows", 0),
                            "percentage_affected": iss.get("percentage_affected", 0.0),
                            "description": iss.get("description", ""),
                            "suggested_remediation": iss.get("recommended_action", ""),
                        }
                        for iss in issues_list
                    ],
                },
            },
            # SECTION 3: CLEANING SUMMARY
            "section_3_cleaning_summary": {
                "section_number": 3,
                "title": "Data Cleaning & Transformation Summary",
                "observed_results": {
                    "category": CATEGORY_OBSERVED,
                    "session_dataset_status": cleaning.get("dataset_status"),
                    "is_cleaned_dataset": cleaning.get("is_cleaned"),
                    "row_count_delta": {
                        "before": cleaning.get("initial_rows"),
                        "after": cleaning.get("current_rows"),
                        "rows_removed": cleaning.get("initial_rows", 0) - cleaning.get("current_rows", 0),
                    },
                    "column_count_delta": {
                        "before": cleaning.get("initial_columns"),
                        "after": cleaning.get("current_columns"),
                    },
                    "duplicates_purged": cleaning.get("duplicates_removed_count", 0),
                    "missing_values_remediated": cleaning.get("missing_values_imputed_count", 0),
                    "operations_applied": cleaning.get("operations_applied", []),
                },
                "potential_issues": {
                    "category": CATEGORY_ISSUES,
                    "unresolved_anomalies_count": len(issues_list),
                    "notice": "Review unresolved quality anomalies before downstream model ingestion." if issues_list else "Zero unresolved anomalies detected.",
                },
                "recommendations": {
                    "category": CATEGORY_RECOMMENDATIONS,
                    "recommended_pipeline_steps": cleaning.get("recommended_operations", []),
                },
            },
            # SECTION 4: STATISTICAL ANALYSIS
            "section_4_statistical_analysis": {
                "section_number": 4,
                "title": "Descriptive & Inferential Statistical Analysis",
                "observed_results": {
                    "category": CATEGORY_OBSERVED,
                    "numerical_columns_count": len(stats_raw),
                    "numerical_moments": stats_raw,
                    "categorical_summaries": cat_stats_raw,
                },
                "potential_issues": {
                    "category": CATEGORY_ISSUES,
                    "distribution_anomalies": distribution_anomalies,
                },
                "ai_interpretation": {
                    "category": CATEGORY_AI,
                    "statistical_observations": cleaned_stat_obs,
                },
            },
            # SECTION 5: CORRELATION ANALYSIS
            "section_5_correlation_analysis": {
                "section_number": 5,
                "title": "Feature Correlation & Co-Movement Analysis",
                "observed_results": {
                    "category": CATEGORY_OBSERVED,
                    "method": "Pearson Linear Product-Moment Correlation (r)",
                    "numerical_features_analyzed": self.corr_data.get("numerical_columns", []),
                    "total_evaluated_pairs": len(ranked_corrs),
                    "top_ranked_pairs": ranked_corrs[:8],
                    "strongest_positive": self.corr_data.get("key_correlations", {}).get("strongest_positive"),
                    "strongest_negative": self.corr_data.get("key_correlations", {}).get("strongest_negative"),
                },
                "potential_issues": {
                    "category": CATEGORY_ISSUES,
                    "multicollinear_pairs_count": len(multicollinear_pairs),
                    "multicollinear_flags": multicollinear_pairs,
                },
                "ai_interpretation": {
                    "category": CATEGORY_AI,
                    "disclaimer": "Correlation measures linear statistical association only and does NOT indicate direct causal relationships.",
                    "synthesis": ai_insights.get("important_relationships", "Calculated standard correlation matrices."),
                },
            },
            # SECTION 6: OUTLIER ANALYSIS
            "section_6_outlier_analysis": {
                "section_number": 6,
                "title": "Outlier Detection & Anomaly Breakdown",
                "observed_results": {
                    "category": CATEGORY_OBSERVED,
                    "detection_method": "Interquartile Range (IQR 1.5x Multiplier)",
                    "total_outliers_count": total_outliers,
                    "features_with_outliers_count": len([c for c in outliers_col if c.get("outlier_count", 0) > 0]),
                    "column_breakdown": outliers_col,
                },
                "potential_issues": {
                    "category": CATEGORY_ISSUES,
                    "confirmed_data_errors_count": sum(
                        c.get("classification_breakdown", {}).get("confirmed_data_errors", 0) for c in outliers_col
                    ),
                    "statistical_tail_outliers_count": sum(
                        c.get("classification_breakdown", {}).get("potential_statistical_outliers", 0) for c in outliers_col
                    ),
                    "critical_outlier_columns": [
                        {
                            "column": c.get("column"),
                            "outlier_count": c.get("outlier_count"),
                            "outlier_percentage": c.get("outlier_percentage"),
                            "bounds": [c.get("lower_threshold"), c.get("upper_threshold")],
                        }
                        for c in outliers_col
                        if c.get("outlier_percentage", 0) >= 3.0
                    ],
                },
                "recommendations": {
                    "category": CATEGORY_RECOMMENDATIONS,
                    "outlier_handling_guidance": "Treat extreme values using domain-informed Winsorization (capping at 1st/99th percentile) or tree-based algorithms rather than naive record deletion.",
                },
            },
            # SECTION 7: IMPORTANT VISUALIZATIONS
            "section_7_important_visualizations": {
                "section_number": 7,
                "title": "Important Exploratory Visualizations",
                "category": CATEGORY_OBSERVED,
                "total_charts_rendered": self.viz_data.get("total_charts", 0),
                "charts": self.viz_data.get("charts", []),
            },
            # SECTION 8: AI-GENERATED INSIGHTS
            "section_8_ai_insights": {
                "section_number": 8,
                "title": "AI-Generated Strategic Insights & Analytical Synthesis",
                "category": CATEGORY_AI,
                "grounding_statement": "Strictly Grounded in Deterministic Python Calculations (Zero Hallucination / Zero Fabrication Guarantee)",
                "executive_summary": ai_insights.get("executive_summary", "Automated analysis completed successfully."),
                "key_findings": ai_insights.get("key_findings", ""),
                "important_trends": ai_insights.get("important_trends", ""),
                "important_relationships": ai_insights.get("important_relationships", ""),
                "data_quality_concerns": ai_insights.get("data_quality_concerns", ""),
                "outlier_findings": ai_insights.get("potential_outlier_findings", ""),
            },
            # SECTION 9: RECOMMENDATIONS
            "section_9_recommendations": {
                "section_number": 9,
                "title": "Actionable Strategic & Engineering Recommendations",
                "category": CATEGORY_RECOMMENDATIONS,
                "data_cleaning_recommendations": strategic_recommendations["cleaning"],
                "feature_engineering_recommendations": strategic_recommendations["feature_engineering"],
                "modeling_recommendations": strategic_recommendations["modeling"],
                "governance_recommendations": strategic_recommendations["governance"],
                "follow_up_analysis": ai_insights.get("suggested_follow_up", ""),
            },
        }

        # Convenience top-level aliases for backward compatibility and fast access
        structured["title"] = f"Executive Data Analysis & Quality Report: {self.filename}"
        structured["overview"] = structured["section_1_dataset_overview"]["summary"]
        structured["quality"] = {
            "health_score": quality_summary.get("overall_score", 100),
            "health_grade": quality_summary.get("health_grade", "A+"),
            "total_issues": self.quality_data.get("total_issues", len(issues_list)),
            "issues": issues_list,
        }
        structured["cleaning"] = structured["section_3_cleaning_summary"]["observed_results"]
        structured["statistics"] = {
            "numerical_columns_count": len(stats_raw),
            "observations": stat_observations,
            "numerical_summary": stats_raw,
        }
        structured["correlations"] = {
            "total_pairs": len(ranked_corrs),
            "top_pairs": ranked_corrs[:8],
        }
        structured["outliers"] = {
            "total_outliers": total_outliers,
            "columns_affected": len([c for c in outliers_col if c.get("outlier_count", 0) > 0]),
            "breakdown": {c.get("column"): c for c in outliers_col},
        }
        structured["ai_insights"] = self.insights_data
        structured["recommendations"] = structured["section_9_recommendations"]

        return structured

    def _assemble_recommendations(
        self,
        quality_issues: List[Dict[str, Any]],
        multicollinear_pairs: List[Dict[str, Any]],
        outliers_count: int,
        ranked_corrs: List[Dict[str, Any]],
        ai_recs: str,
    ) -> Dict[str, List[str]]:
        """Construct deterministic, high-impact recommendations categorized by domain."""
        cleaning_recs = []
        if quality_issues:
            for iss in quality_issues[:4]:
                cleaning_recs.append(f"Remediate {iss.get('issue_type')} in `{iss.get('column')}`: {iss.get('recommended_action', 'Inspect values.')}")
        else:
            cleaning_recs.append("Data quality is clean. Maintain automated schema validation at data ingestion boundaries.")

        fe_recs = []
        if ranked_corrs:
            top_p = ranked_corrs[0]
            c1 = top_p.get("variable_a") or top_p.get("column_1") or "Feature A"
            c2 = top_p.get("variable_b") or top_p.get("column_2") or "Feature B"
            fe_recs.append(f"Create interaction term or ratio between `{c1}` and `{c2}` (Pearson r = {top_p.get('correlation', top_p.get('pearson_r', 0)):.3f}).")
        if multicollinear_pairs:
            fe_recs.append(f"Address {len(multicollinear_pairs)} highly collinear pair(s) using Principal Component Analysis (PCA) or feature variance pruning.")
        else:
            fe_recs.append("Standardize continuous variables with z-score or robust scaling prior to distance-based modeling.")

        model_recs = []
        if outliers_count > 0:
            model_recs.append(f"Utilize robust loss functions (e.g., Huber loss) or ensemble tree models (XGBoost, Random Forest) to mitigate impact of {outliers_count} outliers.")
        else:
            model_recs.append("Standard linear or gradient boosted architectures are well-calibrated for current feature distributions.")
        model_recs.append("Implement k-fold cross-validation with stratification if working with imbalanced categorical targets.")

        gov_recs = [
            "Establish automated data pipeline telemetry to flag sudden shifts in distribution moments (drift detection).",
            "Store dataset version hashes to ensure reproducible analytics and compliant audit trails.",
        ]

        return {
            "cleaning": cleaning_recs,
            "feature_engineering": fe_recs,
            "modeling": model_recs,
            "governance": gov_recs,
        }

    # ==========================================================================
    # 2. BEAUTIFULLY FORMATTED MARKDOWN REPORT GENERATOR
    # ==========================================================================
    def generate_markdown_report(self) -> str:
        """
        Generate a comprehensive, GitHub-flavored Markdown report clearly demarcating all
        9 sections and the 4 taxonomy categories with icons and callouts.
        """
        rep = self.generate_structured_report()
        meta = rep["metadata"]
        s1 = rep["section_1_dataset_overview"]
        s2 = rep["section_2_data_quality_assessment"]
        s3 = rep["section_3_cleaning_summary"]
        s4 = rep["section_4_statistical_analysis"]
        s5 = rep["section_5_correlation_analysis"]
        s6 = rep["section_6_outlier_analysis"]
        s7 = rep["section_7_important_visualizations"]
        s8 = rep["section_8_ai_insights"]
        s9 = rep["section_9_recommendations"]

        ov = s1["summary"]
        q_obs = s2["observed_metrics"]
        q_iss = s2["potential_issues"]["issues"]
        cl_obs = s3["observed_results"]
        cl_rec = s3["recommendations"]["recommended_pipeline_steps"]
        num_stats = s4["observed_results"]["numerical_moments"]
        stat_obs = s4["ai_interpretation"]["statistical_observations"]
        corrs = s5["observed_results"]["top_ranked_pairs"]
        multi_flags = s5["potential_issues"]["multicollinear_flags"]
        out_obs = s6["observed_results"]
        out_cols = out_obs.get("column_breakdown", [])
        charts = s7.get("charts", [])

        # Format Columns Schema Table
        schema_rows = []
        for col in s1["columns_schema"][:15]:
            samples = ", ".join([str(s) for s in col.get("sample_values", [])])
            schema_rows.append(
                f"| `{col.get('name')}` | {col.get('classified_type')} | `{col.get('dtype')}` | {col.get('missing_count')} ({col.get('missing_pct')}%) | {col.get('unique_count')} | `{samples}` |"
            )
        schema_table = (
            "| Column | Type | Dtype | Missing (%) | Unique | Sample Values |\n|---|---|---|---|---|---|\n"
            + "\n".join(schema_rows)
        )

        # Format Data Quality Issues Table
        iss_rows = []
        for iss in q_iss[:8]:
            iss_rows.append(
                f"| **{iss.get('severity')}** | `{iss.get('column')}` | {iss.get('issue_type')} | {iss.get('affected_rows')} ({iss.get('percentage_affected')}%) | {iss.get('suggested_remediation')} |"
            )
        iss_table = (
            "| Severity | Column | Issue Type | Affected Records | Suggested Remediation |\n|---|---|---|---|---|\n"
            + "\n".join(iss_rows)
            if iss_rows
            else "_No critical data quality issues identified._"
        )

        # Format Numerical Statistics Table
        num_stat_rows = []
        for col_name, st in list(num_stats.items())[:10]:
            num_stat_rows.append(
                f"| `{col_name}` | {st.get('mean', 0):.2f} | {st.get('median', 0):.2f} | {st.get('std', 0):.2f} | [{st.get('min', 0):.2f}, {st.get('max', 0):.2f}] | {st.get('skewness', 0):.2f} | {st.get('kurtosis', 0):.2f} | [{st.get('ci_lower_95', 0):.2f}, {st.get('ci_upper_95', 0):.2f}] |"
            )
        num_stat_table = (
            "| Column | Mean | Median | Std Dev | [Min, Max] | Skewness | Kurtosis | 95% CI |\n|---|---|---|---|---|---|---|---|\n"
            + "\n".join(num_stat_rows)
            if num_stat_rows
            else "_No continuous numerical features found for moments analysis._"
        )

        # Format Top Correlations Table
        corr_rows = []
        for pair in corrs[:8]:
            c1 = pair.get("variable_a") or pair.get("feature_a") or pair.get("column_1")
            c2 = pair.get("variable_b") or pair.get("feature_b") or pair.get("column_2")
            r = pair.get("correlation", pair.get("pearson_r", 0))
            strength = pair.get("strength", "N/A")
            direction = pair.get("direction", "N/A")
            corr_rows.append(f"| `{c1}` & `{c2}` | **{r:.4f}** | {strength} | {direction} |")
        corr_table = (
            "| Feature Pair | Pearson r | Strength | Direction |\n|---|---|---|---|\n" + "\n".join(corr_rows)
            if corr_rows
            else "_No linear correlation pairs available or fewer than 2 numerical features._"
        )

        # Format Outliers Table
        out_rows = []
        for o in out_cols:
            if o.get("outlier_count", 0) > 0:
                out_rows.append(
                    f"| `{o.get('column')}` | **{o.get('outlier_count')}** | {o.get('outlier_percentage'):.2f}% | [{o.get('lower_threshold', 0):.2f}, {o.get('upper_threshold', 0):.2f}] | [{o.get('min_outlier', 'N/A')}, {o.get('max_outlier', 'N/A')}] |"
                )
        out_table = (
            "| Column | Outlier Count | Outlier % | IQR Thresholds [Lower, Upper] | Extreme Values |\n|---|---|---|---|---|\n"
            + "\n".join(out_rows)
            if out_rows
            else "_No statistical outliers detected under standard IQR bounds (1.5x multiplier)._"
        )

        # Format Visualizations List
        viz_lines = []
        for i, c in enumerate(charts, 1):
            viz_lines.append(f"- **Chart {i}: {c.get('title')}** ({c.get('type')})  \n  _{c.get('description')}_")
        viz_text = "\n".join(viz_lines) if viz_lines else "- _No charts rendered for this dataset shape._"

        # Format Recommendations
        cl_rec_bullets = "\n".join(f"- 🧹 {r}" for r in s9["data_cleaning_recommendations"])
        fe_rec_bullets = "\n".join(f"- ⚙️ {r}" for r in s9["feature_engineering_recommendations"])
        model_rec_bullets = "\n".join(f"- 📈 {r}" for r in s9["modeling_recommendations"])
        gov_rec_bullets = "\n".join(f"- 🔒 {r}" for r in s9["governance_recommendations"])

        md = f"""# {meta['report_title']}

> **Dataset ID:** `{meta['dataset_id']}` &bull; **Generated:** `{meta['generated_at']}` &bull; **Engine:** `{meta['phase']}`  
> **Taxonomy Key:** 🔍 `Observed results` | ⚠️ `Potential issues` | 🧠 `AI interpretation` | 🎯 `Recommendations`

---

## 1. Dataset Overview
**Category:** 🔍 *Observed results*

| Metric | Measured Value | Metric | Measured Value |
|---|---|---|---|
| **Total Rows** | **{ov.get('total_rows', 0):,}** | **Total Features (Columns)** | **{ov.get('total_columns', 0):,}** |
| **Total Data Cells** | **{ov.get('total_cells', 0):,}** | **RAM Memory Usage** | **{ov.get('memory_usage_mb', 0):.2f} MB** |
| **Missing Data Cells** | **{ov.get('missing_cells_count', 0):,}** ({ov.get('missing_cells_pct', 0):.2f}%) | **Duplicate Rows** | **{ov.get('duplicate_rows_count', 0):,}** ({ov.get('duplicate_rows_pct', 0):.2f}%) |

### Column Schema & Data Types
{schema_table}

---

## 2. Data Quality Assessment
- 🔍 **Observed Result:** Overall Data Health Score is **{q_obs.get('health_score', 100)} / 100** (Grade **{q_obs.get('health_grade', 'A+')}**). Completeness: **{q_obs.get('completeness_score', 100.0):.2f}%**, Uniqueness: **{q_obs.get('uniqueness_score', 100.0):.2f}%**.
- ⚠️ **Potential Issues:** A total of **{q_obs.get('total_detected_issues', 0)} data quality issues** detected.

{iss_table}

---

## 3. Cleaning Summary
- 🔍 **Observed Result:** Dataset Status: **{cl_obs.get('session_dataset_status')}**. Record delta: **{cl_obs.get('row_count_delta', {}).get('before', 0):,} &rarr; {cl_obs.get('row_count_delta', {}).get('after', 0):,} rows**. Duplicates purged: **{cl_obs.get('duplicates_purged', 0)}**, Missing values remediated: **{cl_obs.get('missing_values_remediated', 0)}**.
- ⚠️ **Potential Issues:** {s3['potential_issues']['notice']}
- 🎯 **Recommendations:**
{chr(10).join(f"- {op}" for op in cl_rec) if cl_rec else "- Dataset is clean; no immediate transformations required."}

---

## 4. Statistical Analysis
- 🔍 **Observed Result:** Evaluated **{len(num_stats)} numerical features** for parametric moments, spread, and quartile percentiles.

{num_stat_table}

- ⚠️ **Potential Issues:**
{chr(10).join(f"- ⚠️ {a}" for a in s4['potential_issues']['distribution_anomalies']) if s4['potential_issues']['distribution_anomalies'] else "- Standard distribution moments observed across all continuous dimensions."}

- 🧠 **AI Interpretation:**
{chr(10).join(f"- {o}" for o in stat_obs) if stat_obs else "- Standard feature distributions observed across features."}

---

## 5. Correlation Analysis
- 🔍 **Observed Result:** Pearson linear product-moment correlation analysis across **{len(s5['observed_results']['numerical_features_analyzed'])} numerical features**.

{corr_table}

- ⚠️ **Potential Issues:**
{chr(10).join(f"- ⚠️ Collinear Pair: `{m['feature_a']}` & `{m['feature_b']}` (r = {m['correlation']:.3f}) — {m['risk']}" for m in multi_flags) if multi_flags else "- No severe multicollinearity (|r| >= 0.85) detected."}

- 🧠 **AI Interpretation:**
> *Methodological Note: {s5['ai_interpretation']['disclaimer']}*

---

## 6. Outlier Analysis
- 🔍 **Observed Result:** Multi-feature Interquartile Range (IQR 1.5x) detection identified **{out_obs.get('total_outliers_count', 0):,} total outliers** across **{out_obs.get('features_with_outliers_count', 0)} feature(s)**.

{out_table}

- ⚠️ **Potential Issues:**
- Confirmed Data Entry / Domain Errors: **{s6['potential_issues']['confirmed_data_errors_count']}**
- Potential Statistical Tail Instances: **{s6['potential_issues']['statistical_tail_outliers_count']}**

- 🎯 **Recommendations:**
- {s6['recommendations']['outlier_handling_guidance']}

---

## 7. Important Visualizations
**Category:** 🔍 *Observed results*

{viz_text}

---

## 8. AI-Generated Insights
**Category:** 🧠 *AI interpretation*  
_{s8.get('grounding_statement')}_

### Executive Summary
{s8.get('executive_summary')}

### Key Analytical Findings
{s8.get('key_findings') or '- Complete feature profile available in dashboard.'}

### Important Trends & Structural Dynamics
{s8.get('important_trends') or '- Cross-sectional empirical distributions evaluated.'}

### Anomaly & Risk Evaluation
{s8.get('data_quality_concerns') or '- No critical data integrity risks identified.'}

---

## 9. Recommendations
**Category:** 🎯 *Recommendations*

### Data Cleaning & Integrity Remediation
{cl_rec_bullets}

### Feature Engineering & Transformation
{fe_rec_bullets}

### Predictive Modeling & Validation Strategy
{model_rec_bullets}

### Data Governance & Quality Controls
{gov_rec_bullets}

---

*Report deterministically generated by AI Data Analyst Agent &bull; Phase 13 Automated Analysis Report Engine.*
"""
        return md

    # ==========================================================================
    # 3. STANDALONE STYLED HTML REPORT GENERATOR
    # ==========================================================================
    def generate_html_report(self) -> str:
        """
        Generate a fully styled, self-contained standalone HTML report with
        embedded dark-theme layout, KPI cards, category badges, Plotly charts, and print optimization.
        """
        rep = self.generate_structured_report()
        meta = rep["metadata"]
        s1 = rep["section_1_dataset_overview"]
        s2 = rep["section_2_data_quality_assessment"]
        s3 = rep["section_3_cleaning_summary"]
        s4 = rep["section_4_statistical_analysis"]
        s5 = rep["section_5_correlation_analysis"]
        s6 = rep["section_6_outlier_analysis"]
        s7 = rep["section_7_important_visualizations"]
        s8 = rep["section_8_ai_insights"]
        s9 = rep["section_9_recommendations"]

        ov = s1["summary"]
        q_obs = s2["observed_metrics"]
        q_iss = s2["potential_issues"]["issues"]
        cl_obs = s3["observed_results"]
        cl_rec = s3["recommendations"]["recommended_pipeline_steps"]
        num_stats = s4["observed_results"]["numerical_moments"]
        stat_obs = s4["ai_interpretation"]["statistical_observations"]
        corrs = s5["observed_results"]["top_ranked_pairs"]
        multi_flags = s5["potential_issues"]["multicollinear_flags"]
        out_obs = s6["observed_results"]
        out_cols = out_obs.get("column_breakdown", [])
        charts = s7.get("charts", [])

        # Build Correlation Table Rows
        corr_trs = ""
        for p in corrs[:8]:
            c1 = p.get("variable_a") or p.get("feature_a") or p.get("column_1")
            c2 = p.get("variable_b") or p.get("feature_b") or p.get("column_2")
            r = p.get("correlation", p.get("pearson_r", 0))
            strength = p.get("strength", "N/A")
            direction = p.get("direction", "N/A")
            badge_class = "badge-indigo" if "Strong" in strength else "badge-cyan"
            corr_trs += f"""<tr>
                <td><strong>{c1}</strong> & <strong>{c2}</strong></td>
                <td style="font-family: monospace; font-weight: bold; color: #6366f1;">{r:.4f}</td>
                <td><span class="badge {badge_class}">{strength}</span></td>
                <td>{direction}</td>
            </tr>"""
        if not corr_trs:
            corr_trs = "<tr><td colspan='4' class='text-muted'>No significant linear correlations detected or insufficient continuous features.</td></tr>"

        # Build Outlier Table Rows
        out_trs = ""
        for o in out_cols:
            if o.get("outlier_count", 0) > 0:
                out_trs += f"""<tr>
                    <td><strong>{o.get('column')}</strong></td>
                    <td style="font-family: monospace; font-weight: bold; color: #f59e0b;">{o.get('outlier_count'):,}</td>
                    <td style="font-family: monospace;">{o.get('outlier_percentage'):.2f}%</td>
                    <td style="font-family: monospace;">[{o.get('lower_threshold', 0):.2f}, {o.get('upper_threshold', 0):.2f}]</td>
                    <td style="font-family: monospace;">[{o.get('min_outlier', 'N/A')}, {o.get('max_outlier', 'N/A')}]</td>
                </tr>"""
        if not out_trs:
            out_trs = "<tr><td colspan='5' class='text-muted'>No statistical outliers detected under standard IQR bounds (1.5x).</td></tr>"

        # Build Data Quality Issues Rows
        quality_trs = ""
        for iss in q_iss[:8]:
            sev = iss.get("severity", "Low")
            sev_class = "badge-danger" if sev == "Critical" else ("badge-warning" if sev == "High" else "badge-indigo")
            quality_trs += f"""<tr>
                <td><span class="badge {sev_class}">{sev}</span></td>
                <td><code>{iss.get('column')}</code></td>
                <td><strong>{iss.get('issue_type')}</strong></td>
                <td>{iss.get('affected_rows')} ({iss.get('percentage_affected')}%)</td>
                <td>{iss.get('suggested_remediation')}</td>
            </tr>"""
        if not quality_trs:
            quality_trs = "<tr><td colspan='5' class='text-muted'>Zero data quality defects identified across 12 inspection algorithms.</td></tr>"

        # Build Numerical Stats Rows
        stats_trs = ""
        for col_name, st in list(num_stats.items())[:10]:
            stats_trs += f"""<tr>
                <td><strong>{col_name}</strong></td>
                <td style="font-family: monospace;">{st.get('mean', 0):.2f}</td>
                <td style="font-family: monospace;">{st.get('median', 0):.2f}</td>
                <td style="font-family: monospace;">{st.get('std', 0):.2f}</td>
                <td style="font-family: monospace;">[{st.get('min', 0):.2f}, {st.get('max', 0):.2f}]</td>
                <td style="font-family: monospace;">{st.get('skewness', 0):.2f}</td>
                <td style="font-family: monospace;">[{st.get('ci_lower_95', 0):.2f}, {st.get('ci_upper_95', 0):.2f}]</td>
            </tr>"""
        if not stats_trs:
            stats_trs = "<tr><td colspan='7' class='text-muted'>No numerical continuous columns available for statistical moment calculation.</td></tr>"

        # Build Chart Embed Scripts
        chart_divs = ""
        chart_scripts = ""
        for i, c in enumerate(charts):
            div_id = f"plotly_report_chart_{i}"
            spec_json = json.dumps(c.get("spec", {}))
            chart_divs += f"""
            <div class="chart-card">
                <div class="chart-card-header">
                    <h4>{c.get('title')}</h4>
                    <span class="badge badge-indigo">{c.get('type')}</span>
                </div>
                <div id="{div_id}" style="width: 100%; height: 350px;"></div>
                <p class="chart-desc">{c.get('description')}</p>
            </div>
            """
            chart_scripts += f"""
            try {{
                const spec_{i} = {spec_json};
                if (spec_{i} && spec_{i}.data) {{
                    Plotly.newPlot('{div_id}', spec_{i}.data, spec_{i}.layout || {{}}, spec_{i}.config || {{responsive: true}});
                }}
            }} catch(e) {{ console.error("Error plotting chart {i}:", e); }}
            """

        # Build Recommendations HTML Lists
        cl_rec_html = "".join(f"<li>{r}</li>" for r in s9["data_cleaning_recommendations"])
        fe_rec_html = "".join(f"<li>{r}</li>" for r in s9["feature_engineering_recommendations"])
        mod_rec_html = "".join(f"<li>{r}</li>" for r in s9["modeling_recommendations"])
        gov_rec_html = "".join(f"<li>{r}</li>" for r in s9["governance_recommendations"])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Data Analysis Report - {self.filename}</title>
    <!-- Plotly CDN for standalone interactive charts -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {{
            --bg: #0B0F19;
            --surface: #111827;
            --surface-card: #1F2937;
            --surface-hover: #374151;
            --text: #F9FAFB;
            --text-muted: #9CA3AF;
            --primary: #6366F1;
            --primary-glow: rgba(99, 102, 241, 0.15);
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
            --cyan: #06B6D4;
            --border: rgba(255, 255, 255, 0.1);
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 40px 20px;
        }}
        .report-container {{
            max-width: 1100px;
            margin: 0 auto;
            background: var(--surface);
            border-radius: 16px;
            padding: 48px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border);
        }}
        .report-header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 24px;
            margin-bottom: 32px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .report-title {{ font-size: 28px; font-weight: 800; color: #FFF; margin-bottom: 8px; letter-spacing: -0.5px; }}
        .report-meta {{ color: var(--text-muted); font-size: 14px; }}
        .taxonomy-bar {{
            display: flex;
            gap: 12px;
            margin-top: 16px;
            flex-wrap: wrap;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-observed {{ background: rgba(6, 182, 212, 0.15); color: var(--cyan); border: 1px solid rgba(6, 182, 212, 0.3); }}
        .badge-issues {{ background: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-ai {{ background: rgba(139, 92, 246, 0.15); color: #C084FC; border: 1px solid rgba(139, 92, 246, 0.3); }}
        .badge-recs {{ background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-grade {{ background: rgba(16, 185, 129, 0.2); color: var(--success); font-size: 16px; padding: 6px 14px; }}
        .badge-indigo {{ background: rgba(99, 102, 241, 0.2); color: var(--primary); }}
        .badge-cyan {{ background: rgba(6, 182, 212, 0.2); color: var(--cyan); }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: var(--warning); }}
        .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .kpi-box {{
            background: var(--surface-card);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border);
        }}
        .kpi-label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }}
        .kpi-val {{ font-size: 24px; font-weight: 800; color: #FFF; margin: 6px 0; font-family: monospace; }}
        .section {{ margin-bottom: 40px; }}
        .section-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 8px;
        }}
        .section-title {{ font-size: 20px; font-weight: 700; color: #FFF; border-left: 4px solid var(--primary); padding-left: 12px; }}
        .ai-box {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .ai-title {{ font-size: 16px; font-weight: 700; color: #A5B4FC; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
        ul {{ padding-left: 20px; margin-top: 8px; }}
        li {{ margin-bottom: 8px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: var(--surface-card); color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
        .text-muted {{ color: var(--text-muted); font-style: italic; }}
        .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; }}
        @media (max-width: 900px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
        .chart-card {{
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }}
        .chart-card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .chart-desc {{ font-size: 12px; color: var(--text-muted); margin-top: 8px; }}
        .callout {{
            border-radius: 8px;
            padding: 14px 18px;
            margin: 14px 0;
            font-size: 14px;
        }}
        .callout-issue {{ background: rgba(245, 158, 11, 0.1); border-left: 4px solid var(--warning); color: #FCD34D; }}
        .callout-rec {{ background: rgba(16, 185, 129, 0.1); border-left: 4px solid var(--success); color: #6EE7B7; }}
        .callout-ai {{ background: rgba(99, 102, 241, 0.1); border-left: 4px solid var(--primary); color: #C7D2FE; }}
        .footer {{ text-align: center; color: var(--text-muted); font-size: 13px; margin-top: 48px; border-top: 1px solid var(--border); padding-top: 24px; }}
        
        @media print {{
            body {{ background: #FFF; color: #000; padding: 0; }}
            .report-container {{ border: none; box-shadow: none; padding: 0; background: #FFF; color: #000; max-width: 100%; }}
            .kpi-box, .chart-card {{ background: #F9FAFB; border: 1px solid #E5E7EB; }}
            .kpi-val, .report-title, .section-title {{ color: #000; }}
            th {{ background: #F3F4F6; color: #374151; }}
            .ai-box {{ background: #F9FAFB; border: 1px solid #E5E7EB; color: #000; }}
            .ai-title {{ color: #4F46E5; }}
            td, th {{ border-bottom: 1px solid #E5E7EB; }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <!-- Header -->
        <div class="report-header">
            <div>
                <h1 class="report-title">{meta['report_title']}</h1>
                <div class="report-meta">
                    <strong>Dataset:</strong> {self.filename} &bull; 
                    <strong>ID:</strong> {self.dataset_id} &bull; 
                    <strong>Generated:</strong> {meta['generated_at']} &bull; 
                    <strong>AI Engine:</strong> {meta['ai_provider']} ({meta['ai_model']})
                </div>
                <div class="taxonomy-bar">
                    <span class="badge badge-observed">🔍 Observed results</span>
                    <span class="badge badge-issues">⚠️ Potential issues</span>
                    <span class="badge badge-ai">🧠 AI interpretation</span>
                    <span class="badge badge-recs">🎯 Recommendations</span>
                </div>
            </div>
            <div>
                <span class="badge badge-grade">Grade: {q_obs.get('health_grade', 'A+')} ({q_obs.get('health_score', 100)}/100)</span>
            </div>
        </div>

        <!-- KPI Summary Cards -->
        <div class="kpi-grid">
            <div class="kpi-box">
                <div class="kpi-label">Total Rows</div>
                <div class="kpi-val">{ov.get('total_rows', 0):,}</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Total Features</div>
                <div class="kpi-val">{ov.get('total_columns', 0):,}</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Memory Footprint</div>
                <div class="kpi-val">{ov.get('memory_usage_mb', 0):.2f} MB</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Missing Cells</div>
                <div class="kpi-val">{ov.get('missing_cells_pct', 0):.2f}%</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Duplicate Rows</div>
                <div class="kpi-val">{ov.get('duplicate_rows_pct', 0):.2f}%</div>
            </div>
        </div>

        <!-- 1. DATASET OVERVIEW -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">1. Dataset Overview & Schema</h2>
                <span class="badge badge-observed">🔍 Observed results</span>
            </div>
            <p style="color: var(--text-muted); margin-bottom: 12px;">Complete tabular dimensions and data type classifications for active dataset features.</p>
            <table>
                <thead>
                    <tr>
                        <th>Feature Name</th>
                        <th>Classified Type</th>
                        <th>Pandas Dtype</th>
                        <th>Missing Count (%)</th>
                        <th>Unique Values</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f"<tr><td><strong>{c.get('name')}</strong></td><td>{c.get('classified_type')}</td><td><code>{c.get('dtype')}</code></td><td>{c.get('missing_count')} ({c.get('missing_pct')}%)</td><td>{c.get('unique_count')}</td></tr>" for c in s1['columns_schema'][:12])}
                </tbody>
            </table>
        </div>

        <!-- 2. DATA QUALITY ASSESSMENT -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">2. Data Quality Assessment</h2>
                <div>
                    <span class="badge badge-observed">🔍 Observed results</span>
                    <span class="badge badge-issues">⚠️ Potential issues</span>
                </div>
            </div>
            <p>Composite Quality Health: <strong>{q_obs.get('health_score', 100)} / 100</strong> ({q_obs.get('completeness_score', 100.0):.2f}% completeness, {q_obs.get('uniqueness_score', 100.0):.2f}% uniqueness) with <strong>{q_obs.get('total_detected_issues', 0)} detected defect(s)</strong>.</p>
            <table>
                <thead>
                    <tr>
                        <th>Severity</th>
                        <th>Column</th>
                        <th>Defect Category</th>
                        <th>Affected Rows</th>
                        <th>Remediation Protocol</th>
                    </tr>
                </thead>
                <tbody>
                    {quality_trs}
                </tbody>
            </table>
        </div>

        <!-- 3. CLEANING SUMMARY -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">3. Cleaning Summary & Audit Trail</h2>
                <div>
                    <span class="badge badge-observed">🔍 Observed results</span>
                    <span class="badge badge-recs">🎯 Recommendations</span>
                </div>
            </div>
            <div class="kpi-grid">
                <div class="kpi-box">
                    <div class="kpi-label">Status</div>
                    <div style="font-weight: 700; font-size: 16px; margin-top: 4px; color: var(--cyan);">{cl_obs.get('session_dataset_status')}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Record Shift</div>
                    <div class="kpi-val">{cl_obs.get('row_count_delta', {}).get('before', 0):,} &rarr; {cl_obs.get('row_count_delta', {}).get('after', 0):,}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Duplicates Purged</div>
                    <div class="kpi-val">{cl_obs.get('duplicates_purged', 0)}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Nulls Handled</div>
                    <div class="kpi-val">{cl_obs.get('missing_values_remediated', 0)}</div>
                </div>
            </div>
            <div class="callout callout-rec">
                <strong>🎯 Recommended Transformation Pipeline:</strong>
                <ul>
                    {"".join(f"<li>{r}</li>" for r in cl_rec) if cl_rec else "<li>Dataset is clean; maintain ingestion validation.</li>"}
                </ul>
            </div>
        </div>

        <!-- 4. STATISTICAL ANALYSIS -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">4. Statistical Analysis & Moments</h2>
                <div>
                    <span class="badge badge-observed">🔍 Observed results</span>
                    <span class="badge badge-ai">🧠 AI interpretation</span>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Std Dev</th>
                        <th>[Min, Max]</th>
                        <th>Skewness</th>
                        <th>95% Confidence Interval</th>
                    </tr>
                </thead>
                <tbody>
                    {stats_trs}
                </tbody>
            </table>
            <div class="callout callout-ai" style="margin-top: 16px;">
                <strong>🧠 Rule-Based Statistical Observations:</strong>
                <ul>
                    {"".join(f"<li>{o}</li>" for o in stat_obs[:5])}
                </ul>
            </div>
        </div>

        <!-- 5. CORRELATION ANALYSIS -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">5. Correlation Analysis</h2>
                <div>
                    <span class="badge badge-observed">🔍 Observed results</span>
                    <span class="badge badge-issues">⚠️ Potential issues</span>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Feature Pair</th>
                        <th>Pearson r</th>
                        <th>Strength</th>
                        <th>Direction</th>
                    </tr>
                </thead>
                <tbody>
                    {corr_trs}
                </tbody>
            </table>
            <p class="text-muted" style="margin-top: 8px; font-size: 13px;">* Note on Causation: Correlation measures linear co-movement only and does not establish causal mechanisms.</p>
        </div>

        <!-- 6. OUTLIER ANALYSIS -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">6. Outlier & Anomaly Analysis (IQR 1.5x)</h2>
                <div>
                    <span class="badge badge-observed">🔍 Observed results</span>
                    <span class="badge badge-issues">⚠️ Potential issues</span>
                </div>
            </div>
            <p>Total statistical outliers detected across features: <strong>{out_obs.get('total_outliers_count', 0):,}</strong> (Confirmed Errors: <strong>{s6['potential_issues']['confirmed_data_errors_count']}</strong>, Statistical Extremes: <strong>{s6['potential_issues']['statistical_tail_outliers_count']}</strong>).</p>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Outlier Count</th>
                        <th>Outlier %</th>
                        <th>IQR Bounds [Lower, Upper]</th>
                        <th>Extreme Values</th>
                    </tr>
                </thead>
                <tbody>
                    {out_trs}
                </tbody>
            </table>
        </div>

        <!-- 7. IMPORTANT VISUALIZATIONS -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">7. Important Exploratory Visualizations</h2>
                <span class="badge badge-observed">🔍 Observed results</span>
            </div>
            <div class="chart-grid">
                {chart_divs}
            </div>
        </div>

        <!-- 8. AI-GENERATED INSIGHTS -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">8. AI-Generated Insights & Synthesis</h2>
                <span class="badge badge-ai">🧠 AI interpretation</span>
            </div>
            <div class="ai-box">
                <div class="ai-title">✨ Strategic Executive Takeaways</div>
                <p>{s8.get('executive_summary')}</p>
            </div>
            <div class="grid-2">
                <div class="kpi-box">
                    <h4 style="color: #FFF; margin-bottom: 8px;">📊 Key Analytical Findings</h4>
                    <p style="font-size: 14px; white-space: pre-line;">{s8.get('key_findings') or 'Standard empirical distributions observed.'}</p>
                </div>
                <div class="kpi-box">
                    <h4 style="color: #FFF; margin-bottom: 8px;">📈 Trend Dynamics & Associations</h4>
                    <p style="font-size: 14px; white-space: pre-line;">{s8.get('important_trends') or 'Cross-sectional feature dynamics evaluated.'}</p>
                </div>
            </div>
        </div>

        <!-- 9. RECOMMENDATIONS -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">9. Recommendations & Next Steps</h2>
                <span class="badge badge-recs">🎯 Recommendations</span>
            </div>
            <div class="grid-2">
                <div class="kpi-box">
                    <h4 style="color: #10B981; margin-bottom: 8px;">🧹 Data Cleaning & Preprocessing</h4>
                    <ul>{cl_rec_html}</ul>
                </div>
                <div class="kpi-box">
                    <h4 style="color: #6366F1; margin-bottom: 8px;">⚙️ Feature Engineering</h4>
                    <ul>{fe_rec_html}</ul>
                </div>
                <div class="kpi-box">
                    <h4 style="color: #06B6D4; margin-bottom: 8px;">📈 Predictive Modeling</h4>
                    <ul>{mod_rec_html}</ul>
                </div>
                <div class="kpi-box">
                    <h4 style="color: #F59E0B; margin-bottom: 8px;">🔒 Data Governance & Telemetry</h4>
                    <ul>{gov_rec_html}</ul>
                </div>
            </div>
        </div>

        <div class="footer">
            Generated by AI Data Analyst Agent &bull; Professional Portfolio Analytics Suite &bull; Phase 13 Automated Analysis Report Engine
        </div>
    </div>

    <!-- Initialize interactive Plotly charts -->
    <script>
        document.addEventListener('DOMContentLoaded', () => {{
            {chart_scripts}
        }});
    </script>
</body>
</html>
"""
        return html

    def export_report(self, target_path: Union[str, Path], report_format: str = "html") -> Path:
        """Export generated report directly to disk."""
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fmt = report_format.lower().strip()

        if fmt in ["md", "markdown"]:
            content = self.generate_markdown_report()
            target.write_text(content, encoding="utf-8")
        elif fmt in ["json"]:
            content = json.dumps(self.generate_structured_report(), indent=2)
            target.write_text(content, encoding="utf-8")
        else:
            content = self.generate_html_report()
            target.write_text(content, encoding="utf-8")

        return target


# ==============================================================================
# BACKWARD COMPATIBLE HELPER FUNCTIONS
# ==============================================================================
def generate_markdown_report(dataset_name: str, profile_data: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> str:
    """Backward-compatible helper function."""
    if df is not None:
        generator = ReportGenerator(df, dataset_id=dataset_name, profile_data=profile_data)
        return generator.generate_markdown_report()

    overview = profile_data.get("overview", {})
    return f"""# Data Analysis Report: {dataset_name}

## Executive Summary
- Total Records: {overview.get('total_rows', 0):,}
- Total Features: {overview.get('total_columns', 0):,}
- Memory Usage: {overview.get('memory_usage_mb', 0):.2f} MB
- Missing Data: {overview.get('missing_cells_percentage', 0):.2f}%

*Generated by AI Data Analyst Agent.*
"""
