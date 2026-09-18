"""
Intelligent Analysis Planner (Phase 11 Module).
Generates explicit multi-step analytical plans, evaluates visualization necessity,
selects optimal chart types, and coordinates deterministic execution steps.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import pandas as pd

from visualization.charts import inspect_column_types

logger = logging.getLogger(__name__)


class AnalysisPlanStep:
    """Represents an individual step in the analysis plan."""

    def __init__(
        self,
        step_number: int,
        action: str,
        description: str,
        status: str = "completed",
    ):
        self.step_number = step_number
        self.action = action
        self.description = description
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "action": self.action,
            "description": self.description,
            "status": self.status,
            "display": f"{self.step_number}. {self.action}",
        }


class AnalysisPlan:
    """Structured plan returned by the Analysis Planner."""

    def __init__(
        self,
        query: str,
        intent: str,
        tool_name: str,
        tool_params: Dict[str, Any],
        steps: List[str],
        requires_visualization: bool,
        recommended_chart_type: str,
        visualization_reasoning: str,
        target_columns: Optional[List[str]] = None,
        structured_steps: Optional[List[Dict[str, Any]]] = None,
    ):
        self.query = query
        self.intent = intent
        self.tool_name = tool_name
        self.tool_params = tool_params
        self.steps = steps  # List of formatted strings e.g. ["1. Detect date column", ...]
        self.requires_visualization = requires_visualization
        self.recommended_chart_type = recommended_chart_type
        self.visualization_reasoning = visualization_reasoning
        self.target_columns = target_columns or []
        self.structured_steps = structured_steps or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "steps": self.steps,
            "total_steps": len(self.steps),
            "requires_visualization": self.requires_visualization,
            "recommended_chart_type": self.recommended_chart_type,
            "visualization_reasoning": self.visualization_reasoning,
            "target_columns": self.target_columns,
            "structured_steps": self.structured_steps,
        }


class AnalysisPlanner:
    """
    Intelligent Analysis Planner & Visualization Decision Engine.
    Evaluates questions, determines necessary analytical steps, and decides if visualization is required.
    """

    @staticmethod
    def plan(
        query: str,
        df: pd.DataFrame,
        tool_name: str,
        tool_params: Dict[str, Any],
        resolved_columns: Optional[Dict[str, Any]] = None,
    ) -> AnalysisPlan:
        """
        Synthesize a tailored multi-step analysis plan and visualization decision.
        """
        q = query.strip()
        q_lower = q.lower()
        schema = inspect_column_types(df)
        num_cols = schema.get("numerical", [])
        cat_cols = schema.get("categorical", [])
        dt_cols = schema.get("datetime", [])

        # -------------------------------------------------------------
        # 1. TIME SERIES / TEMPORAL TRENDS
        # E.g. "How have sales changed over time?"
        # 1. Detect date column
        # 2. Detect sales column
        # 3. Perform time aggregation
        # 4. Generate line chart
        # 5. Explain trend
        # -------------------------------------------------------------
        if tool_name == "time_series_analysis" or any(k in q_lower for k in ("over time", "trend", "change over", "changed over", "timeline", "by month", "by year")):
            date_col = tool_params.get("date_col") or (dt_cols[0] if dt_cols else "Date column")
            val_col = tool_params.get("value_col") or (num_cols[0] if num_cols else "Metric column")
            freq = tool_params.get("freq", "auto")
            agg = tool_params.get("agg_func", "sum")

            steps = [
                f"Detect date column ({date_col})",
                f"Detect sales / target metric column ({val_col})",
                f"Perform time aggregation ({agg} over chronological intervals)",
                "Generate line chart",
                "Explain trend",
            ]
            formatted_steps = [f"{i+1}. {s}" for i, s in enumerate(steps)]

            return AnalysisPlan(
                query=query,
                intent="time_series_trend",
                tool_name="time_series_analysis",
                tool_params={"date_col": date_col, "value_col": val_col, "freq": freq, "agg_func": agg},
                steps=formatted_steps,
                requires_visualization=True,
                recommended_chart_type="line",
                visualization_reasoning="Chronological progression and temporal growth require an interactive line chart to highlight inflection points, peak periods, and troughs.",
                target_columns=[date_col, val_col],
                structured_steps=[
                    {"step": 1, "action": "Detect date column", "target": str(date_col)},
                    {"step": 2, "action": "Detect metric column", "target": str(val_col)},
                    {"step": 3, "action": "Perform time aggregation", "method": f"resample_{agg}"},
                    {"step": 4, "action": "Generate line chart", "chart_type": "line"},
                    {"step": 5, "action": "Explain trend", "output": "growth_rate_and_direction"},
                ],
            )

        # -------------------------------------------------------------
        # 2. CORRELATION / RELATIONSHIP
        # E.g. "Is advertising spend related to revenue?"
        # 1. Identify advertising spend
        # 2. Identify revenue
        # 3. Calculate Pearson correlation
        # 4. Generate scatter plot
        # 5. Explain relationship without claiming causation
        # -------------------------------------------------------------
        if tool_name == "correlation_analysis" or any(k in q_lower for k in ("related", "relate", "correlation", "correlate", "association", "depend on", "linked with")):
            c1 = tool_params.get("col1")
            c2 = tool_params.get("col2")

            if c1 and c2:
                # Specific pairwise relationship -> Scatter plot
                steps = [
                    f"Identify first variable ({c1})",
                    f"Identify second variable ({c2})",
                    "Calculate Pearson correlation coefficient (r)",
                    "Generate scatter plot",
                    "Explain relationship without claiming causation",
                ]
                formatted_steps = [f"{i+1}. {s}" for i, s in enumerate(steps)]

                return AnalysisPlan(
                    query=query,
                    intent="bivariate_correlation",
                    tool_name="correlation_analysis",
                    tool_params={"col1": c1, "col2": c2, "method": tool_params.get("method", "pearson")},
                    steps=formatted_steps,
                    requires_visualization=True,
                    recommended_chart_type="scatter",
                    visualization_reasoning="Evaluating continuous bivariate associations requires a scatter plot with distribution markers to inspect co-movement and linear dispersion.",
                    target_columns=[c1, c2],
                    structured_steps=[
                        {"step": 1, "action": "Identify first variable", "column": c1},
                        {"step": 2, "action": "Identify second variable", "column": c2},
                        {"step": 3, "action": "Calculate Pearson correlation", "method": "pearson"},
                        {"step": 4, "action": "Generate scatter plot", "chart_type": "scatter"},
                        {"step": 5, "action": "Explain relationship without claiming causation", "rule": "non_causation_mandate"},
                    ],
                )
            else:
                # Global dataset-wide correlation ranking -> Correlation Heatmap
                steps = [
                    "Identify all continuous numerical columns",
                    "Calculate symmetric Pearson correlation matrix",
                    "Rank strongest positive and negative associations",
                    "Generate correlation heatmap",
                    "Explain top relationships without claiming causation",
                ]
                formatted_steps = [f"{i+1}. {s}" for i, s in enumerate(steps)]

                return AnalysisPlan(
                    query=query,
                    intent="multivariate_correlation_ranking",
                    tool_name="correlation_analysis",
                    tool_params={"threshold": tool_params.get("threshold", 0.0)},
                    steps=formatted_steps,
                    requires_visualization=True,
                    recommended_chart_type="heatmap",
                    visualization_reasoning="Synthesizing multi-variable relationships across all features requires an annotated correlation heatmap to highlight cluster associations.",
                    target_columns=num_cols[:5],
                    structured_steps=[
                        {"step": 1, "action": "Identify continuous numerical columns", "count": len(num_cols)},
                        {"step": 2, "action": "Calculate Pearson correlation matrix", "method": "pearson"},
                        {"step": 3, "action": "Rank association pairs", "ordering": "descending_abs_r"},
                        {"step": 4, "action": "Generate correlation heatmap", "chart_type": "heatmap"},
                        {"step": 5, "action": "Explain top relationships without claiming causation", "rule": "non_causation_mandate"},
                    ],
                )

        # -------------------------------------------------------------
        # 3. GROUPBY / CATEGORICAL RANKING / HIGHEST REVENUE
        # E.g. "Which category has the highest revenue?"
        # 1. Identify category column
        # 2. Identify revenue column
        # 3. Group by category
        # 4. Aggregate revenue
        # 5. Sort descending
        # 6. Generate bar chart
        # 7. Explain result
        # -------------------------------------------------------------
        if tool_name == "groupby_analysis" or any(k in q_lower for k in ("which ", "highest", "lowest", "best", "worst", "top ", "per category", "by region")):
            grp_col = tool_params.get("group_by_col") or (cat_cols[0] if cat_cols else "Category column")
            tgt_col = tool_params.get("target_col") or (num_cols[0] if num_cols else "Revenue column")
            agg_func = tool_params.get("agg_func", "sum")
            sort_desc = tool_params.get("sort_desc", True)
            sort_label = "descending (highest first)" if sort_desc else "ascending (lowest first)"

            steps = [
                f"Identify category / grouping column ({grp_col})",
                f"Identify revenue / metric column ({tgt_col})",
                f"Group by {grp_col}",
                f"Aggregate {tgt_col} ({agg_func})",
                f"Sort {sort_label}",
                "Generate bar chart",
                "Explain result",
            ]
            formatted_steps = [f"{i+1}. {s}" for i, s in enumerate(steps)]

            return AnalysisPlan(
                query=query,
                intent="categorical_groupby_ranking",
                tool_name="groupby_analysis",
                tool_params={
                    "group_by_col": grp_col,
                    "target_col": tgt_col,
                    "agg_func": agg_func,
                    "sort_desc": sort_desc,
                    "limit": tool_params.get("limit", 10),
                },
                steps=formatted_steps,
                requires_visualization=True,
                recommended_chart_type="bar",
                visualization_reasoning="Ranking and comparing categorical performance tiers is most effectively presented using a sorted interactive bar chart with percentage shares.",
                target_columns=[grp_col, tgt_col],
                structured_steps=[
                    {"step": 1, "action": f"Identify grouping column", "column": grp_col},
                    {"step": 2, "action": f"Identify metric column", "column": tgt_col},
                    {"step": 3, "action": f"Group by {grp_col}", "operation": "groupby"},
                    {"step": 4, "action": f"Aggregate {tgt_col}", "function": agg_func},
                    {"step": 5, "action": f"Sort {sort_label}", "ascending": not sort_desc},
                    {"step": 6, "action": "Generate bar chart", "chart_type": "bar"},
                    {"step": 7, "action": "Explain result", "output": "top_performer_and_breakdown"},
                ],
            )

        # -------------------------------------------------------------
        # 4. OUTLIER / ANOMALY DETECTION
        # E.g. "Are there unusual values?"
        # 1. Identify target continuous column(s)
        # 2. Calculate IQR (1.5x) or Z-score thresholds
        # 3. Detect extreme value anomalies
        # 4. Generate box plot chart
        # 5. Explain outlier findings and valid boundaries
        # -------------------------------------------------------------
        if tool_name == "outlier_analysis" or any(k in q_lower for k in ("unusual", "outlier", "anomal", "extreme value", "irregular")):
            col_name = tool_params.get("column_name") or (num_cols[0] if num_cols else "All numerical columns")
            method = tool_params.get("method", "iqr")

            steps = [
                f"Identify continuous numerical column ({col_name})",
                f"Calculate statistical anomaly boundaries ({method.upper()} 1.5x method)",
                "Detect outlier instances beyond thresholds",
                "Generate box plot chart with jittered points",
                "Explain outlier findings and valid statistical range",
            ]
            formatted_steps = [f"{i+1}. {s}" for i, s in enumerate(steps)]

            return AnalysisPlan(
                query=query,
                intent="outlier_detection",
                tool_name="outlier_analysis",
                tool_params={"column_name": tool_params.get("column_name"), "method": method, "threshold": 1.5},
                steps=formatted_steps,
                requires_visualization=True,
                recommended_chart_type="box",
                visualization_reasoning="Anomalies and distribution tails are best visualized with an interactive box plot clearly demarcating quartiles, whiskers, and individual outlier points.",
                target_columns=[col_name] if col_name != "All numerical columns" else num_cols,
                structured_steps=[
                    {"step": 1, "action": "Identify numerical feature", "column": col_name},
                    {"step": 2, "action": "Calculate outlier boundaries", "method": method},
                    {"step": 3, "action": "Detect outlier instances", "operation": "filter_extremes"},
                    {"step": 4, "action": "Generate box plot chart", "chart_type": "box"},
                    {"step": 5, "action": "Explain outlier findings", "output": "outlier_counts_and_bounds"},
                ],
            )

        # -------------------------------------------------------------
        # 5. DISTRIBUTION / HISTOGRAM / SPREAD
        # E.g. "What is the distribution of sales?"
        # 1. Identify target feature
        # 2. Compute histogram bin intervals
        # 3. Calculate skewness, kurtosis, and dispersion
        # 4. Generate distribution histogram chart
        # 5. Explain distribution shape and symmetry
        # -------------------------------------------------------------
        if tool_name == "distribution_analysis" or any(k in q_lower for k in ("distribution", "histogram", "spread", "skew", "kurtosis")):
            col_name = tool_params.get("column_name") or (num_cols[0] if num_cols else df.columns[0])

            steps = [
                f"Identify target column ({col_name})",
                "Compute histogram bin intervals and frequency counts",
                "Calculate skewness, kurtosis, and spread metrics",
                "Generate distribution histogram chart",
                "Explain distribution shape, dominant bin, and symmetry",
            ]
            formatted_steps = [f"{i+1}. {s}" for i, s in enumerate(steps)]

            return AnalysisPlan(
                query=query,
                intent="distribution_histogram",
                tool_name="distribution_analysis",
                tool_params={"column_name": col_name, "bins": tool_params.get("bins", 10)},
                steps=formatted_steps,
                requires_visualization=True,
                recommended_chart_type="histogram",
                visualization_reasoning="Inspecting feature modality, spread, and skewness requires a binned histogram chart with count overlays.",
                target_columns=[col_name],
                structured_steps=[
                    {"step": 1, "action": "Identify target column", "column": col_name},
                    {"step": 2, "action": "Compute histogram bins", "bins": 10},
                    {"step": 3, "action": "Calculate skewness and kurtosis", "metrics": ["mean", "median", "skew", "kurtosis"]},
                    {"step": 4, "action": "Generate distribution histogram", "chart_type": "histogram"},
                    {"step": 5, "action": "Explain distribution shape", "output": "skewness_and_spread"},
                ],
            )

        # -------------------------------------------------------------
        # 6. SCALAR AGGREGATION ANALYSIS (NO CHART REQUIRED)
        # E.g. "What is the average profit?", "What is total revenue?"
        # 1. Identify target numerical column
        # 2. Clean and parse numeric records
        # 3. Calculate requested aggregation metric (mean/sum/median)
        # 4. Explain statistical result
        # -> requires_visualization: False
        # -------------------------------------------------------------
        if tool_name == "aggregation_analysis" or any(k in q_lower for k in ("average", "mean", "avg", "total", "sum of", "median", "minimum", "maximum")):
            tgt_col = tool_params.get("target_col") or (num_cols[0] if num_cols else "Column")
            agg = tool_params.get("agg_func", "mean")

            steps = [
                f"Identify target numerical column ({tgt_col})",
                "Clean non-numeric records and handle null values",
                f"Calculate deterministic {agg} aggregation and descriptive moments",
                "Explain statistical result and contextual summary range",
            ]
            formatted_steps = [f"{i+1}. {s}" for i, s in enumerate(steps)]

            return AnalysisPlan(
                query=query,
                intent="scalar_aggregation",
                tool_name="aggregation_analysis",
                tool_params={"target_col": tgt_col, "agg_func": agg},
                steps=formatted_steps,
                requires_visualization=False,
                recommended_chart_type="none",
                visualization_reasoning="Direct scalar metrics (averages, totals, medians) are cleanly communicated via concise KPI summaries without requiring a standalone chart.",
                target_columns=[tgt_col],
                structured_steps=[
                    {"step": 1, "action": "Identify numerical column", "column": tgt_col},
                    {"step": 2, "action": "Clean numeric series", "operation": "dropna_and_coerce"},
                    {"step": 3, "action": f"Calculate {agg}", "function": agg},
                    {"step": 4, "action": "Explain statistical result", "output": "scalar_kpi_card"},
                ],
            )

        # -------------------------------------------------------------
        # 7. DATASET OVERVIEW / SUMMARY (NO CHART REQUIRED)
        # E.g. "Give me an overview of the dataset"
        # -------------------------------------------------------------
        if tool_name == "dataset_summary" or any(k in q_lower for k in ("overview", "how many rows", "describe dataset", "shape of")):
            steps = [
                "Inspect total dataset dimensions (rows and columns)",
                "Evaluate missing cell rate and duplicate row count",
                "Classify schema columns into data types (numerical, categorical, datetime)",
                "Synthesize high-level dataset health and profile summary",
            ]
            formatted_steps = [f"{i+1}. {s}" for i, s in enumerate(steps)]

            return AnalysisPlan(
                query=query,
                intent="dataset_overview",
                tool_name="dataset_summary",
                tool_params={},
                steps=formatted_steps,
                requires_visualization=False,
                recommended_chart_type="none",
                visualization_reasoning="High-level tabular metadata and schema breakdown are best presented in structured summary cards.",
                target_columns=list(df.columns),
                structured_steps=[
                    {"step": 1, "action": "Inspect dimensions", "metrics": ["total_rows", "total_cols"]},
                    {"step": 2, "action": "Evaluate data integrity", "metrics": ["missing_pct", "duplicate_pct"]},
                    {"step": 3, "action": "Classify column types", "operation": "schema_inference"},
                    {"step": 4, "action": "Synthesize summary", "output": "dataset_profile_card"},
                ],
            )

        # -------------------------------------------------------------
        # 8. INVESTIGATE FURTHER / RECOMMENDATIONS
        # -------------------------------------------------------------
        if tool_name == "investigate_further":
            steps = [
                "Audit outlier bounds across continuous metrics",
                "Scan pairwise correlation matrix for strong associations",
                "Evaluate categorical cardinality across primary metrics",
                "Synthesize prioritized data-driven exploration leads",
            ]
            formatted_steps = [f"{i+1}. {s}" for i, s in enumerate(steps)]

            return AnalysisPlan(
                query=query,
                intent="investigation_leads",
                tool_name="investigate_further",
                tool_params={},
                steps=formatted_steps,
                requires_visualization=False,
                recommended_chart_type="none",
                visualization_reasoning="Actionable investigation recommendations are presented as prioritized markdown action cards.",
                target_columns=list(df.columns)[:5],
                structured_steps=[
                    {"step": 1, "action": "Audit outlier bounds", "operation": "scan_iqr"},
                    {"step": 2, "action": "Scan correlation matrix", "operation": "scan_pearson"},
                    {"step": 3, "action": "Evaluate category variance", "operation": "scan_variance"},
                    {"step": 4, "action": "Synthesize leads", "output": "priority_recommendations"},
                ],
            )

        # -------------------------------------------------------------
        # 9. SINGLE COLUMN SUMMARY
        # -------------------------------------------------------------
        col_name = tool_params.get("column_name", df.columns[0] if len(df.columns) > 0 else "Column")
        is_num = col_name in num_cols
        is_cat = col_name in cat_cols

        steps = [
            f"Locate target column ({col_name})",
            "Calculate data completeness, missingness, and uniqueness",
            f"Compute {'statistical moments (mean, std, quartiles)' if is_num else 'frequency distribution and top categories'}",
            "Explain column findings",
        ]
        formatted_steps = [f"{i+1}. {s}" for i, s in enumerate(steps)]

        return AnalysisPlan(
            query=query,
            intent="column_summary",
            tool_name="column_summary",
            tool_params={"column_name": col_name},
            steps=formatted_steps,
            requires_visualization=is_cat or is_num,
            recommended_chart_type="bar" if is_cat else ("box" if is_num else "none"),
            visualization_reasoning=f"Visualizing the spread of '{col_name}' highlights category proportions or statistical dispersion.",
            target_columns=[col_name],
            structured_steps=[
                {"step": 1, "action": "Locate column", "column": col_name},
                {"step": 2, "action": "Calculate completeness", "metrics": ["valid_count", "missing_pct"]},
                {"step": 3, "action": "Compute statistics", "type": "numerical" if is_num else "categorical"},
                {"step": 4, "action": "Explain column findings", "output": "column_summary_card"},
            ],
        )
