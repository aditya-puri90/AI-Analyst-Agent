"""
Natural Language Dataset Q&A Engine (Phase 10 Module).
Translates user natural language questions into deterministic Python analysis tool calls,
executes calculations safely on Pandas DataFrames, synthesizes factually grounded explanations,
generates interactive Plotly visualizations, and manages per-dataset conversation session history.
"""

import re
import json
import logging
import math
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

from config.settings import Config
from analysis.profiler import infer_column_type, _safe_json_value
from analysis.statistics import StatisticalAnalysisEngine
from analysis.correlation import CorrelationAnalysisEngine
from analysis.outliers import OutlierDetectionEngine
from visualization.charts import inspect_column_types
from agent.prompts import (
    QA_ROUTER_SYSTEM_PROMPT,
    QA_EXPLAINER_SYSTEM_PROMPT,
    build_qa_router_prompt,
    build_qa_explainer_prompt,
)
from utils.file_handler import load_dataset

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. COLUMN RESOLVER & SCHEMA UTILITIES
# ==============================================================================

COMMON_COLUMN_ALIASES = {
    "sales": ["sales", "sales_amount", "revenue", "turnover", "total_sales", "sale_amount", "gross_sales"],
    "revenue": ["revenue", "sales_amount", "sales", "total_revenue", "gross_revenue", "income"],
    "profit": ["profit", "net_profit", "margin", "net_margin", "operating_profit", "earnings"],
    "discount": ["discount", "discount_pct", "discount_amount", "discount_percentage", "rebate"],
    "quantity": ["quantity", "qty", "volume", "units", "units_sold", "count", "items_count"],
    "price": ["price", "unit_price", "cost", "unit_cost", "item_price", "rate", "amount"],
    "age": ["customer_age", "age", "user_age", "client_age", "buyer_age"],
    "date": ["order_date", "date", "created_at", "timestamp", "transaction_date", "datetime", "purchase_date", "time"],
    "category": ["category", "product_category", "item_category", "segment", "type", "department", "group"],
    "region": ["region", "territory", "location", "country", "state", "city", "area", "zone"],
    "product": ["product", "product_name", "product_id", "item", "item_name", "sku", "title"],
    "customer": ["customer", "customer_id", "customer_name", "client", "user", "user_id"],
    "payment": ["payment_method", "payment_type", "payment", "mode_of_payment"],
    "returned": ["is_returned", "returned", "return_status", "is_refunded"],
}


def normalize_token(text: str) -> str:
    """Normalize string token for fuzzy matching."""
    return re.sub(r"[^a-zA-Z0-9]", "", str(text).lower().strip())


def resolve_column_name(
    df: pd.DataFrame,
    target: Optional[str],
    preferred_type: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve a user-provided or LLM-provided column candidate to an exact DataFrame column.
    Supports exact matching, case-insensitive matching, stripped normalization, and semantic alias mapping.
    """
    if not target or not isinstance(target, str):
        return None

    target_clean = target.strip()
    if target_clean in df.columns:
        return target_clean

    target_norm = normalize_token(target_clean)
    col_norms = {normalize_token(c): c for c in df.columns}

    # 1. Exact normalized match (e.g. "sales amount" -> "Sales_Amount")
    if target_norm in col_norms:
        return col_norms[target_norm]

    # 2. Substring matching in column names
    for norm_c, orig_c in col_norms.items():
        if target_norm and (target_norm in norm_c or norm_c in target_norm):
            return orig_c

    # 3. Check semantic aliases
    for concept, aliases in COMMON_COLUMN_ALIASES.items():
        if target_norm in [normalize_token(a) for a in aliases] or target_norm == concept:
            for alias in aliases:
                alias_norm = normalize_token(alias)
                if alias_norm in col_norms:
                    return col_norms[alias_norm]

    # 4. Fallback if preferred_type is requested
    if preferred_type:
        schema = inspect_column_types(df)
        candidates = schema.get(preferred_type, [])
        if candidates:
            return candidates[0]

    return None


def get_dataset_column_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Build compact column schema dictionary for LLM routing."""
    schema_by_type = inspect_column_types(df)
    cols_meta = {}
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        c_type = "unknown"
        for t, cols in schema_by_type.items():
            if col in cols:
                c_type = t
                break
        sample_vals = df[col].dropna().head(3).tolist()
        cols_meta[col] = {
            "type": c_type,
            "dtype": dtype_str,
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique()),
            "sample_values": [_safe_json_value(v) for v in sample_vals],
        }
    return cols_meta


# ==============================================================================
# 2. SAFE DETERMINISTIC PYTHON ANALYSIS TOOLS
# ==============================================================================

def dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive dataset metadata, record volume, dimensionality,
    memory footprint, missingness, and column taxonomy.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 2)
    total_cells = total_rows * total_cols
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / max(1, total_cells)) * 100, 2)
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / max(1, total_rows)) * 100, 2)

    schema = inspect_column_types(df)
    columns_breakdown = []
    for col in df.columns:
        c_type = "categorical"
        for t, cols in schema.items():
            if col in cols:
                c_type = t
                break
        columns_breakdown.append({
            "name": col,
            "type": c_type,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(float(df[col].isna().mean() * 100), 2),
            "unique_values": int(df[col].nunique()),
        })

    return {
        "status": "success",
        "tool": "dataset_summary",
        "total_rows": total_rows,
        "total_columns": total_cols,
        "memory_usage_mb": memory_mb,
        "total_missing_cells": missing_cells,
        "overall_missing_pct": missing_pct,
        "duplicate_rows_count": duplicate_rows,
        "duplicate_rows_pct": duplicate_pct,
        "column_types_count": {k: len(v) for k, v in schema.items()},
        "columns": columns_breakdown,
    }


def column_summary(df: pd.DataFrame, column_name: str) -> Dict[str, Any]:
    """
    Compute exhaustive statistical, frequency, or temporal summary for a single column.
    """
    col = resolve_column_name(df, column_name)
    if not col:
        return {
            "status": "unsupported",
            "tool": "column_summary",
            "reason": f"Column '{column_name}' was not found in the dataset.",
            "available_columns": list(df.columns),
        }

    series = df[col]
    total_count = len(series)
    missing_count = int(series.isna().sum())
    missing_pct = round((missing_count / max(1, total_count)) * 100, 2)
    valid_count = total_count - missing_count
    unique_count = int(series.nunique())

    schema = inspect_column_types(df)
    is_num = col in schema.get("numerical", []) or pd.api.types.is_numeric_dtype(series)
    is_dt = col in schema.get("datetime", [])
    is_cat = col in schema.get("categorical", []) or col in schema.get("boolean", []) or col in schema.get("id", [])

    result: Dict[str, Any] = {
        "status": "success",
        "tool": "column_summary",
        "column_name": col,
        "total_rows": total_count,
        "valid_rows": valid_count,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "unique_count": unique_count,
    }

    if is_dt:
        dt_clean = pd.to_datetime(series, errors="coerce").dropna()
        if not dt_clean.empty:
            result.update({
                "column_type": "datetime",
                "earliest_date": dt_clean.min().strftime("%Y-%m-%d"),
                "latest_date": dt_clean.max().strftime("%Y-%m-%d"),
                "total_days_span": int((dt_clean.max() - dt_clean.min()).days),
            })
    elif is_num:
        clean_num = pd.to_numeric(series, errors="coerce").dropna()
        if not clean_num.empty:
            mean_val = round(float(clean_num.mean()), 4)
            std_val = round(float(clean_num.std(ddof=1) if len(clean_num) > 1 else 0.0), 4)
            median_val = round(float(clean_num.median()), 4)
            min_val = round(float(clean_num.min()), 4)
            max_val = round(float(clean_num.max()), 4)
            q25 = round(float(clean_num.quantile(0.25)), 4)
            q75 = round(float(clean_num.quantile(0.75)), 4)
            iqr_val = round(q75 - q25, 4)
            skew_val = round(float(clean_num.skew()), 4) if len(clean_num) > 2 else 0.0
            kurt_val = round(float(clean_num.kurtosis()), 4) if len(clean_num) > 3 else 0.0

            result.update({
                "column_type": "numerical",
                "mean": mean_val,
                "median": median_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "q25": q25,
                "q75": q75,
                "iqr": iqr_val,
                "skewness": skew_val,
                "kurtosis": kurt_val,
                "sum": round(float(clean_num.sum()), 4),
            })
    else:
        clean_cat = series.dropna().astype(str)
        top_counts = clean_cat.value_counts().head(5)
        top_categories = []
        for cat, count in top_counts.items():
            pct = round((count / max(1, valid_count)) * 100, 2)
            top_categories.append({"category": cat, "count": int(count), "percentage": pct})

        result.update({
            "column_type": "categorical",
            "mode": str(clean_cat.mode().iloc[0]) if not clean_cat.empty else "N/A",
            "top_categories": top_categories,
        })

    return result


def groupby_analysis(
    df: pd.DataFrame,
    group_by_col: str,
    target_col: Optional[str] = None,
    agg_func: str = "sum",
    sort_desc: bool = True,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Compute grouped aggregations (e.g. 'Which category has the highest sales?', 'Which region performs best?').
    Calculates exact totals, means, counts, rankings, and percentage shares.
    """
    group_col = resolve_column_name(df, group_by_col)
    if not group_col:
        # Check if target was swapped with group
        swap_group = resolve_column_name(df, target_col)
        swap_target = resolve_column_name(df, group_by_col)
        if swap_group and swap_target:
            group_col, target_col = swap_group, swap_target

    if not group_col:
        schema = inspect_column_types(df)
        cat_cols = schema.get("categorical", []) + schema.get("boolean", [])
        return {
            "status": "unsupported",
            "tool": "groupby_analysis",
            "reason": f"Grouping column '{group_by_col}' was not found in the dataset.",
            "suggested_grouping_columns": cat_cols,
        }

    schema = inspect_column_types(df)
    num_cols = schema.get("numerical", [])

    # Resolve target column
    metric_col = None
    if target_col:
        metric_col = resolve_column_name(df, target_col)
        if not metric_col and target_col.lower() not in ("count", "rows", "records"):
            return {
                "status": "unsupported",
                "tool": "groupby_analysis",
                "reason": f"Metric column '{target_col}' was not found in the dataset.",
                "suggested_metric_columns": num_cols,
            }

    # Normalize agg function
    func_map = {
        "avg": "mean",
        "average": "mean",
        "mean": "mean",
        "sum": "sum",
        "total": "sum",
        "count": "count",
        "median": "median",
        "min": "min",
        "minimum": "min",
        "max": "max",
        "maximum": "max",
        "std": "std",
    }
    normalized_func = func_map.get(str(agg_func).lower().strip(), "sum")

    # If no metric column specified, select best available or count
    if not metric_col:
        if normalized_func == "count":
            metric_col = group_col
        elif num_cols:
            metric_col = num_cols[0]
        else:
            normalized_func = "count"
            metric_col = group_col

    # Execute groupby
    clean_df = df[[group_col, metric_col]].dropna(subset=[group_col]).copy()
    if normalized_func != "count":
        clean_df[metric_col] = pd.to_numeric(clean_df[metric_col], errors="coerce")
        clean_df = clean_df.dropna(subset=[metric_col])

    if clean_df.empty:
        return {
            "status": "error",
            "tool": "groupby_analysis",
            "reason": f"No valid records found for grouping '{group_col}' by '{metric_col}'.",
        }

    if normalized_func == "count":
        grouped = clean_df.groupby(group_col).size()
    else:
        grouped = clean_df.groupby(group_col)[metric_col].agg(normalized_func)

    grouped = grouped.sort_values(ascending=not sort_desc)
    total_val = float(clean_df[metric_col].sum()) if normalized_func == "sum" else float(len(clean_df))

    groups_data = []
    for rank, (grp_name, val) in enumerate(grouped.items(), start=1):
        v = float(val)
        pct = round((v / max(0.00001, total_val)) * 100, 2) if (total_val > 0 and normalized_func in ("sum", "count")) else None
        formatted_str = f"${v:,.2f}" if "sales" in metric_col.lower() or "revenue" in metric_col.lower() or "price" in metric_col.lower() else f"{v:,.2f}"
        groups_data.append({
            "rank": rank,
            "group": str(grp_name),
            "value": round(v, 4),
            "formatted_value": formatted_str,
            "percentage_of_total": pct,
        })

    top_performer = groups_data[0] if groups_data else None
    bottom_performer = groups_data[-1] if len(groups_data) > 1 else None

    return {
        "status": "success",
        "tool": "groupby_analysis",
        "group_by_column": group_col,
        "target_column": metric_col,
        "aggregation_function": normalized_func,
        "total_groups_count": len(groups_data),
        "top_performer": top_performer,
        "bottom_performer": bottom_performer,
        "groups": groups_data[:limit],
    }


def aggregation_analysis(
    df: pd.DataFrame,
    target_col: str,
    agg_func: str = "mean",
) -> Dict[str, Any]:
    """
    Compute specific or multi-moment aggregations across a continuous numerical feature
    (e.g., 'What is the average profit?', 'What is the total sales amount?').
    """
    col = resolve_column_name(df, target_col)
    schema = inspect_column_types(df)
    num_cols = schema.get("numerical", [])

    if not col:
        return {
            "status": "unsupported",
            "tool": "aggregation_analysis",
            "reason": f"Column '{target_col}' was not found in the dataset.",
            "requested_metric": target_col,
            "suggested_columns": num_cols,
        }

    clean_series = pd.to_numeric(df[col], errors="coerce").dropna()
    if clean_series.empty:
        return {
            "status": "error",
            "tool": "aggregation_analysis",
            "reason": f"Column '{col}' contains no valid numerical records.",
        }

    func_map = {
        "avg": "mean",
        "average": "mean",
        "mean": "mean",
        "sum": "sum",
        "total": "sum",
        "median": "median",
        "min": "min",
        "minimum": "min",
        "max": "max",
        "maximum": "max",
        "std": "std",
        "count": "count",
    }
    normalized_func = func_map.get(str(agg_func).lower().strip(), "mean")

    mean_val = round(float(clean_series.mean()), 4)
    median_val = round(float(clean_series.median()), 4)
    sum_val = round(float(clean_series.sum()), 4)
    min_val = round(float(clean_series.min()), 4)
    max_val = round(float(clean_series.max()), 4)
    std_val = round(float(clean_series.std(ddof=1) if len(clean_series) > 1 else 0.0), 4)
    count_val = int(len(clean_series))

    primary_val_map = {
        "mean": mean_val,
        "median": median_val,
        "sum": sum_val,
        "min": min_val,
        "max": max_val,
        "std": std_val,
        "count": count_val,
    }
    primary_result = primary_val_map.get(normalized_func, mean_val)

    # Formatted display
    is_currency = any(k in col.lower() for k in ("sales", "revenue", "price", "profit", "cost", "amount", "salary"))
    fmt_prefix = "$" if is_currency else ""
    fmt_str = f"{fmt_prefix}{primary_result:,.2f}" if isinstance(primary_result, float) else f"{primary_result:,}"

    return {
        "status": "success",
        "tool": "aggregation_analysis",
        "column_name": col,
        "aggregation_function": normalized_func,
        "primary_value": primary_result,
        "formatted_value": fmt_str,
        "summary_metrics": {
            "mean": mean_val,
            "median": median_val,
            "sum": sum_val,
            "min": min_val,
            "max": max_val,
            "std": std_val,
            "count": count_val,
        },
    }


def correlation_analysis(
    df: pd.DataFrame,
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    threshold: float = 0.0,
    method: str = "pearson",
) -> Dict[str, Any]:
    """
    Compute pairwise Pearson linear correlation or find strongest correlated feature pairs.
    Adheres strictly to non-causation reporting rules.
    """
    schema = inspect_column_types(df)
    num_cols = schema.get("numerical", [])

    if len(num_cols) < 2:
        return {
            "status": "unsupported",
            "tool": "correlation_analysis",
            "reason": f"Dataset contains fewer than 2 numerical features ({len(num_cols)} found). Correlation requires at least 2 continuous numerical variables.",
            "numerical_columns": num_cols,
        }

    c1 = resolve_column_name(df, col1) if col1 else None
    c2 = resolve_column_name(df, col2) if col2 else None

    # Specific pair analysis
    if c1 and c2:
        clean_df = df[[c1, c2]].dropna()
        if len(clean_df) < 3:
            return {
                "status": "error",
                "tool": "correlation_analysis",
                "reason": f"Insufficient non-null paired observations between '{c1}' and '{c2}' (n={len(clean_df)}).",
            }
        corr_val = float(clean_df[c1].corr(clean_df[c2], method="pearson"))
        abs_r = abs(corr_val)
        if abs_r >= 0.8:
            strength = "Very Strong"
        elif abs_r >= 0.6:
            strength = "Strong"
        elif abs_r >= 0.4:
            strength = "Moderate"
        elif abs_r >= 0.2:
            strength = "Weak"
        else:
            strength = "Negligible / Very Weak"

        direction = "Positive" if corr_val >= 0 else "Negative"

        return {
            "status": "success",
            "tool": "correlation_analysis",
            "analysis_type": "pairwise",
            "column_1": c1,
            "column_2": c2,
            "correlation_coefficient": round(corr_val, 4),
            "strength": strength,
            "direction": direction,
            "sample_size": len(clean_df),
            "non_causation_notice": "Correlation quantifies statistical linear co-movement only and never proves or implies causation.",
        }

    # Global ranked correlation analysis
    engine = CorrelationAnalysisEngine(df)
    corr_results = engine.analyze(threshold=threshold)
    ranked_pairs = corr_results.get("ranked_pairs", [])
    top_pos_list = corr_results.get("key_correlations", {}).get("strongest_positive", [])
    top_neg_list = corr_results.get("key_correlations", {}).get("strongest_negative", [])
    top_pos = top_pos_list[0] if top_pos_list else None
    top_neg = top_neg_list[0] if top_neg_list else None

    return {
        "status": "success",
        "tool": "correlation_analysis",
        "analysis_type": "global_ranking",
        "total_pairs_computed": len(ranked_pairs),
        "strongest_positive": top_pos,
        "strongest_negative": top_neg,
        "top_correlated_pairs": ranked_pairs[:6],
        "non_causation_notice": "All observed correlations describe statistical associations and do not represent causal relationships.",
    }


def outlier_analysis(
    df: pd.DataFrame,
    column_name: Optional[str] = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> Dict[str, Any]:
    """
    Detect anomalies and extreme outliers in numerical columns using IQR or Z-score boundaries.
    """
    schema = inspect_column_types(df)
    num_cols = schema.get("numerical", [])

    if not num_cols:
        return {
            "status": "unsupported",
            "tool": "outlier_analysis",
            "reason": "No numerical continuous columns available for outlier detection.",
        }

    engine = OutlierDetectionEngine(df)
    outlier_res = engine.analyze(method=method, param=threshold)

    # Specific column requested
    if column_name:
        col = resolve_column_name(df, column_name)
        if not col:
            return {
                "status": "unsupported",
                "tool": "outlier_analysis",
                "reason": f"Column '{column_name}' was not found in dataset.",
                "available_numerical_columns": num_cols,
            }

        col_data = outlier_res.get("column_details", {}).get(col)
        if not col_data:
            return {
                "status": "error",
                "tool": "outlier_analysis",
                "reason": f"Failed to compute outlier metrics for '{col}'.",
            }

        sample_vals = [o.get("value") for o in col_data.get("outliers", [])[:6]]

        return {
            "status": "success",
            "tool": "outlier_analysis",
            "analysis_type": "single_column",
            "column_name": col,
            "method": method.upper(),
            "outlier_count": col_data.get("outlier_count", 0),
            "outlier_percentage": col_data.get("outlier_percentage", 0.0),
            "lower_threshold": col_data.get("lower_threshold"),
            "upper_threshold": col_data.get("upper_threshold"),
            "sample_outlier_values": sample_vals,
            "severity": col_data.get("status", "Clean"),
        }

    # Dataset-wide outlier overview
    cols_dict = outlier_res.get("column_details", {})
    cols_with_outliers = [c for c in cols_dict.values() if c.get("outlier_count", 0) > 0]
    cols_with_outliers.sort(key=lambda x: x.get("outlier_count", 0), reverse=True)

    formatted_cols = []
    for c in cols_with_outliers[:6]:
        formatted_cols.append({
            "column": c.get("column_name"),
            "outlier_count": c.get("outlier_count"),
            "outlier_percentage": c.get("outlier_percentage"),
            "lower_threshold": c.get("lower_threshold"),
            "upper_threshold": c.get("upper_threshold"),
            "severity": c.get("status", "Moderate Outliers"),
        })

    return {
        "status": "success",
        "tool": "outlier_analysis",
        "analysis_type": "dataset_wide",
        "total_outliers_count": outlier_res.get("total_outlier_instances", 0),
        "columns_with_outliers_count": len(cols_with_outliers),
        "columns_analyzed": formatted_cols,
    }


def time_series_analysis(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    freq: str = "auto",
    agg_func: str = "sum",
) -> Dict[str, Any]:
    """
    Perform chronological time series analysis, calculating temporal progression,
    growth rates, peak periods, and trough periods.
    """
    schema = inspect_column_types(df)
    dt_cols = schema.get("datetime", [])
    num_cols = schema.get("numerical", [])

    resolved_dt = resolve_column_name(df, date_col) if date_col else (dt_cols[0] if dt_cols else None)
    if not resolved_dt:
        return {
            "status": "unsupported",
            "tool": "time_series_analysis",
            "reason": "No datetime or temporal column was detected in this dataset.",
            "suggested_analysis": "Use cross-sectional groupby_analysis or categorical distribution_analysis instead.",
        }

    resolved_val = resolve_column_name(df, value_col) if value_col else (num_cols[0] if num_cols else None)
    if not resolved_val:
        return {
            "status": "unsupported",
            "tool": "time_series_analysis",
            "reason": f"Numerical metric column '{value_col or 'N/A'}' was not found for time series aggregation.",
            "available_numerical_columns": num_cols,
        }

    # Prepare DataFrame
    clean_df = df[[resolved_dt, resolved_val]].copy()
    clean_df["parsed_dt"] = pd.to_datetime(clean_df[resolved_dt], errors="coerce")
    clean_df[resolved_val] = pd.to_numeric(clean_df[resolved_val], errors="coerce")
    clean_df = clean_df.dropna(subset=["parsed_dt", resolved_val]).sort_values("parsed_dt")

    if len(clean_df) < 2:
        return {
            "status": "error",
            "tool": "time_series_analysis",
            "reason": "Insufficient non-null timestamped records for time series evaluation.",
        }

    min_dt = clean_df["parsed_dt"].min()
    max_dt = clean_df["parsed_dt"].max()
    span_days = int((max_dt - min_dt).days)

    # Determine frequency
    if freq == "auto":
        if span_days > 730:
            resample_freq = "YE" if hasattr(pd.offsets, "YearEnd") else "Y"
            period_fmt = "%Y"
        elif span_days > 60:
            resample_freq = "ME" if hasattr(pd.offsets, "MonthEnd") else "M"
            period_fmt = "%Y-%m"
        else:
            resample_freq = "D"
            period_fmt = "%Y-%m-%d"
    else:
        resample_freq = freq
        period_fmt = "%Y-%m-%d"

    clean_df = clean_df.set_index("parsed_dt")
    try:
        ts_series = clean_df[resolved_val].resample(resample_freq).agg(agg_func).dropna()
    except Exception:
        ts_series = clean_df[resolved_val].resample("M").agg(agg_func).dropna()
        period_fmt = "%Y-%m"

    if ts_series.empty:
        return {
            "status": "error",
            "tool": "time_series_analysis",
            "reason": "Resampling yielded zero non-null temporal buckets.",
        }

    timeline = []
    for dt_idx, val in ts_series.items():
        dt_str = dt_idx.strftime(period_fmt)
        timeline.append({"period": dt_str, "value": round(float(val), 2)})

    peak_entry = max(timeline, key=lambda x: x["value"])
    trough_entry = min(timeline, key=lambda x: x["value"])

    first_val = timeline[0]["value"]
    last_val = timeline[-1]["value"]
    growth_pct = round(((last_val - first_val) / max(0.0001, abs(first_val))) * 100, 2) if first_val != 0 else 0.0

    trend_direction = "Upward / Growth" if growth_pct > 5.0 else ("Downward / Decline" if growth_pct < -5.0 else "Stable / Sideways")

    return {
        "status": "success",
        "tool": "time_series_analysis",
        "date_column": resolved_dt,
        "value_column": resolved_val,
        "time_span_days": span_days,
        "start_date": min_dt.strftime("%Y-%m-%d"),
        "end_date": max_dt.strftime("%Y-%m-%d"),
        "periods_count": len(timeline),
        "overall_growth_pct": growth_pct,
        "trend_direction": trend_direction,
        "peak_period": peak_entry,
        "trough_period": trough_entry,
        "timeline": timeline[:36],
    }


def distribution_analysis(
    df: pd.DataFrame,
    column_name: str,
    bins: int = 10,
) -> Dict[str, Any]:
    """
    Analyze the distribution shape, bin intervals, skewness, and spread of a numerical or categorical feature.
    """
    col = resolve_column_name(df, column_name)
    if not col:
        return {
            "status": "unsupported",
            "tool": "distribution_analysis",
            "reason": f"Column '{column_name}' was not found in the dataset.",
            "available_columns": list(df.columns),
        }

    schema = inspect_column_types(df)
    is_num = col in schema.get("numerical", []) or pd.api.types.is_numeric_dtype(df[col])

    if is_num:
        clean_s = pd.to_numeric(df[col], errors="coerce").dropna()
        if clean_s.empty:
            return {
                "status": "error",
                "tool": "distribution_analysis",
                "reason": f"Column '{col}' has no valid numeric data points.",
            }

        counts, bin_edges = np.histogram(clean_s, bins=max(3, min(bins, 25)))
        hist_bins = []
        for i in range(len(counts)):
            hist_bins.append({
                "bin_start": round(float(bin_edges[i]), 2),
                "bin_end": round(float(bin_edges[i+1]), 2),
                "range_label": f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}",
                "count": int(counts[i]),
                "percentage": round(float(counts[i] / len(clean_s)) * 100, 2),
            })

        dominant_bin = max(hist_bins, key=lambda x: x["count"])
        skew_val = round(float(clean_s.skew()), 4) if len(clean_s) > 2 else 0.0
        kurt_val = round(float(clean_s.kurtosis()), 4) if len(clean_s) > 3 else 0.0

        if abs(skew_val) < 0.5:
            skew_label = "Approximately Symmetric"
        elif skew_val > 0.5:
            skew_label = "Right-skewed (Positively skewed - tail on high values)"
        else:
            skew_label = "Left-skewed (Negatively skewed - tail on low values)"

        return {
            "status": "success",
            "tool": "distribution_analysis",
            "column_name": col,
            "column_type": "numerical",
            "sample_size": len(clean_s),
            "mean": round(float(clean_s.mean()), 4),
            "median": round(float(clean_s.median()), 4),
            "std": round(float(clean_s.std(ddof=1) if len(clean_s) > 1 else 0.0), 4),
            "skewness": skew_val,
            "skewness_classification": skew_label,
            "kurtosis": kurt_val,
            "dominant_bin": dominant_bin,
            "bins": hist_bins,
        }
    else:
        # Categorical distribution
        clean_s = df[col].dropna().astype(str)
        val_counts = clean_s.value_counts().head(10)
        cat_bins = []
        for cat, cnt in val_counts.items():
            cat_bins.append({
                "category": cat,
                "count": int(cnt),
                "percentage": round(float(cnt / max(1, len(clean_s))) * 100, 2),
            })
        return {
            "status": "success",
            "tool": "distribution_analysis",
            "column_name": col,
            "column_type": "categorical",
            "sample_size": len(clean_s),
            "unique_categories_count": int(clean_s.nunique()),
            "dominant_category": cat_bins[0] if cat_bins else None,
            "categories": cat_bins,
        }


def investigate_further(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate prioritized data-driven exploration leads based on anomalies,
    skewed features, strong correlations, and missingness signals.
    """
    recommendations = []

    # 1. Check correlations
    corr_engine = CorrelationAnalysisEngine(df)
    corr_res = corr_engine.analyze(threshold=0.0)
    top_pos_list = corr_res.get("key_correlations", {}).get("strongest_positive", [])
    if top_pos_list:
        top_pos = top_pos_list[0]
        recommendations.append({
            "topic": "Feature Co-movement & Predictive Modeling",
            "priority": "High",
            "finding": f"Strong association between `{top_pos.get('variable_a')}` and `{top_pos.get('variable_b')}` (r = {top_pos.get('correlation')}).",
            "suggested_query": "What are the strongest correlations?",
            "action": "Investigate underlying operational mechanics connecting these metrics.",
        })

    # 2. Check outliers
    outlier_engine = OutlierDetectionEngine(df)
    outlier_res = outlier_engine.analyze(method="iqr")
    cols_dict = outlier_res.get("column_details", {})
    cols_with_outliers = [c for c in cols_dict.values() if c.get("outlier_count", 0) > 0]
    if cols_with_outliers:
        top_out = cols_with_outliers[0]
        recommendations.append({
            "topic": "Statistical Outlier Remediation",
            "priority": "High",
            "finding": f"`{top_out.get('column_name')}` has {top_out.get('outlier_count')} outliers ({top_out.get('outlier_percentage')}%) violating IQR thresholds.",
            "suggested_query": "Are there unusual values?",
            "action": "Verify if high values represent VIP customers, recording anomalies, or valid variance.",
        })

    # 3. Check categorical group drivers
    schema = inspect_column_types(df)
    cat_cols = schema.get("categorical", [])
    num_cols = schema.get("numerical", [])
    if cat_cols and num_cols:
        recommendations.append({
            "topic": "Subgroup Segment Performance",
            "priority": "Medium",
            "finding": f"High diversity in `{cat_cols[0]}` across primary metric `{num_cols[0]}`.",
            "suggested_query": f"Which {cat_cols[0]} has the highest {num_cols[0]}?",
            "action": "Perform subgroup decomposition to identify leading and lagging cohorts.",
        })

    # 4. Datetime trend decomposition
    dt_cols = schema.get("datetime", [])
    if dt_cols and num_cols:
        recommendations.append({
            "topic": "Temporal Dynamics & Seasonality",
            "priority": "Medium",
            "finding": f"Timestamp feature `{dt_cols[0]}` permits longitudinal analysis.",
            "suggested_query": f"How has {num_cols[0]} changed over time?",
            "action": "Evaluate monthly trends, peak velocity periods, and seasonal dips.",
        })

    return {
        "status": "success",
        "tool": "investigate_further",
        "total_recommendations": len(recommendations),
        "recommendations": recommendations,
    }


# ==============================================================================
# 3. DETERMINISTIC INTENT / TOOL ROUTER (ZERO-API-KEY FALLBACK)
# ==============================================================================

def route_query_deterministic(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Intelligent NLP rule-based intent router.
    Parses natural language queries, resolves columns via fuzzy matching,
    and returns tool selection and parameters.
    """
    q = query.lower().strip()
    schema = inspect_column_types(df)
    num_cols = schema.get("numerical", [])
    cat_cols = schema.get("categorical", []) + schema.get("boolean", [])
    dt_cols = schema.get("datetime", [])

    # 1. Dataset overview / summary intent
    if any(k in q for k in ("overview", "how many rows", "how many columns", "shape of", "dataset summary", "describe dataset", "what is in this data", "tell me about this dataset")):
        return {"tool": "dataset_summary", "params": {}}

    # 2. What should I investigate / next steps
    if any(k in q for k in ("investigate", "recommend", "next step", "what should i look at", "further analysis", "explore further", "what to explore")):
        return {"tool": "investigate_further", "params": {}}

    # 3. Outlier / anomaly intent
    if any(k in q for k in ("unusual", "outlier", "anomal", "extreme value", "odd value", "abnormal", "irregular")):
        # Check if a specific column is referenced
        target_col = None
        for col in df.columns:
            if normalize_token(col) in normalize_token(q) or col.lower() in q:
                target_col = col
                break
        return {"tool": "outlier_analysis", "params": {"column_name": target_col}}

    # 4. Correlation / relationship intent
    if any(k in q for k in ("correlation", "correlate", "relationship", "relate", "association", "linked with", "depend on", "co-movement")):
        matched_cols = []
        for col in df.columns:
            if normalize_token(col) in normalize_token(q) or col.lower() in q:
                matched_cols.append(col)
        c1 = matched_cols[0] if len(matched_cols) > 0 else None
        c2 = matched_cols[1] if len(matched_cols) > 1 else None
        return {"tool": "correlation_analysis", "params": {"col1": c1, "col2": c2}}

    # 5. Time series / trend intent
    if any(k in q for k in ("over time", "trend", "change over", "changed over", "monthly", "daily", "yearly", "historical", "time series", "progression", "timeline", "by date", "by month")):
        metric_col = None
        for col in num_cols:
            if normalize_token(col) in normalize_token(q) or col.lower() in q:
                metric_col = col
                break
        if not metric_col and ("sales" in q or "revenue" in q):
            metric_col = resolve_column_name(df, "sales")
        date_col = dt_cols[0] if dt_cols else None
        return {"tool": "time_series_analysis", "params": {"date_col": date_col, "value_col": metric_col}}

    # 6. Distribution / spread / histogram intent
    if any(k in q for k in ("distribution", "spread", "histogram", "skew", "kurtosis", "range of", "dispersion")):
        matched_col = None
        for col in df.columns:
            if normalize_token(col) in normalize_token(q) or col.lower() in q:
                matched_col = col
                break
        if not matched_col:
            matched_col = num_cols[0] if num_cols else df.columns[0]
        return {"tool": "distribution_analysis", "params": {"column_name": matched_col}}

    # 7. Groupby / Top performer / Highest / Lowest per category intent
    # e.g., "Which category has the highest sales?", "Which region performs best?", "Which products have highest revenue?"
    if any(k in q for k in ("which ", "highest", "lowest", "best", "worst", "top ", "per ", "by ", "most ", "least ", "breakdown by", "compare")):
        # Find group column
        group_candidate = None
        for col in cat_cols:
            if normalize_token(col) in normalize_token(q) or col.lower() in q:
                group_candidate = col
                break

        # Check semantic aliases for group
        if not group_candidate:
            if "category" in q:
                group_candidate = resolve_column_name(df, "Category")
            elif "region" in q:
                group_candidate = resolve_column_name(df, "Region")
            elif "product" in q or "item" in q:
                group_candidate = resolve_column_name(df, "Product")
            elif "payment" in q:
                group_candidate = resolve_column_name(df, "Payment_Method")

        # Find target numerical metric
        metric_candidate = None
        for col in num_cols:
            if normalize_token(col) in normalize_token(q) or col.lower() in q:
                metric_candidate = col
                break

        if not metric_candidate:
            if "sales" in q or "highest" in q or "perform" in q:
                metric_candidate = resolve_column_name(df, "Sales_Amount") or (num_cols[0] if num_cols else None)
            elif "profit" in q:
                metric_candidate = resolve_column_name(df, "Profit")
            elif "revenue" in q:
                metric_candidate = resolve_column_name(df, "Revenue") or resolve_column_name(df, "Sales_Amount")

        # Determine sort direction & agg function
        sort_desc = not any(k in q for k in ("lowest", "worst", "least", "bottom", "smallest", "minimum"))
        agg_func = "mean" if any(k in q for k in ("average", "mean", "avg")) else "sum"

        if group_candidate:
            return {
                "tool": "groupby_analysis",
                "params": {
                    "group_by_col": group_candidate,
                    "target_col": metric_candidate,
                    "agg_func": agg_func,
                    "sort_desc": sort_desc,
                }
            }

    # 8. Single Aggregation intent
    # e.g., "What is the average profit?", "What is total sales?", "What is the max customer age?"
    if any(k in q for k in ("average", "mean", "avg", "total", "sum", "median", "minimum", "min ", "maximum", "max ")):
        agg_func = "mean"
        if any(k in q for k in ("total", "sum")):
            agg_func = "sum"
        elif "median" in q:
            agg_func = "median"
        elif any(k in q for k in ("minimum", "min ")):
            agg_func = "min"
        elif any(k in q for k in ("maximum", "max ")):
            agg_func = "max"

        matched_col = None
        for col in df.columns:
            if normalize_token(col) in normalize_token(q) or col.lower() in q:
                matched_col = col
                break

        # Check semantic search if literal column not found
        if not matched_col:
            for alias_key in ("profit", "sales", "revenue", "discount", "age", "quantity", "price"):
                if alias_key in q:
                    matched_col = resolve_column_name(df, alias_key)
                    if not matched_col and alias_key == "profit":
                        # Explicit check for missing profit column
                        matched_col = "profit"
                    break

        if matched_col:
            return {
                "tool": "aggregation_analysis",
                "params": {"target_col": matched_col, "agg_func": agg_func},
            }

    # 9. Single column summary intent
    for col in df.columns:
        if normalize_token(col) in normalize_token(q) or col.lower() in q:
            return {"tool": "column_summary", "params": {"column_name": col}}

    # 10. Fallback: Dataset Summary
    return {"tool": "dataset_summary", "params": {}}


# ==============================================================================
# 4. DETERMINISTIC EXPLANATION SYNTHESIZER (ZERO-HALLUCINATION FALLBACK)
# ==============================================================================

def synthesize_tool_explanation(
    query: str,
    tool_name: str,
    tool_params: Dict[str, Any],
    tool_result: Dict[str, Any],
    df: pd.DataFrame,
) -> Tuple[str, List[str]]:
    """
    Generate authentic, factually grounded natural language explanation from deterministic tool output.
    Guarantees 100% adherence to zero-hallucination mandate and returns 2-3 logical follow-up questions.
    """
    status = tool_result.get("status", "success")

    # -------------------------------------------------------------
    # Case A: Unsupported Query (e.g. Missing Column)
    # -------------------------------------------------------------
    if status == "unsupported":
        reason = tool_result.get("reason", "The requested analysis is not supported by the available columns.")
        avail_cols = list(df.columns)
        num_cols = inspect_column_types(df).get("numerical", [])
        cat_cols = inspect_column_types(df).get("categorical", [])

        explanation = f"""### ⚠️ Analysis Limitation
**{reason}**

#### Available Dataset Features:
- **Numerical Metrics**: {", ".join([f"`{c}`" for c in num_cols]) if num_cols else "None"}
- **Categorical Dimensions**: {", ".join([f"`{c}`" for c in cat_cols]) if cat_cols else "None"}

You can rephrase your question using any of the available columns listed above."""

        followups = [
            f"What is the average {num_cols[0]}?" if num_cols else "Give me an overview of the dataset",
            f"Which {cat_cols[0]} has the highest {num_cols[0]}?" if (cat_cols and num_cols) else "What are the strongest correlations?",
            "What should I investigate further?",
        ]
        return explanation, followups

    # -------------------------------------------------------------
    # Case B: Error
    # -------------------------------------------------------------
    if status == "error":
        reason = tool_result.get("reason", "An error occurred during deterministic calculation.")
        return f"### ❌ Calculation Error\n\n{reason}", ["Give me an overview of the dataset", "What should I investigate further?"]

    # -------------------------------------------------------------
    # Case C: Groupby Analysis
    # -------------------------------------------------------------
    if tool_name == "groupby_analysis":
        group_col = tool_result.get("group_by_column")
        metric_col = tool_result.get("target_column")
        agg_func = tool_result.get("aggregation_function")
        top_p = tool_result.get("top_performer")
        bot_p = tool_result.get("bottom_performer")
        groups = tool_result.get("groups", [])

        top_desc = f"**`{top_p.get('group')}`** with a {agg_func} of **{top_p.get('formatted_value')}**" if top_p else "N/A"
        pct_str = f" ({top_p.get('percentage_of_total')}% of total)" if top_p and top_p.get("percentage_of_total") is not None else ""

        table_rows = ["| Rank | " + str(group_col) + " | " + f"{agg_func.capitalize()} {metric_col}" + " | Share |", "| :---: | :--- | :---: | :---: |"]
        for g in groups[:5]:
            share_text = f"{g.get('percentage_of_total')}%" if g.get('percentage_of_total') is not None else "—"
            table_rows.append(f"| #{g.get('rank')} | **{g.get('group')}** | {g.get('formatted_value')} | {share_text} |")
        table_md = "\n".join(table_rows)

        explanation = f"""### 🏆 Leading Segment: {top_desc}{pct_str}

Based on the verified calculations grouping by **`{group_col}`** against **`{metric_col}`**:
- The highest performing group is {top_desc}{pct_str}.
{f"- The lowest performing group is **`{bot_p.get('group')}`** with **{bot_p.get('formatted_value')}**." if bot_p else ""}
- Total distinct **`{group_col}`** groups analyzed: **{tool_result.get('total_groups_count')}**.

#### Performance Breakdown:
{table_md}
"""
        followups = [
            f"What is the average {metric_col} overall?",
            f"Are there unusual values in {metric_col}?",
            f"How has {metric_col} changed over time?" if inspect_column_types(df).get("datetime") else "What are the strongest correlations?",
        ]
        return explanation, followups

    # -------------------------------------------------------------
    # Case D: Aggregation Analysis
    # -------------------------------------------------------------
    if tool_name == "aggregation_analysis":
        col_name = tool_result.get("column_name")
        func_name = tool_result.get("aggregation_function")
        fmt_val = tool_result.get("formatted_value")
        metrics = tool_result.get("summary_metrics", {})

        explanation = f"""### 📊 {func_name.capitalize()} {col_name}: **{fmt_val}**

The deterministic calculation for **`{col_name}`** yields:
- **{func_name.capitalize()}**: **{fmt_val}** (calculated over {metrics.get('count', 0):,} records)
- **Median**: **{metrics.get('median'):,}** | **Standard Deviation**: **{metrics.get('std'):,}**
- **Recorded Range**: Min: **{metrics.get('min'):,}** to Max: **{metrics.get('max'):,}**
- **Total Aggregate (Sum)**: **{metrics.get('sum'):,}**
"""
        schema = inspect_column_types(df)
        cat_cols = schema.get("categorical", [])
        followups = [
            f"Which {cat_cols[0]} has the highest {col_name}?" if cat_cols else f"What is the distribution of {col_name}?",
            f"Are there unusual values in {col_name}?",
            "What should I investigate further?",
        ]
        return explanation, followups

    # -------------------------------------------------------------
    # Case E: Correlation Analysis
    # -------------------------------------------------------------
    if tool_name == "correlation_analysis":
        if tool_result.get("analysis_type") == "pairwise":
            c1 = tool_result.get("column_1")
            c2 = tool_result.get("column_2")
            r_val = tool_result.get("correlation_coefficient")
            strength = tool_result.get("strength")
            direction = tool_result.get("direction")

            explanation = f"""### 🔗 Correlation: **r = {r_val}** ({strength} {direction})

The Pearson linear correlation between **`{c1}`** and **`{c2}`** is **r = {r_val}** (evaluated across n = {tool_result.get('sample_size'):,} pairs).
- **Strength & Direction**: **{strength} {direction.lower()} association**.
- **Interpretation**: Higher values of `{c1}` tend to co-occur with {direction.lower()} values of `{c2}`.

> **Methodological Grounding Notice**: Correlation measures statistical linear association only and never indicates a direct causal relationship.
"""
        else:
            top_pos = tool_result.get("strongest_positive")
            top_neg = tool_result.get("strongest_negative")
            pairs = tool_result.get("top_correlated_pairs", [])

            pos_desc = f"**`{top_pos.get('variable_a')}` & `{top_pos.get('variable_b')}`** (r = **+{top_pos.get('correlation')}**)" if top_pos else "None"
            neg_desc = f"**`{top_neg.get('variable_a')}` & `{top_neg.get('variable_b')}`** (r = **{top_neg.get('correlation')}**)" if top_neg else "None"

            pairs_list = "\n".join([
                f"- **`{p.get('variable_a')}` & `{p.get('variable_b')}`**: Pearson r = **{p.get('correlation')}** ({p.get('strength') or 'Moderate'} {p.get('direction')})"
                for p in pairs[:4]
            ])

            explanation = f"""### 🔗 Top Linear Correlations

- **Strongest Positive Association**: {pos_desc}
- **Strongest Negative Association**: {neg_desc}

#### Key Correlated Feature Pairs:
{pairs_list}

> **Methodological Grounding Notice**: All reported correlations indicate empirical co-movement and do not imply causal mechanisms.
"""
        followups = [
            "Are there unusual values in these columns?",
            "What should I investigate further?",
            "Give me an overview of the dataset",
        ]
        return explanation, followups

    # -------------------------------------------------------------
    # Case F: Outlier Analysis
    # -------------------------------------------------------------
    if tool_name == "outlier_analysis":
        if tool_result.get("analysis_type") == "single_column":
            col_name = tool_result.get("column_name")
            cnt = tool_result.get("outlier_count", 0)
            pct = tool_result.get("outlier_percentage", 0.0)
            lb = tool_result.get("lower_threshold")
            ub = tool_result.get("upper_threshold")
            samples = tool_result.get("sample_outlier_values", [])

            explanation = f"""### ⚠️ Outlier Report for `{col_name}`: **{cnt} Anomalies** ({pct}%)

Statistical anomaly detection using the **IQR (1.5x) Method** identified **{cnt} outliers** ({pct}% of records) outside the normal bounds:
- **Valid Statistical Range**: [Lower Threshold: **{lb}**, Upper Threshold: **{ub}**]
- **Sample Outlier Values**: {", ".join([f"**{v}**" for v in samples]) if samples else "None"}
- **Assessment**: {'Outliers indicate extreme tails or potential data recording errors that may skew downstream averages.' if cnt > 0 else 'Values fall neatly within normal distribution bounds.'}
"""
        else:
            total_cnt = tool_result.get("total_outliers_count", 0)
            cols_cnt = tool_result.get("columns_with_outliers_count", 0)
            cols = tool_result.get("columns_analyzed", [])

            cols_list = "\n".join([
                f"- **`{c.get('column')}`**: **{c.get('outlier_count')} outliers** ({c.get('outlier_percentage')}%) | Normal bounds: [{c.get('lower_threshold')}, {c.get('upper_threshold')}]"
                for c in cols
            ]) if cols else "- No statistical outliers detected beyond 1.5x IQR thresholds."

            explanation = f"""### ⚠️ Dataset Outlier Scan: **{total_cnt} Total Anomalies** Across {cols_cnt} Columns

The automated scan evaluated all numerical dimensions for statistical anomalies:
{cols_list}

**Recommendation**: Consider winsorizing or filtering extreme values in high-outlier columns prior to regression modeling.
"""
        followups = [
            "What should I investigate further?",
            "What are the strongest correlations?",
            "What is the average profit?" if "profit" in df.columns else "Give me an overview of the dataset",
        ]
        return explanation, followups

    # -------------------------------------------------------------
    # Case G: Time Series Analysis
    # -------------------------------------------------------------
    if tool_name == "time_series_analysis":
        date_col = tool_result.get("date_column")
        val_col = tool_result.get("value_column")
        growth = tool_result.get("overall_growth_pct", 0.0)
        direction = tool_result.get("trend_direction", "Stable")
        peak = tool_result.get("peak_period", {})
        trough = tool_result.get("trough_period", {})

        explanation = f"""### 📈 Temporal Trend for `{val_col}`: **{direction}** ({growth:+.2f}%)

Evaluating chronological progression along **`{date_col}`** over **{tool_result.get('time_span_days')} days** ({tool_result.get('start_date')} to {tool_result.get('end_date')}):
- **Overall Trajectory**: **{direction}** with an overall net change of **{growth:+.2f}%**.
- **Peak Period**: **{peak.get('period')}** reaching **{peak.get('value'):,}**.
- **Trough Period**: **{trough.get('period')}** at **{trough.get('value'):,}**.
- **Total Periodic Intervals**: **{tool_result.get('periods_count')}** intervals evaluated.
"""
        followups = [
            f"Which category has the highest {val_col}?" if inspect_column_types(df).get("categorical") else f"What is the distribution of {val_col}?",
            f"Are there unusual values in {val_col}?",
            "What should I investigate further?",
        ]
        return explanation, followups

    # -------------------------------------------------------------
    # Case H: Distribution Analysis
    # -------------------------------------------------------------
    if tool_name == "distribution_analysis":
        col_name = tool_result.get("column_name")
        c_type = tool_result.get("column_type")
        if c_type == "numerical":
            mean_v = tool_result.get("mean")
            med_v = tool_result.get("median")
            skew_v = tool_result.get("skewness")
            skew_class = tool_result.get("skewness_classification")
            dom_bin = tool_result.get("dominant_bin", {})

            explanation = f"""### 📊 Distribution of `{col_name}`: **{skew_class}**

- **Central Tendency**: Mean = **{mean_v:,}** | Median = **{med_v:,}** | Std Dev = **{tool_result.get('std'):,}**
- **Shape**: Skewness = **{skew_v}** ({skew_class}) | Kurtosis = **{tool_result.get('kurtosis')}**
- **Dominant Concentration Range**: **{dom_bin.get('range_label')}** containing **{dom_bin.get('count'):,} rows** ({dom_bin.get('percentage')}% of dataset).
"""
        else:
            dom_cat = tool_result.get("dominant_category", {})
            explanation = f"""### 📊 Distribution of `{col_name}`: **{tool_result.get('unique_categories_count')} Unique Classes**

- **Dominant Category**: **'{dom_cat.get('category')}'** representing **{dom_cat.get('percentage')}%** ({dom_cat.get('count'):,} rows).
- **Total Categories**: **{tool_result.get('unique_categories_count')}**.
"""
        followups = [
            f"Are there unusual values in {col_name}?",
            "What are the strongest correlations?",
            "What should I investigate further?",
        ]
        return explanation, followups

    # -------------------------------------------------------------
    # Case I: Dataset Summary
    # -------------------------------------------------------------
    if tool_name == "dataset_summary":
        rows = tool_result.get("total_rows", 0)
        cols = tool_result.get("total_columns", 0)
        mb = tool_result.get("memory_usage_mb", 0)
        miss_pct = tool_result.get("overall_missing_pct", 0)
        dups = tool_result.get("duplicate_rows_count", 0)
        type_counts = tool_result.get("column_types_count", {})

        explanation = f"""### 📋 Dataset Profile Overview

- **Dimensions**: **{rows:,} rows** × **{cols} columns** (~**{mb} MB** memory footprint).
- **Data Integrity**: **{miss_pct}% overall missing cells** | **{dups} duplicate rows**.
- **Feature Breakdown**: {type_counts.get('numerical', 0)} numerical, {type_counts.get('categorical', 0)} categorical, {type_counts.get('datetime', 0)} datetime, {type_counts.get('boolean', 0)} boolean.

#### Available Columns:
{", ".join([f"`{c['name']}` ({c['type']})" for c in tool_result.get('columns', [])])}
"""
        schema = inspect_column_types(df)
        num_cols = schema.get("numerical", [])
        cat_cols = schema.get("categorical", [])
        followups = [
            f"Which {cat_cols[0]} has the highest {num_cols[0]}?" if (cat_cols and num_cols) else "What are the strongest correlations?",
            f"What is the average {num_cols[0]}?" if num_cols else "Are there unusual values?",
            "What should I investigate further?",
        ]
        return explanation, followups

    # -------------------------------------------------------------
    # Case J: Investigate Further
    # -------------------------------------------------------------
    if tool_name == "investigate_further":
        recs = tool_result.get("recommendations", [])
        recs_list = "\n".join([
            f"#### {i+1}. [{r.get('priority')} Priority] {r.get('topic')}\n- **Signal**: {r.get('finding')}\n- **Action**: {r.get('action')}\n- **Quick Ask**: *\"{r.get('suggested_query')}\"*"
            for i, r in enumerate(recs)
        ]) if recs else "- The dataset is homogeneous with no critical anomalies detected."

        explanation = f"""### 🔍 Prioritized Investigation Recommendations

Based on statistical signals, outlier bounds, correlation matrices, and distribution skews across the dataset:

{recs_list}
"""
        followups = [
            recs[0].get("suggested_query") if len(recs) > 0 else "What are the strongest correlations?",
            recs[1].get("suggested_query") if len(recs) > 1 else "Are there unusual values?",
            "Give me an overview of the dataset",
        ]
        return explanation, followups

    # Fallback
    return f"### Analytical Result\n\n```json\n{json.dumps(tool_result, indent=2, default=str)}\n```", ["What should I investigate further?"]


# ==============================================================================
# 5. VISUALIZATION SPEC BUILDER FOR Q&A
# ==============================================================================

def build_qa_visualization_spec(
    tool_name: str,
    tool_params: Dict[str, Any],
    tool_result: Dict[str, Any],
    df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """
    Generate responsive, dark-glassmorphic Plotly visualization specifications tailored to the tool result.
    Returns None if visualization is not appropriate.
    """
    if tool_result.get("status") != "success":
        return None

    DARK_LAYOUT = {
        "paper_bgcolor": "rgba(15, 23, 42, 0)",
        "plot_bgcolor": "rgba(15, 23, 42, 0.4)",
        "font": {"family": "Plus Jakarta Sans, sans-serif", "color": "#cbd5e1", "size": 12},
        "margin": {"l": 50, "r": 30, "t": 40, "b": 50},
        "xaxis": {
            "gridcolor": "rgba(255, 255, 255, 0.06)",
            "zerolinecolor": "rgba(255, 255, 255, 0.12)",
        },
        "yaxis": {
            "gridcolor": "rgba(255, 255, 255, 0.06)",
            "zerolinecolor": "rgba(255, 255, 255, 0.12)",
        },
        "autosize": True,
    }

    # 1. Groupby Analysis -> Bar Chart
    if tool_name == "groupby_analysis":
        groups = tool_result.get("groups", [])[:10]
        if not groups:
            return None
        x_vals = [g.get("group") for g in groups]
        y_vals = [g.get("value") for g in groups]
        text_vals = [g.get("formatted_value") for g in groups]
        group_col = tool_result.get("group_by_column")
        metric_col = tool_result.get("target_column")
        agg_func = tool_result.get("aggregation_function")

        data = [{
            "type": "bar",
            "x": x_vals,
            "y": y_vals,
            "text": text_vals,
            "textposition": "auto",
            "marker": {
                "color": y_vals,
                "colorscale": [[0, "#6366f1"], [1, "#a855f7"]],
                "line": {"color": "rgba(255, 255, 255, 0.2)", "width": 1},
            },
            "hovertemplate": f"<b>%{{x}}</b><br>{agg_func.capitalize()} {metric_col}: %{{text}}<extra></extra>",
        }]
        layout = dict(DARK_LAYOUT)
        layout["title"] = f"{agg_func.capitalize()} {metric_col} by {group_col}"
        return {"data": data, "layout": layout, "chart_type": "bar"}

    # 2. Time Series Analysis -> Line Chart with Area Fill
    if tool_name == "time_series_analysis":
        timeline = tool_result.get("timeline", [])
        if not timeline:
            return None
        x_vals = [t.get("period") for t in timeline]
        y_vals = [t.get("value") for t in timeline]
        val_col = tool_result.get("value_column")

        data = [{
            "type": "scatter",
            "mode": "lines+markers",
            "x": x_vals,
            "y": y_vals,
            "line": {"color": "#38bdf8", "width": 3, "shape": "spline"},
            "marker": {"color": "#6366f1", "size": 6},
            "fill": "tozeroy",
            "fillcolor": "rgba(56, 189, 248, 0.12)",
            "hovertemplate": f"Period: %{{x}}<br>{val_col}: %{{y:,.2f}}<extra></extra>",
        }]
        layout = dict(DARK_LAYOUT)
        layout["title"] = f"{val_col} Progression Over Time"
        return {"data": data, "layout": layout, "chart_type": "line"}

    # 3. Correlation Analysis -> Scatter with trendline or Heatmap
    if tool_name == "correlation_analysis" and tool_result.get("analysis_type") == "pairwise":
        c1 = tool_result.get("column_1")
        c2 = tool_result.get("column_2")
        if c1 in df.columns and c2 in df.columns:
            clean_df = df[[c1, c2]].dropna().head(300)
            data = [{
                "type": "scatter",
                "mode": "markers",
                "x": clean_df[c1].tolist(),
                "y": clean_df[c2].tolist(),
                "marker": {
                    "color": "#818cf8",
                    "size": 7,
                    "opacity": 0.7,
                    "line": {"color": "rgba(255, 255, 255, 0.3)", "width": 1},
                },
                "hovertemplate": f"{c1}: %{{x}}<br>{c2}: %{{y}}<extra></extra>",
            }]
            layout = dict(DARK_LAYOUT)
            layout["title"] = f"Correlation: {c1} vs {c2} (r = {tool_result.get('correlation_coefficient')})"
            layout["xaxis"]["title"] = c1
            layout["yaxis"]["title"] = c2
            return {"data": data, "layout": layout, "chart_type": "scatter"}

    # 4. Outlier Analysis -> Boxplot
    if tool_name == "outlier_analysis" and tool_result.get("analysis_type") == "single_column":
        col_name = tool_result.get("column_name")
        if col_name in df.columns:
            clean_s = pd.to_numeric(df[col_name], errors="coerce").dropna()
            data = [{
                "type": "box",
                "y": clean_s.tolist(),
                "name": col_name,
                "boxpoints": "outliers",
                "marker": {"color": "#ef4444", "outliercolor": "#f87171"},
                "line": {"color": "#6366f1"},
                "fillcolor": "rgba(99, 102, 241, 0.2)",
            }]
            layout = dict(DARK_LAYOUT)
            layout["title"] = f"Outlier Distribution for {col_name}"
            return {"data": data, "layout": layout, "chart_type": "box"}

    # 5. Distribution Analysis -> Histogram
    if tool_name == "distribution_analysis" and tool_result.get("column_type") == "numerical":
        bins_data = tool_result.get("bins", [])
        if bins_data:
            x_labels = [b.get("range_label") for b in bins_data]
            y_counts = [b.get("count") for b in bins_data]
            data = [{
                "type": "bar",
                "x": x_labels,
                "y": y_counts,
                "marker": {
                    "color": "#10b981",
                    "line": {"color": "rgba(255, 255, 255, 0.2)", "width": 1},
                },
                "hovertemplate": "Range: %{x}<br>Count: %{y}<extra></extra>",
            }]
            layout = dict(DARK_LAYOUT)
            layout["title"] = f"Distribution: {tool_result.get('column_name')}"
            return {"data": data, "layout": layout, "chart_type": "histogram"}

    return None


# ==============================================================================
# 6. CONVERSATION SESSION MEMORY MANAGER
# ==============================================================================

class ConversationSessionManager:
    """
    Maintains and manages in-memory multi-turn conversation session history
    per active dataset.
    """

    def __init__(self):
        # Key: dataset_id -> List[Dict[str, Any]]
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

    def get_history(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation turns for the dataset."""
        return self._sessions.get(dataset_id, [])

    def add_turn(
        self,
        dataset_id: str,
        user_query: str,
        assistant_explanation: str,
        tool_name: str,
        tool_params: Dict[str, Any],
        tool_result: Dict[str, Any],
        visualization: Optional[Dict[str, Any]] = None,
        followups: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Record a completed question-answer interaction."""
        if dataset_id not in self._sessions:
            self._sessions[dataset_id] = []

        turn_id = f"msg_{len(self._sessions[dataset_id]) + 1}_{datetime.utcnow().strftime('%H%M%S')}"
        turn_entry = {
            "id": turn_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_query": user_query,
            "assistant_explanation": assistant_explanation,
            "tool_executed": tool_name,
            "tool_params": tool_params,
            "tool_result": tool_result,
            "visualization": visualization,
            "followups": followups or [],
        }
        self._sessions[dataset_id].append(turn_entry)
        return turn_entry

    def clear_history(self, dataset_id: str) -> bool:
        """Clear conversation history for the given dataset."""
        if dataset_id in self._sessions:
            self._sessions[dataset_id] = []
            return True
        return False

    def get_suggested_questions(self, df: pd.DataFrame, dataset_id: str = "dataset") -> List[str]:
        """
        Dynamically generate tailored prompt suggestions based on actual column schema.
        """
        schema = inspect_column_types(df)
        num_cols = schema.get("numerical", [])
        cat_cols = schema.get("categorical", [])
        dt_cols = schema.get("datetime", [])

        suggestions = []

        # 1. Groupby suggestion
        if cat_cols and num_cols:
            suggestions.append(f"Which {cat_cols[0]} has the highest {num_cols[0]}?")
        elif cat_cols:
            suggestions.append(f"What is the breakdown of records by {cat_cols[0]}?")

        # 2. Aggregation suggestion
        if num_cols:
            suggestions.append(f"What is the average {num_cols[0]}?")

        # 3. Top region / performer
        region_col = resolve_column_name(df, "Region")
        sales_col = resolve_column_name(df, "Sales") or (num_cols[0] if num_cols else None)
        if region_col and sales_col:
            suggestions.append("Which region performs best?")
        elif len(cat_cols) > 1 and num_cols:
            suggestions.append(f"Which {cat_cols[1]} performs best by {num_cols[0]}?")

        # 4. Correlations
        if len(num_cols) >= 2:
            suggestions.append("What are the strongest correlations?")

        # 5. Outliers
        if num_cols:
            suggestions.append("Are there unusual values?")

        # 6. Time series
        if dt_cols and num_cols:
            suggestions.append(f"How has {num_cols[0]} changed over time?")

        # 7. Products / High revenue
        prod_col = resolve_column_name(df, "Product")
        if prod_col:
            suggestions.append("Which products have the highest revenue?")

        # 8. Investigation lead
        suggestions.append("What should I investigate further?")

        return suggestions[:8]


# Global session manager instance
session_manager = ConversationSessionManager()


# ==============================================================================
# 7. MAIN QUESTION ROUTER CLASS
# ==============================================================================

class QuestionRouter:
    """
    Main Natural-Language Dataset Q&A Router.
    Routes user questions to deterministic tools, executes calculations,
    and synthesizes grounded explanations with optional visualizations.
    """

    AVAILABLE_TOOLS = {
        "dataset_summary": dataset_summary,
        "column_summary": column_summary,
        "groupby_analysis": groupby_analysis,
        "aggregation_analysis": aggregation_analysis,
        "correlation_analysis": correlation_analysis,
        "outlier_analysis": outlier_analysis,
        "time_series_analysis": time_series_analysis,
        "distribution_analysis": distribution_analysis,
        "investigate_further": investigate_further,
    }

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = (provider or Config.AI_PROVIDER or "deterministic_fallback").lower()
        self.api_key = api_key
        self.model = model or Config.LLM_MODEL or "gemini-1.5-pro"

        if not self.api_key:
            if self.provider == "gemini":
                self.api_key = Config.GEMINI_API_KEY
            elif self.provider == "openai":
                self.api_key = Config.OPENAI_API_KEY
            elif self.provider == "anthropic":
                self.api_key = Config.ANTHROPIC_API_KEY

    def is_provider_configured(self, provider: Optional[str] = None) -> bool:
        """Check if LLM provider has an active API key."""
        p = (provider or self.provider).lower()
        if p == "deterministic_fallback":
            return True
        if p == "gemini":
            k = self.api_key if p == self.provider else Config.GEMINI_API_KEY
            return bool(k and k.strip() and not k.startswith("your_"))
        if p == "openai":
            k = self.api_key if p == self.provider else Config.OPENAI_API_KEY
            return bool(k and k.strip() and not k.startswith("your_"))
        if p == "anthropic":
            k = self.api_key if p == self.provider else Config.ANTHROPIC_API_KEY
            return bool(k and k.strip() and not k.startswith("your_"))
        return False

    def ask(
        self,
        df: pd.DataFrame,
        query: str,
        dataset_id: str = "dataset",
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience alias for route_and_execute."""
        return self.route_and_execute(df, query, dataset_id=dataset_id, provider=provider, model=model)

    def answer_question(
        self,
        df: pd.DataFrame,
        query: str,
        dataset_id: str = "dataset",
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience alias for route_and_execute."""
        return self.route_and_execute(df, query, dataset_id=dataset_id, provider=provider, model=model)

    def route_and_execute(
        self,
        df: pd.DataFrame,
        query: str,
        dataset_id: str = "dataset",
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Complete end-to-end pipeline:
        1. Select tool via LLM or Deterministic NLP Router
        2. Execute tool safely on DataFrame
        3. Synthesize factually grounded explanation
        4. Generate interactive Plotly visualization spec
        5. Record conversation turn in session memory
        """
        start_time = datetime.utcnow()
        active_prov = (provider or self.provider).lower()
        active_model = model or self.model

        logger.info("Routing user question '%s' for dataset '%s' (Provider: %s)", query, dataset_id, active_prov)

        # 1. Intent Routing & Tool Selection
        routed_intent = None
        routing_mode = "deterministic_nlp"

        if self.is_provider_configured(active_prov) and active_prov != "deterministic_fallback":
            try:
                routed_intent = self._llm_route_intent(query, df, dataset_id, provider=active_prov, model=active_model)
                routing_mode = f"llm_{active_prov}"
            except Exception as e:
                logger.warning("LLM intent routing failed (%s). Falling back to deterministic NLP router.", e)
                routed_intent = route_query_deterministic(query, df)
                routing_mode = "fallback_nlp"
        else:
            routed_intent = route_query_deterministic(query, df)

        tool_name = routed_intent.get("tool", "dataset_summary")
        tool_params = routed_intent.get("params", {})

        # Handle unsupported tool request directly from router
        if tool_name == "unsupported_analysis":
            tool_result = {
                "status": "unsupported",
                "tool": "unsupported_analysis",
                "reason": tool_params.get("reason", "Analysis not supported with available columns."),
                "suggested_columns": tool_params.get("suggested_columns", list(df.columns)),
            }
        else:
            # 2. Execute deterministic calculation tool
            tool_func = self.AVAILABLE_TOOLS.get(tool_name, dataset_summary)
            try:
                tool_result = tool_func(df, **tool_params)
            except Exception as e:
                logger.error("Error executing tool %s: %s", tool_name, e, exc_info=True)
                tool_result = {
                    "status": "error",
                    "tool": tool_name,
                    "reason": f"Tool execution failed: {str(e)}",
                }

        # 3. Grounded Explanation Synthesis
        explanation = ""
        followups = []
        explanation_mode = "deterministic"

        if self.is_provider_configured(active_prov) and active_prov != "deterministic_fallback" and tool_result.get("status") == "success":
            try:
                explanation, followups = self._llm_explain_result(query, tool_name, tool_params, tool_result, df, provider=active_prov, model=active_model)
                explanation_mode = f"llm_{active_prov}"
            except Exception as e:
                logger.warning("LLM explanation synthesis failed (%s). Falling back to deterministic explainer.", e)
                explanation, followups = synthesize_tool_explanation(query, tool_name, tool_params, tool_result, df)
                explanation_mode = "fallback_deterministic"
        else:
            explanation, followups = synthesize_tool_explanation(query, tool_name, tool_params, tool_result, df)

        # 4. Optional Visualization Spec
        viz_spec = build_qa_visualization_spec(tool_name, tool_params, tool_result, df)

        duration_sec = round((datetime.utcnow() - start_time).total_seconds(), 3)

        # 5. Record to session history
        turn_data = session_manager.add_turn(
            dataset_id=dataset_id,
            user_query=query,
            assistant_explanation=explanation,
            tool_name=tool_name,
            tool_params=tool_params,
            tool_result=tool_result,
            visualization=viz_spec,
            followups=followups,
        )

        return {
            "success": True,
            "turn_id": turn_data.get("id"),
            "query": query,
            "dataset_id": dataset_id,
            "tool_executed": tool_name,
            "tool_parameters": tool_params,
            "tool_result": tool_result,
            "explanation": explanation,
            "visualization": viz_spec,
            "followups": followups,
            "metadata": {
                "routing_mode": routing_mode,
                "explanation_mode": explanation_mode,
                "duration_seconds": duration_sec,
                "grounding": "Strictly Grounded in Python Execution (Zero Hallucinations)",
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

    # -------------------------------------------------------------
    # LLM Router & Explainer Internals
    # -------------------------------------------------------------
    def _llm_route_intent(
        self,
        query: str,
        df: pd.DataFrame,
        dataset_id: str,
        provider: str,
        model: str,
    ) -> Dict[str, Any]:
        """Ask LLM to return tool name and parameters in JSON format."""
        schema_info = get_dataset_column_schema(df)
        history = session_manager.get_history(dataset_id)
        prompt = build_qa_router_prompt(query, schema_info, history)

        response_text = self._dispatch_llm(prompt, system_prompt=QA_ROUTER_SYSTEM_PROMPT, provider=provider, model=model)

        # Strip markdown code blocks if wrapped
        clean_json_str = response_text.strip()
        if clean_json_str.startswith("```"):
            clean_json_str = re.sub(r"^```(?:json)?\n?", "", clean_json_str)
            clean_json_str = re.sub(r"\n?```$", "", clean_json_str).strip()

        parsed = json.loads(clean_json_str)
        if isinstance(parsed, dict) and "tool" in parsed:
            return parsed
        raise ValueError(f"Invalid LLM router format: {response_text}")

    def _llm_explain_result(
        self,
        query: str,
        tool_name: str,
        tool_params: Dict[str, Any],
        tool_result: Dict[str, Any],
        df: pd.DataFrame,
        provider: str,
        model: str,
    ) -> Tuple[str, List[str]]:
        """Ask LLM to explain the deterministic calculation result."""
        schema_info = get_dataset_column_schema(df)
        prompt = build_qa_explainer_prompt(query, tool_name, tool_params, tool_result, schema_info)

        response_text = self._dispatch_llm(prompt, system_prompt=QA_EXPLAINER_SYSTEM_PROMPT, provider=provider, model=model)

        # Extract follow-up bullet suggestions if present
        followups = []
        lines = response_text.splitlines()
        for line in lines:
            if line.strip().startswith("- ") and "?" in line:
                q_text = line.strip().lstrip("- *").strip().strip('"').strip("'")
                if len(q_text) > 8 and q_text not in followups:
                    followups.append(q_text)

        if not followups:
            # Fallback default followups
            _, fallback_followups = synthesize_tool_explanation(query, tool_name, tool_params, tool_result, df)
            followups = fallback_followups

        return response_text, followups[:3]

    def _dispatch_llm(self, user_prompt: str, system_prompt: str, provider: str, model: str) -> str:
        """Dispatch prompt to configured LLM API provider."""
        p = provider.lower()
        if p == "gemini":
            api_key = self.api_key or Config.GEMINI_API_KEY
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                genai_model = genai.GenerativeModel(
                    model_name=model if "gemini" in model else "gemini-1.5-pro",
                    system_instruction=system_prompt,
                )
                res = genai_model.generate_content(
                    user_prompt,
                    generation_config={"temperature": 0.1, "max_output_tokens": 2048},
                )
                if res and res.text:
                    return res.text
            except ImportError:
                pass

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model if 'gemini' in model else 'gemini-1.5-pro'}:generateContent?key={api_key}"
            payload = {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
            }
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            raise ValueError(f"Empty Gemini response: {data}")

        elif p == "openai":
            api_key = self.api_key or Config.OPENAI_API_KEY
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model or "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 2048,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

        elif p == "anthropic":
            api_key = self.api_key or Config.ANTHROPIC_API_KEY
            url = "https://api.anthropic.com/v1/messages"
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
            payload = {
                "model": model or "claude-3-5-sonnet-20241022",
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.1,
                "max_tokens": 2048,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data["content"][0]["text"]

        raise ValueError(f"Unsupported LLM provider '{provider}'")


# ==============================================================================
# 8. CONVENIENCE FUNCTION
# ==============================================================================

def route_user_question(
    query: str,
    dataset_id: str,
    df: Optional[pd.DataFrame] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Primary interface for routing user question on a dataset.
    Loads dataset if not passed and runs QuestionRouter.
    """
    if df is None:
        loaded_df, err = load_dataset(dataset_id)
        if err or loaded_df is None:
            return {
                "success": False,
                "error": f"Failed to load dataset '{dataset_id}': {err}",
            }
        df = loaded_df

    router = QuestionRouter(provider=provider, model=model)
    return router.route_and_execute(df, query, dataset_id=dataset_id, provider=provider, model=model)
