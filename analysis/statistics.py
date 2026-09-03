"""
Statistical Analysis Engine for AI Data Analyst Agent (Phase 5).
Performs pure Python deterministic descriptive, inferential, and distribution statistics.
Calculates parametric and non-parametric moments, quartiles, IQR, skewness, kurtosis,
confidence intervals, percentiles, categorical distributions, datetime periodicity,
and rule-based domain statistical observations without relying on LLMs.
"""

import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from analysis.profiler import infer_column_type, infer_column_types

logger = logging.getLogger(__name__)


def _safe_json_value(val: Any) -> Any:
    """
    Ensure any numpy/pandas scalar or object is safely serializable to standard JSON.
    Converts NaN/Inf to None, datetime/timestamps to ISO strings, numpy types to Python primitives.
    """
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass

    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 4)
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        try:
            return pd.to_datetime(val).isoformat()
        except Exception:
            return str(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (pd.Timedelta, np.timedelta64)):
        return str(val)
    return str(val)


def analyze_numerical_column(series: pd.Series, col_name: str = "", confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Perform deep statistical analysis on a single numerical series.
    
    Calculates:
    - Central tendency: Mean, Median, Mode
    - Dispersion: Standard Deviation, Variance, Min, Max, Range, IQR, Coefficient of Variation, SEM
    - Quartiles & Percentiles: Q1, Q2 (Median), Q3, P1, P5, P10, P25, P50, P75, P90, P95, P99
    - Higher moments: Fisher-Pearson Skewness, Fisher Excess Kurtosis
    - Inferential: Confidence Intervals for the Mean at specified confidence level (e.g. 95%)
    - Outlier bounds via 1.5x IQR and outlier counts
    - Normality assessment via D'Agostino-Pearson / Shapiro-Wilk test
    
    Robust against:
    - Missing values (NaN/None)
    - Constant columns (variance=0)
    - Small sample sizes (n=0, n=1, n=2)
    - Dirty non-numeric strings with numeric coercion
    
    Args:
        series: Pandas Series to analyze.
        col_name: Column name identifier.
        confidence_level: Desired confidence level for mean CI (default 0.95).
        
    Returns:
        Dict[str, Any]: Comprehensive numerical statistics dictionary.
    """
    name = col_name or (str(series.name) if series.name else "numerical_column")
    total_count = len(series)

    # 1. Clean and convert to numeric float series
    if pd.api.types.is_numeric_dtype(series):
        clean_s = series.dropna().astype(float)
    else:
        # Attempt cleaning symbols (e.g. $, %, commas)
        cleaned_str = series.dropna().astype(str).str.replace(r"[$,€£%\s]", "", regex=True)
        clean_s = pd.to_numeric(cleaned_str, errors="coerce").dropna().astype(float)

    n_valid = len(clean_s)
    missing_count = total_count - n_valid
    missing_percentage = round(missing_count / total_count * 100, 2) if total_count > 0 else 0.0

    # Base structure for empty or invalid series
    if n_valid == 0:
        return {
            "name": name,
            "data_type": "Numerical",
            "total_count": total_count,
            "valid_count": 0,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "is_constant": False,
            "is_empty": True,
            "mean": None,
            "median": None,
            "mode": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "range": None,
            "q1": None,
            "q2": None,
            "q3": None,
            "iqr": None,
            "skewness": None,
            "kurtosis": None,
            "confidence_interval": {
                "level": confidence_level,
                "lower": None,
                "upper": None,
                "margin_of_error": None,
            },
            "percentiles": {f"p{p}": None for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
            "outliers": {
                "lower_bound": None,
                "upper_bound": None,
                "count": 0,
                "percentage": 0.0,
            },
            "normality": {
                "is_normal": None,
                "test_statistic": None,
                "p_value": None,
                "distribution_shape": "Undefined (Empty Data)",
            },
            "dispersion_metrics": {
                "cv_percentage": None,
                "sem": None,
                "zeros_count": 0,
                "zeros_percentage": 0.0,
            },
        }

    # 2. Central Tendency & Basic Bounds
    mean_val = float(clean_s.mean())
    median_val = float(clean_s.median())
    min_val = float(clean_s.min())
    max_val = float(clean_s.max())
    range_val = float(max_val - min_val)

    # Mode calculation (handling continuous floats or multiple modes)
    try:
        mode_series = clean_s.mode()
        if len(mode_series) > 0 and len(mode_series) < n_valid:
            mode_val = float(mode_series.iloc[0])
        else:
            mode_val = median_val  # Fallback if all values distinct
    except Exception:
        mode_val = median_val

    # Constant column check
    is_constant = bool(clean_s.nunique() <= 1)

    # 3. Dispersion Metrics
    if n_valid >= 2 and not is_constant:
        std_val = float(clean_s.std(ddof=1))
        var_val = float(clean_s.var(ddof=1))
        sem_val = float(std_val / math.sqrt(n_valid))
        cv_pct = float(round((std_val / abs(mean_val) * 100), 2)) if mean_val != 0 else None
    else:
        std_val = 0.0 if is_constant else None
        var_val = 0.0 if is_constant else None
        sem_val = 0.0 if is_constant else None
        cv_pct = 0.0 if is_constant else None

    # 4. Quartiles & Percentiles
    q1_val = float(clean_s.quantile(0.25))
    q2_val = float(clean_s.quantile(0.50))
    q3_val = float(clean_s.quantile(0.75))
    iqr_val = float(q3_val - q1_val)

    pct_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentiles_dict = {}
    for p in pct_keys:
        percentiles_dict[f"p{p}"] = _safe_json_value(float(clean_s.quantile(p / 100.0)))

    # 5. Higher-Order Moments (Skewness & Kurtosis)
    # Fisher excess kurtosis: normal = 0.0
    if n_valid >= 3 and not is_constant and std_val and std_val > 0:
        try:
            skew_val = float(stats.skew(clean_s, bias=False))
            if math.isnan(skew_val) or math.isinf(skew_val):
                skew_val = 0.0
        except Exception:
            skew_val = float(clean_s.skew()) if n_valid >= 3 else 0.0

        try:
            kurt_val = float(stats.kurtosis(clean_s, bias=False, fisher=True))
            if math.isnan(kurt_val) or math.isinf(kurt_val):
                kurt_val = 0.0
        except Exception:
            kurt_val = float(clean_s.kurt()) if n_valid >= 4 else 0.0
    else:
        skew_val = 0.0 if is_constant else None
        kurt_val = 0.0 if is_constant else None

    # 6. Confidence Interval for the Mean
    ci_lower = None
    ci_upper = None
    margin_of_error = None

    if n_valid >= 2 and std_val is not None:
        if is_constant or std_val == 0.0:
            ci_lower = mean_val
            ci_upper = mean_val
            margin_of_error = 0.0
        else:
            try:
                # Student's t-distribution interval
                t_crit = stats.t.ppf((1 + confidence_level) / 2.0, df=n_valid - 1)
                margin_of_error = float(t_crit * sem_val)
                ci_lower = float(mean_val - margin_of_error)
                ci_upper = float(mean_val + margin_of_error)
            except Exception:
                # Fallback to standard normal interval
                margin_of_error = float(1.96 * sem_val)
                ci_lower = float(mean_val - margin_of_error)
                ci_upper = float(mean_val + margin_of_error)

    # 7. Outliers Analysis via 1.5x IQR
    lower_outlier_bound = float(q1_val - (1.5 * iqr_val))
    upper_outlier_bound = float(q3_val + (1.5 * iqr_val))
    outlier_mask = (clean_s < lower_outlier_bound) | (clean_s > upper_outlier_bound)
    outlier_count = int(outlier_mask.sum())
    outlier_pct = round(outlier_count / n_valid * 100, 2) if n_valid > 0 else 0.0

    # 8. Normality & Distribution Assessment
    is_normal = None
    norm_stat = None
    norm_p = None
    shape_desc = "Normal"

    if n_valid >= 8 and not is_constant:
        try:
            # D'Agostino-Pearson omnibus test
            k2, p_val = stats.normaltest(clean_s)
            norm_stat = float(k2) if not math.isnan(k2) else None
            norm_p = float(p_val) if not math.isnan(p_val) else None
            is_normal = bool(norm_p is not None and norm_p > 0.05)
        except Exception:
            try:
                # Shapiro-Wilk test for smaller sample sizes
                sh_stat, sh_p = stats.shapiro(clean_s.head(500))
                norm_stat = float(sh_stat)
                norm_p = float(sh_p)
                is_normal = bool(norm_p > 0.05)
            except Exception:
                pass

    # Characterize distribution shape
    if is_constant:
        shape_desc = "Constant (Zero Variance)"
    elif n_valid < 5:
        shape_desc = "Insufficient Sample Size"
    elif skew_val is not None and abs(skew_val) > 1.5:
        shape_desc = "Highly Skewed Right (Positive Tail)" if skew_val > 0 else "Highly Skewed Left (Negative Tail)"
    elif skew_val is not None and abs(skew_val) > 0.5:
        shape_desc = "Moderately Skewed Right" if skew_val > 0 else "Moderately Skewed Left"
    elif kurt_val is not None and kurt_val > 2.0:
        shape_desc = "Leptokurtic (Heavy Tails & Sharp Peak)"
    elif kurt_val is not None and kurt_val < -1.0:
        shape_desc = "Platykurtic (Light Tails & Flat Peak)"
    elif is_normal:
        shape_desc = "Approximately Normal (Gaussian)"
    else:
        shape_desc = "Symmetric / Non-Gaussian"

    # Zeros count
    zeros_count = int((clean_s == 0).sum())
    zeros_pct = round(zeros_count / n_valid * 100, 2) if n_valid > 0 else 0.0

    return {
        "name": name,
        "data_type": "Numerical",
        "total_count": total_count,
        "valid_count": n_valid,
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
        "is_constant": is_constant,
        "is_empty": False,
        "mean": _safe_json_value(mean_val),
        "median": _safe_json_value(median_val),
        "mode": _safe_json_value(mode_val),
        "std": _safe_json_value(std_val),
        "variance": _safe_json_value(var_val),
        "min": _safe_json_value(min_val),
        "max": _safe_json_value(max_val),
        "range": _safe_json_value(range_val),
        "q1": _safe_json_value(q1_val),
        "q2": _safe_json_value(q2_val),
        "q3": _safe_json_value(q3_val),
        "iqr": _safe_json_value(iqr_val),
        "skewness": _safe_json_value(skew_val),
        "kurtosis": _safe_json_value(kurt_val),
        "confidence_interval": {
            "level": confidence_level,
            "lower": _safe_json_value(ci_lower),
            "upper": _safe_json_value(ci_upper),
            "margin_of_error": _safe_json_value(margin_of_error),
        },
        "percentiles": percentiles_dict,
        "outliers": {
            "lower_bound": _safe_json_value(lower_outlier_bound),
            "upper_bound": _safe_json_value(upper_outlier_bound),
            "count": outlier_count,
            "percentage": outlier_pct,
        },
        "normality": {
            "is_normal": is_normal,
            "test_statistic": _safe_json_value(norm_stat),
            "p_value": _safe_json_value(norm_p),
            "distribution_shape": shape_desc,
        },
        "dispersion_metrics": {
            "cv_percentage": _safe_json_value(cv_pct),
            "sem": _safe_json_value(sem_val),
            "zeros_count": zeros_count,
            "zeros_percentage": zeros_pct,
        },
    }


def analyze_categorical_column(series: pd.Series, col_name: str = "", top_n: int = 10) -> Dict[str, Any]:
    """
    Perform statistical analysis on a categorical or discrete column.
    
    Calculates:
    - Total valid count & missing count
    - Cardinality / Number of unique categories
    - Value frequencies and percentage distribution
    - Top N categories with counts, percentages, and cumulative percentages
    - Mode category and mode dominance percentage
    - Shannon Entropy (measure of category diversity/uncertainty)
    - Rare categories count (< 1% frequency)
    
    Args:
        series: Pandas Series to analyze.
        col_name: Column name identifier.
        top_n: Number of top categories to return in detail.
        
    Returns:
        Dict[str, Any]: Categorical statistics dictionary.
    """
    name = col_name or (str(series.name) if series.name else "categorical_column")
    total_count = len(series)
    valid_series = series.dropna()
    n_valid = len(valid_series)
    missing_count = total_count - n_valid
    missing_percentage = round(missing_count / total_count * 100, 2) if total_count > 0 else 0.0

    if n_valid == 0:
        return {
            "name": name,
            "data_type": "Categorical",
            "total_count": total_count,
            "valid_count": 0,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "num_categories": 0,
            "most_frequent_category": None,
            "mode_frequency": 0,
            "mode_percentage": 0.0,
            "entropy": None,
            "rare_categories_count": 0,
            "top_categories": [],
        }

    val_counts = valid_series.value_counts()
    num_categories = int(len(val_counts))
    most_frequent = val_counts.index[0] if num_categories > 0 else None
    mode_freq = int(val_counts.iloc[0]) if num_categories > 0 else 0
    mode_pct = round(mode_freq / n_valid * 100, 2) if n_valid > 0 else 0.0

    # Shannon Entropy calculation: H = -sum(p * log2(p))
    probs = val_counts / n_valid
    entropy_val = float(-np.sum(probs * np.log2(probs + 1e-12))) if num_categories > 1 else 0.0

    # Rare categories (< 1% frequency)
    rare_count = int((probs < 0.01).sum())

    # Build top categories with cumulative share
    top_categories = []
    cum_pct = 0.0
    for cat_val, count in val_counts.head(top_n).items():
        cat_pct = round(int(count) / n_valid * 100, 2)
        cum_pct = round(cum_pct + cat_pct, 2)
        top_categories.append({
            "category": _safe_json_value(cat_val),
            "count": int(count),
            "percentage": cat_pct,
            "cumulative_percentage": min(100.0, cum_pct),
        })

    return {
        "name": name,
        "data_type": "Categorical",
        "total_count": total_count,
        "valid_count": n_valid,
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
        "num_categories": num_categories,
        "most_frequent_category": _safe_json_value(most_frequent),
        "mode_frequency": mode_freq,
        "mode_percentage": mode_pct,
        "entropy": _safe_json_value(entropy_val),
        "rare_categories_count": rare_count,
        "top_categories": top_categories,
    }


def analyze_datetime_column(series: pd.Series, col_name: str = "") -> Dict[str, Any]:
    """
    Perform temporal and frequency statistical analysis on a datetime series.
    
    Calculates:
    - Earliest & Latest date bounds
    - Total duration span (days, months, years)
    - Records over time distributions (by Year, Month, Day of Week, Hour)
    - Inferred time cadence/frequency (Daily, Weekly, Monthly, Hourly, Irregular)
    
    Args:
        series: Pandas Series to analyze.
        col_name: Column name identifier.
        
    Returns:
        Dict[str, Any]: Datetime statistics dictionary.
    """
    name = col_name or (str(series.name) if series.name else "datetime_column")
    total_count = len(series)
    valid_series = series.dropna()

    if pd.api.types.is_datetime64_any_dtype(series):
        dt_series = valid_series
    else:
        try:
            dt_series = pd.to_datetime(valid_series, errors="coerce", format="mixed").dropna()
        except Exception:
            dt_series = pd.to_datetime(valid_series, errors="coerce").dropna()

    n_valid = len(dt_series)
    missing_count = total_count - n_valid
    missing_percentage = round(missing_count / total_count * 100, 2) if total_count > 0 else 0.0

    if n_valid == 0:
        return {
            "name": name,
            "data_type": "Datetime",
            "total_count": total_count,
            "valid_count": 0,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "min_date": None,
            "max_date": None,
            "date_range_days": 0,
            "date_range_formatted": "0 days",
            "inferred_frequency": "Unknown",
            "records_by_year": {},
            "records_by_month": {},
            "records_by_day_of_week": {},
        }

    min_date = dt_series.min()
    max_date = dt_series.max()
    delta = max_date - min_date
    total_days = int(delta.days) if hasattr(delta, "days") else 0

    if total_days >= 365:
        years = round(total_days / 365.25, 1)
        range_str = f"{total_days:,} days (~{years} years)"
    elif total_days >= 30:
        months = round(total_days / 30.4, 1)
        range_str = f"{total_days:,} days (~{months} months)"
    else:
        range_str = f"{total_days:,} days"

    # Frequency / Cadence Detection
    inferred_freq = "Irregular / Mixed"
    if n_valid >= 3:
        sorted_dt = dt_series.sort_values()
        diffs = sorted_dt.diff().dropna()
        # Look at median delta
        median_sec = diffs.dt.total_seconds().median() if hasattr(diffs.dt, "total_seconds") else 0
        if 3500 <= median_sec <= 3700:
            inferred_freq = "Hourly"
        elif 82800 <= median_sec <= 90000:
            inferred_freq = "Daily"
        elif 580000 <= median_sec <= 630000:
            inferred_freq = "Weekly"
        elif 2400000 <= median_sec <= 2700000:
            inferred_freq = "Monthly"
        elif median_sec >= 30000000:
            inferred_freq = "Annual"

    # Temporal breakdowns
    try:
        years_dist = dt_series.dt.year.value_counts().sort_index().to_dict()
        years_dist = {str(k): int(v) for k, v in years_dist.items()}
    except Exception:
        years_dist = {}

    try:
        months_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                      7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        month_counts = dt_series.dt.month.value_counts().sort_index().to_dict()
        months_dist = {months_map.get(k, str(k)): int(v) for k, v in month_counts.items()}
    except Exception:
        months_dist = {}

    try:
        days_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                    4: "Friday", 5: "Saturday", 6: "Sunday"}
        dow_counts = dt_series.dt.dayofweek.value_counts().sort_index().to_dict()
        dow_dist = {days_map.get(k, str(k)): int(v) for k, v in dow_counts.items()}
    except Exception:
        dow_dist = {}

    return {
        "name": name,
        "data_type": "Datetime",
        "total_count": total_count,
        "valid_count": n_valid,
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
        "min_date": _safe_json_value(min_date),
        "max_date": _safe_json_value(max_date),
        "date_range_days": total_days,
        "date_range_formatted": range_str,
        "inferred_frequency": inferred_freq,
        "records_by_year": years_dist,
        "records_by_month": months_dist,
        "records_by_day_of_week": dow_dist,
    }


def generate_statistical_observations(
    numerical_stats: Dict[str, Any],
    categorical_stats: Dict[str, Any],
    datetime_stats: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generate deterministic, rule-based statistical observations and data insights.
    Identifies key statistical properties, anomalies, distributional shapes, and trends
    strictly in Python without calling an LLM.
    
    Returns structured observation objects with category, severity, and text description.
    """
    observations = []

    # 1. Numerical Observations
    for col_name, n_stat in numerical_stats.items():
        if n_stat.get("is_empty"):
            observations.append({
                "column": col_name,
                "type": "Quality Warning",
                "severity": "high",
                "title": f"Column '{col_name}' is Completely Empty",
                "message": f"Feature '{col_name}' contains no non-null numerical values.",
            })
            continue

        if n_stat.get("is_constant"):
            observations.append({
                "column": col_name,
                "type": "Zero Variance",
                "severity": "medium",
                "title": f"Zero-Variance Feature: '{col_name}'",
                "message": f"Column '{col_name}' is constant with uniform value {n_stat.get('mean')}. It provides zero discriminative variance.",
            })
            continue

        # Skewness insights
        skew = n_stat.get("skewness")
        mean = n_stat.get("mean")
        med = n_stat.get("median")
        if skew is not None and abs(skew) >= 1.5:
            direction = "right (positive tail)" if skew > 0 else "left (negative tail)"
            comparison = "Mean > Median" if skew > 0 else "Mean < Median"
            observations.append({
                "column": col_name,
                "type": "Distribution Shape",
                "severity": "info",
                "title": f"High Skewness Detected in '{col_name}'",
                "message": f"Feature '{col_name}' is heavily skewed to the {direction} (Skewness = {skew}, {comparison}: Mean={mean} vs Median={med}). Consider log or Box-Cox transformation for linear modeling.",
            })

        # Kurtosis & Heavy Tails
        kurt = n_stat.get("kurtosis")
        if kurt is not None and kurt > 3.0:
            observations.append({
                "column": col_name,
                "type": "Heavy Tails",
                "severity": "info",
                "title": f"Heavy-Tailed Distribution in '{col_name}'",
                "message": f"Excess kurtosis of {kurt} indicates a leptokurtic distribution with extreme outliers and heavy tails compared to normal Gaussian data.",
            })

        # Outlier Detection
        outliers = n_stat.get("outliers", {})
        outlier_cnt = outliers.get("count", 0)
        outlier_pct = outliers.get("percentage", 0.0)
        if outlier_cnt > 0 and outlier_pct >= 2.0:
            observations.append({
                "column": col_name,
                "type": "Outlier Alert",
                "severity": "warning" if outlier_pct > 5.0 else "info",
                "title": f"Statistical Outliers in '{col_name}'",
                "message": f"Found {outlier_cnt:,} outliers ({outlier_pct}%) outside the 1.5x IQR boundaries [{outliers.get('lower_bound')}, {outliers.get('upper_bound')}].",
            })

        # High Dispersion (Coefficient of Variation)
        disp = n_stat.get("dispersion_metrics", {})
        cv = disp.get("cv_percentage")
        if cv is not None and cv > 100.0:
            observations.append({
                "column": col_name,
                "type": "High Dispersion",
                "severity": "info",
                "title": f"High Relative Dispersion in '{col_name}'",
                "message": f"High coefficient of variation ({cv}%), indicating substantial standard deviation relative to the mean.",
            })

    # 2. Categorical Observations
    for col_name, c_stat in categorical_stats.items():
        mode_cat = c_stat.get("most_frequent_category")
        mode_pct = c_stat.get("mode_percentage", 0.0)
        num_cats = c_stat.get("num_categories", 0)

        # Dominant Category (Class Imbalance)
        if mode_pct >= 70.0 and num_cats > 1:
            observations.append({
                "column": col_name,
                "type": "Class Imbalance",
                "severity": "warning",
                "title": f"Dominant Category in '{col_name}'",
                "message": f"Category '{mode_cat}' dominates {mode_pct}% of the dataset across {num_cats} total categories, showing heavy category concentration.",
            })

        # High Cardinality Categorical
        if num_cats > 50:
            observations.append({
                "column": col_name,
                "type": "High Cardinality",
                "severity": "info",
                "title": f"High Cardinality in '{col_name}'",
                "message": f"Contains {num_cats} distinct categories. May require target encoding or grouping of rare categories for machine learning models.",
            })

    # 3. Datetime Observations
    for col_name, d_stat in datetime_stats.items():
        if d_stat.get("valid_count", 0) > 0:
            cadence = d_stat.get("inferred_frequency", "Mixed")
            span = d_stat.get("date_range_formatted", "")
            observations.append({
                "column": col_name,
                "type": "Temporal Span",
                "severity": "info",
                "title": f"Temporal Coverage in '{col_name}'",
                "message": f"Data spans {span} from {d_stat.get('min_date')} to {d_stat.get('max_date')} with an inferred {cadence} recording cadence.",
            })

    # Fallback if dataset is completely uniform
    if not observations:
        observations.append({
            "column": "Dataset",
            "type": "General Quality",
            "severity": "info",
            "title": "Clean Uniform Statistical Profile",
            "message": "Dataset exhibits balanced distributions without severe outliers, heavy skewness, or zero-variance features.",
        })

    return observations


class StatisticalAnalysisEngine:
    """
    Enterprise-grade Statistical Analysis Engine.
    Coordinates deep numerical moments, quartiles, confidence intervals,
    percentiles, categorical distributions, datetime temporal breakdowns,
    and rule-based statistical observations on arbitrary tabular datasets.
    """

    def __init__(self, df: pd.DataFrame, dataset_id: str = ""):
        """
        Initialize the statistical analysis engine with a DataFrame.
        
        Args:
            df: Input pandas DataFrame.
            dataset_id: Optional dataset identifier.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("StatisticalAnalysisEngine requires a pandas DataFrame instance.")
        self.df = df
        self.dataset_id = dataset_id
        self._stats_cache: Optional[Dict[str, Any]] = None

    def analyze(self, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Execute comprehensive statistical analysis across all columns in the dataset.
        
        Returns:
            Dict[str, Any]: Structured statistical analysis payload.
        """
        if self._stats_cache is not None:
            return self._stats_cache

        column_types = infer_column_types(self.df)

        numerical_results: Dict[str, Any] = {}
        categorical_results: Dict[str, Any] = {}
        datetime_results: Dict[str, Any] = {}

        for col in self.df.columns:
            col_name = str(col)
            series = self.df[col]
            ctype = column_types.get(col_name, "Other")

            if ctype == "Numerical":
                numerical_results[col_name] = analyze_numerical_column(
                    series, col_name=col_name, confidence_level=confidence_level
                )
            elif ctype == "Categorical" or ctype == "Boolean":
                categorical_results[col_name] = analyze_categorical_column(
                    series, col_name=col_name
                )
            elif ctype == "Datetime":
                datetime_results[col_name] = analyze_datetime_column(
                    series, col_name=col_name
                )
            else:
                # Other / Text fallback
                if pd.api.types.is_numeric_dtype(series):
                    numerical_results[col_name] = analyze_numerical_column(
                        series, col_name=col_name, confidence_level=confidence_level
                    )
                else:
                    categorical_results[col_name] = analyze_categorical_column(
                        series, col_name=col_name
                    )

        # Generate rule-based statistical observations
        observations = generate_statistical_observations(
            numerical_results, categorical_results, datetime_results
        )

        result = {
            "dataset_id": self.dataset_id,
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "column_types_breakdown": {
                "numerical_count": len(numerical_results),
                "categorical_count": len(categorical_results),
                "datetime_count": len(datetime_results),
            },
            "numerical_statistics": numerical_results,
            "categorical_statistics": categorical_results,
            "datetime_statistics": datetime_results,
            "statistical_observations": observations,
        }

        self._stats_cache = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Return the statistical analysis as a structured Python dictionary."""
        return self.analyze()

    def numerical_summary_dataframe(self) -> pd.DataFrame:
        """
        Export numerical descriptive statistics as a clean pandas DataFrame.
        
        Returns:
            pd.DataFrame: Tabular summary of all numerical moments and intervals.
        """
        stats_data = self.analyze()
        rows = []
        for col_name, n in stats_data["numerical_statistics"].items():
            ci = n.get("confidence_interval", {})
            outliers = n.get("outliers", {})
            rows.append({
                "column": col_name,
                "valid_count": n.get("valid_count"),
                "missing_count": n.get("missing_count"),
                "mean": n.get("mean"),
                "median": n.get("median"),
                "mode": n.get("mode"),
                "std": n.get("std"),
                "variance": n.get("variance"),
                "min": n.get("min"),
                "max": n.get("max"),
                "range": n.get("range"),
                "q1": n.get("q1"),
                "q3": n.get("q3"),
                "iqr": n.get("iqr"),
                "skewness": n.get("skewness"),
                "kurtosis": n.get("kurtosis"),
                "ci_95_lower": ci.get("lower"),
                "ci_95_upper": ci.get("upper"),
                "outliers_count": outliers.get("count"),
            })
        return pd.DataFrame(rows)

    def categorical_summary_dataframe(self) -> pd.DataFrame:
        """
        Export categorical statistics as a clean pandas DataFrame.
        
        Returns:
            pd.DataFrame: Tabular summary of categorical features and distributions.
        """
        stats_data = self.analyze()
        rows = []
        for col_name, c in stats_data["categorical_statistics"].items():
            rows.append({
                "column": col_name,
                "valid_count": c.get("valid_count"),
                "missing_count": c.get("missing_count"),
                "num_categories": c.get("num_categories"),
                "most_frequent_category": c.get("most_frequent_category"),
                "mode_frequency": c.get("mode_frequency"),
                "mode_percentage": c.get("mode_percentage"),
                "entropy": c.get("entropy"),
                "rare_categories_count": c.get("rare_categories_count"),
            })
        return pd.DataFrame(rows)


def compute_descriptive_statistics(df: pd.DataFrame, dataset_id: str = "") -> Dict[str, Any]:
    """
    Functional entry point: compute comprehensive statistical analysis on a DataFrame.
    
    Args:
        df: Input pandas DataFrame.
        dataset_id: Optional ID string.
        
    Returns:
        Dict[str, Any]: Complete structured statistical analysis dictionary.
    """
    engine = StatisticalAnalysisEngine(df, dataset_id=dataset_id)
    return engine.to_dict()
