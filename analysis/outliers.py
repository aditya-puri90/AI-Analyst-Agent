"""
Outlier & Anomaly Detection Engine for AI Data Analyst Agent (Phase 7).
Provides enterprise-grade, deterministic outlier detection across numerical features
using Interquartile Range (IQR), Z-Score, and Modified Z-Score (MAD) methods.

Features:
- Method selection (IQR standard 1.5x / extreme 3.0x, Z-Score 3.0 / 2.5, Modified Z-Score)
- Comprehensive column metrics (N, Valid N, Outlier Count, Outlier Pct, Lower & Upper Thresholds)
- Statistical edge case detection & warnings (Zero Std, Zero IQR, Small Sample Size, High Skewness)
- Clear distinction between Potential Statistical Outliers and Confirmed Data Errors
- Non-destructive remediation operations (Review, Keep, Remove, Cap) generating separate processed datasets
"""

import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from analysis.profiler import infer_column_types

logger = logging.getLogger(__name__)

# Heuristic patterns for domain constraints
NON_NEGATIVE_PATTERNS = [
    "age", "price", "cost", "salary", "wage", "income", "amount", "revenue",
    "quantity", "qty", "count", "units", "items", "tenure", "experience",
    "duration", "distance", "height", "weight", "length", "width", "depth",
    "speed", "rate_per", "orders", "visits", "clicks", "impressions",
]

PERCENTAGE_PATTERNS = ["pct", "percent", "percentage", "discount_pct", "tax_rate", "rate_pct"]

SENTINEL_VALUES = {-999, -9999, -99, -1, 999, 9999, 99999, 999999}


def _safe_json_value(val: Any) -> Any:
    """Safely convert numpy/pandas scalars into JSON-serializable primitives."""
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
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    return str(val)


def classify_anomaly_type(
    val: float,
    col_name: str,
    lower_bound: float,
    upper_bound: float,
    series_min: float,
    series_max: float,
) -> Tuple[bool, str, str]:
    """
    Distinguish whether an anomalous value is a Confirmed/Likely Data Error
    or a Potential Statistical Outlier based on domain rules and heuristics.

    Returns:
        Tuple[is_data_error: bool, classification_label: str, rationale: str]
    """
    col_lower = col_name.lower()

    # Rule 1: Sentinel missingness codes
    if round(val) in SENTINEL_VALUES and val not in [0, 1]:
        return (
            True,
            "Confirmed Data Error",
            f"Value {val} matches a standard missingness sentinel placeholder code ({round(val)}).",
        )

    # Rule 2: Non-negative domain violations
    for pat in NON_NEGATIVE_PATTERNS:
        if pat in col_lower and val < 0:
            return (
                True,
                "Confirmed Data Error",
                f"Negative value ({val}) in naturally non-negative feature '{col_name}'.",
            )

    # Rule 3: Percentage / Rate range violations
    for pat in PERCENTAGE_PATTERNS:
        if pat in col_lower:
            # If percentage scale is 0 - 100
            if series_max > 1.0 and (val < 0 or val > 100):
                return (
                    True,
                    "Confirmed Data Error",
                    f"Percentage value ({val}%) exceeds allowable 0-100% boundary.",
                )
            # If ratio scale is 0.0 - 1.0
            if series_max <= 1.0 and (val < 0.0 or val > 1.0):
                return (
                    True,
                    "Confirmed Data Error",
                    f"Ratio value ({val}) exceeds allowable [0.0, 1.0] interval.",
                )

    # Rule 4: Domain specific physical impossibility
    if "age" in col_lower:
        if val > 125 or val < 0:
            return (
                True,
                "Confirmed Data Error",
                f"Age value ({val}) is physically improbable for human demographics.",
            )
    if "month" in col_lower and (val < 1 or val > 12):
        return (
            True,
            "Confirmed Data Error",
            f"Month value ({val}) outside valid range [1, 12].",
        )
    if "hour" in col_lower and (val < 0 or val > 24):
        return (
            True,
            "Confirmed Data Error",
            f"Hour value ({val}) outside valid range [0, 24].",
        )

    # Default: Statistically extreme, but plausible domain observation
    direction = "above upper threshold" if val > upper_bound else "below lower threshold"
    bound_val = upper_bound if val > upper_bound else lower_bound
    return (
        False,
        "Potential Outlier",
        f"Extreme statistical observation ({val}) positioned {direction} ({round(bound_val, 2)}).",
    )


def extract_clean_numeric_series(series: pd.Series) -> pd.Series:
    """Extract a cleaned float series, handling string currencies/percentages and excluding booleans."""
    if pd.api.types.is_bool_dtype(series):
        return pd.Series(np.nan, index=series.index, dtype=float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    else:
        cleaned_str = series.dropna().astype(str).str.replace(r"[$,€£%\s]", "", regex=True)
        return pd.to_numeric(cleaned_str, errors="coerce").astype(float)


def detect_column_outliers_iqr(
    series: pd.Series,
    col_name: str,
    multiplier: float = 1.5,
) -> Dict[str, Any]:
    """
    Detect outliers using the Interquartile Range (IQR / Tukey's Fences) method.

    Calculation:
    - Q1 = 25th percentile, Q3 = 75th percentile
    - IQR = Q3 - Q1
    - Lower Threshold = Q1 - (multiplier * IQR)
    - Upper Threshold = Q3 + (multiplier * IQR)
    - Outlier: value < Lower Threshold or value > Upper Threshold

    Edge Case Warnings:
    - Small sample size (N < 10)
    - Zero IQR (IQR == 0.0, heavily concentrated/zero-inflated distributions)
    - Zero variance / constant column (min == max)
    """
    total_obs = len(series)
    clean_s = extract_clean_numeric_series(series)
    valid_s = clean_s.dropna()
    valid_count = len(valid_s)
    missing_count = total_obs - valid_count

    warnings = []

    if valid_count == 0:
        return {
            "column_name": col_name,
            "method": "iqr",
            "multiplier": multiplier,
            "total_observations": total_obs,
            "valid_observations": 0,
            "missing_count": missing_count,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "lower_threshold": None,
            "upper_threshold": None,
            "q1": None,
            "median": None,
            "q3": None,
            "iqr": None,
            "confirmed_error_count": 0,
            "potential_outlier_count": 0,
            "outliers": [],
            "outlier_indices": [],
            "warnings": ["Column contains zero non-null numerical observations."],
            "status": "No Data",
        }

    # Statistical moments & quartiles
    q1_val = float(np.percentile(valid_s, 25))
    median_val = float(np.percentile(valid_s, 50))
    q3_val = float(np.percentile(valid_s, 75))
    iqr_val = float(q3_val - q1_val)
    min_val = float(valid_s.min())
    max_val = float(valid_s.max())

    lower_thresh = float(q1_val - (multiplier * iqr_val))
    upper_thresh = float(q3_val + (multiplier * iqr_val))

    # Check Warnings
    if valid_count < 10:
        warnings.append(
            f"Small sample size warning: Only {valid_count} valid observations available. "
            f"Quartile estimation and IQR outlier boundaries may be unstable."
        )

    if min_val == max_val:
        warnings.append(
            f"Zero variance warning: Column is constant (all values = {min_val}). No outliers exist."
        )

    if iqr_val == 0.0 and min_val != max_val:
        warnings.append(
            f"Zero IQR warning (IQR = 0.0): Over 50% of observational values are identical ({median_val}). "
            f"Standard IQR fences collapse to a single point. Consider using Z-Score or verifying distribution."
        )

    # Detect outliers
    outlier_mask = (clean_s < lower_thresh) | (clean_s > upper_thresh)
    outlier_series = clean_s[outlier_mask]
    outlier_count = int(len(outlier_series))
    outlier_pct = round((outlier_count / valid_count) * 100, 2) if valid_count > 0 else 0.0

    # Build outlier records with error diagnosis
    outliers_list = []
    outlier_indices = []
    confirmed_error_cnt = 0
    potential_outlier_cnt = 0

    for idx, val in outlier_series.items():
        is_err, class_label, rationale = classify_anomaly_type(
            val=float(val),
            col_name=col_name,
            lower_bound=lower_thresh,
            upper_bound=upper_thresh,
            series_min=min_val,
            series_max=max_val,
        )
        if is_err:
            confirmed_error_cnt += 1
        else:
            potential_outlier_cnt += 1

        direction = "below_lower" if val < lower_thresh else "above_upper"
        dist = abs(float(val) - (lower_thresh if val < lower_thresh else upper_thresh))

        outlier_record = {
            "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "value": _safe_json_value(val),
            "lower_threshold": _safe_json_value(lower_thresh),
            "upper_threshold": _safe_json_value(upper_thresh),
            "direction": direction,
            "deviation_from_threshold": _safe_json_value(dist),
            "is_data_error": is_err,
            "classification": class_label,
            "rationale": rationale,
        }
        outliers_list.append(outlier_record)
        outlier_indices.append(idx)

    # Status badge
    if outlier_count == 0:
        status = "Clean"
    elif confirmed_error_cnt > 0:
        status = "Data Errors Detected"
    elif outlier_pct > 5.0:
        status = "High Outliers"
    else:
        status = "Moderate Outliers"

    return {
        "column_name": col_name,
        "method": "iqr",
        "multiplier": multiplier,
        "total_observations": total_obs,
        "valid_observations": valid_count,
        "missing_count": missing_count,
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_pct,
        "lower_threshold": _safe_json_value(lower_thresh),
        "upper_threshold": _safe_json_value(upper_thresh),
        "q1": _safe_json_value(q1_val),
        "median": _safe_json_value(median_val),
        "q3": _safe_json_value(q3_val),
        "iqr": _safe_json_value(iqr_val),
        "min": _safe_json_value(min_val),
        "max": _safe_json_value(max_val),
        "confirmed_error_count": confirmed_error_cnt,
        "potential_outlier_count": potential_outlier_cnt,
        "outliers": outliers_list,
        "outlier_indices": [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in outlier_indices],
        "warnings": warnings,
        "status": status,
    }


def detect_column_outliers_zscore(
    series: pd.Series,
    col_name: str,
    threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Detect outliers using the Parametric Z-Score (Standard Deviation) method.

    Calculation:
    - Mean (mu), Standard Deviation (sigma, ddof=1)
    - Z = (x - mu) / sigma
    - Lower Threshold = mu - (threshold * sigma)
    - Upper Threshold = mu + (threshold * sigma)
    - Outlier: |Z| > threshold

    Edge Case Warnings:
    - Small sample size (N < 30, sample variance and Gaussian assumptions weak)
    - Zero standard deviation (sigma == 0.0, division by zero)
    - High Skewness (|skew| > 1.0, noting z-score bias from heavy tails)
    """
    total_obs = len(series)
    clean_s = extract_clean_numeric_series(series)
    valid_s = clean_s.dropna()
    valid_count = len(valid_s)
    missing_count = total_obs - valid_count

    warnings = []

    if valid_count == 0:
        return {
            "column_name": col_name,
            "method": "zscore",
            "threshold": threshold,
            "total_observations": total_obs,
            "valid_observations": 0,
            "missing_count": missing_count,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "lower_threshold": None,
            "upper_threshold": None,
            "mean": None,
            "std": None,
            "skewness": None,
            "confirmed_error_count": 0,
            "potential_outlier_count": 0,
            "outliers": [],
            "outlier_indices": [],
            "warnings": ["Column contains zero non-null numerical observations."],
            "status": "No Data",
        }

    mean_val = float(valid_s.mean())
    std_val = float(valid_s.std(ddof=1)) if valid_count > 1 else 0.0
    min_val = float(valid_s.min())
    max_val = float(valid_s.max())

    # Check Warnings
    if valid_count < 30:
        warnings.append(
            f"Small sample size warning: Only {valid_count} valid observations available. "
            f"Z-score standard deviation and normality assumptions require caution for N < 30."
        )

    if std_val == 0.0 or math.isnan(std_val) or min_val == max_val:
        warnings.append(
            f"Zero standard deviation warning (std = 0.0): Column is constant. "
            f"Z-scores cannot be computed (division by zero). Outlier count is 0."
        )
        return {
            "column_name": col_name,
            "method": "zscore",
            "threshold": threshold,
            "total_observations": total_obs,
            "valid_observations": valid_count,
            "missing_count": missing_count,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "lower_threshold": _safe_json_value(mean_val),
            "upper_threshold": _safe_json_value(mean_val),
            "mean": _safe_json_value(mean_val),
            "std": 0.0,
            "skewness": 0.0,
            "min": _safe_json_value(min_val),
            "max": _safe_json_value(max_val),
            "confirmed_error_count": 0,
            "potential_outlier_count": 0,
            "outliers": [],
            "outlier_indices": [],
            "warnings": warnings,
            "status": "Constant Feature",
        }

    skew_val = float(stats.skew(valid_s, bias=False)) if valid_count >= 3 else 0.0

    if abs(skew_val) > 1.0:
        warnings.append(
            f"High skewness warning (skew = {skew_val:.2f}): Feature distribution is substantially skewed. "
            f"Z-score bounds may under-detect or bias outliers; consider the IQR method for skewed data."
        )

    lower_thresh = float(mean_val - (threshold * std_val))
    upper_thresh = float(mean_val + (threshold * std_val))

    # Detect outliers via Z-scores
    z_scores = (clean_s - mean_val) / std_val
    outlier_mask = z_scores.abs() > threshold
    outlier_series = clean_s[outlier_mask]
    outlier_count = int(len(outlier_series))
    outlier_pct = round((outlier_count / valid_count) * 100, 2) if valid_count > 0 else 0.0

    outliers_list = []
    outlier_indices = []
    confirmed_error_cnt = 0
    potential_outlier_cnt = 0

    for idx, val in outlier_series.items():
        z = float(z_scores.loc[idx])
        is_err, class_label, rationale = classify_anomaly_type(
            val=float(val),
            col_name=col_name,
            lower_bound=lower_thresh,
            upper_bound=upper_thresh,
            series_min=min_val,
            series_max=max_val,
        )
        if is_err:
            confirmed_error_cnt += 1
        else:
            potential_outlier_cnt += 1

        direction = "below_lower" if val < lower_thresh else "above_upper"
        dist = abs(float(val) - (lower_thresh if val < lower_thresh else upper_thresh))

        outlier_record = {
            "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "value": _safe_json_value(val),
            "z_score": _safe_json_value(z),
            "lower_threshold": _safe_json_value(lower_thresh),
            "upper_threshold": _safe_json_value(upper_thresh),
            "direction": direction,
            "deviation_from_threshold": _safe_json_value(dist),
            "is_data_error": is_err,
            "classification": class_label,
            "rationale": rationale,
        }
        outliers_list.append(outlier_record)
        outlier_indices.append(idx)

    # Status badge
    if outlier_count == 0:
        status = "Clean"
    elif confirmed_error_cnt > 0:
        status = "Data Errors Detected"
    elif outlier_pct > 5.0:
        status = "High Outliers"
    else:
        status = "Moderate Outliers"

    return {
        "column_name": col_name,
        "method": "zscore",
        "threshold": threshold,
        "total_observations": total_obs,
        "valid_observations": valid_count,
        "missing_count": missing_count,
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_pct,
        "lower_threshold": _safe_json_value(lower_thresh),
        "upper_threshold": _safe_json_value(upper_thresh),
        "mean": _safe_json_value(mean_val),
        "std": _safe_json_value(std_val),
        "skewness": _safe_json_value(skew_val),
        "min": _safe_json_value(min_val),
        "max": _safe_json_value(max_val),
        "confirmed_error_count": confirmed_error_cnt,
        "potential_outlier_count": potential_outlier_cnt,
        "outliers": outliers_list,
        "outlier_indices": [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in outlier_indices],
        "warnings": warnings,
        "status": status,
    }


def detect_column_outliers_modified_zscore(
    series: pd.Series,
    col_name: str,
    threshold: float = 3.5,
) -> Dict[str, Any]:
    """
    Detect outliers using the Modified Z-Score (Median Absolute Deviation / Boris Iglewicz & David Hoaglin) method.
    Robust against existing outliers that distort the standard mean and variance.

    Calculation:
    - Median (M)
    - MAD = Median(|x_i - M|)
    - Modified Z_i = 0.6745 * |x_i - M| / MAD (if MAD > 0)
    """
    total_obs = len(series)
    clean_s = extract_clean_numeric_series(series)
    valid_s = clean_s.dropna()
    valid_count = len(valid_s)
    missing_count = total_obs - valid_count

    warnings = []

    if valid_count == 0:
        return {
            "column_name": col_name,
            "method": "modified_zscore",
            "threshold": threshold,
            "total_observations": total_obs,
            "valid_observations": 0,
            "missing_count": missing_count,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "lower_threshold": None,
            "upper_threshold": None,
            "median": None,
            "mad": None,
            "confirmed_error_count": 0,
            "potential_outlier_count": 0,
            "outliers": [],
            "outlier_indices": [],
            "warnings": ["Column contains zero non-null numerical observations."],
            "status": "No Data",
        }

    med_val = float(valid_s.median())
    abs_deviations = (valid_s - med_val).abs()
    mad_val = float(abs_deviations.median())
    min_val = float(valid_s.min())
    max_val = float(valid_s.max())

    if valid_count < 10:
        warnings.append(
            f"Small sample size warning: Only {valid_count} valid observations available for MAD computation."
        )

    if mad_val == 0.0:
        # Fallback to mean absolute deviation or zero std
        warnings.append(
            "Zero MAD warning (MAD = 0.0): Over 50% of values are identical to the median. "
            "Modified Z-Score cannot be computed without non-zero dispersion."
        )
        return {
            "column_name": col_name,
            "method": "modified_zscore",
            "threshold": threshold,
            "total_observations": total_obs,
            "valid_observations": valid_count,
            "missing_count": missing_count,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "lower_threshold": _safe_json_value(med_val),
            "upper_threshold": _safe_json_value(med_val),
            "median": _safe_json_value(med_val),
            "mad": 0.0,
            "min": _safe_json_value(min_val),
            "max": _safe_json_value(max_val),
            "confirmed_error_count": 0,
            "potential_outlier_count": 0,
            "outliers": [],
            "outlier_indices": [],
            "warnings": warnings,
            "status": "Zero MAD",
        }

    # Bound computation: 0.6745 * (x - med) / mad = threshold => x = med +/- (threshold * mad / 0.6745)
    delta = (threshold * mad_val) / 0.6745
    lower_thresh = float(med_val - delta)
    upper_thresh = float(med_val + delta)

    mod_z = 0.6745 * (clean_s - med_val).abs() / mad_val
    outlier_mask = mod_z > threshold
    outlier_series = clean_s[outlier_mask]
    outlier_count = int(len(outlier_series))
    outlier_pct = round((outlier_count / valid_count) * 100, 2) if valid_count > 0 else 0.0

    outliers_list = []
    outlier_indices = []
    confirmed_error_cnt = 0
    potential_outlier_cnt = 0

    for idx, val in outlier_series.items():
        is_err, class_label, rationale = classify_anomaly_type(
            val=float(val),
            col_name=col_name,
            lower_bound=lower_thresh,
            upper_bound=upper_thresh,
            series_min=min_val,
            series_max=max_val,
        )
        if is_err:
            confirmed_error_cnt += 1
        else:
            potential_outlier_cnt += 1

        direction = "below_lower" if val < lower_thresh else "above_upper"
        dist = abs(float(val) - (lower_thresh if val < lower_thresh else upper_thresh))

        outlier_record = {
            "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "value": _safe_json_value(val),
            "modified_z_score": _safe_json_value(float(mod_z.loc[idx])),
            "lower_threshold": _safe_json_value(lower_thresh),
            "upper_threshold": _safe_json_value(upper_thresh),
            "direction": direction,
            "deviation_from_threshold": _safe_json_value(dist),
            "is_data_error": is_err,
            "classification": class_label,
            "rationale": rationale,
        }
        outliers_list.append(outlier_record)
        outlier_indices.append(idx)

    status = "Clean" if outlier_count == 0 else ("Data Errors Detected" if confirmed_error_cnt > 0 else "Moderate Outliers")

    return {
        "column_name": col_name,
        "method": "modified_zscore",
        "threshold": threshold,
        "total_observations": total_obs,
        "valid_observations": valid_count,
        "missing_count": missing_count,
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_pct,
        "lower_threshold": _safe_json_value(lower_thresh),
        "upper_threshold": _safe_json_value(upper_thresh),
        "median": _safe_json_value(med_val),
        "mad": _safe_json_value(mad_val),
        "min": _safe_json_value(min_val),
        "max": _safe_json_value(max_val),
        "confirmed_error_count": confirmed_error_cnt,
        "potential_outlier_count": potential_outlier_cnt,
        "outliers": outliers_list,
        "outlier_indices": [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in outlier_indices],
        "warnings": warnings,
        "status": status,
    }


class OutlierDetectionEngine:
    """
    Enterprise-grade Outlier & Anomaly Detection Engine.
    Executes column-wise detection, handles statistical warnings,
    distinguishes errors from extreme values, and supports non-destructive remediation.
    """

    def __init__(self, df: pd.DataFrame, dataset_id: str = ""):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("OutlierDetectionEngine requires a pandas DataFrame instance.")
        self.df = df
        self.dataset_id = dataset_id

    def get_numerical_columns(self) -> List[str]:
        """Identify all numerical columns in the DataFrame, strictly excluding booleans."""
        col_types = infer_column_types(self.df)
        num_cols = []
        for col in self.df.columns:
            col_str = str(col)
            series = self.df[col]
            if pd.api.types.is_bool_dtype(series) or col_types.get(col_str) == "Boolean":
                continue
            if col_types.get(col_str) == "Numerical" or pd.api.types.is_numeric_dtype(series):
                num_cols.append(col_str)
        return num_cols

    def analyze(
        self,
        method: str = "iqr",
        param: Optional[float] = None,
        column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute outlier detection across all numerical columns or a specific column.

        Args:
            method: 'iqr', 'zscore', or 'modified_zscore'
            param: multiplier for IQR (default 1.5) or threshold for Z-score (default 3.0)
            column: optional specific column name to inspect

        Returns:
            Structured summary analysis dictionary.
        """
        norm_method = str(method).lower().strip()
        if norm_method not in ["iqr", "zscore", "modified_zscore"]:
            norm_method = "iqr"

        if param is None:
            if norm_method == "iqr":
                param = 1.5
            elif norm_method == "zscore":
                param = 3.0
            else:
                param = 3.5

        num_cols = self.get_numerical_columns()

        if column:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in dataset.")
            target_cols = [column]
        else:
            target_cols = num_cols

        columns_results: Dict[str, Any] = {}
        all_dataset_warnings: List[str] = []
        total_outlier_count = 0
        total_error_count = 0
        total_potential_count = 0
        columns_with_outliers = 0
        all_outlier_row_indices = set()

        # Dataset level warning
        if len(self.df) < 10:
            all_dataset_warnings.append(
                f"Small dataset warning: Dataset has only {len(self.df)} rows. Outlier detection boundaries may be fragile."
            )

        for col in target_cols:
            series = self.df[col]
            if norm_method == "iqr":
                col_res = detect_column_outliers_iqr(series, col_name=col, multiplier=param)
            elif norm_method == "zscore":
                col_res = detect_column_outliers_zscore(series, col_name=col, threshold=param)
            else:
                col_res = detect_column_outliers_modified_zscore(series, col_name=col, threshold=param)

            columns_results[col] = col_res

            c_outliers = col_res.get("outlier_count", 0)
            c_errors = col_res.get("confirmed_error_count", 0)
            c_potentials = col_res.get("potential_outlier_count", 0)

            if c_outliers > 0:
                columns_with_outliers += 1
                total_outlier_count += c_outliers
                total_error_count += c_errors
                total_potential_count += c_potentials
                for idx in col_res.get("outlier_indices", []):
                    all_outlier_row_indices.add(idx)

            for w in col_res.get("warnings", []):
                all_dataset_warnings.append(f"[{col}] {w}")

        # Summary table items
        summary_table = []
        for col, res in columns_results.items():
            summary_table.append({
                "column_name": col,
                "method": res.get("method"),
                "total_observations": res.get("total_observations"),
                "valid_observations": res.get("valid_observations"),
                "outlier_count": res.get("outlier_count"),
                "outlier_percentage": res.get("outlier_percentage"),
                "lower_threshold": res.get("lower_threshold"),
                "upper_threshold": res.get("upper_threshold"),
                "confirmed_error_count": res.get("confirmed_error_count"),
                "potential_outlier_count": res.get("potential_outlier_count"),
                "status": res.get("status"),
                "warnings_count": len(res.get("warnings", [])),
                "warnings": res.get("warnings", []),
            })

        # Sort summary table by outlier count descending
        summary_table.sort(key=lambda x: x["outlier_count"], reverse=True)

        total_rows = len(self.df)
        affected_rows_count = len(all_outlier_row_indices)
        affected_rows_pct = round((affected_rows_count / total_rows * 100), 2) if total_rows > 0 else 0.0

        return {
            "dataset_id": self.dataset_id,
            "method": norm_method,
            "parameter": param,
            "total_rows": total_rows,
            "numerical_columns_count": len(num_cols),
            "columns_with_outliers_count": columns_with_outliers,
            "clean_columns_count": len(num_cols) - columns_with_outliers,
            "total_outlier_instances": total_outlier_count,
            "total_confirmed_errors": total_error_count,
            "total_potential_outliers": total_potential_count,
            "affected_rows_count": affected_rows_count,
            "affected_rows_percentage": affected_rows_pct,
            "warnings": all_dataset_warnings,
            "summary_table": summary_table,
            "column_details": columns_results,
            "action_recommendation": (
                "Review flagged records in the Outlier Inspector. "
                "Confirmed data errors should typically be removed or corrected; "
                "potential outliers should be evaluated in domain context before removal."
            ),
        }

    def remediate_outliers(
        self,
        action: str = "remove",
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        param: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute user-selected outlier remediation on a copy of the dataset.
        Guarantees that the original raw DataFrame is never modified.

        Actions:
        - 'remove': Drop rows containing outliers in the specified columns.
        - 'cap': Winsorize / clamp values exceeding lower and upper thresholds.
        - 'remove_errors_only': Drop only rows containing confirmed data errors.

        Returns:
            Tuple[processed_df: pd.DataFrame, remediation_summary: Dict[str, Any]]
        """
        clean_df = self.df.copy()
        analysis = self.analyze(method=method, param=param)
        target_cols = columns if columns else self.get_numerical_columns()

        rows_before = len(clean_df)
        cols_before = len(clean_df.columns)

        action_norm = str(action).lower().strip()
        transformations_applied = []
        rows_dropped = 0
        values_capped = 0

        if action_norm == "remove":
            # Identify all row indices that have outliers in target columns
            indices_to_drop = set()
            for col in target_cols:
                col_res = analysis["column_details"].get(col, {})
                for idx in col_res.get("outlier_indices", []):
                    if idx in clean_df.index:
                        indices_to_drop.add(idx)

            if indices_to_drop:
                clean_df = clean_df.drop(index=list(indices_to_drop))
                rows_dropped = len(indices_to_drop)
                transformations_applied.append(
                    f"Removed {rows_dropped} rows containing outliers across {len(target_cols)} numerical features."
                )

        elif action_norm == "remove_errors_only":
            indices_to_drop = set()
            for col in target_cols:
                col_res = analysis["column_details"].get(col, {})
                for out_rec in col_res.get("outliers", []):
                    if out_rec.get("is_data_error"):
                        idx = out_rec.get("row_index")
                        if idx in clean_df.index:
                            indices_to_drop.add(idx)

            if indices_to_drop:
                clean_df = clean_df.drop(index=list(indices_to_drop))
                rows_dropped = len(indices_to_drop)
                transformations_applied.append(
                    f"Removed {rows_dropped} rows containing confirmed data errors (retained legitimate statistical outliers)."
                )

        elif action_norm == "cap":
            for col in target_cols:
                col_res = analysis["column_details"].get(col, {})
                lower = col_res.get("lower_threshold")
                upper = col_res.get("upper_threshold")

                if lower is not None and upper is not None and col in clean_df.columns:
                    if pd.api.types.is_numeric_dtype(clean_df[col]):
                        mask_low = clean_df[col] < lower
                        mask_high = clean_df[col] > upper
                        n_low = int(mask_low.sum())
                        n_high = int(mask_high.sum())

                        if n_low > 0 or n_high > 0:
                            clean_df[col] = clean_df[col].clip(lower=lower, upper=upper)
                            values_capped += (n_low + n_high)
                            transformations_applied.append(
                                f"Capped {n_low + n_high} outliers in '{col}' to range [{lower}, {upper}]."
                            )

        summary = {
            "action": action_norm,
            "method": analysis["method"],
            "parameter": analysis["parameter"],
            "target_columns": target_cols,
            "rows_before": rows_before,
            "rows_after": len(clean_df),
            "rows_dropped": rows_dropped,
            "values_capped": values_capped,
            "columns_count": len(clean_df.columns),
            "transformations": transformations_applied,
        }

        return clean_df, summary


def detect_dataset_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    param: Optional[float] = None,
    dataset_id: str = "",
) -> Dict[str, Any]:
    """
    High-level convenience entrypoint for full dataset outlier detection.
    """
    engine = OutlierDetectionEngine(df, dataset_id=dataset_id)
    return engine.analyze(method=method, param=param)
