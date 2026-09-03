"""
Automated Data Quality and Cleaning Engine (Phase 4 Module).
Provides 12 deterministic data quality issue detectors, transformation previews,
a non-destructive cleaning pipeline, and comprehensive cleaning summary reporting.
"""

import math
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Set
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Missing value sentinel tokens in string representations
MISSING_SENTINEL_STRINGS = {
    "", "na", "n/a", "null", "none", "nan", "nil", "?", "-", "--", "undefined", "#n/a", "#na"
}

# Domain keywords that are expected to be strictly non-negative
STRICTLY_POSITIVE_DOMAINS = [
    "age", "price", "cost", "salary", "income", "quantity", "count", "revenue",
    "amount", "spend", "rate", "distance", "height", "weight", "duration"
]


def _safe_val(val: Any) -> Any:
    """Safely format values for JSON serialization without NaN/Inf."""
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
    return str(val)


def _is_strictly_numeric(series: pd.Series) -> bool:
    """
    Check if a series is truly numeric (int or float) and NOT boolean.
    Pandas considers bools as a subtype of numeric, which causes errors in quantile/IQR subtraction.
    """
    if pd.api.types.is_bool_dtype(series):
        return False
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series)


# =====================================================================
# 1. 12 DATA QUALITY ISSUE DETECTION ALGORITHMS
# =====================================================================

def detect_data_quality_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Inspect the dataset and identify potential data quality problems across 12 distinct categories:
    1. Missing values
    2. Duplicate rows
    3. Incorrect/inconsistent data types
    4. Constant columns
    5. Near-constant columns
    6. Inconsistent categorical values
    7. Leading/trailing whitespace
    8. Potential numeric columns stored as strings
    9. Potential datetime columns stored as strings
    10. Invalid numerical values where detectable
    11. Extreme outliers
    12. Columns with very high missing percentages

    Returns:
        List[Dict[str, Any]]: Standardized list of detected issues.
    """
    issues: List[Dict[str, Any]] = []
    total_rows, total_cols = df.shape

    if total_rows == 0:
        return issues

    # -------------------------------------------------------------
    # DETECTOR 2: Duplicate Rows (Dataset-level)
    # -------------------------------------------------------------
    dup_mask = df.duplicated()
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        dup_pct = round((dup_count / total_rows) * 100, 2)
        severity = "High" if dup_pct >= 5.0 else ("Medium" if dup_pct >= 1.0 else "Low")
        issues.append({
            "column": "(Entire Dataset)",
            "issue_type": "duplicate_rows",
            "title": "Duplicate Rows Detected",
            "affected_rows": dup_count,
            "percentage_affected": dup_pct,
            "recommended_action": "Remove duplicate rows to preserve observation uniqueness and prevent biased modeling.",
            "severity": severity,
            "details": {
                "duplicate_count": dup_count,
                "total_rows": total_rows,
            },
        })

    # Column-by-column inspection
    for col in df.columns:
        series = df[col]
        col_name = str(col)
        col_name_lower = col_name.lower()
        non_null_series = series.dropna()
        n_valid = len(non_null_series)
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_rows) * 100, 2)
        is_num = _is_strictly_numeric(series)

        # -------------------------------------------------------------
        # DETECTOR 1 & 12: Missing Values & High Missing Percentages
        # -------------------------------------------------------------
        sentinel_nulls = 0
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            for val in non_null_series:
                if str(val).strip().lower() in MISSING_SENTINEL_STRINGS:
                    sentinel_nulls += 1

        total_missing_in_col = null_count + sentinel_nulls
        total_missing_pct = round((total_missing_in_col / total_rows) * 100, 2)

        if total_missing_pct > 40.0:
            # DETECTOR 12: Very High Missing Percentage
            issues.append({
                "column": col_name,
                "issue_type": "high_missing_percentage",
                "title": "Critically High Missingness",
                "affected_rows": total_missing_in_col,
                "percentage_affected": total_missing_pct,
                "recommended_action": f"Column '{col_name}' has {total_missing_pct}% missing values. Consider dropping or flagging.",
                "severity": "Critical",
                "details": {
                    "null_count": total_missing_in_col,
                    "threshold": 40.0,
                },
            })
        elif total_missing_in_col > 0:
            # DETECTOR 1: Missing Values
            rec_action = (
                f"Impute {total_missing_in_col} missing numerical values using column median"
                if is_num
                else f"Impute {total_missing_in_col} missing categorical values using column mode or 'Unknown'"
            )
            severity = "High" if total_missing_pct >= 20.0 else ("Medium" if total_missing_pct >= 5.0 else "Low")
            issues.append({
                "column": col_name,
                "issue_type": "missing_values",
                "title": "Missing Values Present",
                "affected_rows": total_missing_in_col,
                "percentage_affected": total_missing_pct,
                "recommended_action": rec_action,
                "severity": severity,
                "details": {
                    "standard_nulls": null_count,
                    "sentinel_nulls": sentinel_nulls,
                },
            })

        if n_valid == 0:
            continue

        unique_vals = non_null_series.unique()
        unique_count = len(unique_vals)

        # -------------------------------------------------------------
        # DETECTOR 4: Constant Columns (0 Variance)
        # -------------------------------------------------------------
        if unique_count == 1:
            issues.append({
                "column": col_name,
                "issue_type": "constant_column",
                "title": "Constant (Zero Variance) Column",
                "affected_rows": total_rows,
                "percentage_affected": 100.0,
                "recommended_action": f"Column '{col_name}' has only 1 unique value ('{unique_vals[0]}'). Drop to simplify schema.",
                "severity": "High",
                "details": {"constant_value": _safe_val(unique_vals[0])},
            })
            continue

        # -------------------------------------------------------------
        # DETECTOR 5: Near-Constant Columns (>95% Dominance)
        # -------------------------------------------------------------
        val_counts = non_null_series.value_counts()
        top_count = int(val_counts.iloc[0])
        top_val = val_counts.index[0]
        dominance_pct = round((top_count / n_valid) * 100, 2)

        if dominance_pct >= 95.0 and unique_count > 1:
            affected_minority = n_valid - top_count
            issues.append({
                "column": col_name,
                "issue_type": "near_constant_column",
                "title": "Near-Constant Distribution",
                "affected_rows": top_count,
                "percentage_affected": dominance_pct,
                "recommended_action": f"Top category '{top_val}' dominates {dominance_pct}% of records. Review feature relevance.",
                "severity": "Low",
                "details": {
                    "dominant_value": _safe_val(top_val),
                    "dominance_percentage": dominance_pct,
                    "minority_rows": affected_minority,
                },
            })

        # -------------------------------------------------------------
        # DETECTOR 7: Leading / Trailing Whitespace (String/Object)
        # -------------------------------------------------------------
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            str_series = non_null_series.astype(str)
            whitespace_mask = str_series.str.strip() != str_series
            ws_count = int(whitespace_mask.sum())
            if ws_count > 0:
                ws_pct = round((ws_count / total_rows) * 100, 2)
                ws_samples = str_series[whitespace_mask].head(3).tolist()
                issues.append({
                    "column": col_name,
                    "issue_type": "leading_trailing_whitespace",
                    "title": "Leading / Trailing Whitespace",
                    "affected_rows": ws_count,
                    "percentage_affected": ws_pct,
                    "recommended_action": f"Strip unneeded leading/trailing spaces across {ws_count} text cells in '{col_name}'.",
                    "severity": "Low",
                    "details": {
                        "examples": ws_samples,
                    },
                })

        # -------------------------------------------------------------
        # DETECTOR 6: Inconsistent Categorical Values (Casing/Spacing)
        # -------------------------------------------------------------
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            clean_str_vals = [str(v).strip() for v in unique_vals if str(v).strip()]
            norm_map: Dict[str, Set[str]] = {}
            for v in clean_str_vals:
                key = v.lower()
                norm_map.setdefault(key, set()).add(v)

            inconsistent_groups = {k: list(v) for k, v in norm_map.items() if len(v) > 1}
            if inconsistent_groups:
                affected_inconsistent = 0
                for _, group_vals in inconsistent_groups.items():
                    sub_counts = non_null_series[non_null_series.astype(str).str.strip().isin(group_vals)].value_counts()
                    if len(sub_counts) > 1:
                        affected_inconsistent += int(sub_counts.iloc[1:].sum())

                if affected_inconsistent > 0:
                    inc_pct = round((affected_inconsistent / total_rows) * 100, 2)
                    issues.append({
                        "column": col_name,
                        "issue_type": "inconsistent_categorical",
                        "title": "Inconsistent Categorical Casing",
                        "affected_rows": affected_inconsistent,
                        "percentage_affected": inc_pct,
                        "recommended_action": f"Standardize {len(inconsistent_groups)} inconsistent category representations into canonical forms in '{col_name}'.",
                        "severity": "Medium",
                        "details": {
                            "inconsistent_groups": inconsistent_groups,
                        },
                    })

        # -------------------------------------------------------------
        # DETECTOR 8: Potential Numeric Columns Stored as Strings
        # -------------------------------------------------------------
        if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and n_valid >= 5:
            sample = non_null_series.head(100).astype(str)
            if not sample.str.contains(r"[-/]\d{2,4}", regex=True).any():
                cleaned_numeric = sample.str.replace(r"[$,€£¥₹%\s]", "", regex=True).str.replace(",", "", regex=False)
                valid_num_mask = pd.to_numeric(cleaned_numeric, errors="coerce").notna()
                num_success_ratio = valid_num_mask.sum() / len(sample)
                
                if num_success_ratio >= 0.8 and not is_num:
                    full_cleaned = non_null_series.astype(str).str.replace(r"[$,€£¥₹%\s]", "", regex=True).str.replace(",", "", regex=False)
                    full_valid_mask = pd.to_numeric(full_cleaned, errors="coerce").notna()
                    full_success_count = int(full_valid_mask.sum())
                    full_pct = round((full_success_count / total_rows) * 100, 2)

                    issues.append({
                        "column": col_name,
                        "issue_type": "potential_numeric_as_string",
                        "title": "Numeric Column Formatted as Text",
                        "affected_rows": full_success_count,
                        "percentage_affected": full_pct,
                        "recommended_action": f"Clean numeric formatting symbols (currencies/commas/percentages) and cast '{col_name}' to float/int.",
                        "severity": "High",
                        "details": {
                            "success_ratio": round(num_success_ratio * 100, 2),
                            "sample_raw": sample.head(3).tolist(),
                        },
                    })

        # -------------------------------------------------------------
        # DETECTOR 9: Potential Datetime Columns Stored as Strings
        # -------------------------------------------------------------
        if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and n_valid >= 5:
            sample = non_null_series.head(100).astype(str)
            has_date_separators = sample.str.contains(r"[-/:\s,]", regex=True).any()
            if has_date_separators and not pd.api.types.is_datetime64_any_dtype(series):
                try:
                    try:
                        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    except Exception:
                        parsed = pd.to_datetime(sample, errors="coerce")

                    dt_success_ratio = parsed.notna().sum() / len(sample)
                    if dt_success_ratio >= 0.8:
                        try:
                            try:
                                full_parsed = pd.to_datetime(non_null_series, errors="coerce", format="mixed")
                            except Exception:
                                full_parsed = pd.to_datetime(non_null_series, errors="coerce")
                            full_dt_count = int(full_parsed.notna().sum())
                        except Exception:
                            full_dt_count = int(len(sample) * dt_success_ratio)

                        dt_pct = round((full_dt_count / total_rows) * 100, 2)
                        issues.append({
                            "column": col_name,
                            "issue_type": "potential_datetime_as_string",
                            "title": "Datetime Stored as Text",
                            "affected_rows": full_dt_count,
                            "percentage_affected": dt_pct,
                            "recommended_action": f"Parse and cast '{col_name}' strings to standardized datetime timestamp format.",
                            "severity": "Medium",
                            "details": {
                                "success_ratio": round(dt_success_ratio * 100, 2),
                                "sample_dates": sample.head(3).tolist(),
                            },
                        })
                except Exception:
                    pass

        # -------------------------------------------------------------
        # DETECTOR 3: Incorrect / Inconsistent Data Types
        # -------------------------------------------------------------
        if pd.api.types.is_object_dtype(series) and n_valid >= 10:
            type_set = {type(v).__name__ for v in non_null_series.head(100)}
            if len(type_set) > 1 and "float" in type_set and "str" in type_set:
                issues.append({
                    "column": col_name,
                    "issue_type": "inconsistent_data_types",
                    "title": "Inconsistent Mixed Python Types",
                    "affected_rows": total_rows,
                    "percentage_affected": 100.0,
                    "recommended_action": f"Column '{col_name}' contains mixed types {list(type_set)}. Standardize to single dtype.",
                    "severity": "Medium",
                    "details": {"detected_types": list(type_set)},
                })

        # -------------------------------------------------------------
        # DETECTOR 10: Invalid Numerical Values (Inf, Sentinel Codes, Negative Domains)
        # -------------------------------------------------------------
        if is_num:
            num_clean = pd.to_numeric(series, errors="coerce")
            
            inf_count = int(np.isinf(num_clean).sum())
            sentinel_num_mask = num_clean.isin([-999, -9999, 9999, 99999])
            sentinel_num_count = int(sentinel_num_mask.sum())

            is_strictly_positive = any(dom in col_name_lower for dom in STRICTLY_POSITIVE_DOMAINS)
            negative_count = int((num_clean < 0).sum()) if is_strictly_positive else 0

            invalid_total = inf_count + sentinel_num_count + negative_count
            if invalid_total > 0:
                inv_pct = round((invalid_total / total_rows) * 100, 2)
                reasons = []
                if inf_count > 0:
                    reasons.append(f"{inf_count} infinite values")
                if sentinel_num_count > 0:
                    reasons.append(f"{sentinel_num_count} sentinel missing codes (-999/9999)")
                if negative_count > 0:
                    reasons.append(f"{negative_count} negative values in domain '{col_name}'")

                issues.append({
                    "column": col_name,
                    "issue_type": "invalid_numerical_values",
                    "title": "Invalid Numerical Values Detected",
                    "affected_rows": invalid_total,
                    "percentage_affected": inv_pct,
                    "recommended_action": f"Replace invalid values ({', '.join(reasons)}) in '{col_name}' with NaN for proper imputation.",
                    "severity": "High",
                    "details": {
                        "inf_count": inf_count,
                        "sentinel_count": sentinel_num_count,
                        "negative_count": negative_count,
                        "reasons": reasons,
                    },
                })

        # -------------------------------------------------------------
        # DETECTOR 11: Extreme Outliers (IQR extreme fence: Q1 - 3*IQR, Q3 + 3*IQR)
        # -------------------------------------------------------------
        if is_num and n_valid >= 20:
            num_valid = non_null_series[np.isfinite(non_null_series)].astype(float)
            if len(num_valid) >= 20:
                q1 = float(num_valid.quantile(0.25))
                q3 = float(num_valid.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    lower_extreme = q1 - 3.0 * iqr
                    upper_extreme = q3 + 3.0 * iqr
                    extreme_mask = (num_valid < lower_extreme) | (num_valid > upper_extreme)
                    extreme_count = int(extreme_mask.sum())

                    if extreme_count > 0:
                        outlier_pct = round((extreme_count / total_rows) * 100, 2)
                        issues.append({
                            "column": col_name,
                            "issue_type": "extreme_outliers",
                            "title": "Extreme Outliers (3x IQR Rule)",
                            "affected_rows": extreme_count,
                            "percentage_affected": outlier_pct,
                            "recommended_action": f"Inspect or winsorize/cap {extreme_count} extreme values outside range [{round(lower_extreme, 2)}, {round(upper_extreme, 2)}] in '{col_name}'.",
                            "severity": "Low",
                            "details": {
                                "lower_fence": round(lower_extreme, 2),
                                "upper_fence": round(upper_extreme, 2),
                                "outlier_count": extreme_count,
                                "q1": round(q1, 2),
                                "q3": round(q3, 2),
                                "iqr": round(iqr, 2),
                            },
                        })

    return issues


# =====================================================================
# 2. TRANSFORMATION PREVIEW GENERATOR
# =====================================================================

def generate_cleaning_preview(df: pd.DataFrame, issues: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Generate an illustrative before-and-after transformation preview mapping:
    Original value → Proposed cleaned value
    """
    if issues is None:
        issues = detect_data_quality_issues(df)

    previews: List[Dict[str, Any]] = []

    for issue in issues:
        col = issue.get("column")
        if not col or col == "(Entire Dataset)" or col not in df.columns:
            continue

        issue_type = issue.get("issue_type")
        series = df[col]
        non_null_series = series.dropna()

        # 1. Whitespace transformations
        if issue_type == "leading_trailing_whitespace":
            str_series = non_null_series.astype(str)
            ws_mask = str_series.str.strip() != str_series
            samples = str_series[ws_mask].unique()[:4]
            for s in samples:
                previews.append({
                    "column": col,
                    "category": "Whitespace Standardization",
                    "original_value": s,
                    "cleaned_value": s.strip(),
                    "rule": "Strip leading & trailing whitespace",
                })

        # 2. Categorical standardization
        elif issue_type == "inconsistent_categorical":
            str_series = non_null_series.astype(str).str.strip()
            norm_map: Dict[str, Set[str]] = {}
            for v in str_series.unique():
                norm_map.setdefault(v.lower(), set()).add(v)
            
            for _, raw_set in list(norm_map.items())[:3]:
                if len(raw_set) > 1:
                    counts = str_series[str_series.isin(raw_set)].value_counts()
                    canonical = str(counts.index[0]).strip().title()
                    for raw_val in raw_set:
                        if raw_val != canonical:
                            previews.append({
                                "column": col,
                                "category": "Categorical Normalization",
                                "original_value": raw_val,
                                "cleaned_value": canonical,
                                "rule": "Normalize casing to Title Case",
                            })

        # 3. Numeric string conversions
        elif issue_type == "potential_numeric_as_string":
            sample_vals = non_null_series.astype(str).head(10)
            cleaned_samples = sample_vals.str.replace(r"[$,€£¥₹%\s]", "", regex=True).str.replace(",", "", regex=False)
            valid_nums = pd.to_numeric(cleaned_samples, errors="coerce")
            count = 0
            for orig, clean_num in zip(sample_vals, valid_nums):
                if pd.notna(clean_num) and str(orig) != str(clean_num):
                    previews.append({
                        "column": col,
                        "category": "Numeric Type Casting",
                        "original_value": orig,
                        "cleaned_value": _safe_val(clean_num),
                        "rule": "Strip currency/formatting symbols and cast to numeric",
                    })
                    count += 1
                    if count >= 3:
                        break

        # 4. Datetime conversions
        elif issue_type == "potential_datetime_as_string":
            sample_vals = non_null_series.astype(str).head(10)
            try:
                parsed = pd.to_datetime(sample_vals, errors="coerce")
                count = 0
                for orig, dt_val in zip(sample_vals, parsed):
                    if pd.notna(dt_val):
                        formatted_dt = dt_val.strftime("%Y-%m-%d %H:%M:%S") if (dt_val.hour != 0 or dt_val.minute != 0) else dt_val.strftime("%Y-%m-%d")
                        if str(orig) != formatted_dt:
                            previews.append({
                                "column": col,
                                "category": "Datetime Parsing",
                                "original_value": orig,
                                "cleaned_value": formatted_dt,
                                "rule": "Standardize to ISO datetime",
                            })
                            count += 1
                            if count >= 3:
                                break
            except Exception:
                pass

        # 5. Missing value imputation preview
        elif issue_type == "missing_values":
            if _is_strictly_numeric(series):
                med = float(non_null_series.median()) if len(non_null_series) > 0 else 0.0
                previews.append({
                    "column": col,
                    "category": "Missing Value Imputation",
                    "original_value": "NULL / NaN",
                    "cleaned_value": f"{round(med, 2)} (Median)",
                    "rule": "Fill numerical missing with median",
                })
            else:
                mode_val = str(non_null_series.mode().iloc[0]) if len(non_null_series) > 0 else "Unknown"
                previews.append({
                    "column": col,
                    "category": "Missing Value Imputation",
                    "original_value": "NULL / NaN",
                    "cleaned_value": f"'{mode_val}' (Mode)",
                    "rule": "Fill categorical missing with mode",
                })

        # 6. Invalid numerical replacements
        elif issue_type == "invalid_numerical_values":
            details = issue.get("details", {})
            if details.get("sentinel_count", 0) > 0:
                previews.append({
                    "column": col,
                    "category": "Invalid Value Replacement",
                    "original_value": "-999 / 9999",
                    "cleaned_value": "NaN → Median",
                    "rule": "Replace sentinel code with NaN for median imputation",
                })
            if details.get("inf_count", 0) > 0:
                previews.append({
                    "column": col,
                    "category": "Invalid Value Replacement",
                    "original_value": "inf / -inf",
                    "cleaned_value": "NaN → Median",
                    "rule": "Replace infinity with NaN for median imputation",
                })

    return {
        "total_preview_pairs": len(previews),
        "previews": previews,
    }


# =====================================================================
# 3. CONFIGURABLE CLEANING PIPELINE & SUMMARY ENGINE
# =====================================================================

def execute_cleaning_pipeline(
    df: pd.DataFrame,
    operations: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply requested cleaning transformations non-destructively on a copy of the dataset.

    Configurable operations supported:
    - remove_duplicates: bool (Default: True)
    - fill_numeric_missing: str (Default: 'median', options: 'median', 'mean', 'zero', 'drop', 'none')
    - fill_categorical_missing: str (Default: 'mode', options: 'mode', 'unknown', 'drop', 'none')
    - standardize_whitespace: bool (Default: True)
    - standardize_categorical: bool (Default: True)
    - convert_numeric_strings: bool (Default: True)
    - convert_datetime_strings: bool (Default: True)
    - handle_invalid_numerical: bool (Default: True)
    - drop_high_missing_columns: bool (Default: False)
    - drop_constant_columns: bool (Default: False)
    - cap_outliers: bool (Default: False)

    Args:
        df: Input raw pandas DataFrame (remains unmodified).
        operations: Dictionary of requested boolean flags and strategy options.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
        (Cleaned DataFrame, Comprehensive Cleaning Summary Dictionary)
    """
    if operations is None:
        operations = {
            "remove_duplicates": True,
            "fill_numeric_missing": "median",
            "fill_categorical_missing": "mode",
            "standardize_whitespace": True,
            "standardize_categorical": True,
            "convert_numeric_strings": True,
            "convert_datetime_strings": True,
            "handle_invalid_numerical": True,
            "drop_high_missing_columns": False,
            "drop_constant_columns": False,
            "cap_outliers": False,
        }

    clean_df = df.copy()

    rows_before = len(clean_df)
    cols_before = len(clean_df.columns)

    duplicates_removed = 0
    missing_values_handled = 0
    columns_converted = 0
    values_standardized = 0
    columns_dropped: List[str] = []
    operations_applied: List[str] = []
    column_transformations: Dict[str, List[str]] = {str(c): [] for c in clean_df.columns}

    # -------------------------------------------------------------
    # Step 1: Standardize Whitespace across string/object columns
    # -------------------------------------------------------------
    if operations.get("standardize_whitespace", True):
        ws_applied = False
        for col in clean_df.columns:
            if pd.api.types.is_object_dtype(clean_df[col]) or pd.api.types.is_string_dtype(clean_df[col]):
                series = clean_df[col]
                mask = series.notna()
                if mask.any():
                    str_vals = series[mask].astype(str)
                    stripped = str_vals.str.strip()
                    diff_count = int((str_vals != stripped).sum())
                    if diff_count > 0:
                        clean_df.loc[mask, col] = stripped
                        values_standardized += diff_count
                        column_transformations[str(col)].append(f"Stripped whitespace from {diff_count} cells")
                        ws_applied = True
        if ws_applied:
            operations_applied.append("Standardized leading/trailing whitespace")

    # -------------------------------------------------------------
    # Step 2: Standardize Categorical Values (Casing & Normalization)
    # -------------------------------------------------------------
    if operations.get("standardize_categorical", True):
        cat_applied = False
        for col in clean_df.columns:
            if pd.api.types.is_object_dtype(clean_df[col]) or pd.api.types.is_string_dtype(clean_df[col]):
                series = clean_df[col]
                mask = series.notna()
                if mask.sum() >= 5:
                    str_vals = series[mask].astype(str)
                    unique_strs = str_vals.unique()
                    
                    norm_map: Dict[str, Set[str]] = {}
                    for v in unique_strs:
                        norm_map.setdefault(v.lower(), set()).add(v)
                    
                    col_changes = 0
                    mapping_dict: Dict[str, str] = {}
                    for _, raw_set in norm_map.items():
                        if len(raw_set) > 1:
                            counts = str_vals[str_vals.isin(raw_set)].value_counts()
                            canonical = str(counts.index[0]).strip().title()
                            for raw_val in raw_set:
                                if raw_val != canonical:
                                    mapping_dict[raw_val] = canonical
                    
                    if mapping_dict:
                        for orig_val, target_val in mapping_dict.items():
                            match_mask = mask & (clean_df[col].astype(str) == orig_val)
                            change_n = int(match_mask.sum())
                            clean_df.loc[match_mask, col] = target_val
                            col_changes += change_n

                        values_standardized += col_changes
                        column_transformations[str(col)].append(f"Standardized {len(mapping_dict)} inconsistent categorical values ({col_changes} cells)")
                        cat_applied = True

        if cat_applied:
            operations_applied.append("Standardized categorical value casing")

    # -------------------------------------------------------------
    # Step 3: Convert Valid Numeric Strings to Numeric Dtype
    # -------------------------------------------------------------
    if operations.get("convert_numeric_strings", True):
        num_conv_applied = False
        for col in clean_df.columns:
            if (pd.api.types.is_object_dtype(clean_df[col]) or pd.api.types.is_string_dtype(clean_df[col])):
                series = clean_df[col]
                non_nulls = series.dropna()
                if len(non_nulls) >= 5:
                    sample = non_nulls.head(100).astype(str)
                    if not sample.str.contains(r"[-/]\d{2,4}", regex=True).any():
                        cleaned_sample = sample.str.replace(r"[$,€£¥₹%\s]", "", regex=True).str.replace(",", "", regex=False)
                        sample_nums = pd.to_numeric(cleaned_sample, errors="coerce")
                        if sample_nums.notna().sum() / len(sample) >= 0.8:
                            full_cleaned = series.astype(str).str.replace(r"[$,€£¥₹%\s]", "", regex=True).str.replace(",", "", regex=False)
                            converted = pd.to_numeric(full_cleaned, errors="coerce")
                            converted[series.isna()] = np.nan
                            clean_df[col] = converted
                            columns_converted += 1
                            column_transformations[str(col)].append("Converted string formatted numbers to numeric float/int")
                            num_conv_applied = True
        if num_conv_applied:
            operations_applied.append("Converted valid numeric text strings to numeric dtype")

    # -------------------------------------------------------------
    # Step 4: Convert Valid Date Strings to Datetime Dtype
    # -------------------------------------------------------------
    if operations.get("convert_datetime_strings", True):
        dt_conv_applied = False
        for col in clean_df.columns:
            if (pd.api.types.is_object_dtype(clean_df[col]) or pd.api.types.is_string_dtype(clean_df[col])):
                series = clean_df[col]
                non_nulls = series.dropna()
                if len(non_nulls) >= 5:
                    sample = non_nulls.head(100).astype(str)
                    if sample.str.contains(r"[-/:\s,]", regex=True).any():
                        try:
                            try:
                                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                            except Exception:
                                parsed = pd.to_datetime(sample, errors="coerce")

                            if parsed.notna().sum() / len(sample) >= 0.8:
                                try:
                                    clean_df[col] = pd.to_datetime(clean_df[col], errors="coerce", format="mixed")
                                except Exception:
                                    clean_df[col] = pd.to_datetime(clean_df[col], errors="coerce")

                                columns_converted += 1
                                column_transformations[str(col)].append("Parsed and converted text date strings to Datetime")
                                dt_conv_applied = True
                        except Exception:
                            pass
        if dt_conv_applied:
            operations_applied.append("Converted valid date strings to datetime timestamps")

    # -------------------------------------------------------------
    # Step 5: Handle Invalid Numerical Values (Inf, Sentinels -> NaN)
    # -------------------------------------------------------------
    if operations.get("handle_invalid_numerical", True):
        inv_applied = False
        for col in clean_df.columns:
            if _is_strictly_numeric(clean_df[col]):
                series = clean_df[col]
                inf_mask = np.isinf(series)
                if inf_mask.any():
                    clean_df.loc[inf_mask, col] = np.nan
                    values_standardized += int(inf_mask.sum())
                    column_transformations[str(col)].append(f"Replaced {int(inf_mask.sum())} infinite values with NaN")
                    inv_applied = True

                sentinel_mask = series.isin([-999, -9999, 9999, 99999])
                if sentinel_mask.any():
                    clean_df.loc[sentinel_mask, col] = np.nan
                    values_standardized += int(sentinel_mask.sum())
                    column_transformations[str(col)].append(f"Replaced {int(sentinel_mask.sum())} sentinel codes with NaN")
                    inv_applied = True

                col_name_lower = str(col).lower()
                if any(dom in col_name_lower for dom in STRICTLY_POSITIVE_DOMAINS):
                    neg_mask = clean_df[col] < 0
                    if neg_mask.any():
                        clean_df.loc[neg_mask, col] = np.nan
                        values_standardized += int(neg_mask.sum())
                        column_transformations[str(col)].append(f"Replaced {int(neg_mask.sum())} invalid negative values with NaN")
                        inv_applied = True

        if inv_applied:
            operations_applied.append("Replaced invalid numerical values (inf, sentinels, domain anomalies) with NaN")

    # -------------------------------------------------------------
    # Step 6: Drop Constant Columns (Optional)
    # -------------------------------------------------------------
    if operations.get("drop_constant_columns", False):
        const_cols = [c for c in clean_df.columns if clean_df[c].nunique(dropna=False) <= 1 or clean_df[c].nunique(dropna=True) == 1]
        if const_cols:
            clean_df.drop(columns=const_cols, inplace=True)
            columns_dropped.extend([str(c) for c in const_cols])
            operations_applied.append(f"Dropped {len(const_cols)} constant zero-variance column(s)")

    # -------------------------------------------------------------
    # Step 7: Drop High Missing Columns (Optional)
    # -------------------------------------------------------------
    if operations.get("drop_high_missing_columns", False):
        high_miss_cols = []
        thresh = operations.get("high_missing_threshold", 0.5)
        for col in clean_df.columns:
            if clean_df[col].isna().sum() / len(clean_df) >= thresh:
                high_miss_cols.append(col)
        if high_miss_cols:
            clean_df.drop(columns=high_miss_cols, inplace=True)
            columns_dropped.extend([str(c) for c in high_miss_cols])
            operations_applied.append(f"Dropped {len(high_miss_cols)} column(s) with >{int(thresh*100)}% missing values")

    # -------------------------------------------------------------
    # Step 8: Remove Duplicate Rows
    # -------------------------------------------------------------
    if operations.get("remove_duplicates", True):
        dup_mask = clean_df.duplicated()
        duplicates_removed = int(dup_mask.sum())
        if duplicates_removed > 0:
            clean_df = clean_df.drop_duplicates().reset_index(drop=True)
            operations_applied.append(f"Removed {duplicates_removed} duplicate rows")

    # -------------------------------------------------------------
    # Step 9: Fill Numerical Missing Values (Median / Mean / Zero)
    # -------------------------------------------------------------
    num_impute_strategy = operations.get("fill_numeric_missing", "median")
    if num_impute_strategy and num_impute_strategy != "none":
        for col in clean_df.columns:
            if _is_strictly_numeric(clean_df[col]):
                n_miss = int(clean_df[col].isna().sum())
                if n_miss > 0:
                    fill_val = 0.0
                    if num_impute_strategy == "median":
                        fill_val = float(clean_df[col].median()) if clean_df[col].notna().any() else 0.0
                    elif num_impute_strategy == "mean":
                        fill_val = float(clean_df[col].mean()) if clean_df[col].notna().any() else 0.0
                    elif num_impute_strategy == "zero":
                        fill_val = 0.0
                    elif num_impute_strategy == "drop":
                        clean_df = clean_df.dropna(subset=[col]).reset_index(drop=True)
                        missing_values_handled += n_miss
                        column_transformations[str(col)].append(f"Dropped {n_miss} rows with missing numeric values")
                        continue

                    clean_df[col] = clean_df[col].fillna(fill_val)
                    missing_values_handled += n_miss
                    column_transformations[str(col)].append(f"Imputed {n_miss} missing values with {num_impute_strategy} ({round(fill_val, 2)})")

        if missing_values_handled > 0:
            operations_applied.append(f"Imputed numerical missing values using '{num_impute_strategy}'")

    # -------------------------------------------------------------
    # Step 10: Fill Categorical Missing Values (Mode / Unknown)
    # -------------------------------------------------------------
    cat_impute_strategy = operations.get("fill_categorical_missing", "mode")
    if cat_impute_strategy and cat_impute_strategy != "none":
        cat_miss_count = 0
        for col in clean_df.columns:
            if not _is_strictly_numeric(clean_df[col]):
                n_miss = int(clean_df[col].isna().sum())
                if n_miss > 0:
                    if cat_impute_strategy == "mode":
                        mode_series = clean_df[col].dropna().mode()
                        fill_val = mode_series.iloc[0] if len(mode_series) > 0 else "Unknown"
                    elif cat_impute_strategy == "unknown":
                        fill_val = "Unknown"
                    elif cat_impute_strategy == "drop":
                        clean_df = clean_df.dropna(subset=[col]).reset_index(drop=True)
                        cat_miss_count += n_miss
                        column_transformations[str(col)].append(f"Dropped {n_miss} rows with missing categorical values")
                        continue
                    else:
                        fill_val = "Unknown"

                    clean_df[col] = clean_df[col].fillna(fill_val)
                    cat_miss_count += n_miss
                    column_transformations[str(col)].append(f"Imputed {n_miss} missing values with mode/unknown ('{fill_val}')")

        if cat_miss_count > 0:
            missing_values_handled += cat_miss_count
            operations_applied.append(f"Imputed categorical missing values using '{cat_impute_strategy}'")

    # -------------------------------------------------------------
    # Step 11: Outlier Capping (Optional Winsorization)
    # -------------------------------------------------------------
    if operations.get("cap_outliers", False):
        outliers_capped = 0
        for col in clean_df.columns:
            if _is_strictly_numeric(clean_df[col]):
                series = clean_df[col].dropna().astype(float)
                if len(series) >= 20:
                    q1 = float(series.quantile(0.25))
                    q3 = float(series.quantile(0.75))
                    iqr = q3 - q1
                    if iqr > 0:
                        lower = q1 - 3.0 * iqr
                        upper = q3 + 3.0 * iqr
                        cap_mask = (clean_df[col] < lower) | (clean_df[col] > upper)
                        n_capped = int(cap_mask.sum())
                        if n_capped > 0:
                            clean_df[col] = clean_df[col].clip(lower=lower, upper=upper)
                            outliers_capped += n_capped
                            column_transformations[str(col)].append(f"Capped {n_capped} extreme outliers to [{round(lower, 2)}, {round(upper, 2)}]")
        if outliers_capped > 0:
            operations_applied.append(f"Capped {outliers_capped} extreme numerical outliers")

    rows_after = len(clean_df)
    cols_after = len(clean_df.columns)

    summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "columns_before": cols_before,
        "columns_after": cols_after,
        "duplicates_removed": duplicates_removed,
        "missing_values_handled": missing_values_handled,
        "columns_converted": columns_converted,
        "values_standardized": values_standardized,
        "columns_dropped": columns_dropped,
        "operations_applied": operations_applied,
        "column_transformations": {k: v for k, v in column_transformations.items() if v},
    }

    return clean_df, summary


# Legacy aliases for compatibility
generate_cleaning_recommendations = detect_data_quality_issues
