"""
Dataset Profiling and Schema Inference Engine.
Performs deterministic exploratory data analysis, 5-category type inference,
missing value analysis, parametric & non-parametric statistical computing,
categorical modes, datetime spans, data quality scoring, and memory auditing.
Returns structured Python dictionaries and DataFrames (no HTML).
"""

import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

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


def infer_column_type(series: pd.Series) -> str:
    """
    Automatically classify a Pandas Series into one of 5 distinct semantic categories:
    - 'Numerical': Continuous or discrete integers, floating points, decimals.
    - 'Categorical': Discrete categories, enums, low/moderate cardinality text labels.
    - 'Datetime': Parseable dates, timestamps, or date/time strings.
    - 'Boolean': Native bools, binary flags (0/1, True/False, yes/no, Y/N).
    - 'Other': High-cardinality free text, UUIDs, complex unclassified objects.
    
    Args:
        series: Pandas Series to analyze.
        
    Returns:
        str: One of 'Numerical', 'Categorical', 'Datetime', 'Boolean', 'Other'.
    """
    # Drop NAs for type inference
    valid_series = series.dropna()
    n_valid = len(valid_series)

    if n_valid == 0:
        return "Other"

    # 1. Check if native boolean dtype
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"

    # 2. Check binary flags or boolean-like values
    unique_vals = set(valid_series.unique())
    unique_count = len(unique_vals)

    # Boolean representation checks
    if unique_vals.issubset({0, 1, 0.0, 1.0}):
        # In small or large series, strict 0 and 1 with bool-like name or binary nature
        col_name = str(series.name).lower() if hasattr(series, "name") and series.name else ""
        if any(bool_kw in col_name for bool_kw in ["is_", "has_", "flag", "active", "enabled", "selected"]) or unique_count <= 2:
            return "Boolean"

    str_unique_lower = {str(v).strip().lower() for v in unique_vals}
    bool_sets = [
        {"true", "false"},
        {"yes", "no"},
        {"y", "n"},
        {"t", "f"},
        {"1", "0"},
    ]
    if any(str_unique_lower.issubset(b_set) for b_set in bool_sets):
        return "Boolean"

    # 3. Check native Datetime types
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return "Datetime"

    # 4. Attempt Datetime parsing on object / string columns
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = valid_series.head(100).astype(str)
        # Avoid pure numbers being parsed as unix timestamps unless they have date separators
        has_date_separators = sample.str.contains(r"[-/:\s,]", regex=True).any()
        if has_date_separators:
            try:
                # Try format='mixed' (pandas >= 2.0) or default fallback
                try:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                except Exception:
                    parsed = pd.to_datetime(sample, errors="coerce")

                success_ratio = parsed.notna().sum() / len(sample)
                if success_ratio >= 0.8:
                    return "Datetime"
            except Exception:
                pass

    # 5. Check native Numerical types
    if pd.api.types.is_numeric_dtype(series):
        # Check if it's coded categorical (e.g. rating 1-5 with 5 discrete values)
        # Only treat as categorical if integer with tiny cardinality and not an ID
        if pd.api.types.is_integer_dtype(series) and unique_count <= 5 and n_valid >= 30:
            col_name = str(series.name).lower() if hasattr(series, "name") and series.name else ""
            if "rating" in col_name or "score" in col_name or "level" in col_name or "class" in col_name:
                return "Categorical"
        return "Numerical"

    # Check if object series is convertible to numeric (e.g., "$1,200.50", "45%")
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = valid_series.head(100).astype(str)
        cleaned_sample = sample.str.replace(r"[$,€£%\s]", "", regex=True)
        try:
            converted = pd.to_numeric(cleaned_sample, errors="coerce")
            if converted.notna().sum() / len(sample) >= 0.9:
                return "Numerical"
        except Exception:
            pass

    # 6. Distinguish between Categorical and Other (Free Text / High-Cardinality IDs)
    unique_ratio = unique_count / n_valid if n_valid > 0 else 0
    col_name_str = str(series.name).lower() if hasattr(series, "name") and series.name else ""

    # Clear ID column names with high uniqueness
    is_id_name = any(id_tag in col_name_str for id_tag in ["_id", "id_", "uuid", "guid", "code", "identifier", "key", "token"]) or col_name_str == "id"
    if is_id_name and (unique_ratio >= 0.5 or unique_count >= 10):
        return "Other"

    # Strict 100% unique key in medium/large datasets
    if unique_count == n_valid and n_valid >= 15:
        return "Other"

    # High average text length suggests free text/notes/descriptions
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        avg_char_length = valid_series.astype(str).str.len().mean()
        if avg_char_length > 60 and unique_ratio > 0.4:
            return "Other"

    # Discrete categories (low cardinality count)
    if unique_count <= 50:
        return "Categorical"

    # Moderate cardinality with low uniqueness ratio (< 30%)
    if unique_ratio <= 0.30:
        return "Categorical"

    return "Other"


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer semantic types for all columns in a DataFrame.
    
    Args:
        df: Pandas DataFrame.
        
    Returns:
        Dict[str, str]: Mapping of column name to classified type ('Numerical', 'Categorical', 'Datetime', 'Boolean', 'Other').
    """
    return {str(col): infer_column_type(df[col]) for col in df.columns}


class DatasetProfiler:
    """
    Enterprise-grade reusable dataset profiling engine.
    Computes dataset-level metrics, data quality health scores,
    and deep per-column parametric and non-parametric statistical profiles.
    """

    def __init__(self, df: pd.DataFrame, dataset_id: str = ""):
        """
        Initialize the profiler with a Pandas DataFrame.
        
        Args:
            df: Input pandas DataFrame to profile.
            dataset_id: Optional unique identifier for the dataset.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DatasetProfiler expects a pandas DataFrame instance.")
        self.df = df
        self.dataset_id = dataset_id
        self._profile_cache: Optional[Dict[str, Any]] = None

    def calculate_dataset_metrics(self) -> Dict[str, Any]:
        """
        Calculate dataset-level structural and memory metrics.
        
        Returns:
            Dict[str, Any]: Dataset-level metrics.
        """
        total_rows, total_cols = self.df.shape
        total_cells = total_rows * total_cols if total_rows > 0 else 0

        # Missing values
        null_counts = self.df.isnull().sum()
        total_missing_cells = int(null_counts.sum())
        missing_cells_pct = round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0

        rows_with_missing = int((self.df.isnull().any(axis=1)).sum()) if total_rows > 0 else 0
        rows_with_missing_pct = round((rows_with_missing / total_rows * 100), 2) if total_rows > 0 else 0.0

        # Duplicates
        duplicate_rows = int(self.df.duplicated().sum()) if total_rows > 0 else 0
        duplicate_rows_pct = round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0

        # Memory usage
        try:
            memory_bytes = int(self.df.memory_usage(deep=True).sum())
        except Exception:
            memory_bytes = int(self.df.memory_usage().sum())
        
        memory_mb = round(memory_bytes / (1024 * 1024), 3)
        if memory_bytes < 1024:
            memory_formatted = f"{memory_bytes} B"
        elif memory_bytes < 1024 * 1024:
            memory_formatted = f"{round(memory_bytes / 1024, 2)} KB"
        else:
            memory_formatted = f"{memory_mb} MB"

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_cells": total_cells,
            "memory_usage_bytes": memory_bytes,
            "memory_usage_mb": memory_mb,
            "memory_usage_formatted": memory_formatted,
            "total_missing_cells": total_missing_cells,
            "missing_cells_percentage": missing_cells_pct,
            "rows_with_missing": rows_with_missing,
            "rows_with_missing_percentage": rows_with_missing_pct,
            "duplicate_rows": duplicate_rows,
            "duplicate_rows_percentage": duplicate_rows_pct,
        }

    def calculate_quality_score(self, dataset_metrics: Dict[str, Any], column_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute an automated composite Data Quality Health score (0-100) and letter grade.
        
        Evaluates:
        1. Completeness (Penalty for missing cells)
        2. Uniqueness (Penalty for duplicate rows)
        3. Column Integrity (Constant columns, 100% missing columns)
        
        Returns:
            Dict[str, Any]: Data quality summary dictionary.
        """
        total_rows = dataset_metrics["total_rows"]
        total_cols = dataset_metrics["total_columns"]

        if total_rows == 0 or total_cols == 0:
            return {
                "health_score": 0,
                "health_grade": "F",
                "completeness_score": 0.0,
                "uniqueness_score": 0.0,
                "quality_status": "Empty Dataset",
                "warnings": ["Dataset contains no data rows or columns."],
                "passed_checks": 0,
                "total_checks": 5,
            }

        # 1. Completeness score (0 - 100)
        missing_pct = dataset_metrics["missing_cells_percentage"]
        completeness_score = max(0.0, round(100.0 - missing_pct, 2))

        # 2. Uniqueness score (0 - 100)
        dups_pct = dataset_metrics["duplicate_rows_percentage"]
        uniqueness_score = max(0.0, round(100.0 - dups_pct, 2))

        # 3. Check specific column anomalies
        warnings = []
        passed_checks = 0
        total_checks = 5

        # Check 1: Missing values threshold (< 5% is great)
        if missing_pct == 0:
            passed_checks += 1
        elif missing_pct > 15:
            warnings.append(f"High overall missingness: {missing_pct}% of data cells are null.")
        else:
            passed_checks += 1

        # Check 2: Redundant rows (< 1% is great)
        if dups_pct == 0:
            passed_checks += 1
        elif dups_pct > 5:
            warnings.append(f"Significant row duplication detected: {dataset_metrics['duplicate_rows']} duplicate rows ({dups_pct}%).")
        else:
            passed_checks += 1

        # Check 3: Completely empty columns
        empty_cols = [c["name"] for c in column_profiles if c.get("missing_percentage", 0) == 100.0]
        if empty_cols:
            warnings.append(f"Completely empty columns detected: {', '.join(empty_cols[:3])}{'...' if len(empty_cols) > 3 else ''}.")
        else:
            passed_checks += 1

        # Check 4: Constant/Zero-variance columns
        constant_cols = [c["name"] for c in column_profiles if c.get("is_constant", False)]
        if constant_cols:
            warnings.append(f"Constant/zero-variance columns detected: {', '.join(constant_cols[:3])}{'...' if len(constant_cols) > 3 else ''}.")
        else:
            passed_checks += 1

        # Check 5: Skewness or high outlier columns
        high_skew_cols = [c["name"] for c in column_profiles if c.get("numerical_stats") and abs(c["numerical_stats"].get("skewness") or 0) > 3.0]
        if high_skew_cols:
            warnings.append(f"Heavy skewness detected in numerical features: {', '.join(high_skew_cols[:3])}.")
        else:
            passed_checks += 1

        # Composite Health Score Calculation
        raw_score = (completeness_score * 0.50) + (uniqueness_score * 0.30) + ((passed_checks / total_checks * 100) * 0.20)
        health_score = round(max(0.0, min(100.0, raw_score)), 1)

        if health_score >= 95:
            health_grade = "A+"
            quality_status = "Excellent Quality"
        elif health_score >= 85:
            health_grade = "A"
            quality_status = "High Quality"
        elif health_score >= 75:
            health_grade = "B"
            quality_status = "Good Quality"
        elif health_score >= 60:
            health_grade = "C"
            quality_status = "Moderate Quality"
        elif health_score >= 45:
            health_grade = "D"
            quality_status = "Poor Quality"
        else:
            health_grade = "F"
            quality_status = "Critical Issues"

        return {
            "health_score": health_score,
            "health_grade": health_grade,
            "completeness_score": completeness_score,
            "uniqueness_score": uniqueness_score,
            "quality_status": quality_status,
            "warnings": warnings if warnings else ["No significant data quality issues detected."],
            "passed_checks": passed_checks,
            "total_checks": total_checks,
        }

    def profile_column(self, col_name: str) -> Dict[str, Any]:
        """
        Compute deep profiling metrics for an individual column.
        
        Calculates:
        - Common metrics: name, data_type, missing_count, missing_percentage, unique_values, duplicate_values, sample_values
        - Numerical metrics (if Numerical): mean, median, std, min, max, q1, q3, iqr, skewness
        - Categorical metrics (if Categorical): number of categories, most frequent category, mode frequency
        - Datetime metrics (if Datetime): min_date, max_date, date_range
        - Boolean metrics (if Boolean): true/false counts and percentages
        
        Args:
            col_name: Name of the column.
            
        Returns:
            Dict[str, Any]: Column profile dictionary.
        """
        series = self.df[col_name]
        total_rows = len(self.df)
        col_type = infer_column_type(series)

        # Non-null and missing values
        null_count = int(series.isnull().sum())
        non_null_count = int(total_rows - null_count)
        null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        non_null_pct = round((non_null_count / total_rows * 100), 2) if total_rows > 0 else 0.0

        # Unique & Duplicate counts
        valid_series = series.dropna()
        unique_vals = valid_series.unique()
        unique_count = len(unique_vals)
        unique_pct = round((unique_count / total_rows * 100), 2) if total_rows > 0 else 0.0

        # Duplicate values count where meaningful: non_null_count - unique_count
        duplicate_count = max(0, non_null_count - unique_count) if non_null_count > unique_count else 0
        duplicate_pct = round((duplicate_count / non_null_count * 100), 2) if non_null_count > 0 else 0.0

        # Sample values (up to 5 distinct representative values)
        sample_values = [_safe_json_value(v) for v in unique_vals[:5]]

        base_profile: Dict[str, Any] = {
            "name": str(col_name),
            "data_type": str(series.dtype),
            "pandas_dtype": str(series.dtype),
            "classified_type": col_type,
            "inferred_type": col_type,
            "non_null_count": non_null_count,
            "non_null_percentage": non_null_pct,
            "missing_count": null_count,
            "missing_percentage": null_pct,
            "unique_count": unique_count,
            "unique_percentage": unique_pct,
            "duplicate_count": duplicate_count,
            "duplicate_percentage": duplicate_pct,
            "is_constant": unique_count <= 1,
            "is_unique": unique_count == total_rows and total_rows > 0,
            "sample_values": sample_values,
            "numerical_stats": None,
            "categorical_stats": None,
            "datetime_stats": None,
            "boolean_stats": None,
            "other_stats": None,
        }

        # -----------------------------------------------------------------
        # 1. Numerical Calculations
        # -----------------------------------------------------------------
        if col_type == "Numerical" and non_null_count > 0:
            try:
                # Convert to numeric safely
                if pd.api.types.is_numeric_dtype(series):
                    num_series = valid_series.astype(float)
                else:
                    cleaned = valid_series.astype(str).str.replace(r"[$,€£%\s]", "", regex=True)
                    num_series = pd.to_numeric(cleaned, errors="coerce").dropna()

                if len(num_series) > 0:
                    mean_val = float(num_series.mean())
                    median_val = float(num_series.median())
                    std_val = float(num_series.std(ddof=1)) if len(num_series) > 1 else 0.0
                    min_val = float(num_series.min())
                    max_val = float(num_series.max())

                    q1_val = float(num_series.quantile(0.25))
                    q3_val = float(num_series.quantile(0.75))
                    iqr_val = float(q3_val - q1_val)

                    # Skewness (requires at least 3 points for sample skewness)
                    skew_val = float(num_series.skew()) if len(num_series) >= 3 else 0.0
                    if math.isnan(skew_val) or math.isinf(skew_val):
                        skew_val = 0.0

                    base_profile["numerical_stats"] = {
                        "mean": _safe_json_value(mean_val),
                        "median": _safe_json_value(median_val),
                        "std": _safe_json_value(std_val),
                        "min": _safe_json_value(min_val),
                        "max": _safe_json_value(max_val),
                        "q1": _safe_json_value(q1_val),
                        "q3": _safe_json_value(q3_val),
                        "iqr": _safe_json_value(iqr_val),
                        "skewness": _safe_json_value(skew_val),
                        "zeros_count": int((num_series == 0).sum()),
                        "zeros_percentage": round(float((num_series == 0).sum()) / len(num_series) * 100, 2),
                    }
            except Exception as e:
                logger.warning("Failed to compute numerical stats for column %s: %s", col_name, e)

        # -----------------------------------------------------------------
        # 2. Categorical Calculations
        # -----------------------------------------------------------------
        if col_type == "Categorical" and non_null_count > 0:
            try:
                val_counts = valid_series.value_counts()
                num_categories = int(len(val_counts))
                most_frequent = val_counts.index[0] if num_categories > 0 else None
                freq_most_frequent = int(val_counts.iloc[0]) if num_categories > 0 else 0
                freq_pct = round((freq_most_frequent / non_null_count * 100), 2) if non_null_count > 0 else 0.0

                top_categories = []
                for cat_val, count in val_counts.head(10).items():
                    top_categories.append({
                        "category": _safe_json_value(cat_val),
                        "count": int(count),
                        "percentage": round((int(count) / non_null_count * 100), 2) if non_null_count > 0 else 0.0,
                    })

                base_profile["categorical_stats"] = {
                    "num_categories": num_categories,
                    "most_frequent_category": _safe_json_value(most_frequent),
                    "frequency_most_frequent": freq_most_frequent,
                    "frequency_percentage": freq_pct,
                    "top_categories": top_categories,
                }
            except Exception as e:
                logger.warning("Failed to compute categorical stats for column %s: %s", col_name, e)

        # -----------------------------------------------------------------
        # 3. Datetime Calculations
        # -----------------------------------------------------------------
        if col_type == "Datetime" and non_null_count > 0:
            try:
                if pd.api.types.is_datetime64_any_dtype(series):
                    dt_series = valid_series
                else:
                    try:
                        dt_series = pd.to_datetime(valid_series, errors="coerce", format="mixed").dropna()
                    except Exception:
                        dt_series = pd.to_datetime(valid_series, errors="coerce").dropna()

                if len(dt_series) > 0:
                    min_date = dt_series.min()
                    max_date = dt_series.max()
                    date_range_delta = max_date - min_date

                    total_days = date_range_delta.days if hasattr(date_range_delta, "days") else 0
                    if total_days >= 365:
                        years = round(total_days / 365.25, 1)
                        range_str = f"{total_days:,} days (~{years} years)"
                    elif total_days >= 30:
                        months = round(total_days / 30.4, 1)
                        range_str = f"{total_days:,} days (~{months} months)"
                    else:
                        range_str = f"{total_days:,} days"

                    base_profile["datetime_stats"] = {
                        "min_date": _safe_json_value(min_date),
                        "max_date": _safe_json_value(max_date),
                        "date_range": range_str,
                        "date_range_days": total_days,
                    }
            except Exception as e:
                logger.warning("Failed to compute datetime stats for column %s: %s", col_name, e)

        # -----------------------------------------------------------------
        # 4. Boolean Calculations
        # -----------------------------------------------------------------
        if col_type == "Boolean" and non_null_count > 0:
            try:
                # Normalize values to boolean
                true_count = 0
                false_count = 0
                for v in valid_series:
                    str_v = str(v).strip().lower()
                    if v is True or str_v in ["true", "1", "1.0", "yes", "y", "t"]:
                        true_count += 1
                    else:
                        false_count += 1

                true_pct = round((true_count / non_null_count * 100), 2) if non_null_count > 0 else 0.0
                false_pct = round((false_count / non_null_count * 100), 2) if non_null_count > 0 else 0.0

                base_profile["boolean_stats"] = {
                    "true_count": true_count,
                    "false_count": false_count,
                    "true_percentage": true_pct,
                    "false_percentage": false_pct,
                }
            except Exception as e:
                logger.warning("Failed to compute boolean stats for column %s: %s", col_name, e)

        # -----------------------------------------------------------------
        # 5. Other (Text/ID) Calculations
        # -----------------------------------------------------------------
        if col_type == "Other" and non_null_count > 0:
            try:
                lengths = valid_series.astype(str).str.len()
                base_profile["other_stats"] = {
                    "avg_length": round(float(lengths.mean()), 1),
                    "min_length": int(lengths.min()),
                    "max_length": int(lengths.max()),
                }
            except Exception:
                pass

        return base_profile

    def profile(self) -> Dict[str, Any]:
        """
        Execute full profiling of the dataset.
        
        Returns:
            Dict[str, Any]: Complete structured profile containing dataset metrics,
                            data quality score, column-by-column profiles,
                            and type classification breakdown.
        """
        if self._profile_cache is not None:
            return self._profile_cache

        dataset_metrics = self.calculate_dataset_metrics()
        column_profiles = [self.profile_column(col) for col in self.df.columns]
        quality_score = self.calculate_quality_score(dataset_metrics, column_profiles)

        # Count types
        type_counts: Dict[str, int] = {
            "Numerical": 0,
            "Categorical": 0,
            "Datetime": 0,
            "Boolean": 0,
            "Other": 0,
        }
        for col_p in column_profiles:
            t = col_p["classified_type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        # Head preview (first 10 rows clean)
        head_records = self.df.head(10).replace({np.nan: None}).to_dict(orient="records")
        clean_head = [{k: _safe_json_value(v) for k, v in row.items()} for row in head_records]

        result = {
            "dataset_id": self.dataset_id,
            "overview": dataset_metrics,
            "quality": quality_score,
            "type_counts": type_counts,
            "columns": column_profiles,
            "preview_head": clean_head,
        }

        self._profile_cache = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Return the complete profile as a structured Python dictionary."""
        return self.profile()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the column-level profiling metrics into a structured summary pandas DataFrame.
        
        Returns:
            pd.DataFrame: Tabular summary of all column profiles.
        """
        profile_data = self.profile()
        rows = []
        for c in profile_data["columns"]:
            row = {
                "column_name": c["name"],
                "inferred_type": c["inferred_type"],
                "pandas_dtype": c["pandas_dtype"],
                "non_null_count": c["non_null_count"],
                "missing_count": c["missing_count"],
                "missing_percentage": c["missing_percentage"],
                "unique_count": c["unique_count"],
                "duplicate_count": c["duplicate_count"],
            }
            # Flatten type-specific stats
            if c.get("numerical_stats"):
                ns = c["numerical_stats"]
                row.update({
                    "mean": ns.get("mean"),
                    "median": ns.get("median"),
                    "std": ns.get("std"),
                    "min": ns.get("min"),
                    "max": ns.get("max"),
                    "q1": ns.get("q1"),
                    "q3": ns.get("q3"),
                    "iqr": ns.get("iqr"),
                    "skewness": ns.get("skewness"),
                })
            elif c.get("categorical_stats"):
                cs = c["categorical_stats"]
                row.update({
                    "num_categories": cs.get("num_categories"),
                    "most_frequent_category": cs.get("most_frequent_category"),
                    "frequency_most_frequent": cs.get("frequency_most_frequent"),
                })
            elif c.get("datetime_stats"):
                ds = c["datetime_stats"]
                row.update({
                    "min_date": ds.get("min_date"),
                    "max_date": ds.get("max_date"),
                    "date_range": ds.get("date_range"),
                })
            elif c.get("boolean_stats"):
                bs = c["boolean_stats"]
                row.update({
                    "true_count": bs.get("true_count"),
                    "false_count": bs.get("false_count"),
                    "true_percentage": bs.get("true_percentage"),
                })
            rows.append(row)
        return pd.DataFrame(rows)


def profile_dataset(df: pd.DataFrame, dataset_id: str = "") -> Dict[str, Any]:
    """
    Generate a full structural profile of a dataset using DatasetProfiler.
    
    Args:
        df: Input DataFrame.
        dataset_id: Optional ID of the dataset.
        
    Returns:
        Dict[str, Any]: Structured profiling dictionary.
    """
    profiler = DatasetProfiler(df, dataset_id=dataset_id)
    return profiler.to_dict()


def profile_column(series: pd.Series, col_name: str = "") -> Dict[str, Any]:
    """
    Profile an individual pandas Series.
    
    Args:
        series: Series to profile.
        col_name: Optional column name override.
        
    Returns:
        Dict[str, Any]: Column profile dictionary.
    """
    name = col_name or (str(series.name) if series.name else "column")
    df = pd.DataFrame({name: series})
    profiler = DatasetProfiler(df)
    return profiler.profile_column(name)


def get_dataset_preview(df: pd.DataFrame, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
    """
    Get paginated rows of the DataFrame for interactive tabular UI preview.
    
    Args:
        df: Input DataFrame.
        page: Current page (1-indexed).
        page_size: Number of records per page.
        
    Returns:
        Dict[str, Any]: Paginated records, total count, columns list.
    """
    total_rows = len(df)
    page = max(1, page)
    page_size = min(max(5, page_size), 100)

    total_pages = math.ceil(total_rows / page_size) if total_rows > 0 else 1
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    subset = df.iloc[start_idx:end_idx]
    records = subset.replace({np.nan: None}).to_dict(orient="records")
    clean_records = [{k: _safe_json_value(v) for k, v in row.items()} for row in records]

    return {
        "columns": [str(c) for c in df.columns],
        "rows": clean_records,
        "pagination": {
            "current_page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
    }
