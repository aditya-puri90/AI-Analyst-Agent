"""
Dataset Profiling and Schema Inference Engine.
Performs deterministic exploratory data analysis, type inference, missing value analysis,
and memory auditing on arbitrary tabular datasets.
"""

import math
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def infer_column_type(series: pd.Series) -> str:
    """
    Infer the semantic data type of a Pandas Series dynamically.
    
    Categories:
    - 'numerical': integer or floating point values.
    - 'boolean': boolean, binary 0/1, or yes/no / true/false.
    - 'datetime': parseable dates or timestamps.
    - 'categorical': discrete categories with low/medium cardinality.
    - 'id_or_text': high-cardinality strings, codes, or free text.
    
    Args:
        series: Pandas Series to analyze.
        
    Returns:
        str: Inferred semantic type.
    """
    # Drop NAs for type inference
    valid_series = series.dropna()
    n_valid = len(valid_series)

    if n_valid == 0:
        return "unknown"

    # 1. Check if native boolean or binary string/integer
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    unique_vals = set(valid_series.unique())
    if unique_vals.issubset({0, 1, 0.0, 1.0}):
        # Could be binary/boolean flag
        return "boolean"
    if unique_vals.issubset({"true", "false", "True", "False", "TRUE", "FALSE", "yes", "no", "Yes", "No", "Y", "N"}):
        return "boolean"

    # 2. Check native numerical types
    if pd.api.types.is_numeric_dtype(series):
        # If integer with very low cardinality (< 5% unique and unique < 10), could be coded categorical
        unique_count = len(unique_vals)
        if unique_count <= 5 and n_valid >= 20:
            return "categorical"
        return "numerical"

    # 3. Check for datetime parseability
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Attempt datetime conversion on string sample
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = valid_series.head(100).astype(str)
        # Avoid purely numeric strings being parsed as unix epochs unless they look like dates
        has_date_separators = sample.str.contains(r"[-/:\s,]", regex=True).any()
        if has_date_separators:
            try:
                # Convert sample with pd.to_datetime using format='mixed' if supported
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().sum() / len(sample) >= 0.8:
                    return "datetime"
            except Exception:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() / len(sample) >= 0.8:
                        return "datetime"
                except Exception:
                    pass

        # 4. Check if string values are convertible to numerical (e.g. "$1,200.50" or "45%")
        cleaned_sample = sample.str.replace(r"[$,€£%\s]", "", regex=True)
        try:
            converted = pd.to_numeric(cleaned_sample, errors="coerce")
            if converted.notna().sum() / len(sample) >= 0.9:
                return "numerical"
        except Exception:
            pass

    # 5. Distinguish between Categorical and ID/Free-Text
    unique_count = len(unique_vals)
    unique_ratio = unique_count / n_valid if n_valid > 0 else 0
    col_name_str = str(series.name).lower() if hasattr(series, "name") and series.name else ""

    # Clear ID column names with unique values
    is_id_name = any(id_tag in col_name_str for id_tag in ["_id", "id_", "uuid", "guid", "code", "identifier", "key"]) or col_name_str == "id"
    if is_id_name and (unique_ratio >= 0.6 or unique_count >= 10):
        return "id_or_text"

    # Strict 100% unique key in larger datasets
    if unique_count == n_valid and n_valid >= 10:
        return "id_or_text"

    # Discrete categories (low cardinality count)
    if unique_count <= 25:
        return "categorical"

    # High percentage of duplicates
    if unique_ratio < 0.30 or unique_count <= 50:
        return "categorical"
    
    # High cardinality fallback
    return "id_or_text"


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer semantic types for all columns in a DataFrame.
    
    Args:
        df: Pandas DataFrame.
        
    Returns:
        Dict[str, str]: Mapping of column name to inferred type.
    """
    return {col: infer_column_type(df[col]) for col in df.columns}


def _safe_json_value(val: Any) -> Any:
    """Ensure data is serializable to standard JSON."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 4)
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    return str(val)


def profile_dataset(df: pd.DataFrame, dataset_id: str = "") -> Dict[str, Any]:
    """
    Generate a full structural profile of a dataset.
    
    Args:
        df: Input DataFrame.
        dataset_id: Optional ID of the dataset.
        
    Returns:
        Dict[str, Any]: Comprehensive profile metrics and column breakdown.
    """
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols if total_rows > 0 else 0

    # Missingness analysis
    null_counts = df.isnull().sum()
    total_missing_cells = int(null_counts.sum())
    missing_cells_pct = round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
    rows_with_missing = int((df.isnull().any(axis=1)).sum())
    rows_with_missing_pct = round((rows_with_missing / total_rows * 100), 2) if total_rows > 0 else 0.0

    # Duplication analysis
    duplicate_rows = int(df.duplicated().sum())
    duplicate_rows_pct = round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0

    # Memory calculation
    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_mb = round(memory_bytes / (1024 * 1024), 3)

    # Inferred types
    inferred_types = infer_column_types(df)

    # Type breakdown summary
    type_counts: Dict[str, int] = {}
    for col_type in inferred_types.values():
        type_counts[col_type] = type_counts.get(col_type, 0) + 1

    # Column-level profiles
    columns_profile = []
    for col in df.columns:
        series = df[col]
        col_type = inferred_types.get(col, "unknown")
        col_null_count = int(null_counts[col])
        col_null_pct = round((col_null_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        unique_vals = series.dropna().unique()
        unique_count = len(unique_vals)
        unique_pct = round((unique_count / total_rows * 100), 2) if total_rows > 0 else 0.0

        # Sample values (up to 5 distinct values)
        sample_vals = [_safe_json_value(v) for v in unique_vals[:5]]

        col_meta = {
            "name": str(col),
            "inferred_type": col_type,
            "pandas_dtype": str(series.dtype),
            "non_null_count": int(total_rows - col_null_count),
            "null_count": col_null_count,
            "null_percentage": col_null_pct,
            "unique_count": unique_count,
            "unique_percentage": unique_pct,
            "is_constant": unique_count <= 1,
            "is_unique": unique_count == total_rows and total_rows > 0,
            "sample_values": sample_vals,
        }
        columns_profile.append(col_meta)

    # Head and Tail previews
    head_records = df.head(10).replace({np.nan: None}).to_dict(orient="records")
    clean_head = [{k: _safe_json_value(v) for k, v in row.items()} for row in head_records]

    profile_result = {
        "dataset_id": dataset_id,
        "overview": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_cells": total_cells,
            "memory_usage_mb": memory_mb,
            "memory_usage_bytes": memory_bytes,
            "total_missing_cells": total_missing_cells,
            "missing_cells_percentage": missing_cells_pct,
            "rows_with_missing": rows_with_missing,
            "rows_with_missing_percentage": rows_with_missing_pct,
            "duplicate_rows": duplicate_rows,
            "duplicate_rows_percentage": duplicate_rows_pct,
            "type_counts": type_counts,
        },
        "columns": columns_profile,
        "preview_head": clean_head,
    }

    return profile_result


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
