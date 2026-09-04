"""
Correlation Analysis Engine for AI Data Analyst Agent (Phase 6).
Computes deterministic Pearson correlation matrices, handles edge cases (missing values,
constant zero-variance features, small samples), classifies correlation strength and direction,
identifies strongest positive and negative associations, supports user-defined threshold filtering,
and enforces strict non-causation scientific guidelines.
"""

import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from analysis.profiler import infer_column_types

logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = (
    "Statistical Notice: Correlation measures the strength and direction of a linear association "
    "between two numerical variables. Correlation does NOT imply causation. A strong statistical "
    "association between variables does not establish that changes in one variable cause changes in the other."
)


def _safe_json_value(val: Any) -> Any:
    """Safely convert numpy/pandas types to JSON-serializable Python types."""
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


def classify_correlation_strength(r: Optional[float]) -> str:
    """
    Classify the magnitude of correlation coefficient |r| into standard statistical tiers:
    - 0.80 <= |r| <= 1.00: Very strong
    - 0.60 <= |r| < 0.80: Strong
    - 0.40 <= |r| < 0.60: Moderate
    - 0.20 <= |r| < 0.40: Weak
    - 0.00 <= |r| < 0.20: Very weak
    """
    if r is None or math.isnan(r):
        return "Undefined"
    
    abs_r = abs(float(r))
    if abs_r >= 0.80:
        return "Very strong"
    elif abs_r >= 0.60:
        return "Strong"
    elif abs_r >= 0.40:
        return "Moderate"
    elif abs_r >= 0.20:
        return "Weak"
    else:
        return "Very weak"


def classify_correlation_direction(r: Optional[float], tolerance: float = 1e-4) -> str:
    """
    Classify the sign of the correlation coefficient:
    - Positive (r > 0)
    - Negative (r < 0)
    - Neutral (r == 0 or undefined)
    """
    if r is None or math.isnan(r):
        return "Neutral"
    
    val = float(r)
    if val > tolerance:
        return "Positive"
    elif val < -tolerance:
        return "Negative"
    else:
        return "Neutral"


def extract_numerical_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Identify and extract clean numeric columns from an arbitrary DataFrame.
    Returns:
    - Cleaned numeric DataFrame
    - List of valid numeric column names
    - List of constant / zero-variance column names
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(), [], []

    column_types = infer_column_types(df)
    numeric_cols = []
    constant_cols = []
    cleaned_dict = {}

    for col in df.columns:
        col_name = str(col)
        series = df[col]
        ctype = column_types.get(col_name, "Other")

        # Check if numeric dtype or classified as Numerical
        if ctype == "Numerical" or pd.api.types.is_numeric_dtype(series):
            try:
                if pd.api.types.is_numeric_dtype(series):
                    clean_s = pd.to_numeric(series, errors="coerce")
                else:
                    # Clean currency/percentage formatting
                    cleaned_str = series.dropna().astype(str).str.replace(r"[$,€£%\s]", "", regex=True)
                    clean_s = pd.to_numeric(cleaned_str, errors="coerce")
                
                # Check for constant column or zero non-null values
                non_nulls = clean_s.dropna()
                if len(non_nulls) == 0:
                    continue
                
                if non_nulls.nunique() <= 1 or non_nulls.std() == 0.0:
                    constant_cols.append(col_name)
                
                numeric_cols.append(col_name)
                cleaned_dict[col_name] = clean_s
            except Exception as e:
                logger.debug("Could not convert column '%s' to numeric: %s", col_name, e)
                continue

    if not cleaned_dict:
        return pd.DataFrame(), [], []

    num_df = pd.DataFrame(cleaned_dict, index=df.index)
    return num_df, numeric_cols, constant_cols


def calculate_pearson_correlation_matrix(
    df: pd.DataFrame,
    min_periods: int = 2
) -> Dict[str, Any]:
    """
    Compute full symmetric Pearson correlation matrix with pairwise observation counts.
    Safely handles missing values and constant features without throwing exceptions.
    
    Returns structured dict with columns, matrix data, and pairwise sample sizes.
    """
    num_df, numeric_cols, constant_cols = extract_numerical_dataframe(df)
    
    if len(numeric_cols) == 0:
        return {
            "columns": [],
            "matrix": {},
            "sample_sizes": {},
            "constant_columns": [],
            "total_numerical_columns": 0,
        }

    # Compute pairwise Pearson correlation
    corr_df = num_df.corr(method="pearson", min_periods=min_periods)
    
    matrix_dict: Dict[str, Dict[str, Optional[float]]] = {}
    sample_sizes: Dict[str, Dict[str, int]] = {}

    for col_a in numeric_cols:
        matrix_dict[col_a] = {}
        sample_sizes[col_a] = {}
        
        is_const_a = col_a in constant_cols
        series_a = num_df[col_a]

        for col_b in numeric_cols:
            is_const_b = col_b in constant_cols
            series_b = num_df[col_b]

            # Calculate pairwise valid non-null count
            valid_mask = series_a.notna() & series_b.notna()
            n_pair = int(valid_mask.sum())
            sample_sizes[col_a][col_b] = n_pair

            if col_a == col_b:
                matrix_dict[col_a][col_b] = 1.0 if not is_const_a else 1.0
                continue

            if is_const_a or is_const_b or n_pair < min_periods:
                matrix_dict[col_a][col_b] = 0.0
                continue

            val = corr_df.loc[col_a, col_b] if col_a in corr_df.index and col_b in corr_df.columns else None
            if val is None or pd.isna(val) or math.isnan(val):
                # Try direct scipy computation as fallback
                try:
                    r_calc, _ = stats.pearsonr(series_a[valid_mask], series_b[valid_mask])
                    matrix_dict[col_a][col_b] = _safe_json_value(r_calc)
                except Exception:
                    matrix_dict[col_a][col_b] = 0.0
            else:
                matrix_dict[col_a][col_b] = _safe_json_value(val)

    return {
        "columns": numeric_cols,
        "matrix": matrix_dict,
        "sample_sizes": sample_sizes,
        "constant_columns": constant_cols,
        "total_numerical_columns": len(numeric_cols),
    }


def compute_ranked_correlation_pairs(
    matrix_data: Dict[str, Any],
    threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Extract all unique unordered pairs (Var A, Var B) where A != B,
    calculate strength, direction, and sort by absolute correlation descending.
    Filters pairs with |r| >= threshold.
    """
    columns = matrix_data.get("columns", [])
    matrix = matrix_data.get("matrix", {})
    sample_sizes = matrix_data.get("sample_sizes", {})

    pairs = []
    seen_pairs = set()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_a = columns[i]
            col_b = columns[j]

            pair_key = tuple(sorted([col_a, col_b]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            r_val = matrix.get(col_a, {}).get(col_b)
            n_obs = sample_sizes.get(col_a, {}).get(col_b, 0)

            if r_val is None:
                r_num = 0.0
            else:
                r_num = float(r_val)

            abs_r = abs(r_num)
            if abs_r < threshold:
                continue

            strength = classify_correlation_strength(r_num)
            direction = classify_correlation_direction(r_num)

            pairs.append({
                "variable_a": col_a,
                "variable_b": col_b,
                "pair_label": f"{col_a} ↔ {col_b}",
                "correlation": _safe_json_value(r_num),
                "absolute_correlation": _safe_json_value(abs_r),
                "strength": strength,
                "direction": direction,
                "sample_size": n_obs,
            })

    # Sort pairs by absolute correlation descending
    pairs.sort(key=lambda x: (x["absolute_correlation"] if x["absolute_correlation"] is not None else 0.0), reverse=True)
    return pairs


def generate_key_correlations(
    ranked_pairs: List[Dict[str, Any]],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Extract strongest positive and strongest negative correlations,
    and generate deterministic non-causal analytical highlight cards.
    """
    if not ranked_pairs:
        return {
            "strongest_positive": [],
            "strongest_negative": [],
            "key_highlights": [],
            "summary_text": "No significant numerical correlations detected.",
        }

    positive_pairs = [p for p in ranked_pairs if (p["correlation"] or 0) > 0]
    negative_pairs = [p for p in ranked_pairs if (p["correlation"] or 0) < 0]

    # Positive sorted descending by correlation
    positive_pairs.sort(key=lambda x: (x["correlation"] or 0), reverse=True)
    # Negative sorted ascending by correlation (most negative first)
    negative_pairs.sort(key=lambda x: (x["correlation"] or 0))

    top_positive = positive_pairs[:top_k]
    top_negative = negative_pairs[:top_k]

    # Build key highlight objects
    highlights = []

    for p in top_positive:
        r = p["correlation"]
        highlights.append({
            "pair": f"{p['variable_a']} ↔ {p['variable_b']}",
            "variable_a": p["variable_a"],
            "variable_b": p["variable_b"],
            "correlation": r,
            "direction": "Positive",
            "strength": p["strength"],
            "badge_class": "badge-positive",
            "explanation": (
                f"As '{p['variable_a']}' increases, '{p['variable_b']}' tends to increase linearly "
                f"(r = {r:.4f}, {p['strength'].lower()} positive association). Non-causal: Does not imply direct causation."
            ),
        })

    for p in top_negative:
        r = p["correlation"]
        highlights.append({
            "pair": f"{p['variable_a']} ↔ {p['variable_b']}",
            "variable_a": p["variable_a"],
            "variable_b": p["variable_b"],
            "correlation": r,
            "direction": "Negative",
            "strength": p["strength"],
            "badge_class": "badge-negative",
            "explanation": (
                f"As '{p['variable_a']}' increases, '{p['variable_b']}' tends to decrease linearly "
                f"(r = {r:.4f}, {p['strength'].lower()} inverse association). Non-causal: Does not imply direct causation."
            ),
        })

    # Summary statement
    if highlights:
        top_lead = highlights[0]
        summary_text = (
            f"Strongest observed association is between {top_lead['pair']} (r = {top_lead['correlation']}, "
            f"{top_lead['strength']} {top_lead['direction']})."
        )
    else:
        summary_text = "No strong linear correlations identified across numerical features."

    return {
        "strongest_positive": top_positive,
        "strongest_negative": top_negative,
        "key_highlights": highlights,
        "summary_text": summary_text,
    }


class CorrelationAnalysisEngine:
    """
    Enterprise-grade Correlation Analysis Engine.
    Coordinates automated numerical feature extraction, Pearson correlation matrix computation,
    strength & direction categorization, threshold filtering, ranked association tables,
    and non-causal analytical insights.
    """

    def __init__(self, df: pd.DataFrame, dataset_id: str = ""):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("CorrelationAnalysisEngine requires a pandas DataFrame instance.")
        self.df = df
        self.dataset_id = dataset_id
        self._analysis_cache: Optional[Dict[str, Any]] = None

    def analyze(self, threshold: float = 0.0) -> Dict[str, Any]:
        """
        Execute complete correlation analysis pipeline.
        
        Args:
            threshold: Minimum absolute correlation coefficient |r| to filter pairs (0.0 to 1.0).
            
        Returns:
            Dict[str, Any]: Structured correlation analysis result payload.
        """
        # 1. Compute base matrix
        matrix_data = calculate_pearson_correlation_matrix(self.df)
        columns = matrix_data["columns"]
        
        # 2. Extract ranked pairs with threshold
        all_pairs = compute_ranked_correlation_pairs(matrix_data, threshold=0.0)
        filtered_pairs = compute_ranked_correlation_pairs(matrix_data, threshold=threshold)

        # 3. Categorize counts by strength
        strength_counts = {
            "very_strong": sum(1 for p in all_pairs if p["strength"] == "Very strong"),
            "strong": sum(1 for p in all_pairs if p["strength"] == "Strong"),
            "moderate": sum(1 for p in all_pairs if p["strength"] == "Moderate"),
            "weak": sum(1 for p in all_pairs if p["strength"] == "Weak"),
            "very_weak": sum(1 for p in all_pairs if p["strength"] == "Very weak"),
        }

        # 4. Generate Key Correlations
        key_correlations = generate_key_correlations(all_pairs, top_k=5)

        result = {
            "dataset_id": self.dataset_id,
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "total_numerical_columns": len(columns),
            "numerical_columns": columns,
            "constant_columns": matrix_data["constant_columns"],
            "total_pairs_count": len(all_pairs),
            "filtered_pairs_count": len(filtered_pairs),
            "current_threshold": threshold,
            "strength_counts": strength_counts,
            "correlation_matrix": matrix_data["matrix"],
            "sample_sizes": matrix_data["sample_sizes"],
            "ranked_pairs": filtered_pairs,
            "all_ranked_pairs": all_pairs,
            "key_correlations": key_correlations,
            "disclaimer": DISCLAIMER_TEXT,
        }

        return result

    def to_dict(self, threshold: float = 0.0) -> Dict[str, Any]:
        """Return analysis result as a structured dictionary."""
        return self.analyze(threshold=threshold)

    def ranked_pairs_dataframe(self, threshold: float = 0.0) -> pd.DataFrame:
        """
        Export ranked correlation pairs as a clean pandas DataFrame.
        """
        analysis = self.analyze(threshold=threshold)
        rows = []
        for p in analysis["ranked_pairs"]:
            rows.append({
                "Variable A": p["variable_a"],
                "Variable B": p["variable_b"],
                "Correlation": p["correlation"],
                "Strength": p["strength"],
                "Direction": p["direction"],
                "Sample Size": p["sample_size"],
            })
        return pd.DataFrame(rows)

    def matrix_dataframe(self) -> pd.DataFrame:
        """
        Export Pearson correlation matrix as a pandas DataFrame.
        """
        matrix_data = calculate_pearson_correlation_matrix(self.df)
        cols = matrix_data["columns"]
        matrix = matrix_data["matrix"]
        data = [[matrix.get(r, {}).get(c, 0.0) for c in cols] for r in cols]
        return pd.DataFrame(data, index=cols, columns=cols)


def compute_correlation_analysis(
    df: pd.DataFrame,
    dataset_id: str = "",
    threshold: float = 0.0
) -> Dict[str, Any]:
    """
    Functional entry point for correlation analysis on a DataFrame.
    """
    engine = CorrelationAnalysisEngine(df, dataset_id=dataset_id)
    return engine.analyze(threshold=threshold)
