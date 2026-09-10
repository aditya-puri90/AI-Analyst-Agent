"""
Interactive Plotly Visualization Engine & Automatic Chart Recommendation System.
Phase 8 of AI Data Analyst Agent.

Provides:
1. Column-type and semantic role inspection (Numerical, Categorical, Datetime, Boolean, ID).
2. Rule-based automatic chart recommendation engine:
   - Single Numerical: Histogram, Box Plot
   - Single Categorical: Bar Chart, Donut / Category Distribution
   - Numerical + Numerical: Scatter Plot with Correlation & Trend
   - Datetime + Numerical: Time Series Line / Area Chart
   - Categorical + Numerical: Grouped / Aggregated Bar Chart
   - Datetime + Categorical: Time-Based Category Distribution
3. Intelligent chart candidate pruning, ranking, and priority scoring.
4. Interactive custom chart builder ("Build Your Own Chart") supporting multiple types & aggregations.
5. High-performance, dark-theme Plotly JSON specifications with rich tooltips, annotations, and responsive layouts.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import math
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ==============================================================================
# Design System & Plotly Dark Theme Palette
# ==============================================================================

DARK_THEME_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Plus Jakarta Sans, -apple-system, sans-serif", "color": "#cbd5e1"},
    "margin": {"l": 55, "r": 35, "t": 50, "b": 60, "pad": 4},
    "xaxis": {
        "tickfont": {"color": "#94a3b8", "size": 10, "family": "JetBrains Mono, monospace"},
        "gridcolor": "rgba(255, 255, 255, 0.05)",
        "zeroline": False,
        "linecolor": "rgba(255, 255, 255, 0.1)",
    },
    "yaxis": {
        "tickfont": {"color": "#94a3b8", "size": 10, "family": "JetBrains Mono, monospace"},
        "gridcolor": "rgba(255, 255, 255, 0.05)",
        "zeroline": False,
        "linecolor": "rgba(255, 255, 255, 0.1)",
    },
    "hoverlabel": {
        "bgcolor": "#1e293b",
        "bordercolor": "#6366f1",
        "font": {"color": "#f8fafc", "family": "Plus Jakarta Sans, sans-serif", "size": 12},
    },
    "legend": {
        "bgcolor": "rgba(15, 23, 42, 0.7)",
        "bordercolor": "rgba(255, 255, 255, 0.08)",
        "borderwidth": 1,
        "font": {"color": "#cbd5e1", "size": 11},
    },
}

PLOTLY_CONFIG = {
    "responsive": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "data_chart", "height": 600, "width": 900, "scale": 2},
}

COLOR_PALETTE = [
    "#6366f1",  # Indigo
    "#06b6d4",  # Cyan
    "#10b981",  # Emerald
    "#f59e0b",  # Amber
    "#f43f5e",  # Rose
    "#a855f7",  # Purple
    "#ec4899",  # Pink
    "#3b82f6",  # Blue
    "#14b8a6",  # Teal
    "#84cc16",  # Lime
    "#eab308",  # Yellow
    "#8b5cf6",  # Violet
]


def _merge_layout(custom_layout: Dict[str, Any], height: int = 380) -> Dict[str, Any]:
    """Helper to merge custom layout overrides into the standardized dark theme layout."""
    import copy
    layout = copy.deepcopy(DARK_THEME_LAYOUT)
    layout["height"] = height
    for k, v in custom_layout.items():
        if isinstance(v, dict) and k in layout and isinstance(layout[k], dict):
            layout[k].update(v)
        else:
            layout[k] = v
    return layout


# ==============================================================================
# 1. Column Semantic Role & Schema Inspection
# ==============================================================================

def inspect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Inspect all columns in a DataFrame and classify them into semantic roles:
    - 'numerical': Continuous/discrete numbers with non-zero variance.
    - 'categorical': Low to moderate cardinality categorical labels.
    - 'datetime': Parseable timestamps and temporal sequences.
    - 'boolean': Binary flags (True/False, 0/1, Yes/No).
    - 'identifier': High-cardinality ID columns (e.g. Order_ID, UUIDs).
    """
    classified = {
        "numerical": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "identifier": [],
    }

    if df.empty:
        return classified

    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        clean_s = series.dropna()
        n_valid = len(clean_s)

        if n_valid == 0:
            continue

        n_unique = clean_s.nunique()
        col_str = str(col).lower()

        # 1. Datetime check (native or parseable)
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
            classified["datetime"].append(col)
            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sample = clean_s.head(50).astype(str)
            if sample.str.contains(r"[-/:\s,]", regex=True).any():
                try:
                    try:
                        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    except Exception:
                        parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() / len(sample) >= 0.8:
                        classified["datetime"].append(col)
                        continue
                except Exception:
                    pass

        # 2. Boolean check
        if pd.api.types.is_bool_dtype(series):
            classified["boolean"].append(col)
            continue

        unique_vals = set(clean_s.unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}) and (n_unique <= 2):
            if any(bool_kw in col_str for bool_kw in ["is_", "has_", "flag", "active", "enabled", "selected"]):
                classified["boolean"].append(col)
                continue

        str_unique_lower = {str(v).strip().lower() for v in unique_vals}
        if any(str_unique_lower.issubset(b_set) for b_set in [{"true", "false"}, {"yes", "no"}, {"y", "n"}]):
            classified["boolean"].append(col)
            continue

        # 3. Check for identifier columns
        is_id_name = any(id_kw in col_str for id_kw in ["_id", "id_", "uuid", "guid", "ssn", "key", "code"]) or col_str == "id"
        if is_id_name and not pd.api.types.is_numeric_dtype(series):
            classified["identifier"].append(col)
            continue

        if (n_unique > 50 and n_unique / max(1, n_rows) >= 0.85) and not pd.api.types.is_numeric_dtype(series):
            classified["identifier"].append(col)
            continue

        # 4. Numerical check
        if pd.api.types.is_numeric_dtype(series):
            # Check if sequential ID like 1, 2, 3...
            if is_id_name and n_unique / max(1, n_rows) >= 0.95:
                classified["identifier"].append(col)
            else:
                num_s = pd.to_numeric(clean_s, errors="coerce").dropna()
                if len(num_s) > 0 and num_s.std() > 0:
                    classified["numerical"].append(col)
            continue

        # 5. Categorical check
        if n_unique <= 60 or (n_unique / max(1, n_rows) <= 0.5):
            classified["categorical"].append(col)
        else:
            classified["identifier"].append(col)

    return classified


# ==============================================================================
# 2. Rule-Based Plotly Chart Generators
# ==============================================================================

def build_histogram_spec(
    df: pd.DataFrame,
    col: str,
    bins: Optional[int] = None,
    height: int = 380,
) -> Dict[str, Any]:
    """
    Generate an interactive Plotly Histogram with distribution statistics,
    mean/median annotations, and missing value indicators.
    """
    if col not in df.columns:
        return _empty_chart_spec(f"Column '{col}' not found in dataset.")

    series = pd.to_numeric(df[col], errors="coerce")
    clean_s = series.dropna()

    if len(clean_s) == 0:
        return _empty_chart_spec(f"No valid numerical data in '{col}'.")

    missing_count = len(series) - len(clean_s)
    mean_val = float(clean_s.mean())
    median_val = float(clean_s.median())
    std_val = float(clean_s.std()) if len(clean_s) > 1 else 0.0

    n_bins = bins or min(35, max(12, int(len(clean_s) ** 0.5)))

    trace = {
        "type": "histogram",
        "x": clean_s.tolist(),
        "name": f"{col} Frequency",
        "marker": {
            "color": "rgba(99, 102, 241, 0.65)",
            "line": {"color": "#818cf8", "width": 1.5},
        },
        "opacity": 0.85,
        "nbinsx": n_bins,
        "hovertemplate": (
            f"<b>{col} Range:</b> %{{x}}<br>"
            "<b>Count:</b> %{y}<br>"
            "<extra></extra>"
        ),
    }

    shapes = [
        # Mean line
        {
            "type": "line",
            "x0": mean_val,
            "x1": mean_val,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": "#34d399", "width": 2, "dash": "dash"},
        },
        # Median line
        {
            "type": "line",
            "x0": median_val,
            "x1": median_val,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": "#fbbf24", "width": 2, "dash": "dot"},
        },
    ]

    annotations = [
        {
            "x": mean_val,
            "y": 1.05,
            "yref": "paper",
            "text": f"Mean: {mean_val:.2f}",
            "showarrow": False,
            "font": {"color": "#34d399", "size": 10, "family": "JetBrains Mono"},
            "bgcolor": "rgba(15, 23, 42, 0.8)",
            "borderpad": 2,
        },
        {
            "x": median_val,
            "y": 0.95,
            "yref": "paper",
            "text": f"Median: {median_val:.2f}",
            "showarrow": False,
            "font": {"color": "#fbbf24", "size": 10, "family": "JetBrains Mono"},
            "bgcolor": "rgba(15, 23, 42, 0.8)",
            "borderpad": 2,
        },
    ]

    subtitle = f"N = {len(clean_s):,}" + (f" • {missing_count} missing rows excluded" if missing_count > 0 else "")

    layout = _merge_layout({
        "title": {
            "text": f"<b>Distribution Histogram of {col}</b><br><sup style='color:#94a3b8;'>{subtitle}</sup>",
            "font": {"size": 14, "color": "#f8fafc"},
            "x": 0.02,
        },
        "xaxis": {"title": {"text": col, "font": {"color": "#cbd5e1", "size": 11}}},
        "yaxis": {"title": {"text": "Observation Count", "font": {"color": "#cbd5e1", "size": 11}}},
        "shapes": shapes,
        "annotations": annotations,
        "showlegend": False,
    }, height=height)

    return {"data": [trace], "layout": layout, "config": PLOTLY_CONFIG}


def build_single_boxplot_spec(
    df: pd.DataFrame,
    col: str,
    height: int = 380,
) -> Dict[str, Any]:
    """
    Generate an interactive Plotly Box Plot for a single numerical column
    showing median, quartiles, IQR fences, and individual outlier markers.
    """
    if col not in df.columns:
        return _empty_chart_spec(f"Column '{col}' not found in dataset.")

    series = pd.to_numeric(df[col], errors="coerce")
    clean_s = series.dropna()

    if len(clean_s) == 0:
        return _empty_chart_spec(f"No valid numerical data in '{col}'.")

    missing_count = len(series) - len(clean_s)
    plot_s = clean_s if len(clean_s) <= 5000 else clean_s.sample(5000, random_state=42)

    trace = {
        "type": "box",
        "y": plot_s.tolist(),
        "name": col,
        "boxpoints": "outliers",
        "jitter": 0.35,
        "pointpos": -1.8,
        "fillcolor": "rgba(99, 102, 241, 0.25)",
        "line": {"color": "#818cf8", "width": 2},
        "marker": {
            "size": 6,
            "color": "#f43f5e",
            "outliercolor": "#fb7185",
            "line": {"color": "#ffffff", "width": 0.8},
        },
        "boxmean": True,
        "hovertemplate": (
            f"<b>{col}</b><br>"
            "Value: %{y}<br>"
            "<extra></extra>"
        ),
    }

    subtitle = f"Min: {clean_s.min():.2f} • Q1: {clean_s.quantile(0.25):.2f} • Median: {clean_s.median():.2f} • Q3: {clean_s.quantile(0.75):.2f} • Max: {clean_s.max():.2f}"

    layout = _merge_layout({
        "title": {
            "text": f"<b>Box Plot & Quartile Spread of {col}</b><br><sup style='color:#94a3b8;'>{subtitle}</sup>",
            "font": {"size": 14, "color": "#f8fafc"},
            "x": 0.02,
        },
        "yaxis": {"title": {"text": col, "font": {"color": "#cbd5e1", "size": 11}}},
        "xaxis": {"showticklabels": False},
        "showlegend": False,
    }, height=height)

    return {"data": [trace], "layout": layout, "config": PLOTLY_CONFIG}


def build_bar_chart_spec(
    df: pd.DataFrame,
    col: str,
    top_n: int = 15,
    height: int = 380,
) -> Dict[str, Any]:
    """
    Generate an interactive Plotly Bar Chart for a categorical or boolean column.
    Renders top N categories with count labels, percentages, and handles "+ Other".
    """
    if col not in df.columns:
        return _empty_chart_spec(f"Column '{col}' not found in dataset.")

    series = df[col].astype(str).replace({"nan": np.nan, "None": np.nan})
    clean_s = series.dropna()

    if len(clean_s) == 0:
        return _empty_chart_spec(f"No valid categorical data in '{col}'.")

    missing_count = len(series) - len(clean_s)
    val_counts = clean_s.value_counts()
    total_valid = len(clean_s)

    if len(val_counts) > top_n:
        top_counts = val_counts.head(top_n)
        other_sum = val_counts.iloc[top_n:].sum()
        x_vals = list(top_counts.index) + ["Other (Aggregated)"]
        y_vals = [int(v) for v in top_counts.values] + [int(other_sum)]
    else:
        x_vals = list(val_counts.index)
        y_vals = [int(v) for v in val_counts.values]

    pct_vals = [round((y / total_valid) * 100, 1) for y in y_vals]
    hover_text = [f"<b>{x}</b><br>Count: {y:,}<br>Share: {pct}%" for x, y, pct in zip(x_vals, y_vals, pct_vals)]

    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(x_vals))]

    trace = {
        "type": "bar",
        "x": x_vals,
        "y": y_vals,
        "text": [f"{y:,}" for y in y_vals],
        "textposition": "auto",
        "textfont": {"color": "#f8fafc", "family": "JetBrains Mono", "size": 10},
        "marker": {
            "color": colors,
            "line": {"color": "rgba(255, 255, 255, 0.2)", "width": 1},
        },
        "hoverinfo": "text",
        "text": hover_text,
    }

    subtitle = f"{len(val_counts)} Unique Categories • Total N = {total_valid:,}" + (f" • {missing_count} missing rows excluded" if missing_count > 0 else "")

    layout = _merge_layout({
        "title": {
            "text": f"<b>Frequency Distribution of {col}</b><br><sup style='color:#94a3b8;'>{subtitle}</sup>",
            "font": {"size": 14, "color": "#f8fafc"},
            "x": 0.02,
        },
        "xaxis": {
            "title": {"text": col, "font": {"color": "#cbd5e1", "size": 11}},
            "tickangle": -35 if len(x_vals) > 5 else 0,
        },
        "yaxis": {"title": {"text": "Frequency / Count", "font": {"color": "#cbd5e1", "size": 11}}},
        "showlegend": False,
    }, height=height)

    return {"data": [trace], "layout": layout, "config": PLOTLY_CONFIG}


def build_category_distribution_spec(
    df: pd.DataFrame,
    col: str,
    top_n: int = 10,
    height: int = 380,
) -> Dict[str, Any]:
    """
    Generate an interactive Plotly Donut / Pie chart showing category proportions.
    """
    if col not in df.columns:
        return _empty_chart_spec(f"Column '{col}' not found in dataset.")

    series = df[col].astype(str).replace({"nan": np.nan, "None": np.nan})
    clean_s = series.dropna()

    if len(clean_s) == 0:
        return _empty_chart_spec(f"No valid categorical data in '{col}'.")

    val_counts = clean_s.value_counts()
    total_valid = len(clean_s)

    if len(val_counts) > top_n:
        top_counts = val_counts.head(top_n)
        other_sum = val_counts.iloc[top_n:].sum()
        labels = list(top_counts.index) + ["Other (Aggregated)"]
        values = [int(v) for v in top_counts.values] + [int(other_sum)]
    else:
        labels = list(val_counts.index)
        values = [int(v) for v in val_counts.values]

    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(labels))]

    trace = {
        "type": "pie",
        "labels": labels,
        "values": values,
        "hole": 0.52,
        "marker": {
            "colors": colors,
            "line": {"color": "#0f172a", "width": 2},
        },
        "textinfo": "percent+label",
        "textposition": "inside",
        "insidetextorientation": "radial",
        "hovertemplate": (
            "<b>Category:</b> %{label}<br>"
            "<b>Count:</b> %{value:,}<br>"
            "<b>Percentage:</b> %{percent}<br>"
            "<extra></extra>"
        ),
    }

    layout = _merge_layout({
        "title": {
            "text": f"<b>Category Share of {col}</b><br><sup style='color:#94a3b8;'>N = {total_valid:,} records</sup>",
            "font": {"size": 14, "color": "#f8fafc"},
            "x": 0.02,
        },
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "y": -0.15,
            "x": 0.5,
            "xanchor": "center",
        },
        "annotations": [{
            "font": {"size": 13, "color": "#f8fafc", "family": "Plus Jakarta Sans"},
            "showarrow": False,
            "text": f"<b>{len(labels)}</b><br><span style='font-size:10px;color:#94a3b8;'>Classes</span>",
            "x": 0.5,
            "y": 0.5,
        }],
    }, height=height)

    return {"data": [trace], "layout": layout, "config": PLOTLY_CONFIG}


def build_scatter_spec(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    height: int = 400,
) -> Dict[str, Any]:
    """
    Generate an interactive Plotly Scatter Plot for two numerical columns,
    annotating Pearson correlation (r) and optional trendline.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return _empty_chart_spec(f"Columns '{x_col}' and/or '{y_col}' not found.")

    sub_cols = [x_col, y_col] + ([color_col] if color_col and color_col in df.columns else [])
    sub_df = df[sub_cols].dropna().copy()

    # Cast to numeric
    sub_df[x_col] = pd.to_numeric(sub_df[x_col], errors="coerce")
    sub_df[y_col] = pd.to_numeric(sub_df[y_col], errors="coerce")
    sub_df = sub_df.dropna(subset=[x_col, y_col])

    if len(sub_df) == 0:
        return _empty_chart_spec(f"No valid pairwise observations for '{x_col}' and '{y_col}'.")

    plot_df = sub_df if len(sub_df) <= 3000 else sub_df.sample(3000, random_state=42)

    # Compute correlation
    r_val = float(sub_df[x_col].corr(sub_df[y_col])) if len(sub_df) > 1 else 0.0
    r_str = f"r = {r_val:+.3f}" if not math.isnan(r_val) else "r = N/A"

    traces = []

    if color_col and color_col in plot_df.columns:
        unique_groups = plot_df[color_col].unique()[:8]
        for i, grp in enumerate(unique_groups):
            grp_df = plot_df[plot_df[color_col] == grp]
            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "x": grp_df[x_col].tolist(),
                "y": grp_df[y_col].tolist(),
                "name": str(grp),
                "marker": {
                    "size": 7,
                    "color": color,
                    "opacity": 0.8,
                    "line": {"color": "#ffffff", "width": 0.6},
                },
                "hovertemplate": (
                    f"<b>{color_col}:</b> {grp}<br>"
                    f"<b>{x_col}:</b> %{{x}}<br>"
                    f"<b>{y_col}:</b> %{{y}}<br>"
                    "<extra></extra>"
                ),
            })
    else:
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": plot_df[x_col].tolist(),
            "y": plot_df[y_col].tolist(),
            "name": "Observations",
            "marker": {
                "size": 7,
                "color": "rgba(99, 102, 241, 0.75)",
                "opacity": 0.8,
                "line": {"color": "#818cf8", "width": 1},
            },
            "hovertemplate": (
                f"<b>{x_col}:</b> %{{x}}<br>"
                f"<b>{y_col}:</b> %{{y}}<br>"
                "<extra></extra>"
            ),
        })

    # Add linear trendline if correlation is defined
    if not math.isnan(r_val) and len(sub_df) >= 5 and sub_df[x_col].std() > 0:
        try:
            x_vals = sub_df[x_col].values
            y_vals = sub_df[y_col].values
            poly_fit = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
            y_line = np.polyval(poly_fit, x_line)

            traces.append({
                "type": "scatter",
                "mode": "lines",
                "x": x_line.tolist(),
                "y": y_line.tolist(),
                "name": "Linear Trend",
                "line": {"color": "#34d399", "width": 2, "dash": "dash"},
                "hoverinfo": "none",
            })
        except Exception:
            pass

    subtitle = f"Pearson Correlation: <b>{r_str}</b> • N = {len(sub_df):,} observations"

    layout = _merge_layout({
        "title": {
            "text": f"<b>Bivariate Relationship: {y_col} vs. {x_col}</b><br><sup style='color:#94a3b8;'>{subtitle}</sup>",
            "font": {"size": 14, "color": "#f8fafc"},
            "x": 0.02,
        },
        "xaxis": {"title": {"text": x_col, "font": {"color": "#cbd5e1", "size": 11}}},
        "yaxis": {"title": {"text": y_col, "font": {"color": "#cbd5e1", "size": 11}}},
        "showlegend": bool(color_col and color_col in plot_df.columns),
    }, height=height)

    return {"data": traces, "layout": layout, "config": PLOTLY_CONFIG}


def build_line_chart_spec(
    df: pd.DataFrame,
    date_col: str,
    num_col: str,
    agg: str = "mean",
    height: int = 400,
) -> Dict[str, Any]:
    """
    Generate an interactive Plotly Time Series Line / Area Chart for Datetime + Numerical features.
    Sorts dates chronologically and aggregates if excessive timestamps exist.
    """
    if date_col not in df.columns or num_col not in df.columns:
        return _empty_chart_spec(f"Columns '{date_col}' and/or '{num_col}' not found.")

    sub_df = df[[date_col, num_col]].dropna().copy()
    sub_df[date_col] = pd.to_datetime(sub_df[date_col], errors="coerce")
    sub_df[num_col] = pd.to_numeric(sub_df[num_col], errors="coerce")
    sub_df = sub_df.dropna().sort_values(by=date_col)

    if len(sub_df) == 0:
        return _empty_chart_spec(f"No valid temporal data for '{date_col}' and '{num_col}'.")

    if len(sub_df) > 300 or sub_df[date_col].duplicated().any():
        try:
            agg_func = "sum" if agg.lower() == "sum" else "mean"
            grouped = sub_df.groupby(sub_df[date_col].dt.date)[num_col].agg(agg_func).reset_index()
            grouped.columns = [date_col, num_col]
            plot_df = grouped
        except Exception:
            plot_df = sub_df
    else:
        plot_df = sub_df

    x_dates = [str(d) for d in plot_df[date_col]]
    y_vals = [round(float(v), 2) for v in plot_df[num_col]]

    trace = {
        "type": "scatter",
        "mode": "lines+markers",
        "x": x_dates,
        "y": y_vals,
        "name": f"{agg.capitalize()} of {num_col}",
        "line": {"color": "#06b6d4", "width": 2.5, "shape": "spline"},
        "marker": {"size": 5, "color": "#22d3ee", "line": {"color": "#ffffff", "width": 0.8}},
        "fill": "tozeroy",
        "fillcolor": "rgba(6, 182, 212, 0.12)",
        "hovertemplate": (
            f"<b>Date:</b> %{{x}}<br>"
            f"<b>{num_col} ({agg.capitalize()}):</b> %{{y}}<br>"
            "<extra></extra>"
        ),
    }

    subtitle = f"Chronological Trend across {len(plot_df):,} periods • Total points: {len(sub_df):,}"

    layout = _merge_layout({
        "title": {
            "text": f"<b>Time Series: {num_col} over {date_col}</b><br><sup style='color:#94a3b8;'>{subtitle}</sup>",
            "font": {"size": 14, "color": "#f8fafc"},
            "x": 0.02,
        },
        "xaxis": {
            "title": {"text": date_col, "font": {"color": "#cbd5e1", "size": 11}},
            "tickangle": -30,
        },
        "yaxis": {"title": {"text": f"{num_col} ({agg.capitalize()})", "font": {"color": "#cbd5e1", "size": 11}}},
        "showlegend": False,
    }, height=height)

    return {"data": [trace], "layout": layout, "config": PLOTLY_CONFIG}


def build_grouped_bar_spec(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    agg: str = "mean",
    top_n: int = 15,
    height: int = 400,
) -> Dict[str, Any]:
    """
    Generate an interactive Plotly Grouped / Aggregated Bar Chart for Categorical + Numerical features.
    Computes mean/sum/median by category and sorts descending.
    """
    if cat_col not in df.columns or num_col not in df.columns:
        return _empty_chart_spec(f"Columns '{cat_col}' and/or '{num_col}' not found.")

    sub_df = df[[cat_col, num_col]].dropna().copy()
    sub_df[cat_col] = sub_df[cat_col].astype(str)
    sub_df[num_col] = pd.to_numeric(sub_df[num_col], errors="coerce")
    sub_df = sub_df.dropna()

    if len(sub_df) == 0:
        return _empty_chart_spec(f"No valid observations for '{cat_col}' and '{num_col}'.")

    agg_lower = agg.lower()
    if agg_lower not in ["mean", "sum", "median", "count", "min", "max", "std"]:
        agg_lower = "mean"

    grouped = sub_df.groupby(cat_col)[num_col].agg(agg_lower).reset_index()
    grouped = grouped.sort_values(by=num_col, ascending=False)

    if len(grouped) > top_n:
        top_grp = grouped.head(top_n)
        other_agg = sub_df[~sub_df[cat_col].isin(top_grp[cat_col])][num_col].agg(agg_lower)
        x_vals = list(top_grp[cat_col]) + ["Other (Aggregated)"]
        y_vals = [round(float(v), 2) for v in top_grp[num_col]] + [round(float(other_agg), 2)]
    else:
        x_vals = list(grouped[cat_col])
        y_vals = [round(float(v), 2) for v in grouped[num_col]]

    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(x_vals))]

    trace = {
        "type": "bar",
        "x": x_vals,
        "y": y_vals,
        "text": [f"{y:,.1f}" for y in y_vals],
        "textposition": "auto",
        "textfont": {"color": "#f8fafc", "family": "JetBrains Mono", "size": 10},
        "marker": {
            "color": colors,
            "line": {"color": "rgba(255, 255, 255, 0.15)", "width": 1},
        },
        "hovertemplate": (
            f"<b>{cat_col}:</b> %{{x}}<br>"
            f"<b>{num_col} ({agg_lower.capitalize()}):</b> %{{y}}<br>"
            "<extra></extra>"
        ),
    }

    subtitle = f"Aggregated by <b>{agg_lower.capitalize()}</b> across {len(x_vals)} categories • N = {len(sub_df):,}"

    layout = _merge_layout({
        "title": {
            "text": f"<b>{agg_lower.capitalize()} of {num_col} by {cat_col}</b><br><sup style='color:#94a3b8;'>{subtitle}</sup>",
            "font": {"size": 14, "color": "#f8fafc"},
            "x": 0.02,
        },
        "xaxis": {
            "title": {"text": cat_col, "font": {"color": "#cbd5e1", "size": 11}},
            "tickangle": -35 if len(x_vals) > 5 else 0,
        },
        "yaxis": {"title": {"text": f"{agg_lower.capitalize()} ({num_col})", "font": {"color": "#cbd5e1", "size": 11}}},
        "showlegend": False,
    }, height=height)

    return {"data": [trace], "layout": layout, "config": PLOTLY_CONFIG}


def build_time_category_spec(
    df: pd.DataFrame,
    date_col: str,
    cat_col: str,
    top_n_cats: int = 6,
    height: int = 420,
) -> Dict[str, Any]:
    """
    Generate an interactive Plotly Stacked Bar / Multi-Series Line Chart for Datetime + Categorical features.
    Visualizes categorical breakdown and frequency shifts over time periods.
    """
    if date_col not in df.columns or cat_col not in df.columns:
        return _empty_chart_spec(f"Columns '{date_col}' and/or '{cat_col}' not found.")

    sub_df = df[[date_col, cat_col]].dropna().copy()
    sub_df[date_col] = pd.to_datetime(sub_df[date_col], errors="coerce")
    sub_df[cat_col] = sub_df[cat_col].astype(str)
    sub_df = sub_df.dropna().sort_values(by=date_col)

    if len(sub_df) == 0:
        return _empty_chart_spec(f"No valid data for '{date_col}' and '{cat_col}'.")

    # Select top categories
    top_cats = sub_df[cat_col].value_counts().head(top_n_cats).index.tolist()
    sub_df["cat_clean"] = sub_df[cat_col].apply(lambda x: x if x in top_cats else "Other")

    time_span_days = (sub_df[date_col].max() - sub_df[date_col].min()).days
    if time_span_days > 180:
        sub_df["period"] = sub_df[date_col].dt.to_period("M").astype(str)
    else:
        sub_df["period"] = sub_df[date_col].dt.to_period("D").astype(str)

    cross_tab = pd.crosstab(sub_df["period"], sub_df["cat_clean"])
    periods = [str(p) for p in cross_tab.index]

    traces = []
    categories_to_plot = [c for c in top_cats if c in cross_tab.columns] + (["Other"] if "Other" in cross_tab.columns else [])

    for i, cat in enumerate(categories_to_plot):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        counts = cross_tab[cat].tolist()
        traces.append({
            "type": "bar",
            "x": periods,
            "y": counts,
            "name": cat,
            "marker": {"color": color},
            "hovertemplate": (
                f"<b>Period:</b> %{{x}}<br>"
                f"<b>{cat}:</b> %{{y}} records<br>"
                "<extra></extra>"
            ),
        })

    subtitle = f"Temporal Volume by Category across {len(periods)} periods • Top {len(top_cats)} categories"

    layout = _merge_layout({
        "title": {
            "text": f"<b>Temporal Category Analysis: {cat_col} over {date_col}</b><br><sup style='color:#94a3b8;'>{subtitle}</sup>",
            "font": {"size": 14, "color": "#f8fafc"},
            "x": 0.02,
        },
        "barmode": "stack",
        "xaxis": {
            "title": {"text": "Time Period", "font": {"color": "#cbd5e1", "size": 11}},
            "tickangle": -35,
        },
        "yaxis": {"title": {"text": "Record Count", "font": {"color": "#cbd5e1", "size": 11}}},
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "y": -0.25,
            "x": 0.5,
            "xanchor": "center",
        },
    }, height=height)

    return {"data": traces, "layout": layout, "config": PLOTLY_CONFIG}


def _empty_chart_spec(message: str) -> Dict[str, Any]:
    """Return a clean placeholder Plotly spec when data is missing or invalid."""
    return {
        "data": [],
        "layout": _merge_layout({
            "title": {"text": message, "font": {"color": "#94a3b8", "size": 12}},
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
        }, height=280),
        "config": {"responsive": True, "displayModeBar": False},
    }


# ==============================================================================
# 3. Chart Recommendation Engine
# ==============================================================================

class ChartRecommendationEngine:
    """
    Intelligent chart recommendation system:
    - Analyzes dataset schema, data types, variance, correlation, and temporal spans.
    - Evaluates all statistical pairing rules.
    - Prunes useless / low-information combinations.
    - Computes relevance priority scores and human-readable analytical rationales.
    - Produces production-ready Plotly chart specifications.
    """

    def __init__(self, df: pd.DataFrame, dataset_id: Optional[str] = None):
        self.df = df
        self.dataset_id = dataset_id or "default"
        self.schema = inspect_column_types(df)

    def recommend(
        self,
        limit: int = 10,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate ranked, non-redundant chart recommendations for the dataset.

        Args:
            limit: Maximum number of charts to return (default 10).
            category_filter: Optional filter ('Distributions', 'Relationships', 'Categorical', 'Trends').

        Returns:
            List of recommendation dicts containing metadata, ranking, rationale, and plotly_spec.
        """
        if self.df.empty:
            return []

        candidates: List[Dict[str, Any]] = []

        num_cols = self.schema["numerical"]
        cat_cols = self.schema["categorical"] + self.schema["boolean"]
        dt_cols = self.schema["datetime"]

        # -------------------------------------------------------------
        # 1. Single Numerical Recommendations (Histogram & Box Plot)
        # -------------------------------------------------------------
        for col in num_cols:
            series = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if len(series) < 3:
                continue

            std_val = float(series.std())
            if std_val == 0:
                continue

            skew_val = abs(float(series.skew())) if len(series) > 2 else 0.0
            completeness = len(series) / len(self.df)

            # Histogram candidate
            hist_score = 65 + min(25, skew_val * 10) + (completeness * 10)
            hist_rationale = (
                f"Feature '{col}' exhibits {'significant right-tail skewness' if series.skew() > 1 else 'left-tail skewness' if series.skew() < -1 else 'a balanced bell-curve distribution'} (skew = {series.skew():.2f}, std = {std_val:.2f})."
            )
            candidates.append({
                "id": f"hist_{col}",
                "title": f"Distribution of {col}",
                "chart_type": "histogram",
                "category": "Distributions",
                "score": round(hist_score, 1),
                "priority": "High" if hist_score >= 80 else "Medium",
                "rationale": hist_rationale,
                "columns_used": [col],
                "generator": lambda c=col: build_histogram_spec(self.df, c),
            })

            # Box Plot candidate
            iqr = float(series.quantile(0.75) - series.quantile(0.25))
            box_score = 60 + (15 if iqr > 0 else 0) + (completeness * 10)
            box_rationale = f"Displays median ({series.median():.2f}), interquartile range (IQR = {iqr:.2f}), and outlier spread for '{col}'."
            candidates.append({
                "id": f"box_{col}",
                "title": f"Box Plot of {col}",
                "chart_type": "box",
                "category": "Distributions",
                "score": round(box_score, 1),
                "priority": "High" if box_score >= 80 else "Medium",
                "rationale": box_rationale,
                "columns_used": [col],
                "generator": lambda c=col: build_single_boxplot_spec(self.df, c),
            })

        # -------------------------------------------------------------
        # 2. Single Categorical Recommendations (Bar & Donut)
        # -------------------------------------------------------------
        for col in cat_cols:
            series = self.df[col].dropna()
            n_unique = series.nunique()
            if n_unique < 2 or n_unique > 30:
                continue

            val_counts = series.value_counts()
            top_share = val_counts.iloc[0] / len(series) if len(series) > 0 else 1.0

            # Frequency Bar candidate
            bar_score = 70 + (15 if 2 <= n_unique <= 10 else 5) - (top_share * 10 if top_share > 0.9 else 0)
            bar_rationale = f"Categorical breakdown of '{col}' across {n_unique} unique classes (Top class: '{val_counts.index[0]}' with {val_counts.iloc[0]:,} records)."
            candidates.append({
                "id": f"bar_{col}",
                "title": f"Frequency Breakdown of {col}",
                "chart_type": "bar",
                "category": "Categorical",
                "score": round(bar_score, 1),
                "priority": "High" if bar_score >= 80 else "Medium",
                "rationale": bar_rationale,
                "columns_used": [col],
                "generator": lambda c=col: build_bar_chart_spec(self.df, c),
            })

            # Donut / Distribution candidate (best for 2-8 classes)
            if n_unique <= 8:
                donut_score = 72 + (15 if 2 <= n_unique <= 6 else 0)
                donut_rationale = f"Proportional share and percentage distribution of '{col}' across {n_unique} distinct segments."
                candidates.append({
                    "id": f"donut_{col}",
                    "title": f"Category Share of {col}",
                    "chart_type": "pie",
                    "category": "Categorical",
                    "score": round(donut_score, 1),
                    "priority": "High" if donut_score >= 80 else "Medium",
                    "rationale": donut_rationale,
                    "columns_used": [col],
                    "generator": lambda c=col: build_category_distribution_spec(self.df, c),
                })

        # -------------------------------------------------------------
        # 3. Numerical + Numerical Recommendations (Scatter Plot)
        # -------------------------------------------------------------
        if len(num_cols) >= 2:
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    sub = self.df[[c1, c2]].dropna()
                    if len(sub) < 5:
                        continue

                    s1 = pd.to_numeric(sub[c1], errors="coerce")
                    s2 = pd.to_numeric(sub[c2], errors="coerce")
                    valid_mask = s1.notna() & s2.notna()
                    if valid_mask.sum() < 5:
                        continue

                    r_val = float(s1[valid_mask].corr(s2[valid_mask]))
                    abs_r = abs(r_val) if not math.isnan(r_val) else 0.0

                    scatter_score = 55 + (abs_r * 40)
                    scatter_rationale = (
                        f"Linear association between '{c1}' and '{c2}' with Pearson r = {r_val:+.3f} "
                        f"({'strong positive' if r_val > 0.6 else 'strong inverse' if r_val < -0.6 else 'moderate' if abs_r >= 0.3 else 'weak'} correlation)."
                    )
                    candidates.append({
                        "id": f"scatter_{c1}_{c2}",
                        "title": f"{c2} vs. {c1}",
                        "chart_type": "scatter",
                        "category": "Relationships",
                        "score": round(scatter_score, 1),
                        "priority": "High" if abs_r >= 0.5 or scatter_score >= 80 else "Medium" if abs_r >= 0.25 else "Low",
                        "rationale": scatter_rationale,
                        "columns_used": [c1, c2],
                        "generator": lambda x=c1, y=c2: build_scatter_spec(self.df, x, y),
                    })

        # -------------------------------------------------------------
        # 4. Datetime + Numerical Recommendations (Time Series Line)
        # -------------------------------------------------------------
        for dt_col in dt_cols:
            for num_col in num_cols:
                sub = self.df[[dt_col, num_col]].dropna()
                if len(sub) < 4:
                    continue

                line_score = 85
                line_rationale = f"Temporal progression and chronological trend of '{num_col}' tracked across time periods in '{dt_col}'."
                candidates.append({
                    "id": f"line_{dt_col}_{num_col}",
                    "title": f"{num_col} Trend over {dt_col}",
                    "chart_type": "line",
                    "category": "Trends",
                    "score": line_score,
                    "priority": "High",
                    "rationale": line_rationale,
                    "columns_used": [dt_col, num_col],
                    "generator": lambda d=dt_col, n=num_col: build_line_chart_spec(self.df, d, n),
                })

        # -------------------------------------------------------------
        # 5. Categorical + Numerical Recommendations (Grouped Bar)
        # -------------------------------------------------------------
        for cat_col in cat_cols:
            n_unique = self.df[cat_col].nunique()
            if n_unique < 2 or n_unique > 20:
                continue

            for num_col in num_cols:
                sub = self.df[[cat_col, num_col]].dropna()
                if len(sub) < 5:
                    continue

                grp_means = sub.groupby(cat_col)[num_col].mean()
                mean_std = float(grp_means.std()) if len(grp_means) > 1 else 0.0

                grp_score = 75 + (10 if mean_std > 0 else 0) + (5 if 3 <= n_unique <= 10 else 0)
                grp_rationale = f"Compares average metric '{num_col}' across '{cat_col}' segments (highest mean: {grp_means.idxmax()} = {grp_means.max():.2f})."
                candidates.append({
                    "id": f"grp_bar_{cat_col}_{num_col}",
                    "title": f"Mean {num_col} by {cat_col}",
                    "chart_type": "bar",
                    "category": "Categorical",
                    "score": round(grp_score, 1),
                    "priority": "High" if grp_score >= 82 else "Medium",
                    "rationale": grp_rationale,
                    "columns_used": [cat_col, num_col],
                    "generator": lambda c=cat_col, n=num_col: build_grouped_bar_spec(self.df, c, n, agg="mean"),
                })

        # -------------------------------------------------------------
        # 6. Datetime + Categorical Recommendations (Time Category)
        # -------------------------------------------------------------
        for dt_col in dt_cols:
            for cat_col in cat_cols:
                n_unique = self.df[cat_col].nunique()
                if n_unique < 2 or n_unique > 8:
                    continue

                sub = self.df[[dt_col, cat_col]].dropna()
                if len(sub) < 8:
                    continue

                time_cat_score = 80
                time_cat_rationale = f"Evaluates how '{cat_col}' category mix and volumes evolve chronologically over '{dt_col}'."
                candidates.append({
                    "id": f"time_cat_{dt_col}_{cat_col}",
                    "title": f"{cat_col} Distribution over {dt_col}",
                    "chart_type": "bar",
                    "category": "Trends",
                    "score": time_cat_score,
                    "priority": "High",
                    "rationale": time_cat_rationale,
                    "columns_used": [dt_col, cat_col],
                    "generator": lambda d=dt_col, c=cat_col: build_time_category_spec(self.df, d, c),
                })

        # -------------------------------------------------------------
        # Ranking, Diversification & Output Assembly
        # -------------------------------------------------------------
        if category_filter:
            candidates = [c for c in candidates if c["category"].lower() == category_filter.lower()]

        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected = []
        seen_cols: Dict[str, int] = {}
        seen_types: Dict[str, int] = {}

        for cand in candidates:
            col_key = "-".join(sorted(cand["columns_used"]))
            c_type = cand["chart_type"]

            if seen_cols.get(col_key, 0) >= 2:
                continue
            if seen_types.get(c_type, 0) >= 4 and len(candidates) > limit:
                continue

            seen_cols[col_key] = seen_cols.get(col_key, 0) + 1
            seen_types[c_type] = seen_types.get(c_type, 0) + 1

            try:
                plotly_spec = cand["generator"]()
            except Exception as e:
                logger.warning("Error generating chart spec for candidate %s: %s", cand["id"], e)
                plotly_spec = _empty_chart_spec(f"Could not generate chart: {e}")

            cand_item = {
                "id": cand["id"],
                "title": cand["title"],
                "chart_type": cand["chart_type"],
                "category": cand["category"],
                "score": cand["score"],
                "priority": cand["priority"],
                "rationale": cand["rationale"],
                "columns_used": cand["columns_used"],
                "plotly_spec": plotly_spec,
            }
            selected.append(cand_item)

            if len(selected) >= limit:
                break

        return selected


def recommend_charts(
    df: pd.DataFrame,
    limit: int = 10,
    category_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convenience functional wrapper around ChartRecommendationEngine."""
    engine = ChartRecommendationEngine(df)
    return engine.recommend(limit=limit, category_filter=category_filter)


# ==============================================================================
# 4. Custom Chart Builder ("Build Your Own Chart")
# ==============================================================================

def build_custom_chart_spec(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    aggregation: Optional[str] = "none",
    title: Optional[str] = None,
    height: int = 440,
) -> Dict[str, Any]:
    """
    Construct a custom interactive Plotly chart specification based on user-selected parameters.

    Args:
        df: Pandas DataFrame
        chart_type: One of 'bar', 'line', 'scatter', 'histogram', 'box', 'pie', 'area', 'heatmap'
        x_col: Selected X-axis column
        y_col: Optional selected Y-axis column
        color_col: Optional selected grouping/color column
        aggregation: Optional aggregation ('none', 'mean', 'sum', 'median', 'count', 'min', 'max', 'std')
        title: Custom chart title override
        height: Chart height in pixels
    """
    if df.empty:
        return _empty_chart_spec("The dataset is empty.")

    if x_col not in df.columns:
        return _empty_chart_spec(f"Selected X column '{x_col}' was not found in dataset.")

    chart_type_lower = (chart_type or "bar").strip().lower()
    agg_lower = (aggregation or "none").strip().lower()

    required_cols = [x_col]
    if y_col and y_col in df.columns:
        required_cols.append(y_col)
    if color_col and color_col in df.columns:
        required_cols.append(color_col)

    sub_df = df[required_cols].dropna(subset=[x_col] + ([y_col] if y_col and y_col in df.columns else [])).copy()

    if len(sub_df) == 0:
        return _empty_chart_spec("No valid observations available for the selected columns.")

    # 1. Histogram
    if chart_type_lower in ["histogram", "dist"]:
        return build_histogram_spec(sub_df, x_col, height=height)

    # 2. Box Plot
    if chart_type_lower == "box":
        if y_col and y_col in sub_df.columns:
            sub_df[y_col] = pd.to_numeric(sub_df[y_col], errors="coerce")
            sub_df = sub_df.dropna(subset=[y_col])
            unique_x = sub_df[x_col].astype(str).unique()[:12]
            traces = []
            for i, x_val in enumerate(unique_x):
                y_data = sub_df[sub_df[x_col].astype(str) == x_val][y_col]
                color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
                traces.append({
                    "type": "box",
                    "y": y_data.tolist(),
                    "name": str(x_val),
                    "marker": {"color": color},
                    "boxmean": True,
                })
            layout = _merge_layout({
                "title": {"text": f"<b>{title or f'Box Plot of {y_col} grouped by {x_col}'}</b>", "x": 0.02},
                "xaxis": {"title": {"text": x_col}},
                "yaxis": {"title": {"text": y_col}},
                "showlegend": len(traces) > 1,
            }, height=height)
            return {"data": traces, "layout": layout, "config": PLOTLY_CONFIG}
        else:
            return build_single_boxplot_spec(sub_df, x_col, height=height)

    # 3. Pie / Donut
    if chart_type_lower in ["pie", "donut"]:
        if y_col and y_col in sub_df.columns and agg_lower != "none":
            sub_df[y_col] = pd.to_numeric(sub_df[y_col], errors="coerce")
            grouped = sub_df.groupby(x_col)[y_col].agg("sum" if agg_lower == "sum" else "mean").head(12)
            labels = list(grouped.index.astype(str))
            values = [round(float(v), 2) for v in grouped.values]
            trace = {
                "type": "pie",
                "labels": labels,
                "values": values,
                "hole": 0.5 if chart_type_lower == "donut" else 0.0,
                "marker": {"colors": COLOR_PALETTE[:len(labels)]},
                "textinfo": "percent+label",
            }
            layout = _merge_layout({
                "title": {"text": f"<b>{title or f'{agg_lower.capitalize()} of {y_col} by {x_col}'}</b>", "x": 0.02},
                "showlegend": True,
            }, height=height)
            return {"data": [trace], "layout": layout, "config": PLOTLY_CONFIG}
        else:
            return build_category_distribution_spec(sub_df, x_col, height=height)

    # 4. Scatter
    if chart_type_lower == "scatter":
        if not y_col or y_col not in sub_df.columns:
            return _empty_chart_spec("Scatter plots require both X and Y columns.")
        return build_scatter_spec(sub_df, x_col, y_col, color_col=color_col, height=height)

    # 5. Line / Area
    if chart_type_lower in ["line", "area"]:
        if not y_col or y_col not in sub_df.columns:
            return _empty_chart_spec("Line charts require both X and Y columns.")

        is_dt = pd.api.types.is_datetime64_any_dtype(sub_df[x_col])
        if not is_dt:
            try:
                sub_df[x_col] = pd.to_datetime(sub_df[x_col], errors="coerce")
                is_dt = sub_df[x_col].notna().sum() / len(sub_df) >= 0.8
            except Exception:
                is_dt = False

        if is_dt:
            return build_line_chart_spec(sub_df, x_col, y_col, agg="sum" if agg_lower == "sum" else "mean", height=height)

        sub_df[y_col] = pd.to_numeric(sub_df[y_col], errors="coerce")
        sub_df = sub_df.dropna(subset=[y_col])

        if agg_lower != "none":
            agg_func = "sum" if agg_lower == "sum" else "mean" if agg_lower == "mean" else "median"
            grouped = sub_df.groupby(x_col)[y_col].agg(agg_func).reset_index()
            x_vals = grouped[x_col].astype(str).tolist()
            y_vals = [round(float(v), 2) for v in grouped[y_col]]
        else:
            plot_df = sub_df.head(500)
            x_vals = plot_df[x_col].astype(str).tolist()
            y_vals = [round(float(v), 2) for v in plot_df[y_col]]

        trace = {
            "type": "scatter",
            "mode": "lines+markers",
            "x": x_vals,
            "y": y_vals,
            "name": y_col,
            "line": {"color": "#6366f1", "width": 2.5},
            "marker": {"size": 5, "color": "#818cf8"},
            "fill": "tozeroy" if chart_type_lower == "area" else "none",
            "fillcolor": "rgba(99, 102, 241, 0.15)" if chart_type_lower == "area" else None,
        }
        layout = _merge_layout({
            "title": {"text": f"<b>{title or f'{y_col} vs. {x_col}'}</b>", "x": 0.02},
            "xaxis": {"title": {"text": x_col}, "tickangle": -35 if len(x_vals) > 6 else 0},
            "yaxis": {"title": {"text": y_col}},
            "showlegend": False,
        }, height=height)
        return {"data": [trace], "layout": layout, "config": PLOTLY_CONFIG}

    # 6. Bar Chart (Default)
    if y_col and y_col in sub_df.columns:
        agg_func = agg_lower if agg_lower in ["mean", "sum", "median", "count", "min", "max", "std"] else "mean"
        return build_grouped_bar_spec(sub_df, x_col, y_col, agg=agg_func, height=height)
    else:
        return build_bar_chart_spec(sub_df, x_col, height=height)


# ==============================================================================
# 5. Backward Compatibility Functions for Correlation & Outliers
# ==============================================================================

def build_correlation_heatmap_spec(
    corr_matrix: Dict[str, Dict[str, Optional[float]]],
    columns: List[str],
    height: int = 480,
) -> Dict[str, Any]:
    """
    Construct a high-performance, dark-theme Plotly Heatmap JSON specification.
    """
    if not columns or not corr_matrix:
        return _empty_chart_spec("No Numerical Features Available for Correlation Heatmap")

    n_cols = len(columns)
    z_values = []
    text_values = []
    annotations = []

    for i, row_col in enumerate(columns):
        row_z = []
        row_text = []
        for j, col_col in enumerate(columns):
            val = corr_matrix.get(row_col, {}).get(col_col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                r_num = 0.0
                display_str = "--"
            else:
                r_num = round(float(val), 4)
                display_str = f"{r_num:.2f}"

            row_z.append(r_num)
            row_text.append(f"{row_col} ↔ {col_col}<br>Pearson r: {r_num:.4f}")

            if n_cols <= 16:
                text_color = "#ffffff" if abs(r_num) > 0.35 else "#cbd5e1"
                annotations.append({
                    "x": col_col,
                    "y": row_col,
                    "text": display_str,
                    "font": {
                        "family": "JetBrains Mono, monospace",
                        "size": 11 if n_cols <= 8 else 9,
                        "color": text_color,
                        "weight": "bold",
                    },
                    "showarrow": False,
                })

        z_values.append(row_z)
        text_values.append(row_text)

    diverging_colorscale = [
        [0.0, "rgb(225, 29, 72)"],     # Rose (-1.0)
        [0.25, "rgb(244, 63, 94)"],
        [0.45, "rgb(51, 65, 85)"],
        [0.50, "rgb(30, 41, 59)"],     # Slate Neutral (0.0)
        [0.55, "rgb(51, 65, 85)"],
        [0.75, "rgb(99, 102, 241)"],
        [1.0, "rgb(16, 185, 129)"],    # Emerald (+1.0)
    ]

    trace = {
        "type": "heatmap",
        "z": z_values,
        "x": columns,
        "y": columns,
        "colorscale": diverging_colorscale,
        "zmin": -1.0,
        "zmax": 1.0,
        "hoverongaps": False,
        "hoverinfo": "text",
        "text": text_values,
        "colorbar": {
            "title": {"text": "Pearson (r)", "font": {"color": "#cbd5e1", "size": 12}},
            "tickvals": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "ticktext": ["-1.0 (Neg)", "-0.5", "0.0", "+0.5", "+1.0 (Pos)"],
            "thickness": 14,
            "len": 0.85,
            "outlinewidth": 0,
        },
    }

    calculated_height = max(height, min(750, 180 + (n_cols * 32)))

    layout = _merge_layout({
        "height": calculated_height,
        "margin": {"l": 120, "r": 60, "t": 40, "b": 100, "pad": 4},
        "xaxis": {"tickangle": -45},
        "yaxis": {"autorange": "reversed"},
        "annotations": annotations,
    }, height=calculated_height)

    return {"data": [trace], "layout": layout, "config": PLOTLY_CONFIG}


def build_outlier_boxplot_spec(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    height: int = 420,
) -> Dict[str, Any]:
    """
    Construct a high-performance, dark-theme Plotly multi-column Box Plot JSON specification.
    """
    if df.empty:
        return _empty_chart_spec("No Data Available for Box Plot")

    if not columns:
        columns = [str(c) for c in df.select_dtypes(include=[np.number]).columns]

    if not columns:
        return _empty_chart_spec("No Numerical Features Available for Box Plot")

    traces = []
    palette = [
        {"fill": "rgba(99, 102, 241, 0.25)", "line": "#818cf8"},
        {"fill": "rgba(16, 185, 129, 0.25)", "line": "#34d399"},
        {"fill": "rgba(6, 182, 212, 0.25)", "line": "#22d3ee"},
        {"fill": "rgba(245, 158, 11, 0.25)", "line": "#fbbf24"},
        {"fill": "rgba(168, 85, 247, 0.25)", "line": "#c084fc"},
        {"fill": "rgba(236, 72, 153, 0.25)", "line": "#f472b6"},
    ]

    for i, col in enumerate(columns[:12]):
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue

        color_item = palette[i % len(palette)]
        sample_s = series if len(series) <= 5000 else series.sample(5000, random_state=42)

        traces.append({
            "type": "box",
            "y": sample_s.tolist(),
            "name": str(col),
            "boxpoints": "outliers",
            "jitter": 0.35,
            "pointpos": -1.8,
            "fillcolor": color_item["fill"],
            "line": {"color": color_item["line"], "width": 2},
            "marker": {
                "size": 5,
                "color": "#f43f5e",
                "outliercolor": "#fb7185",
                "line": {"color": "#f43f5e", "width": 1},
            },
            "boxmean": True,
            "hoverinfo": "y+name",
        })

    layout = _merge_layout({
        "xaxis": {"tickangle": -35 if len(traces) > 4 else 0},
        "showlegend": False,
    }, height=max(height, 380))

    return {"data": traces, "layout": layout, "config": PLOTLY_CONFIG}


def build_outlier_distribution_spec(
    series: pd.Series,
    col_name: str,
    lower_thresh: Optional[float],
    upper_thresh: Optional[float],
    method_name: str = "IQR",
    height: int = 380,
) -> Dict[str, Any]:
    """
    Construct a high-performance Plotly Distribution Histogram & Boundary Chart for a single column.
    """
    clean_s = pd.to_numeric(series, errors="coerce").dropna()

    if len(clean_s) == 0:
        return _empty_chart_spec(f"No valid numerical data in '{col_name}'")

    min_val = float(clean_s.min())
    max_val = float(clean_s.max())
    span = max_val - min_val if max_val > min_val else 1.0
    pad = span * 0.08

    traces = [
        {
            "type": "histogram",
            "x": clean_s.tolist(),
            "name": f"{col_name} Distribution",
            "marker": {
                "color": "rgba(99, 102, 241, 0.55)",
                "line": {"color": "#818cf8", "width": 1.5},
            },
            "opacity": 0.85,
            "nbinsx": min(35, max(12, int(len(clean_s) ** 0.5))),
            "hoverinfo": "x+y",
        }
    ]

    outliers_mask = pd.Series(False, index=clean_s.index)
    if lower_thresh is not None:
        outliers_mask |= (clean_s < lower_thresh)
    if upper_thresh is not None:
        outliers_mask |= (clean_s > upper_thresh)

    outlier_points = clean_s[outliers_mask]
    if len(outlier_points) > 0:
        traces.append({
            "type": "scatter",
            "x": outlier_points.tolist(),
            "y": [0.5] * len(outlier_points),
            "mode": "markers",
            "name": f"Outliers ({len(outlier_points)})",
            "marker": {
                "color": "#f43f5e",
                "size": 8,
                "symbol": "diamond",
                "line": {"color": "#ffffff", "width": 1},
            },
            "hoverinfo": "x+name",
        })

    shapes = []
    annotations = []

    if lower_thresh is not None and not math.isnan(lower_thresh):
        if min_val < lower_thresh:
            shapes.append({
                "type": "rect",
                "x0": min_val - pad,
                "x1": lower_thresh,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "fillcolor": "rgba(244, 63, 94, 0.12)",
                "line": {"width": 0},
            })
        shapes.append({
            "type": "line",
            "x0": lower_thresh,
            "x1": lower_thresh,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": "#f43f5e", "width": 2, "dash": "dash"},
        })
        annotations.append({
            "x": lower_thresh,
            "y": 1.04,
            "yref": "paper",
            "text": f"Lower ({lower_thresh:.2f})",
            "showarrow": False,
            "font": {"color": "#f43f5e", "size": 10, "family": "JetBrains Mono", "weight": "bold"},
            "bgcolor": "rgba(30, 41, 59, 0.8)",
            "borderpad": 2,
        })

    if upper_thresh is not None and not math.isnan(upper_thresh):
        if max_val > upper_thresh:
            shapes.append({
                "type": "rect",
                "x0": upper_thresh,
                "x1": max_val + pad,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "fillcolor": "rgba(244, 63, 94, 0.12)",
                "line": {"width": 0},
            })
        shapes.append({
            "type": "line",
            "x0": upper_thresh,
            "x1": upper_thresh,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": "#f43f5e", "width": 2, "dash": "dash"},
        })
        annotations.append({
            "x": upper_thresh,
            "y": 1.04,
            "yref": "paper",
            "text": f"Upper ({upper_thresh:.2f})",
            "showarrow": False,
            "font": {"color": "#f43f5e", "size": 10, "family": "JetBrains Mono", "weight": "bold"},
            "bgcolor": "rgba(30, 41, 59, 0.8)",
            "borderpad": 2,
        })

    layout = _merge_layout({
        "xaxis": {"title": {"text": col_name}},
        "yaxis": {"title": {"text": "Count / Frequency"}},
        "shapes": shapes,
        "annotations": annotations,
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "y": -0.22,
            "x": 0.5,
            "xanchor": "center",
        },
    }, height=height)

    return {"data": traces, "layout": layout, "config": PLOTLY_CONFIG}


def generate_summary_charts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate standard exploratory charts for the dashboard.
    """
    return recommend_charts(df, limit=8)
