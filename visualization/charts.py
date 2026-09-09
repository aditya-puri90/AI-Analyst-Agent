"""
Interactive Plotly Visualization Engine for AI Data Analyst Agent.
Constructs rich, interactive JSON chart specifications for web rendering,
including correlation heatmaps, distributions, and scatter metrics.
"""

from typing import Dict, Any, List, Optional
import math
import pandas as pd
import numpy as np


def build_correlation_heatmap_spec(
    corr_matrix: Dict[str, Dict[str, Optional[float]]],
    columns: List[str],
    height: int = 480
) -> Dict[str, Any]:
    """
    Construct a high-performance, dark-theme Plotly Heatmap JSON specification.
    
    Features:
    - Custom diverging colorscale (Rose #f43f5e for negative, Slate #1e293b for zero, Indigo/Emerald #10b981 for positive)
    - Clamped z-range [-1.0, 1.0] with symmetric midpoint at 0.0
    - Cell value text annotations
    - Rich interactive hover tooltips
    - Fully responsive container layout
    """
    if not columns or not corr_matrix:
        return {
            "data": [],
            "layout": {
                "title": {"text": "No Numerical Features Available for Correlation Heatmap", "font": {"color": "#94a3b8"}},
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            },
            "config": {"responsive": True, "displayModeBar": False},
        }

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

            # Only add cell text annotations if matrix is not overly massive (e.g. <= 20 columns)
            if n_cols <= 16:
                # Text contrast color
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

    # Custom Diverging Palette: Rose (-1.0) -> Slate (0.0) -> Indigo/Emerald (+1.0)
    diverging_colorscale = [
        [0.0, "rgb(225, 29, 72)"],     # Deep Rose / Red (-1.0)
        [0.25, "rgb(244, 63, 94)"],    # Light Rose (-0.5)
        [0.45, "rgb(51, 65, 85)"],     # Slate Transition
        [0.50, "rgb(30, 41, 59)"],     # Dark Slate Neutral (0.0)
        [0.55, "rgb(51, 65, 85)"],     # Slate Transition
        [0.75, "rgb(99, 102, 241)"],   # Vibrant Indigo (+0.5)
        [1.0, "rgb(16, 185, 129)"],    # Emerald Green (+1.0)
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
            "title": {"text": "Pearson (r)", "font": {"color": "#cbd5e1", "size": 12, "family": "Plus Jakarta Sans"}},
            "tickvals": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "ticktext": ["-1.0 (Neg)", "-0.5", "0.0", "+0.5", "+1.0 (Pos)"],
            "tickfont": {"color": "#94a3b8", "size": 10, "family": "JetBrains Mono"},
            "thickness": 14,
            "len": 0.85,
            "outlinewidth": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
    }

    # Calculate optimal height based on column count
    calculated_height = max(height, min(750, 180 + (n_cols * 32)))

    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "height": calculated_height,
        "margin": {"l": 120, "r": 60, "t": 40, "b": 100, "pad": 4},
        "xaxis": {
            "tickangle": -45,
            "tickfont": {"color": "#cbd5e1", "size": 11, "family": "Plus Jakarta Sans"},
            "gridcolor": "rgba(255,255,255,0.04)",
            "zeroline": False,
        },
        "yaxis": {
            "autorange": "reversed",
            "tickfont": {"color": "#cbd5e1", "size": 11, "family": "Plus Jakarta Sans"},
            "gridcolor": "rgba(255,255,255,0.04)",
            "zeroline": False,
        },
        "annotations": annotations,
    }

    config = {
        "responsive": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }

    return {
        "data": [trace],
        "layout": layout,
        "config": config,
    }


def build_outlier_boxplot_spec(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    height: int = 420,
) -> Dict[str, Any]:
    """
    Construct a high-performance, dark-theme Plotly multi-column Box Plot JSON specification.
    Displays median, quartiles, fences, and individual jittered outlier points.
    """
    if df.empty:
        return {
            "data": [],
            "layout": {
                "title": {"text": "No Data Available for Box Plot", "font": {"color": "#94a3b8"}},
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            },
            "config": {"responsive": True, "displayModeBar": False},
        }

    # Select numerical columns
    if not columns:
        columns = [str(c) for c in df.select_dtypes(include=[np.number]).columns]

    if not columns:
        return {
            "data": [],
            "layout": {
                "title": {"text": "No Numerical Features Available for Box Plot", "font": {"color": "#94a3b8"}},
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            },
            "config": {"responsive": True, "displayModeBar": False},
        }

    traces = []
    # Color palette for multiple box traces
    palette = [
        {"fill": "rgba(99, 102, 241, 0.25)", "line": "#818cf8"},
        {"fill": "rgba(16, 185, 129, 0.25)", "line": "#34d399"},
        {"fill": "rgba(6, 182, 212, 0.25)", "line": "#22d3ee"},
        {"fill": "rgba(245, 158, 11, 0.25)", "line": "#fbbf24"},
        {"fill": "rgba(168, 85, 247, 0.25)", "line": "#c084fc"},
        {"fill": "rgba(236, 72, 153, 0.25)", "line": "#f472b6"},
    ]

    for i, col in enumerate(columns[:12]):  # Limit to 12 columns for visual clarity
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue

        color_item = palette[i % len(palette)]
        # Sample for display if extremely massive dataset
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

    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "height": max(height, 380),
        "margin": {"l": 60, "r": 40, "t": 30, "b": 80, "pad": 4},
        "font": {"family": "Plus Jakarta Sans, sans-serif", "color": "#cbd5e1"},
        "showlegend": False,
        "xaxis": {
            "tickangle": -35 if len(traces) > 4 else 0,
            "tickfont": {"color": "#cbd5e1", "size": 11},
            "gridcolor": "rgba(255,255,255,0.04)",
            "zeroline": False,
        },
        "yaxis": {
            "tickfont": {"color": "#cbd5e1", "size": 11, "family": "JetBrains Mono"},
            "gridcolor": "rgba(255,255,255,0.06)",
            "zeroline": False,
        },
    }

    config = {
        "responsive": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }

    return {
        "data": traces,
        "layout": layout,
        "config": config,
    }


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
    Renders data histogram, threshold boundary dashed lines, and shaded outlier danger zones.
    """
    clean_s = pd.to_numeric(series, errors="coerce").dropna()

    if len(clean_s) == 0:
        return {
            "data": [],
            "layout": {
                "title": {"text": f"No valid numerical data in '{col_name}'", "font": {"color": "#94a3b8"}},
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            },
            "config": {"responsive": True, "displayModeBar": False},
        }

    min_val = float(clean_s.min())
    max_val = float(clean_s.max())
    span = max_val - min_val if max_val > min_val else 1.0
    pad = span * 0.08

    # Primary histogram trace
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

    # Outlier points strip trace
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

    # Lower Threshold Line & Zone
    if lower_thresh is not None and not math.isnan(lower_thresh):
        # Shaded lower zone
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
        # Threshold line
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

    # Upper Threshold Line & Zone
    if upper_thresh is not None and not math.isnan(upper_thresh):
        # Shaded upper zone
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
        # Threshold line
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

    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "height": height,
        "margin": {"l": 50, "r": 40, "t": 40, "b": 60, "pad": 4},
        "font": {"family": "Plus Jakarta Sans, sans-serif", "color": "#cbd5e1"},
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "y": -0.22,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 11, "color": "#94a3b8"},
        },
        "xaxis": {
            "title": {"text": col_name, "font": {"color": "#cbd5e1", "size": 12}},
            "tickfont": {"color": "#94a3b8", "size": 10, "family": "JetBrains Mono"},
            "gridcolor": "rgba(255,255,255,0.05)",
            "zeroline": False,
        },
        "yaxis": {
            "title": {"text": "Count / Frequency", "font": {"color": "#cbd5e1", "size": 12}},
            "tickfont": {"color": "#94a3b8", "size": 10, "family": "JetBrains Mono"},
            "gridcolor": "rgba(255,255,255,0.05)",
            "zeroline": False,
        },
        "shapes": shapes,
        "annotations": annotations,
    }

    config = {
        "responsive": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }

    return {
        "data": traces,
        "layout": layout,
        "config": config,
    }


def generate_summary_charts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate standard exploratory charts for the dashboard.
    """
    return []


