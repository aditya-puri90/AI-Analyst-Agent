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


def generate_summary_charts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate standard exploratory charts for the dashboard.
    """
    return []

