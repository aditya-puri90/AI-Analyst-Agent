"""
Report Generation Engine (Phase 12 Module).
Compiles comprehensive dataset analysis, data quality audit, descriptive statistics,
correlation analysis, outlier findings, and AI insights into Markdown and standalone HTML executive reports.
"""

from typing import Dict, Any, Optional
import datetime
import pandas as pd
import numpy as np

from analysis.profiler import DatasetProfiler
from analysis.cleaning import detect_data_quality_issues
from analysis.statistics import StatisticalAnalysisEngine
from analysis.correlation import CorrelationAnalysisEngine
from analysis.outliers import OutlierDetectionEngine
from agent.analyst_agent import AnalystAgent


class ReportGenerator:
    """
    Comprehensive Executive Report Generator for AI Data Analyst Agent.
    Assembles all analysis engines into executive-level Markdown and HTML reports.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        original_filename: Optional[str] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        quality_data: Optional[Dict[str, Any]] = None,
        stats_data: Optional[Dict[str, Any]] = None,
        corr_data: Optional[Dict[str, Any]] = None,
        outlier_data: Optional[Dict[str, Any]] = None,
        insights_data: Optional[Dict[str, Any]] = None,
    ):
        self.df = df
        self.dataset_id = dataset_id
        self.filename = original_filename or dataset_id
        self.profile_data = profile_data
        self.quality_data = quality_data
        self.stats_data = stats_data
        self.corr_data = corr_data
        self.outlier_data = outlier_data
        self.insights_data = insights_data

    def _ensure_data(self):
        """Lazy-load missing analysis components."""
        if not self.profile_data:
            profiler = DatasetProfiler(self.df, dataset_id=self.dataset_id)
            self.profile_data = profiler.to_dict()

        if not self.quality_data:
            issues = detect_data_quality_issues(self.df)
            self.quality_data = {
                "total_issues": len(issues),
                "issues": issues,
            }

        if not self.stats_data:
            stats_engine = StatisticalAnalysisEngine(self.df, dataset_id=self.dataset_id)
            self.stats_data = stats_engine.analyze()

        if not self.corr_data:
            corr_engine = CorrelationAnalysisEngine(self.df, dataset_id=self.dataset_id)
            self.corr_data = corr_engine.analyze()

        if not self.outlier_data:
            outlier_engine = OutlierDetectionEngine(self.df, dataset_id=self.dataset_id)
            self.outlier_data = outlier_engine.analyze(method="iqr")

        if not self.insights_data:
            agent = AnalystAgent()
            ctx = agent.build_analysis_context(self.df, dataset_id=self.dataset_id)
            self.insights_data = agent.generate_insights(ctx)

    def generate_structured_report(self) -> Dict[str, Any]:
        """Compile a complete JSON structure of the executive report."""
        self._ensure_data()

        overview = self.profile_data.get("overview", {})
        quality_summary = self.profile_data.get("quality_summary", {})
        ranked_pairs = self.corr_data.get("ranked_pairs", [])
        top_correlations = ranked_pairs[:5] if ranked_pairs else []
        col_outliers = self.outlier_data.get("columns", {})
        
        # Outlier counts
        total_outliers = sum(c.get("outlier_count", 0) for c in col_outliers.values())

        return {
            "title": f"Executive Data Analysis Report: {self.filename}",
            "dataset_id": self.dataset_id,
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overview": overview,
            "quality": {
                "health_score": quality_summary.get("overall_score", 100),
                "health_grade": quality_summary.get("health_grade", "A+"),
                "total_issues": self.quality_data.get("total_issues", 0),
                "issues": self.quality_data.get("issues", []),
            },
            "statistics": {
                "numerical_columns_count": len(self.stats_data.get("numerical_statistics", {})),
                "observations": self.stats_data.get("statistical_observations", []),
                "numerical_summary": self.stats_data.get("numerical_statistics", {}),
            },
            "correlations": {
                "total_pairs": len(ranked_pairs),
                "top_pairs": top_correlations,
            },
            "outliers": {
                "total_outliers": total_outliers,
                "columns_affected": len([c for c, v in col_outliers.items() if v.get("outlier_count", 0) > 0]),
                "breakdown": col_outliers,
            },
            "ai_insights": self.insights_data,
        }

    def generate_markdown_report(self) -> str:
        """Generate a complete, beautifully formatted Markdown report."""
        rep = self.generate_structured_report()
        ov = rep["overview"]
        qual = rep["quality"]
        stats = rep["statistics"]
        corrs = rep["correlations"]
        out = rep["outliers"]
        ai = rep["ai_insights"]

        # Format Top Correlations Table
        corr_rows = []
        for pair in corrs.get("top_pairs", []):
            corr_rows.append(
                f"| `{pair.get('feature_a')}` & `{pair.get('feature_b')}` | {pair.get('pearson_r', 0):.4f} | {pair.get('strength', 'N/A')} | {pair.get('direction', 'N/A')} |"
            )
        corr_table = (
            "| Feature Pair | Pearson r | Strength | Direction |\n|---|---|---|---|\n" + "\n".join(corr_rows)
            if corr_rows
            else "_No significant linear correlations detected or insufficient numerical features._"
        )

        # Format Outliers Table
        out_rows = []
        for col_name, o in out.get("breakdown", {}).items():
            if o.get("outlier_count", 0) > 0:
                out_rows.append(
                    f"| `{col_name}` | {o.get('outlier_count', 0)} | {o.get('outlier_percentage', 0):.2f}% | [{o.get('lower_threshold', 0):.2f}, {o.get('upper_threshold', 0):.2f}] |"
                )
        out_table = (
            "| Column | Outlier Count | Outlier % | Bounds [Lower, Upper] |\n|---|---|---|---|\n" + "\n".join(out_rows)
            if out_rows
            else "_No statistical outliers detected under standard IQR bounds (1.5x multiplier)._"
        )

        # Format Statistical Highlights
        obs_bullets = "\n".join(f"- {o.get('message', o)}" for o in stats.get("observations", []))
        if not obs_bullets:
            obs_bullets = "- Standard feature distributions observed across features."

        # Format AI sections
        ai_exec = ai.get("executive_summary", "Automated analysis completed successfully.")
        ai_findings = ai.get("key_findings", [])
        findings_md = "\n".join(f"- **Finding {i+1}**: {f}" for i, f in enumerate(ai_findings)) if ai_findings else "- Comprehensive profile available in dashboard."
        
        ai_risks = ai.get("anomalies_and_risks", [])
        risks_md = "\n".join(f"- ⚠️ {r}" for r in ai_risks) if ai_risks else "- No critical operational risks identified."
        
        ai_recs = ai.get("strategic_recommendations", [])
        recs_md = "\n".join(f"- 🎯 **Action {i+1}**: {r}" for i, r in enumerate(ai_recs)) if ai_recs else "- Continue routine data collection and monitoring."

        md = f"""# Executive Data Analysis & Quality Report

**Dataset:** `{self.filename}`  
**Dataset ID:** `{self.dataset_id}`  
**Report Generated:** `{rep['generated_at']}`  
**AI Engine:** `{ai.get('provider_used', 'Deterministic Synthesis')}`  

---

## 1. Executive Summary & AI Findings

{ai_exec}

### Key Strategic Findings
{findings_md}

### Anomalies, Risks & Data Biases
{risks_md}

### Strategic Recommendations & Action Items
{recs_md}

---

## 2. Dataset Overview & Vital Statistics

| Metric | Value | Detail |
|---|---|---|
| **Total Observational Rows** | **{ov.get('total_rows', 0):,}** | Valid records |
| **Total Features (Columns)** | **{ov.get('total_columns', 0):,}** | Dimensions |
| **Total Data Points** | **{ov.get('total_cells', 0):,}** | Matrix entries |
| **RAM Memory Usage** | **{ov.get('memory_usage_mb', 0):.2f} MB** | In-memory footprint |
| **Duplicate Rows** | **{ov.get('duplicate_rows', 0):,}** ({ov.get('duplicate_rows_percentage', 0):.2f}%) | Redundant rows |
| **Missing Data Cells** | **{ov.get('missing_cells', 0):,}** ({ov.get('missing_cells_percentage', 0):.2f}%) | Null / NaN entries |

---

## 3. Data Quality & Integrity Score

- **Data Health Score:** **{qual.get('health_score', 100)} / 100** (Grade: **{qual.get('health_grade', 'A+')}**)
- **Total Quality Issues Detected:** **{qual.get('total_issues', 0)}**
- **Completeness Rate:** **{100 - ov.get('missing_cells_percentage', 0):.2f}%**
- **Uniqueness Rate:** **{100 - ov.get('duplicate_rows_percentage', 0):.2f}%**

---

## 4. Key Statistical Observations

{obs_bullets}

---

## 5. Top Feature Correlations (Pearson r)

{corr_table}

> **Note on Causation**: Correlation measures linear association strength, not direct causal relationships.

---

## 6. Outlier & Anomaly Summary (IQR 1.5x)

- **Total Detected Outliers:** **{out.get('total_outliers', 0)}**
- **Features with Outliers:** **{out.get('columns_affected', 0)}**

{out_table}

---

*Generated by AI Data Analyst Agent — Modern Professional Analytics Suite.*
"""
        return md

    def generate_html_report(self) -> str:
        """Generate a styled, self-contained standalone HTML report for presentations and printing."""
        rep = self.generate_structured_report()
        ov = rep["overview"]
        qual = rep["quality"]
        stats = rep["statistics"]
        corrs = rep["correlations"]
        out = rep["outliers"]
        ai = rep["ai_insights"]

        # Correlation rows
        corr_trs = ""
        for p in corrs.get("top_pairs", []):
            badge_class = "strength-strong" if "Strong" in p.get("strength", "") else "strength-mod"
            corr_trs += f"""<tr>
                <td><strong>{p.get('feature_a')}</strong> & <strong>{p.get('feature_b')}</strong></td>
                <td style="font-family: monospace; font-weight: bold; color: #6366f1;">{p.get('pearson_r', 0):.4f}</td>
                <td><span class="badge {badge_class}">{p.get('strength', 'N/A')}</span></td>
                <td>{p.get('direction', 'N/A')}</td>
            </tr>"""
        if not corr_trs:
            corr_trs = "<tr><td colspan='4' class='text-muted'>No significant linear correlations detected.</td></tr>"

        # Outlier rows
        out_trs = ""
        for col_name, o in out.get("breakdown", {}).items():
            if o.get("outlier_count", 0) > 0:
                out_trs += f"""<tr>
                    <td><strong>{col_name}</strong></td>
                    <td style="font-family: monospace;">{o.get('outlier_count', 0):,}</td>
                    <td style="font-family: monospace;">{o.get('outlier_percentage', 0):.2f}%</td>
                    <td style="font-family: monospace;">[{o.get('lower_threshold', 0):.2f}, {o.get('upper_threshold', 0):.2f}]</td>
                </tr>"""
        if not out_trs:
            out_trs = "<tr><td colspan='4' class='text-muted'>No statistical outliers detected under standard IQR bounds (1.5x).</td></tr>"

        # Observations
        obs_lis = "".join(f"<li>{o.get('message', o)}</li>" for o in stats.get("observations", []))
        if not obs_lis:
            obs_lis = "<li>Standard distributions observed across numerical and categorical features.</li>"

        # AI findings
        findings_lis = "".join(f"<li>{f}</li>" for f in ai.get("key_findings", []))
        risks_lis = "".join(f"<li>{r}</li>" for r in ai.get("anomalies_and_risks", []))
        recs_lis = "".join(f"<li>{r}</li>" for r in ai.get("strategic_recommendations", []))

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Data Report - {self.filename}</title>
    <style>
        :root {{
            --bg: #0B0F19;
            --surface: #111827;
            --surface-card: #1F2937;
            --text: #F9FAFB;
            --text-muted: #9CA3AF;
            --primary: #6366F1;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
            --border: rgba(255, 255, 255, 0.1);
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 40px 20px;
        }}
        .report-container {{
            max-width: 1000px;
            margin: 0 auto;
            background: var(--surface);
            border-radius: 16px;
            padding: 48px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border);
        }}
        .report-header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 24px;
            margin-bottom: 32px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}
        .report-title {{ font-size: 28px; font-weight: 800; color: #FFF; margin-bottom: 8px; }}
        .report-meta {{ color: var(--text-muted); font-size: 14px; }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-grade {{ background: rgba(16, 185, 129, 0.2); color: var(--success); font-size: 16px; padding: 6px 14px; }}
        .strength-strong {{ background: rgba(99, 102, 241, 0.2); color: var(--primary); }}
        .strength-mod {{ background: rgba(6, 182, 212, 0.2); color: #06b6d4; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .kpi-box {{
            background: var(--surface-card);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border);
        }}
        .kpi-label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }}
        .kpi-val {{ font-size: 24px; font-weight: 800; color: #FFF; margin: 4px 0; font-family: monospace; }}
        .section {{ margin-bottom: 36px; }}
        .section-title {{ font-size: 20px; font-weight: 700; margin-bottom: 16px; color: #FFF; border-left: 4px solid var(--primary); padding-left: 12px; }}
        .ai-box {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .ai-title {{ font-size: 16px; font-weight: 700; color: #A5B4FC; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }}
        ul {{ padding-left: 24px; margin-top: 8px; }}
        li {{ margin-bottom: 8px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: var(--surface-card); color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 12px; }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
        .text-muted {{ color: var(--text-muted); font-style: italic; }}
        .footer {{ text-align: center; color: var(--text-muted); font-size: 13px; margin-top: 40px; border-top: 1px solid var(--border); padding-top: 20px; }}
        @media print {{
            body {{ background: #FFF; color: #000; padding: 0; }}
            .report-container {{ border: none; box-shadow: none; padding: 0; background: #FFF; color: #000; }}
            .kpi-box {{ background: #F3F4F6; border: 1px solid #E5E7EB; }}
            .kpi-val, .report-title, .section-title {{ color: #000; }}
            th {{ background: #F3F4F6; color: #374151; }}
            .ai-box {{ background: #F9FAFB; border: 1px solid #E5E7EB; color: #000; }}
            .ai-title {{ color: #4F46E5; }}
            td, th {{ border-bottom: 1px solid #E5E7EB; }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <div>
                <h1 class="report-title">Executive Data Analysis Report</h1>
                <div class="report-meta">
                    <strong>Dataset:</strong> {self.filename} &bull; 
                    <strong>Dataset ID:</strong> {self.dataset_id} &bull; 
                    <strong>Generated:</strong> {rep['generated_at']}
                </div>
            </div>
            <div>
                <span class="badge badge-grade">Grade: {qual.get('health_grade', 'A+')} ({qual.get('health_score', 100)}/100)</span>
            </div>
        </div>

        <!-- KPI Grid -->
        <div class="kpi-grid">
            <div class="kpi-box">
                <div class="kpi-label">Total Rows</div>
                <div class="kpi-val">{ov.get('total_rows', 0):,}</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Total Columns</div>
                <div class="kpi-val">{ov.get('total_columns', 0):,}</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Memory Footprint</div>
                <div class="kpi-val">{ov.get('memory_usage_mb', 0):.2f} MB</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Missing Data</div>
                <div class="kpi-val">{ov.get('missing_cells_percentage', 0):.2f}%</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Duplicates</div>
                <div class="kpi-val">{ov.get('duplicate_rows_percentage', 0):.2f}%</div>
            </div>
        </div>

        <!-- AI Executive Findings -->
        <div class="section">
            <h2 class="section-title">1. Executive Summary & AI Insights</h2>
            <div class="ai-box">
                <div class="ai-title">✨ Strategic Takeaways & Synthesis</div>
                <p>{ai.get('executive_summary', 'Automated data profile generated successfully.')}</p>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div class="kpi-box">
                    <h4 style="color: #FFF; margin-bottom: 8px;">📊 Key Analytical Findings</h4>
                    <ul>{findings_lis or '<li>Standard distribution pattern identified.</li>'}</ul>
                </div>
                <div class="kpi-box">
                    <h4 style="color: #FFF; margin-bottom: 8px;">🎯 Strategic Recommendations</h4>
                    <ul>{recs_lis or '<li>Monitor continuous metrics regularly.</li>'}</ul>
                </div>
            </div>
        </div>

        <!-- Data Quality Audit -->
        <div class="section">
            <h2 class="section-title">2. Data Quality & Health Audit</h2>
            <p>Overall Health Score: <strong>{qual.get('health_score', 100)} / 100</strong> with <strong>{qual.get('total_issues', 0)}</strong> detected quality issues.</p>
        </div>

        <!-- Statistical Observations -->
        <div class="section">
            <h2 class="section-title">3. Statistical Observations</h2>
            <ul>{obs_lis}</ul>
        </div>

        <!-- Correlations -->
        <div class="section">
            <h2 class="section-title">4. Top Feature Correlations</h2>
            <table>
                <thead>
                    <tr>
                        <th>Feature Pair</th>
                        <th>Pearson r</th>
                        <th>Strength</th>
                        <th>Direction</th>
                    </tr>
                </thead>
                <tbody>
                    {corr_trs}
                </tbody>
            </table>
        </div>

        <!-- Outliers -->
        <div class="section">
            <h2 class="section-title">5. Outlier Detection Summary (IQR 1.5x)</h2>
            <p>Total statistical outliers detected across features: <strong>{out.get('total_outliers', 0):,}</strong></p>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Outlier Count</th>
                        <th>Outlier %</th>
                        <th>Bounds [Lower, Upper]</th>
                    </tr>
                </thead>
                <tbody>
                    {out_trs}
                </tbody>
            </table>
        </div>

        <div class="footer">
            Generated by AI Data Analyst Agent &bull; Professional Portfolio Analytics Suite
        </div>
    </div>
</body>
</html>
"""
        return html


def generate_markdown_report(dataset_name: str, profile_data: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> str:
    """Backward-compatible helper function."""
    if df is not None:
        generator = ReportGenerator(df, dataset_id=dataset_name, profile_data=profile_data)
        return generator.generate_markdown_report()

    overview = profile_data.get("overview", {})
    return f"""# Data Analysis Report: {dataset_name}

## Executive Summary
- Total Records: {overview.get('total_rows', 0):,}
- Total Features: {overview.get('total_columns', 0):,}
- Memory Usage: {overview.get('memory_usage_mb', 0):.2f} MB
- Missing Data: {overview.get('missing_cells_percentage', 0):.2f}%

*Generated by AI Data Analyst Agent.*
"""
