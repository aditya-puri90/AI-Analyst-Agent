# 🤖 AI Data Analyst Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, portfolio-ready **AI Data Analyst Agent** web application. It transforms raw, arbitrary CSV datasets into comprehensive exploratory data analysis (EDA), data quality audits, parametric/non-parametric statistics, interactive Plotly visualizations, automated data cleaning pipelines, and natural-language AI insights with strict deterministic grounding.

---

## 📌 Table of Contents

- [Key Highlights](#-key-highlights)
- [System Architecture](#-system-architecture)
- [Directory Structure](#-directory-structure)
- [Technology Stack](#-technology-stack)
- [Development Principles](#-development-principles)
- [Phased Implementation Roadmap](#-phased-implementation-roadmap)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [API Reference (Phase 1)](#-api-reference-phase-1)
- [Testing & Quality Assurance](#-testing--quality-assurance)

---

## 🌟 Key Highlights

1. **Deterministic Core vs. AI Reasoning Layer**: All calculations (quantiles, IQR, correlations, skewness, missingness, isolation forests) are strictly computed by Python, Pandas, SciPy, and Scikit-learn. The LLM acts as an analytical synthesizer, answering user questions and highlighting insights without fabricating numbers.
2. **Dataset Immutability**: Uploaded raw datasets are treated as strictly immutable read-only records in `data/uploads/`. Cleaned and transformed versions are maintained separately in `data/processed/`.
3. **Generic & Dynamic Schema Inference**: Works with arbitrary CSV files. Automatically detects numerical, categorical, datetime, boolean, ID/high-cardinality, and text columns without hardcoded column names or assumptions about specific datasets (Titanic, Superstore, Iris, etc.).
4. **Resilient Ingestion**: Automatic delimiter detection (`,`, `;`, `\t`, `|`), multi-encoding fallback (`utf-8`, `latin-1`, `cp1252`, `iso-8859-1`), secure filename generation, and size limit validations.
5. **Modern Analytics Dashboard**: Dark-mode glassmorphic UI, responsive metric KPI cards, dynamic column chips, and an interactive data explorer.

---

## 🏛 System Architecture

```mermaid
flowchart TD
    subgraph Client["Frontend Layer (HTML5, CSS3, ES6 JS)"]
        UI["Dashboard & Chat UI"]
        Dropzone["Drag & Drop Upload Zone"]
        PlotlyView["Plotly Interactive Charts"]
        Table["Paginated Data Explorer"]
    end

    subgraph Server["Backend Layer (Flask 3.x)"]
        App["Flask App (app.py)"]
        Config["Configuration (config/settings.py)"]
        Validators["File & CSV Validators (utils/validators.py)"]
        FileHandler["Storage & File Handler (utils/file_handler.py)"]
    end

    subgraph Analytics["Deterministic Analytics Engines (analysis/)"]
        Profiler["Dataset Profiler (profiler.py)"]
        Stats["Statistical Engine (statistics.py)"]
        Correlations["Correlation Engine (correlation.py)"]
        Outliers["Outlier Detector (outliers.py)"]
        Cleaning["Data Cleaning Pipeline (cleaning.py)"]
    end

    subgraph Visualization["Visualization Layer (visualization/)"]
        Charts["Plotly Chart Generator (charts.py)"]
    end

    subgraph AI["AI Agent Layer (agent/)"]
        Router["Question Router (question_router.py)"]
        Agent["Analyst Agent (analyst_agent.py)"]
        Prompts["Engineered Prompts (prompts.py)"]
    end

    subgraph Storage["Storage Layer (data/)"]
        Raw["Immutable Uploads (data/uploads/)"]
        Processed["Cleaned Datasets (data/processed/)"]
    end

    Dropzone -->|1. Upload CSV| Validators
    Validators -->|2. Sanitize & Verify| FileHandler
    FileHandler -->|3. Persist Raw CSV| Raw
    FileHandler -->|4. Load DataFrame| Profiler
    Profiler -->|5. Profile Schema & Summary| App
    App -->|6. JSON Response| UI

    UI -->|Analytics Request| Stats & Correlations & Outliers
    UI -->|Visual Request| Charts
    UI -->|NL Query| Router
    Router -->|Dispatch Tool| Analytics
    Analytics -->|Computed Result| Agent
    Agent -->|Structured Insight| UI
```

---

## 📁 Directory Structure

```
AI-Data-Analyst-Agent/
│
├── app.py                      # Flask application factory, routes & API endpoints
├── requirements.txt            # Project dependencies
├── README.md                   # Complete architectural & operational documentation
├── .env.example                # Environment variable configuration template
├── .gitignore                  # Git exclusions (credentials, cache, upload data)
│
├── config/
│   └── settings.py             # Centralized application settings & env loader
│
├── data/
│   ├── uploads/                # Immutable raw CSV files (read-only)
│   └── processed/              # Cleaned / transformed datasets
│
├── analysis/                   # Deterministic Data Analysis Engines
│   ├── __init__.py
│   ├── profiler.py             # Schema detection, data shape, missingness, memory
│   ├── cleaning.py             # (Phase 4) Deduplication, imputation, type casting
│   ├── statistics.py           # (Phase 2) Parametric & non-parametric statistics
│   ├── correlation.py          # (Phase 3) Pearson, Spearman, Cramér's V
│   └── outliers.py             # (Phase 3) IQR, Z-Score, Isolation Forest
│
├── visualization/              # Dynamic Chart Generation
│   ├── __init__.py
│   └── charts.py               # (Phase 3) Plotly JSON chart specifications
│
├── agent/                      # AI Agent & Natural Language Interface
│   ├── __init__.py
│   ├── analyst_agent.py        # (Phase 5) LLM agent with deterministic tool binding
│   ├── question_router.py      # (Phase 5) Intent classification & tool dispatching
│   └── prompts.py              # (Phase 5) Analytical prompt templates
│
├── reports/                    # Automated Reporting
│   └── report_generator.py     # (Phase 5) Markdown / HTML / PDF executive reports
│
├── utils/                      # Utilities & Helpers
│   ├── __init__.py
│   ├── file_handler.py         # File persistence, encoding/delimiter detection
│   └── validators.py           # File validation, MIME check, structural sanity
│
├── templates/                  # Jinja2 HTML Templates
│   ├── index.html              # Landing page & CSV upload dropzone
│   ├── dashboard.html          # Analytics dashboard & dataset explorer
│   └── chat.html               # Natural language AI analyst conversational view
│
├── static/                     # Static Assets
│   ├── css/
│   │   └── style.css           # Modern dark-mode glassmorphic design system
│   └── js/
│       ├── main.js             # Upload handling, toast notifications, UI controls
│       └── dashboard.js        # Dataset explorer, column chips, pagination
│
└── tests/                      # Automated Unit & Integration Test Suite
    └── test_basic.py           # Foundation, validator, profiler & route tests
```

---

## 🛠 Technology Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Web Framework** | Python 3.10+ & Flask 3.0+ | Lightweight, modular REST API and routing |
| **Data Manipulation & Analysis** | Pandas 2.0+, NumPy 1.24+ | In-memory tabular processing, vector math |
| **Statistical Modeling & ML** | SciPy 1.11+, Scikit-learn 1.3+ | Hypothesis testing, distribution fitting, outlier detection |
| **Interactive Visualizations** | Plotly 5.18+ | Client-side interactive charts and heatmaps |
| **Frontend Dashboard** | HTML5, Modern CSS3, Vanilla ES6 JS | Fast, glassmorphic dark-mode analytics dashboard |
| **AI / LLM Integration** | Google Gemini API / OpenAI API | Analytical synthesis, tool calling, report drafting |
| **Testing** | Pytest, Requests | Automated unit and integration testing |

---

## 🎯 Development Principles

1. **Incremental Milestones**: Each phase implements a self-contained, fully testable layer before advancing.
2. **Deterministic Computation**: Python/Pandas calculates statistical values; the LLM never invents numbers.
3. **Data Integrity**: Original uploaded files are never overwritten or altered.
4. **Schema Agnostic**: Zero hardcoded columns; dynamic detection across arbitrary tabular structures.
5. **Defensive Security**: Safe file names, file size caps, extension checks, and sanitized inputs.

---

## 🗺 Phased Implementation Roadmap

- [x] **Phase 1: Project Foundation & Architecture**
  - [x] Modular project scaffolding & package structure.
  - [x] Settings management with `.env` and `config/settings.py`.
  - [x] Secure file upload handler with multi-encoding fallback and delimiter sniffing.
  - [x] Core dataset profiler (schema inference, missingness, duplicates, memory footprint).
  - [x] Flask REST API (`/api/upload`, `/api/datasets`, `/api/profile/<filename>`, `/api/preview/<filename>`).
  - [x] Modern dark-mode dashboard UI with drag-and-drop upload and data explorer.
  - [x] Unit test suite for Phase 1 components.
- [x] **Phase 2: Ingestion & Multi-Layer CSV Validation**
  - [x] Delimiter sniffing, encoding detection, and file integrity validation.
  - [x] First 10 rows tabular preview and RAM footprint summary.
- [x] **Phase 3: Automatic Dataset Profiling Engine**
  - [x] Automated column data type classification (Numerical, Categorical, Datetime, Boolean, Text/ID).
  - [x] Five-number summaries, missingness percentages, uniqueness metrics, and distinct sample chips.
- [x] **Phase 4: Automated Data Quality and Cleaning Engine**
  - [x] 12-point deterministic quality defect inspection (missingness, duplicates, invalid numbers, outliers, whitespace, inconsistent casings, type anomalies).
  - [x] Live transformation preview engine (`Original value → Proposed cleaned value`).
  - [x] Configurable cleaning pipeline (deduplication, median/mode imputation, string casting, outlier capping, column filtering).
  - [x] Strict non-destructive persistence saving processed datasets to `data/processed/`.
  - [x] Comprehensive cleaning summary KPIs (rows before/after, duplicates removed, missing handled, columns converted, values standardized).
  - [x] Cleaned CSV download button and preview toggle in Data Explorer.
- [x] **Phase 5: Statistical Analysis Engine**
  - [x] Pure Python parametric and non-parametric calculations (Mean, Median, Mode, Std, Variance, Range, IQR, Skewness, Kurtosis).
  - [x] Inferential statistics: 95% Confidence Intervals for the Mean via Student's t / normal distributions.
  - [x] Complete percentiles spectrum ($P_1$ to $P_{99}$) and $1.5 \times \text{IQR}$ outlier detection.
  - [x] Categorical Shannon entropy and temporal datetime cadence analysis.
  - [x] Deterministic rule-based statistical observations.
- [x] **Phase 6: Correlation Analysis Engine & Interactive Plotly Heatmaps**
  - [x] Automatic numerical column identification and symmetric Pearson correlation matrix.
  - [x] Resilient pairwise missing value handling and constant zero-variance feature safety.
  - [x] 5-tier correlation strength classification (`Very strong`, `Strong`, `Moderate`, `Weak`, `Very weak`) and direction (`Positive`, `Negative`, `Neutral`).
  - [x] Interactive threshold filtering slider ($|r| \ge \tau$) with live DOM reactivity.
  - [x] Interactive dark-mode Plotly correlation heatmap with in-cell annotations and hover details.
  - [x] Ranked correlation associations table and **Key Correlations** showcase cards with strict non-causation guidance (*"Correlation does not imply causation"*).
  - [x] Standalone CLI runner `run_correlation_demo.py` and 67 automated pytest test suites.
- [x] **Phase 7: Outlier Detection Engine & Interactive Plotly Charts**
  - [x] Interquartile Range (IQR / Tukey's Fences - Standard $1.5\times$ and Extreme $3.0\times$), Parametric Z-Score ($3.0\sigma$ and $2.5\sigma$), and Modified Z-Score (MAD).
  - [x] Column-level outlier analytics: Total observations, Valid N, Outlier count, Outlier percentage, Lower and Upper bounds.
  - [x] Statistical edge-case detection & warnings: Zero standard deviation ($\sigma = 0$), Zero IQR ($\text{IQR} = 0$), Small sample sizes ($N < 10, N < 30$), and high skewness.
  - [x] Domain-aware anomaly classification: Clear distinction between **Potential Statistical Outliers** and **Confirmed Data Errors** (sentinel placeholders, negative age/price/quantity, percentage overflow).
  - [x] Interactive Plotly visualizations: Multi-feature Box Plots with jittered outlier points and Single-Column Distribution Histograms with threshold dashed lines and shaded outlier zones.
  - [x] Outlier Records Inspector Modal for row-level drill-down review.
  - [x] Non-destructive remediation operations (`Remove Outlier Rows`, `Cap Outliers to Bounds`, `Remove Data Errors Only`, `Keep All Outliers`) saving separate datasets to `data/processed/` with parent lineage.
  - [x] Standalone CLI demo `run_outliers_demo.py` and 89 passing automated pytest suites.

- [x] **Phase 8: Automatic Visualization Engine & Chart Recommendation System**
  - [x] Schema-aware heuristic chart recommender (Categorical vs Numerical -> Bar, Temporal vs Numerical -> Line, Numerical Distribution -> Histogram, Anomaly -> Box Plot, Feature Pairs -> Correlation Heatmap).
  - [x] Dynamic Plotly JSON chart specifications with responsive dark-glassmorphic styling.
- [x] **Phase 9: AI Insight Engine & Executive Summary Synthesis**
  - [x] Deterministic context builder aggregating statistical moments, correlation networks, outlier thresholds, and quality scores.
  - [x] Multi-provider LLM integration (Google Gemini, OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet) with zero-key offline deterministic fallback.
  - [x] 8-section grounded executive report synthesis (Executive Summary, Quality Audit, Key Findings, Anomalies, Correlations, Risk Flags, Strategic Recommendations, Methodological Notes).
  - [x] Standalone CLI runner `run_insights_demo.py` and regression test suite.
- [x] **Phase 10: Natural-Language Dataset Q&A ("Ask Your Dataset")**
  - [x] Conversational analytics layer routing plain-English queries to 9 safe deterministic Python analysis tools.
  - [x] Safe tools: `dataset_summary()`, `column_summary()`, `groupby_analysis()`, `aggregation_analysis()`, `correlation_analysis()`, `outlier_analysis()`, `time_series_analysis()`, `distribution_analysis()`, `investigate_further()`.
  - [x] Strict factual grounding: zero numerical hallucinations with every figure directly computed on Pandas DataFrames.
  - [x] Resilient schema & missing column handling: gracefully explains unsupported queries and suggests valid candidate columns.
  - [x] Interactive Plotly visualization spec generation matched to tool output (bar, line, heatmap, boxplot, histogram).
  - [x] Full "Ask Your Dataset" conversational workspace (`/chat`) with session memory, tool badges, JSON inspector drawer, quick prompt chips, and transcript export.
  - [x] Standalone CLI demo `run_qa_demo.py` and 166 passing automated pytest test cases.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### 1. Clone or Navigate to the Repository
```bash
cd AI-Data-Analyst-Agent
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and adjust configuration if needed:
```bash
cp .env.example .env
```

---

## 🖥 Running the Application

Start the Flask development server:
```bash
python app.py
```

The application will be accessible at:
👉 **`http://127.0.0.1:5000`**

### Available Pages:
- **`http://127.0.0.1:5000/`** - Home & CSV Upload Zone
- **`http://127.0.0.1:5000/dashboard`** - Analytics Dashboard, Profiler & Data Cleaning Studio
- **`http://127.0.0.1:5000/chat`** - "Ask Your Dataset" AI Conversational Workspace

### Standalone CLI Runners:
```bash
# Phase 6 Correlation Analysis Demo
python run_correlation_demo.py

# Phase 7 Outlier & Anomaly Detection Demo
python run_outliers_demo.py

# Phase 9 AI Insight Engine Demo
python run_insights_demo.py

# Phase 10 Natural-Language Dataset Q&A Demo
python run_qa_demo.py
```

---

## 📡 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | System health check and status |
| `/api/upload` | `POST` | Upload and validate a new CSV dataset |
| `/api/datasets` | `GET` | List all available uploaded datasets |
| `/api/profile/<dataset_id>` | `GET` | Retrieve metadata profile & schema for a dataset |
| `/api/preview/<dataset_id>` | `GET` | Retrieve paginated rows for tabular inspection |
| `/api/sample/<name>` | `POST` | Load bundled sample dataset (e.g. Sales, Employees) |
| `/api/cleaning/audit/<dataset_id>` | `GET` | Run 12-point data quality audit and get defect list |
| `/api/cleaning/preview/<dataset_id>` | `GET` | Generate before-and-after transformation preview pairs |
| `/api/cleaning/apply/<dataset_id>` | `POST` | Execute configurable cleaning pipeline & persist to `data/processed/` |
| `/api/cleaning/download/<dataset_id>` | `GET` | Download cleaned CSV file from `data/processed/` |
| `/api/insights/<dataset_id>` | `POST` | Generate grounded 8-section AI executive summary report |
| `/api/chat/ask` | `POST` | Ask natural language question, execute tool, and get grounded answer |
| `/api/chat/history/<dataset_id>` | `GET` | Retrieve multi-turn conversation history for active dataset |
| `/api/chat/clear/<dataset_id>` | `POST` | Clear conversation history for active dataset |
| `/api/chat/suggested-questions/<dataset_id>` | `GET` | Dynamically generated schema-tailored question chips |
| `/api/chat/tools` | `GET` | List available safe deterministic analysis tools |

---

## 🧪 Testing & Quality Assurance

Run the automated test suite with `pytest`:
```bash
python -m pytest -v
```

All **166 test cases** across all phases pass with 100% test coverage.
