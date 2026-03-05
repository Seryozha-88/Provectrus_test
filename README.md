# Claude Code Analytics Platform

> **End-to-end analytics platform for Claude Code telemetry data**


---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data Pipeline](#data-pipeline)
- [Dashboard Pages](#dashboard-pages)
- [AI Insights](#ai-insights)
- [ML & Anomaly Detection](#ml--anomaly-detection)
- [Database Schema](#database-schema)
- [Technologies](#technologies)
- [Configuration](#configuration)

---

## Overview

This platform ingests synthetic Claude Code telemetry data (API requests, tool usage, errors, user prompts) from a JSONL log file, processes it through an ETL pipeline into a SQLite database, and presents interactive analytics through a Streamlit dashboard with 7 pages — including an LLM-powered natural language query interface and ML anomaly detection.

### Key Numbers

| Metric | Value |
|--------|-------|
| Total events processed | 454,428 |
| API requests | 118,014 |
| Tool decisions | 151,461 |
| Tool results | 148,418 |
| User prompts | 35,173 |
| API errors | 1,362 |
| Employees | 100 |
| Sessions | 5,000 |
| Total cost tracked | $6,001.43 |
| Date range | 2025-12-03 → 2026-01-31 (60 days) |
| Raw data size | 521 MB JSONL |
| Database size | 115.8 MB SQLite |

---

## Features

### 1. ETL Pipeline (`src/ingest.py`)
- Streaming JSONL parser — memory-efficient for 500+ MB files
- Nested JSON extraction (batch → logEvents → message → event)
- Pydantic validation for all 5 event types
- Automatic type casting (string → numeric)
- Error-resilient: logs and skips malformed records

### 2. Interactive Dashboard (`src/dashboard.py`)
- 7 pages with 20+ interactive Plotly charts
- Real-time filtering by date, practice, level, and model
- KPI cards, time series, heatmaps, pie/bar/stacked charts

### 3. AI-Powered Query Interface (`src/ai_insights.py`)
- Natural language → SQL → human answer pipeline
- OpenRouter API with configurable LLM models
- Safety-checked: read-only queries only
- 10 example questions for quick start

### 4. ML & Anomaly Detection (`src/ml_anomaly.py`)
- IsolationForest anomaly detection (flags unusual sessions)
- DecisionTree practice classification (team prediction)
- KMeans session clustering (behavioral groups)
- LinearRegression cost forecasting (trend prediction)
- 12 engineered features per session, all CPU-only (<3s training)

---

## Architecture

```
┌──────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ generate_fake_data.py│────▶│  Raw Data Files  │────▶│  ETL Pipeline    │
│ (data generator)     │     │  - JSONL (events) │     │  (ingest.py)     │
│                      │     │  - CSV (employees)│     │  - Parse + clean │
└──────────────────────┘     └──────────────────┘     │  - Validate      │
                                                       └────────┬─────────┘
                                                                │
                                                                ▼
┌──────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Streamlit Dashboard │◀────│  Analytics Layer  │◀────│  SQLite Database │
│  (dashboard.py)      │     │  (analytics.py)   │     │  (database.py)   │
│  7 pages + filters   │     │  computed metrics  │     │  6 tables + view │
└──────┬───────────────┘     └──────────────────┘     └──────────────────┘
       │                             ▲
       │  ┌──────────────┐           │
       ├──│ ai_insights  │───────────┘  (NL → SQL → answer)
       │  └──────────────┘
       │  ┌──────────────┐
       └──│ ml_anomaly   │  (IsolationForest, DecisionTree, KMeans, LinearReg)
          └──────────────┘
```

**Layered design:**
- `models.py` — Data validation (Pydantic)
- `database.py` — SQL queries → DataFrames
- `analytics.py` — Computed metrics (cache ratio, efficiency, WoW)
- `dashboard.py` — Streamlit visualization
- `ai_insights.py` — LLM pipeline (OpenRouter)
- `ml_anomaly.py` — scikit-learn models

---

## Project Structure

```
Provectrus(Project)/
│
├── src/
│   ├── __init__.py              # Package init
│   ├── models.py                # Pydantic models for 5 event types (353 lines)
│   ├── ingest.py                # ETL pipeline: JSONL/CSV → SQLite (577 lines)
│   ├── database.py              # DatabaseManager with 13 query methods (693 lines)
│   ├── analytics.py             # AnalyticsEngine with 12 analytics methods (458 lines)
│   ├── ai_insights.py           # AIInsights: NL → SQL → answer (326 lines)
│   ├── ml_anomaly.py            # MLEngine: 4 scikit-learn models (487 lines)
│   └── dashboard.py             # Streamlit app with 7 pages (1,162 lines)
│
├── data/                        # Generated data (gitignored)
│   ├── telemetry_logs.jsonl     # 521 MB, 82,661 batches
│   ├── employees.csv            # 100 employees
│   └── analytics.db             # 115.8 MB SQLite database
│
├── presentations/               # Documentation
│   ├── TECHNICAL_DOCUMENTATION.md
│   └── LLM_USAGE_LOG.md
│
├── generate_fake_data.py        # Assessment-provided data generator
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Modern Python packaging config
├── PROJECT_PLAN.md              # Detailed implementation plan
├── README.md                    # This file
└── .gitignore
```

**Total source code: ~4,056 lines across 7 Python modules.**

---

## Quick Start

### Prerequisites

- Python 3.12+ 
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Seryozha-88/Provectrus_test.git
cd Provectrus_test
```

### 2. Create virtual environment & install dependencies

```bash
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
python generate_fake_data.py --num-users 100 --num-sessions 5000 --days 60 --output-dir data
```

This creates:
- `data/telemetry_logs.jsonl` (~521 MB, ~454K events)
- `data/employees.csv` (100 employees)

### 4. Run ETL pipeline (data ingestion)

```bash
python -m src.ingest
```

Parses JSONL + CSV, validates with Pydantic, loads into SQLite.  
Output: `data/analytics.db` (~116 MB), takes ~24 seconds.

### 5. Launch the dashboard

```bash
streamlit run src/dashboard.py
```

Opens at `http://localhost:8501` with all 7 pages ready.

### 6. (Optional) Test ML models

```bash
python -m src.ml_anomaly
```

Trains all 4 models and prints results (~3 seconds).

---

## Data Pipeline

### Input Format (JSONL)

Each line in `telemetry_logs.jsonl` is a **batch** containing nested events:

```
LINE (batch) → logEvents[] → message (JSON string) → event object
```

This requires **double `json.loads()`** — once for the line, once for each message.

### 5 Event Types

| Event Type | Body Field | Key Attributes |
|-----------|------------|---------------|
| **API Request** | `claude_code.api_request` | model, cost_usd, tokens (in/out/cache), duration_ms |
| **Tool Decision** | `claude_code.tool_use_decision` | tool_name, decision (allow/deny), source |
| **Tool Result** | `claude_code.tool_use_result` | tool_name, success, duration_ms |
| **User Prompt** | `claude_code.user_prompt` | prompt_length, has_images |
| **API Error** | `claude_code.api_error` | error, status_code, attempt, is_retry |

### Processing Steps

1. **Stream** JSONL line by line (memory-efficient)
2. **Extract** logEvents from each batch
3. **Parse** nested JSON message strings
4. **Validate** with Pydantic models (type safety, casting)
5. **Bulk insert** into SQLite with parameterized queries
6. **Create** indexes and session_summary materialized view

---

## Dashboard Pages

### Page 1: 📊 Overview
- KPI cards: total cost, sessions, users, events, error rate
- Daily activity timeline (cost + requests)
- Period summary with averages

### Page 2: 💰 Cost & Tokens
- Cost breakdown by model (pie chart)
- Cost by practice/team (bar chart)
- Cost by seniority level (bar chart)
- Token analysis: input vs output vs cache
- Cache hit ratio metric

### Page 3: 🔧 Tool Usage
- Tool frequency bar chart (top tools)
- Success rate by tool
- Reliability score (70% success + 30% speed)
- Tool decision sources (allow/deny breakdown)

### Page 4: 👥 User Behavior
- Activity heatmap (day × hour)
- User rankings table (cost, sessions, requests)
- Session duration analysis

### Page 5: ⚠️ Errors
- Error type donut chart
- Error by status code
- Error by model
- Retry attempt analysis

### Page 6: 🤖 AI Insights
- Chat interface for natural language queries
- Auto-generates SQL from questions
- Interprets results in plain English
- 10 example questions for quick start
- Expandable SQL + raw results per answer

### Page 7: 🔬 ML & Anomaly Detection
- **Tab 1**: Anomaly Detection — PCA scatter, score histogram, top anomalies table
- **Tab 2**: Practice Classification — feature importance, CV scores, tree rules
- **Tab 3**: Session Clustering — cluster scatter, sizes, profiles heatmap
- **Tab 4**: Cost Forecast — historical + predicted line chart, trend analysis

### Sidebar Filters (Pages 1-5)
- Date range picker
- Practice (team) dropdown
- Seniority level dropdown
- Model dropdown
- ML parameter sliders (Page 7)

---

## AI Insights

The AI query system uses a two-step LLM pipeline:

```
User Question → [LLM: Generate SQL] → Execute SQL → [LLM: Interpret Results] → Answer
```

**Configuration:**
- API: OpenRouter (OpenAI-compatible SDK)
- Default model: `google/gemini-2.0-flash-001`
- Available models: Gemini Flash/Pro, Claude Sonnet/Haiku, GPT-4o-mini, Llama 3.1

**Safety:**
- Only SELECT queries allowed
- Blocked keywords: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, etc.
- Full database schema injected into system prompt

**Example questions:**
- "What is the total cost per team?"
- "Which model has the highest error rate?"
- "Show me the top 5 most expensive sessions"

---

## ML & Anomaly Detection

### Feature Engineering (12 features per session)

| Feature | Description |
|---------|-------------|
| `api_call_count` | Number of API calls in session |
| `total_cost` | Total cost in USD |
| `total_input_tokens` | Sum of input tokens |
| `total_output_tokens` | Sum of output tokens |
| `total_cache_read_tokens` | Sum of cache read tokens |
| `prompt_count` | Number of user prompts |
| `tool_use_count` | Number of tool executions |
| `tool_success_count` | Successful tool executions |
| `duration_minutes` | Session duration |
| `cost_per_api_call` | Engineered: cost / api_calls |
| `tokens_per_api_call` | Engineered: tokens / api_calls |
| `tool_success_rate` | Engineered: successes / tool_uses |

### Models

| Model | Algorithm | Purpose | Key Output |
|-------|-----------|---------|-----------|
| Anomaly Detection | IsolationForest | Flag unusual sessions | 250 anomalies / 5,000 sessions |
| Classification | DecisionTree | Predict team from usage | Feature importance + tree rules |
| Clustering | KMeans (k=4) | Group behavioral patterns | Cluster profiles + PCA scatter |
| Forecasting | LinearRegression | Predict future costs | Trend slope + 30-day forecast |

All models are **CPU-only** (scikit-learn), training time **< 3 seconds**.

---

## Database Schema

### Tables

| Table | Rows | Description |
|-------|------|-------------|
| `api_requests` | 118,014 | Claude API call records |
| `tool_decisions` | 151,461 | Tool allow/deny decisions |
| `tool_results` | 148,418 | Tool execution results |
| `user_prompts` | 35,173 | User input events |
| `api_errors` | 1,362 | API error records |
| `employees` | 100 | User profiles (CSV) |

### Views

- `session_summary` — Pre-aggregated per-session metrics (cost, tokens, API calls, timestamps) joined with employee data

### Indexes

Indexes on `session_id`, `user_email`, `timestamp`, and `model` columns for fast filtering.

---

## Technologies

| Category | Technology | Version |
|----------|-----------|---------|
| Language | Python | 3.12 |
| Web Framework | Streamlit | ≥1.30 |
| Data Validation | Pydantic | ≥2.0 |
| Data Analysis | Pandas | ≥2.0 |
| Visualization | Plotly | ≥5.18 |
| Database | SQLite | 3 (built-in) |
| Machine Learning | scikit-learn | ≥1.4 |
| LLM API | OpenAI SDK (OpenRouter) | ≥1.0 |
| LLM Provider | OpenRouter | — |
| Testing | pytest | ≥8.0 |

---

## Configuration

### Environment Variables

| Variable | Purpose | Where to Set |
|----------|---------|-------------|
| OpenRouter API Key | AI Insights page | Dashboard sidebar input field |

### Default Paths

| Path | Contents |
|------|----------|
| `data/telemetry_logs.jsonl` | Raw telemetry data |
| `data/employees.csv` | Employee profiles |
| `data/analytics.db` | SQLite database |

### ML Parameters (Dashboard Page 7 Sidebar)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Contamination | 0.05 | 0.01–0.20 | Expected anomaly fraction |
| Clusters | 4 | 2–8 | KMeans cluster count |
| Tree depth | 5 | 2–10 | DecisionTree max depth |
| Forecast horizon | 30 | 7–90 | Days to predict ahead |

---

## Git History

| Commit | Phase | Description |
|--------|-------|-------------|
| `e0d19ce` | 0 | Initial project structure and dependencies |
| `c6d0d73` | 1 | Pydantic data models for all event types |
| `468e212` | 2 | ETL pipeline with streaming parser and SQLite loader |
| `974a2e3` | 3-4 | Database query layer and analytics computation engine |
| `69546ef` | 5 | Interactive Streamlit dashboard with 7 pages |
| `2081336` | 6 | LLM-powered AI Insights page with OpenRouter integration |
| `708ac37` | 7 | ML anomaly detection, classification, clustering, forecasting |

---

## License

This project was created as a test assessment for the Provectus Python & Gen AI Internship.
