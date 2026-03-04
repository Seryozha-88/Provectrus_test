# Claude Code Analytics Platform — Implementation Plan

> **Role:** Python & Gen AI Internship @ Provectus
> **Goal:** End-to-end analytics platform for Claude Code telemetry data
> **Created:** March 4, 2026

---

## Table of Contents

1. [What We're Building](#1-what-were-building)
2. [Project Structure](#2-project-structure)
3. [Data Architecture](#3-data-architecture)
4. [Phase-by-Phase Implementation](#4-phase-by-phase-implementation)
5. [Database Schema](#5-database-schema)
6. [Dashboard Pages & Charts](#6-dashboard-pages--charts)
7. [Gen AI Feature Spec](#7-gen-ai-feature-spec)
8. [Key Insights to Extract](#8-key-insights-to-extract)
9. [Deliverables Checklist](#9-deliverables-checklist)
10. [Commands Reference](#10-commands-reference)

---

## 1. What We're Building

```
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ generate_fake_data.py│────►│  Raw Data Files  │────►│  ETL Pipeline    │
│ (provided script)    │     │  - JSONL (events) │     │  (ingest.py)     │
│                      │     │  - CSV (employees)│     │  - Parse nested  │
└─────────────────────┘     └──────────────────┘     │  - Validate      │
                                                      │  - Clean & cast  │
                                                      └────────┬─────────┘
                                                               │
                                                               ▼
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Streamlit Dashboard │◄────│  Analytics Layer │◄────│  SQLite Database │
│  (dashboard.py)      │     │  (analytics.py)  │     │  (database.py)   │
│  - Interactive charts│     │  - Query functions│     │  - 6 tables      │
│  - Filters & KPIs   │     │  - Aggregations  │     │  - Indexes       │
│  - AI Insights tab   │     │  - Cross-cuts    │     │  - Views         │
└─────────────────────┘     └──────────────────┘     └──────────────────┘
                                    ▲
                                    │
                            ┌───────┴────────┐
                            │ ai_insights.py │
                            │ - NL → SQL     │
                            │ - Auto summary │
                            └────────────────┘
```

**In one sentence:** Parse synthetic Claude Code telemetry → store in SQLite → compute analytics → display in Streamlit with an LLM-powered query feature.

---

## 2. Project Structure

```
claude-code-analytics/
│
├── generate_fake_data.py          # Provided data generator (copy from assessment)
│
├── src/
│   ├── __init__.py                # Package init
│   ├── models.py                  # Pydantic models for all data types
│   ├── ingest.py                  # ETL: parse JSONL/CSV → clean → load to DB
│   ├── database.py                # SQLite schema, connection, query repository
│   ├── analytics.py               # Analytical computation functions
│   ├── ai_insights.py             # LLM-powered natural language queries
│   └── dashboard.py               # Streamlit app (entry point)
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py             # Pydantic model validation tests
│   ├── test_ingest.py             # Parsing & cleaning tests
│   └── test_analytics.py          # Analytics function tests
│
├── data/                          # Generated data (gitignored)
│   ├── telemetry_logs.jsonl
│   └── employees.csv
│
├── presentation/
│   └── insights.pdf               # 3-5 slide findings presentation
│
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Modern Python packaging
├── .gitignore
├── README.md                      # Setup instructions + architecture
├── LLM_USAGE_LOG.md              # AI tool usage documentation
└── PROJECT_PLAN.md               # This file
```

**New file for ML:**
```
src/
└── ml_anomaly.py                  # Scikit-learn anomaly detection & classification
```

---

## 3. Data Architecture

### Input: Raw Data Files

#### `telemetry_logs.jsonl` — Nested structure (must be flattened)

```
LINE (batch) → logEvents[] → message (JSON string) → event object
```

Each line:
```json
{
  "messageType": "DATA_MESSAGE",
  "owner": "123456789012",
  "logGroup": "/claude-code/telemetry",
  "logStream": "otel-collector",
  "subscriptionFilters": ["logs-to-s3"],
  "year": 2026, "month": 1, "day": 15,
  "logEvents": [
    {
      "id": "56-digit-number",
      "timestamp": 1737000000000,
      "message": "{...JSON string that must be json.loads()'d...}"
    }
  ]
}
```

Each parsed `message`:
```json
{
  "body": "claude_code.api_request",
  "attributes": {
    "event.timestamp": "2026-01-15T10:30:00.123Z",
    "session.id": "uuid",
    "user.email": "alex.chen@example.com",
    "user.id": "sha256-hash",
    "user.account_uuid": "uuid",
    "organization.id": "uuid",
    "terminal.type": "vscode",
    "event.name": "api_request",
    "model": "claude-opus-4-6",
    "cost_usd": "0.071",
    "duration_ms": "10230",
    "input_tokens": "263",
    "output_tokens": "454",
    "cache_read_tokens": "73099",
    "cache_creation_tokens": "3149"
  },
  "scope": {
    "name": "com.anthropic.claude_code.events",
    "version": "2.1.50"
  },
  "resource": {
    "host.arch": "arm64",
    "host.name": "Alexs-MacBook-Pro.local",
    "os.type": "darwin",
    "os.version": "24.6.0",
    "service.name": "claude-code-None",
    "service.version": "2.1.50",
    "user.email": "",
    "user.practice": "ML Engineering",
    "user.profile": "alex.chen",
    "user.serial": "ABC1234567"
  }
}
```

**CRITICAL parsing notes:**
- `logEvents[].message` is a **string** → must `json.loads()` it
- All numeric values in `attributes` are **strings** → must cast to float/int
- `tool_result_size_bytes` is **optional** (present ~30% of time)
- `resource.user.email` is always `""` → use `attributes.user.email` instead

#### `employees.csv` — Flat table

| Column | Example | Possible Values |
|--------|---------|-----------------|
| email | alex.chen@example.com | {first}.{last}@example.com |
| full_name | Alex Chen | Capitalized |
| practice | ML Engineering | Platform, Data, ML, Backend, Frontend Engineering |
| level | L5 | L1–L10 (bell curve, peak at L5) |
| location | United States | US, Germany, UK, Poland, Canada |

**Join key:** `employees.email` = `attributes["user.email"]`

### 5 Event Types

| Event Type (`body`) | Key Fields | What It Represents |
|---------------------|------------|-------------------|
| `claude_code.api_request` | model, cost_usd, duration_ms, input/output/cache tokens | An LLM API call |
| `claude_code.tool_decision` | tool_name, decision (accept/reject), source | Whether a tool use was approved |
| `claude_code.tool_result` | tool_name, success, duration_ms, size_bytes? | Result of running a tool |
| `claude_code.user_prompt` | prompt_length | User typed a prompt |
| `claude_code.api_error` | model, error message, status_code, attempt | An API call failed |

---

## 4. Phase-by-Phase Implementation

### Phase 0: Project Setup
- [ ] Create folder structure
- [ ] Create `requirements.txt`:
  ```
  streamlit>=1.30.0
  pandas>=2.0.0
  plotly>=5.18.0
  pydantic>=2.0.0
  anthropic>=0.40.0
  scikit-learn>=1.4.0
  pytest>=8.0.0
  ```
- [ ] Create `pyproject.toml` with project metadata
- [ ] Create `.gitignore` (data/, __pycache__, *.db, .env, .venv)
- [ ] `git init` + initial commit
- [ ] Create virtual environment + install deps

**Commit:** `feat: initial project structure and dependencies`

---

### Phase 1: Data Models (`src/models.py`)
- [ ] `Employee` — Pydantic BaseModel with 5 fields
- [ ] `EventScope` — name, version
- [ ] `EventResource` — host.arch, host.name, os.type, os.version, service.*, user.*
- [ ] `BaseEvent` — common attributes (timestamp, session_id, user_email, user_id, account_uuid, org_id, terminal_type) + scope + resource
- [ ] `ApiRequestEvent(BaseEvent)` — model, cost_usd (float), duration_ms (int), input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
- [ ] `ToolDecisionEvent(BaseEvent)` — tool_name, decision, source
- [ ] `ToolResultEvent(BaseEvent)` — tool_name, success (bool), duration_ms, decision_source, decision_type, tool_result_size_bytes (Optional[int])
- [ ] `UserPromptEvent(BaseEvent)` — prompt_length (int)
- [ ] `ApiErrorEvent(BaseEvent)` — model, error, status_code, attempt (int), duration_ms
- [ ] `LogEvent` — id, timestamp, message (raw string)
- [ ] `LogBatch` — messageType, owner, logGroup, logStream, subscriptionFilters, logEvents, year, month, day

**Key Python features to demonstrate:** Type hints, Pydantic validators, Optional fields, field aliases (for dotted attribute names), `model_config` for strictness.

**Commit:** `feat: add Pydantic data models for all event types`

---

### Phase 2: ETL Pipeline (`src/ingest.py`)
- [ ] `parse_employees(csv_path: str) -> list[Employee]`
  - Read CSV with csv.DictReader
  - Validate each row with Employee model
  - Return list
- [ ] `parse_telemetry(jsonl_path: str) -> list[BaseEvent]`
  - Read JSONL line by line (memory-efficient)
  - For each line: `json.loads()` → get `logEvents` array
  - For each logEvent: `json.loads(message)` → parse into correct event type based on `body` field
  - Route to ApiRequestEvent / ToolDecisionEvent / etc.
  - Log and skip malformed records
  - Return categorized events
- [ ] `clean_event(event: dict) -> dict`
  - Cast string numerics to int/float
  - Parse timestamp string to datetime
  - Handle missing optional fields
- [ ] Main `run_pipeline(data_dir, db_path)` function
  - Parse both files
  - Print summary counts
  - Load into database

**Error handling to implement:**
- FileNotFoundError for missing data files
- json.JSONDecodeError for malformed JSON
- pydantic.ValidationError for schema mismatches
- Logging with counts: "Parsed X events, Y errors"

**Commit:** `feat: implement ETL pipeline with validation and error handling`

---

### Phase 3: Database (`src/database.py`)
- [ ] `create_tables(conn)` — DDL for all 6 tables + indexes
- [ ] `insert_employees(conn, employees)`
- [ ] `insert_api_requests(conn, events)`
- [ ] `insert_tool_decisions(conn, events)`
- [ ] `insert_tool_results(conn, events)`
- [ ] `insert_user_prompts(conn, events)`
- [ ] `insert_api_errors(conn, events)`
- [ ] `DatabaseManager` class with query methods:
  - `get_overview_stats() -> dict` — total cost, sessions, users, events, error rate
  - `get_daily_trends(filters) -> DataFrame`
  - `get_cost_by_model(filters) -> DataFrame`
  - `get_cost_by_practice(filters) -> DataFrame`
  - `get_tool_usage(filters) -> DataFrame`
  - `get_tool_success_rates(filters) -> DataFrame`
  - `get_hourly_heatmap(filters) -> DataFrame`
  - `get_error_breakdown(filters) -> DataFrame`
  - `get_user_rankings(filters) -> DataFrame`
  - `get_session_stats(filters) -> DataFrame`
  - `execute_safe_query(sql) -> DataFrame` — for AI queries (read-only)

**Commit:** `feat: add SQLite database layer with schema and query repository`

---

### Phase 4: Analytics (`src/analytics.py`)
- [ ] Wrapper functions that call DatabaseManager and return chart-ready DataFrames
- [ ] Computed metrics:
  - Cache hit ratio: `cache_read / (cache_read + input_tokens)`
  - Token efficiency: `output_tokens / input_tokens`
  - Cost per session: `SUM(cost) GROUP BY session_id`
  - Error rate: `COUNT(errors) / COUNT(api_requests)`
  - Session duration: `MAX(timestamp) - MIN(timestamp) per session`
  - Turns per session: `COUNT(user_prompts) per session`

**Key Analyses:**

| Analysis | SQL Logic | Chart Type |
|----------|-----------|-----------|
| Daily cost trend | SUM(cost_usd) GROUP BY date | Line chart |
| Cost by model | SUM(cost_usd) GROUP BY model | Pie chart |
| Cost by practice | JOIN employees, SUM(cost_usd) GROUP BY practice | Bar chart |
| Cost by level | JOIN employees, SUM(cost_usd) GROUP BY level | Bar chart |
| Tool frequency | COUNT(*) GROUP BY tool_name ORDER BY count DESC | Horizontal bar |
| Tool success rate | AVG(success) GROUP BY tool_name | Heatmap/bar |
| Tool duration | AVG(duration_ms) GROUP BY tool_name | Box plot |
| Peak hours | COUNT(*) GROUP BY HOUR(timestamp) | Heatmap |
| Error types | COUNT(*) GROUP BY error | Donut chart |
| Errors by model | COUNT(*) GROUP BY model, status_code | Stacked bar |
| Sessions per user | COUNT(DISTINCT session_id) GROUP BY user_email | Histogram |
| Model preference by practice | COUNT(*) GROUP BY model, practice | Grouped bar |
| Cost by seniority | AVG(cost) per level | Line chart |

**Commit:** `feat: add analytics computation layer`

---

### Phase 5: Streamlit Dashboard (`src/dashboard.py`)
- [ ] **Sidebar**: Date range picker, practice multi-select, level range slider, location checkboxes, model selector
- [ ] **Page 1 — Overview**:
  - KPI cards: Total Cost ($), Total Sessions, Active Users, Avg Cost/Session, Error Rate (%)
  - Daily activity timeline (events + cost dual-axis)
  - Quick stats table
- [ ] **Page 2 — Cost & Tokens**:
  - Cost trend line (daily, with optional model breakdown)
  - Cost by model pie chart
  - Cost by practice bar chart
  - Token breakdown (input vs output vs cache) stacked bar
  - Cache hit ratio by model gauge/bar
- [ ] **Page 3 — Tool Usage**:
  - Tool frequency horizontal bar (top 17 tools)
  - Tool success rate bar (highlight Bash at 93.3%)
  - Tool avg duration bar (log scale — Task=476s outlier)
  - Decision source breakdown (config 80%, user_temp 15%, etc.)
- [ ] **Page 4 — User Behavior**:
  - Peak usage heatmap (hour × day_of_week)
  - Sessions per user histogram
  - Turns per session distribution
  - Top 10 users by cost/sessions table
  - Practice comparison radar or grouped bar
- [ ] **Page 5 — Errors**:
  - Error rate trend line
  - Error type donut chart
  - Status code distribution bar
  - Errors by model grouped bar
  - Retry attempt distribution
- [ ] **Page 6 — AI Insights** (Gen AI feature):
  - Text input: "Ask a question about the data..."
  - LLM processes → generates SQL → runs → returns answer
  - Display: answer text + optional chart + the SQL used
- [ ] **Page 7 — ML & Anomalies** (scikit-learn feature):
  - Anomaly table: flagged sessions with anomaly scores, sortable
  - Anomaly scatter: PCA 2D plot with anomalies highlighted in red
  - Cluster visualization: PCA/t-SNE scatter colored by cluster
  - Cluster profiles: bar chart showing avg feature values per cluster
  - Decision Tree rules: collapsible text showing how practice is predicted
  - Feature importance: horizontal bar chart (which features matter most)
  - Cost forecast: line chart with historical + forecast trend line
  - Model metrics: accuracy, precision, recall for the classifier

**Commit:** `feat: build interactive Streamlit dashboard with 7 pages`

---

### Phase 6: Gen AI Feature (`src/ai_insights.py`)
- [ ] **Natural Language → SQL → Answer pipeline:**
  1. User types question in English
  2. System prompt provides: full SQLite schema, sample rows, available columns
  3. LLM generates a SQL SELECT query
  4. Execute query against SQLite (read-only)
  5. LLM interprets results and generates human-readable answer
  6. Display answer + chart (if applicable) + SQL used

- [ ] **System prompt template:**
  ```
  You are a data analyst assistant. You have access to a SQLite database
  with the following schema:
  {schema}
  
  The data contains Claude Code telemetry from engineering teams.
  
  Given a user question, generate a SQL SELECT query to answer it.
  Return ONLY the SQL query, no explanation.
  Only use SELECT statements. Never INSERT, UPDATE, DELETE, or DROP.
  ```

- [ ] **Auto-generated insights (bonus):**
  - Feed top-level stats to LLM
  - Generate a "Key Findings" narrative paragraph
  - Display on overview page

- [ ] **Safety:**
  - Validate SQL starts with SELECT
  - Execute with read-only connection
  - Timeout after 5 seconds
  - Catch and display errors gracefully

**Commit:** `feat: add LLM-powered natural language query interface`

---

### Phase 7: ML Anomaly Detection & Classification (`src/ml_anomaly.py`)

This phase adds predictive analytics using **scikit-learn** — directly called out as a bonus in the assessment and highly relevant for a Gen AI internship.

- [ ] **Feature engineering** — build a per-session feature matrix:
  | Feature | Source | Description |
  |---------|--------|-------------|
  | `total_cost` | `SUM(cost_usd)` per session | Total API spend |
  | `total_api_calls` | `COUNT(*)` from api_requests | Number of LLM calls |
  | `total_tokens` | `SUM(input + output + cache)` | Token consumption |
  | `avg_request_duration_ms` | `AVG(duration_ms)` | Avg API latency |
  | `num_tools_used` | `COUNT(*)` from tool_results | Tool call count |
  | `tool_failure_rate` | `1 - AVG(success)` | % failed tool calls |
  | `num_turns` | `COUNT(*)` from user_prompts | User prompt count |
  | `num_errors` | `COUNT(*)` from api_errors | API error count |
  | `session_duration_sec` | `MAX(ts) - MIN(ts)` | Wall-clock duration |
  | `unique_models` | `COUNT(DISTINCT model)` | Model diversity |
  | `avg_prompt_length` | `AVG(prompt_length)` | Avg prompt size |
  | `cache_hit_ratio` | `cache_read / (cache_read + input)` | Cache efficiency |

- [ ] **Anomaly Detection — IsolationForest:**
  ```python
  from sklearn.ensemble import IsolationForest
  
  # Fit on session feature matrix
  model = IsolationForest(contamination=0.05, random_state=42)
  labels = model.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
  scores = model.decision_function(X_scaled)  # anomaly scores
  ```
  - Flag top 5% sessions as anomalous (unusually expensive, long, error-prone)
  - Return anomaly labels + scores per session
  - Display flagged sessions on dashboard with reason

- [ ] **Classification — Decision Tree (User Behavior Profiling):**
  ```python
  from sklearn.tree import DecisionTreeClassifier, export_text
  
  # Predict practice/level from usage patterns
  clf = DecisionTreeClassifier(max_depth=5, random_state=42)
  clf.fit(X_train, y_train)  # y = practice or level
  ```
  - Train a Decision Tree to predict `practice` from session features
  - Show feature importances (which behavior patterns distinguish teams?)
  - Export the tree rules as readable text (`export_text`)
  - Display on dashboard: "What makes ML Engineers different from Backend Engineers?"

- [ ] **Clustering — KMeans (Usage Pattern Groups):**
  ```python
  from sklearn.cluster import KMeans
  from sklearn.preprocessing import StandardScaler
  
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  kmeans = KMeans(n_clusters=4, random_state=42)
  clusters = kmeans.fit_predict(X_scaled)
  ```
  - Group sessions into 3-5 behavioral clusters
  - Label clusters: "Quick lookup", "Deep coding session", "Heavy cost session", "Error-prone session"
  - Visualize with PCA/t-SNE 2D scatter plot

- [ ] **Trend Forecasting — Linear Regression:**
  ```python
  from sklearn.linear_model import LinearRegression
  
  # Predict next-week daily cost from historical trend
  model = LinearRegression()
  model.fit(X_days, y_cost)
  forecast = model.predict(X_future)
  ```
  - Simple daily cost trend extrapolation
  - Show forecast line on the cost trend chart
  - R² score to assess trend reliability

- [ ] **`AnomalyDetector` class with methods:**
  - `build_session_features(db: DatabaseManager) -> pd.DataFrame`
  - `detect_anomalies(features: pd.DataFrame) -> pd.DataFrame` (adds anomaly_label, anomaly_score)
  - `classify_practice(features: pd.DataFrame) -> dict` (accuracy, feature importances, tree rules)
  - `cluster_sessions(features: pd.DataFrame) -> pd.DataFrame` (adds cluster_label)
  - `forecast_cost(db: DatabaseManager, days_ahead: int) -> pd.DataFrame`

**Key Python/ML features to demonstrate:** Pipeline pattern, StandardScaler, train/test split, cross-validation, feature importance visualization, model evaluation metrics.

**Commit:** `feat: add ML anomaly detection, classification, and clustering`

---

### Phase 8: Tests (`tests/`)
- [ ] `test_models.py`:
  - Valid Employee creation
  - Invalid email rejection
  - ApiRequestEvent with all fields
  - Optional field handling (tool_result_size_bytes)
  - Numeric string to float/int casting
- [ ] `test_ingest.py`:
  - Parse a mock JSONL line with 2 logEvents
  - Handle malformed JSON gracefully
  - Parse employees CSV
  - Event routing by body type
- [ ] `test_analytics.py`:
  - Known input → expected aggregation output
  - Empty data handling
  - Filter application
- [ ] `test_ml_anomaly.py`:
  - Feature matrix shape matches expected columns
  - IsolationForest returns valid labels (-1/1)
  - DecisionTree accuracy > baseline (random guessing)
  - Clustering returns expected number of groups

**Commit:** `feat: add pytest unit tests`

---

### Phase 9: Documentation
- [ ] **README.md** — full setup guide, architecture, screenshots
- [ ] **LLM_USAGE_LOG.md** — AI tools used, example prompts, validation methods
- [ ] **presentation/insights.pdf** — 3-5 slides with key findings

**Commit:** `docs: add README, LLM usage log, and insights presentation`

---

### Phase 10: Polish & Final Commit
- [ ] End-to-end run: generate → ingest → dashboard → all pages work
- [ ] ML models run without errors on generated data
- [ ] Squash/rebase commits if needed for clean history
- [ ] Final `git tag v1.0`

---

## 5. Database Schema

```sql
-- Employees table (from CSV)
CREATE TABLE employees (
    email TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    practice TEXT NOT NULL,
    level TEXT NOT NULL,
    location TEXT NOT NULL
);

-- API request events
CREATE TABLE api_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_email TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    cost_usd REAL NOT NULL,
    duration_ms INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cache_creation_tokens INTEGER NOT NULL,
    terminal_type TEXT,
    org_id TEXT,
    scope_version TEXT,
    FOREIGN KEY (user_email) REFERENCES employees(email)
);

-- Tool decision events
CREATE TABLE tool_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_email TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    decision TEXT NOT NULL,
    source TEXT NOT NULL,
    terminal_type TEXT,
    org_id TEXT,
    scope_version TEXT,
    FOREIGN KEY (user_email) REFERENCES employees(email)
);

-- Tool result events
CREATE TABLE tool_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_email TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    success TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    decision_source TEXT,
    decision_type TEXT,
    tool_result_size_bytes INTEGER,
    terminal_type TEXT,
    org_id TEXT,
    scope_version TEXT,
    FOREIGN KEY (user_email) REFERENCES employees(email)
);

-- User prompt events
CREATE TABLE user_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_email TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    prompt_length INTEGER NOT NULL,
    terminal_type TEXT,
    org_id TEXT,
    scope_version TEXT,
    FOREIGN KEY (user_email) REFERENCES employees(email)
);

-- API error events
CREATE TABLE api_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_email TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    error TEXT NOT NULL,
    status_code TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    terminal_type TEXT,
    org_id TEXT,
    scope_version TEXT,
    FOREIGN KEY (user_email) REFERENCES employees(email)
);

-- Indexes for query performance
CREATE INDEX idx_api_requests_session ON api_requests(session_id);
CREATE INDEX idx_api_requests_user ON api_requests(user_email);
CREATE INDEX idx_api_requests_timestamp ON api_requests(timestamp);
CREATE INDEX idx_api_requests_model ON api_requests(model);
CREATE INDEX idx_tool_results_tool ON tool_results(tool_name);
CREATE INDEX idx_tool_results_session ON tool_results(session_id);
CREATE INDEX idx_user_prompts_session ON user_prompts(session_id);
CREATE INDEX idx_api_errors_session ON api_errors(session_id);

-- Useful view: session summary
CREATE VIEW session_summary AS
SELECT
    ar.session_id,
    ar.user_email,
    e.practice,
    e.level,
    e.location,
    MIN(ar.timestamp) as start_time,
    MAX(ar.timestamp) as end_time,
    COUNT(*) as api_call_count,
    SUM(ar.cost_usd) as total_cost,
    SUM(ar.input_tokens) as total_input_tokens,
    SUM(ar.output_tokens) as total_output_tokens,
    SUM(ar.cache_read_tokens) as total_cache_read_tokens
FROM api_requests ar
JOIN employees e ON ar.user_email = e.email
GROUP BY ar.session_id;
```

---

## 6. Dashboard Pages & Charts

### Page 1: Overview
| Component | Chart Type | Data Source |
|-----------|-----------|-------------|
| Total Cost | KPI metric card | `SUM(cost_usd)` |
| Total Sessions | KPI metric card | `COUNT(DISTINCT session_id)` |
| Active Users | KPI metric card | `COUNT(DISTINCT user_email)` |
| Avg Cost/Session | KPI metric card | `AVG(session_cost)` |
| Error Rate | KPI metric card | `errors / api_requests * 100` |
| Daily Activity | Plotly dual-axis line | events/day + cost/day |

### Page 2: Cost & Tokens
| Component | Chart Type |
|-----------|-----------|
| Daily cost trend | `plotly.line` (color by model optional) |
| Cost by model | `plotly.pie` |
| Cost by practice | `plotly.bar` (horizontal) |
| Cost by level | `plotly.bar` |
| Token breakdown by model | `plotly.bar` (stacked: input/output/cache) |
| Cache hit ratio | `plotly.bar` per model |

### Page 3: Tool Usage
| Component | Chart Type |
|-----------|-----------|
| Tool frequency | `plotly.bar` (horizontal, sorted) |
| Success rate by tool | `plotly.bar` (highlight low rates) |
| Avg duration by tool | `plotly.bar` (log scale) |
| Decision source breakdown | `plotly.pie` or `plotly.sunburst` |

### Page 4: User Behavior
| Component | Chart Type |
|-----------|-----------|
| Peak hours heatmap | `plotly.heatmap` (hour × day_of_week) |
| Sessions per user | `plotly.histogram` |
| Top users by cost | `plotly.table` or `st.dataframe` |
| Practice comparison | `plotly.bar` (grouped) |

### Page 5: Errors
| Component | Chart Type |
|-----------|-----------|
| Error rate trend | `plotly.line` |
| Error type breakdown | `plotly.pie` |
| Status code distribution | `plotly.bar` |
| Errors by model | `plotly.bar` (grouped) |

### Page 6: AI Insights
| Component | Type |
|-----------|------|
| Question input | `st.text_input` |
| Answer display | `st.markdown` |
| Generated SQL | `st.code` (collapsible) |
| Result table | `st.dataframe` (if tabular) |

### Page 7: ML & Anomalies
| Component | Chart Type |
|-----------|------------|
| Anomaly sessions table | `st.dataframe` (sortable by score) |
| Anomaly scatter (PCA 2D) | `plotly.scatter` (red = anomaly) |
| Session clusters (PCA 2D) | `plotly.scatter` (colored by cluster) |
| Cluster profiles | `plotly.bar` (grouped, avg features per cluster) |
| Decision Tree rules | `st.code` (collapsible text) |
| Feature importance | `plotly.bar` (horizontal, sorted) |
| Cost forecast | `plotly.line` (historical + dashed forecast) |
| Classifier metrics | `st.metric` cards (accuracy, precision, recall) |

---

## 7. Gen AI Feature Spec

### Natural Language Query Pipeline

```
User Question
     │
     ▼
┌─────────────────────┐
│  System Prompt       │
│  - SQLite schema     │
│  - Column descriptions│
│  - Example queries   │
│  - Rules (SELECT only)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  LLM (Claude API)   │
│  → Generates SQL     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Safety Validation   │
│  - Starts with SELECT│
│  - No DROP/DELETE    │
│  - Timeout 5s       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Execute on SQLite   │
│  → Returns DataFrame │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  LLM Interprets     │
│  → Natural language  │
│    answer + insights │
└──────────┬──────────┘
           │
           ▼
     Display in Dashboard
```

### Example Queries It Should Handle
- "Which engineering practice spends the most on API calls?"
- "What is the error rate for each model?"
- "Who are the top 5 users by total cost?"
- "Which tool fails the most often?"
- "Show me daily cost trends for the last 30 days"
- "Compare token usage between ML Engineering and Backend Engineering"

### Fallback (if no API key)
- Display a "Demo Mode" with pre-computed insights
- Auto-generate 5-6 canned insights from analytics functions
- Still shows the UI, just without live LLM queries

---

## 8. Key Insights to Extract

These are the "stories" to highlight in your presentation:

### Cost Insights
- **Opus models cost ~20x more than Haiku** ($0.07-0.08 vs $0.003 per request)
- **Total spend distribution** across practices and levels
- **Cost per session varies wildly** due to session length differences

### Tool Insights
- **Read and Bash dominate** tool usage (combined ~60% of all tool calls)
- **Bash has the lowest success rate** (93.3%) — implies command failures
- **Task tool takes ~8 minutes avg** — massive outlier vs others

### User Behavior
- **70% of sessions start during business hours** (9am-6pm)
- **Session length follows lognormal** — most short, some very long
- **Senior engineers (L7+) may show different patterns** than juniors

### Error Insights
- **"Request was aborted" is the #1 error** (52% of all errors)
- **Rate limiting (429) is the #2 error** (23%)
- **Error rate is low (~1.2%)** but clustered in bursts

### Practice Comparisons
- Different practices may prefer different models
- Tool usage patterns vary by team (Backend → more Bash, Frontend → more Read)

---

## 9. Deliverables Checklist

### Required
- [ ] **Git Repository** with clean commit history (1 commit per phase)
- [ ] **README.md** with:
  - [ ] Project overview
  - [ ] Architecture diagram
  - [ ] Setup instructions (venv, install, generate data, run dashboard)
  - [ ] Dependencies list
  - [ ] Design decisions explained
- [ ] **Insights Presentation** (3-5 slides PDF):
  - [ ] Slide 1: Executive summary — what the data shows
  - [ ] Slide 2: Cost & token analysis with charts
  - [ ] Slide 3: Developer behavior patterns
  - [ ] Slide 4: Tool usage & errors
  - [ ] Slide 5: Recommendations / Gen AI feature demo
- [ ] **LLM Usage Log** with:
  - [ ] Tools used (GitHub Copilot, Claude)
  - [ ] 5-8 example prompts
  - [ ] How you validated AI output
  - [ ] What you changed from AI suggestions and why

### ML Analytics (Integrated — Phase 7)
- [ ] IsolationForest anomaly detection on session features
- [ ] DecisionTree classifier for practice prediction + feature importances
- [ ] KMeans clustering for session behavior groups
- [ ] Linear regression cost forecasting
- [ ] Dashboard Page 7 with all ML visualizations

### Optional Bonus (extra credit)
- [ ] API endpoints (FastAPI wrapper around analytics)
- [ ] Real-time simulation demo
- [ ] More advanced ML (Random Forest, XGBoost, neural anomaly detection)

---

## 10. Commands Reference

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Generate data
python generate_fake_data.py --num-users 100 --num-sessions 5000 --days 60 --output-dir data

# Run ETL pipeline
python -m src.ingest --data-dir data --db-path data/analytics.db

# Run tests
python -m pytest tests/ -v

# Launch dashboard
streamlit run src/dashboard.py

# Git workflow
git add -A
git commit -m "feat: <description>"
```

---

## Implementation Priority (if time-limited)

| Priority | Phase | Est. Time | Importance |
|----------|-------|-----------|------------|
| 🔴 P0 | Phase 0: Setup | 30 min | Foundation |
| 🔴 P0 | Phase 1: Models | 1 hr | Shows Python skill |
| 🔴 P0 | Phase 2: ETL | 2 hr | Core data pipeline |
| 🔴 P0 | Phase 3: Database | 1.5 hr | Storage layer |
| 🔴 P0 | Phase 5: Dashboard (basic) | 3 hr | Main deliverable |
| 🟡 P1 | Phase 6: Gen AI feature | 2 hr | Role differentiator |
| 🟡 P1 | Phase 7: ML anomaly/classify | 2.5 hr | Shows ML + scikit-learn |
| 🟡 P1 | Phase 9: Docs | 2 hr | Required deliverables |
| 🟢 P2 | Phase 4: Deep analytics | 1.5 hr | Enriches dashboard |
| 🟢 P2 | Phase 8: Tests | 1.5 hr | Shows engineering rigor |
| 🟢 P2 | Phase 10: Polish | 30 min | Final QA |
| ⚪ P3 | Bonus features | 2+ hr | Extra credit |

**Total estimate: ~18-20 hours of focused work**

---

*This plan was created with AI assistance (GitHub Copilot / Claude) as part of the AI-First development approach encouraged by Provectus.*
