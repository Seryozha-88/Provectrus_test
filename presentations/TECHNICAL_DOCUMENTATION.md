# Technical Documentation

> **Claude Code Analytics Platform**
> Provectus Python & Gen AI Internship — Technical Deep Dive

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Flow & ETL Pipeline](#2-data-flow--etl-pipeline)
3. [Data Models (Pydantic)](#3-data-models-pydantic)
4. [Database Layer](#4-database-layer)
5. [Analytics Engine](#5-analytics-engine)
6. [Dashboard (Streamlit)](#6-dashboard-streamlit)
7. [AI Insights (LLM Integration)](#7-ai-insights-llm-integration)
8. [ML & Anomaly Detection](#8-ml--anomaly-detection)
9. [Design Decisions & Trade-offs](#9-design-decisions--trade-offs)
10. [Performance Metrics](#10-performance-metrics)

---

## 1. System Architecture

### Overview

The platform follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  dashboard.py (Streamlit) — 7 pages, Plotly charts, sidebar │
└────────────┬────────────────────────────┬───────────────────┘
             │                            │
┌────────────▼──────────┐  ┌──────────────▼───────────────────┐
│   ANALYTICS LAYER     │  │        AI / ML LAYER              │
│  analytics.py         │  │  ai_insights.py (LLM pipeline)   │
│  - computed metrics   │  │  ml_anomaly.py  (scikit-learn)   │
│  - KPIs, ratios       │  │  - 4 ML models                   │
│  - WoW comparison     │  │  - NL → SQL → answer             │
└────────────┬──────────┘  └──────────────┬───────────────────┘
             │                            │
┌────────────▼────────────────────────────▼───────────────────┐
│                     DATA ACCESS LAYER                        │
│  database.py — DatabaseManager (13 SQL methods)              │
│  - Thread-safe connections (fresh per query)                 │
│  - QueryFilters dataclass for dynamic WHERE clauses          │
│  - Parameterized queries (SQL injection safe)                │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                     STORAGE LAYER                            │
│  SQLite (analytics.db) — 6 tables + 1 view + indexes        │
└─────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

- **Single Responsibility**: Each module has one job
- **Testability**: Database queries can be tested independently from analytics
- **Reusability**: Same data layer could back a REST API or CLI
- **Thread Safety**: Streamlit runs in multi-threaded mode; fresh connections per query

---

## 2. Data Flow & ETL Pipeline

### Input Structure

The JSONL file has a deeply nested structure:

```
telemetry_logs.jsonl
├── Line 1 (batch)
│   ├── messageType: "DATA_MESSAGE"
│   ├── logGroup, logStream, year, month, day
│   └── logEvents: [
│       ├── { id, timestamp, message: "{...JSON string...}" }
│       └── { id, timestamp, message: "{...JSON string...}" }
│   ]
├── Line 2 (batch)
│   └── logEvents: [...]
└── ... (82,661 batches total)
```

Each `message` field is a **JSON string within a JSON string**, requiring double parsing:

```python
# Step 1: Parse the batch line
batch = json.loads(line)

# Step 2: Parse each event message (string → dict)
for log_event in batch["logEvents"]:
    event = json.loads(log_event["message"])  # Second json.loads()
```

### ETL Steps

```
Step 1: EXTRACT
├── Read JSONL line by line (streaming, memory-efficient)
├── Parse batch JSON
└── Extract logEvents array

Step 2: TRANSFORM
├── Parse message JSON string
├── Identify event type by "body" field
├── Validate with Pydantic model (type casting, field aliases)
└── Handle errors gracefully (log & skip)

Step 3: LOAD
├── Create SQLite tables (IF NOT EXISTS)
├── Bulk INSERT with parameterized queries (batch size: 5000)
├── Create indexes on session_id, user_email, timestamp, model
└── Create session_summary materialized view
```

### Performance

| Metric | Value |
|--------|-------|
| Input file size | 521 MB |
| Total batches | 82,661 |
| Events parsed | 454,428 |
| Parse errors | 0 |
| Processing time | ~24 seconds |
| Output DB size | 115.8 MB |
| Memory usage | ~50 MB (streaming) |

---

## 3. Data Models (Pydantic)

### Why Pydantic?

- **Type safety**: Automatic string → int/float casting (`cost_usd: "0.071"` → `0.071`)
- **Validation**: Catches malformed events at parse time
- **Field aliases**: Maps dotted attribute keys to Python names (`"event.timestamp"` → `timestamp`)
- **Documentation**: Models serve as schema documentation

### Model Hierarchy

```python
class Employee(BaseModel):         # From CSV
class EventScope(BaseModel):       # Nested: scope.name, scope.version
class EventResource(BaseModel):    # Nested: host.arch, os.type, etc.

class ApiRequestEvent(BaseModel):  # body = "claude_code.api_request"
class ToolDecisionEvent(BaseModel):# body = "claude_code.tool_use_decision"
class ToolResultEvent(BaseModel):  # body = "claude_code.tool_use_result"
class UserPromptEvent(BaseModel):  # body = "claude_code.user_prompt"
class ApiErrorEvent(BaseModel):    # body = "claude_code.api_error"
```

### Factory Pattern

```python
EVENT_TYPE_MAP = {
    "claude_code.api_request": ApiRequestEvent,
    "claude_code.tool_use_decision": ToolDecisionEvent,
    "claude_code.tool_use_result": ToolResultEvent,
    "claude_code.user_prompt": UserPromptEvent,
    "claude_code.api_error": ApiErrorEvent,
}

def parse_event(raw: dict) -> BaseModel:
    """Route event to correct model by 'body' field."""
    body = raw.get("body", "")
    model_cls = EVENT_TYPE_MAP.get(body)
    if model_cls is None:
        raise ValueError(f"Unknown event type: {body}")
    return model_cls(**raw["attributes"])
```

---

## 4. Database Layer

### Connection Management

```python
class DatabaseManager:
    def _get_conn(self) -> sqlite3.Connection:
        """Create fresh connection per query (thread-safe)."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _query_df(self, sql, params=()):
        """Execute SQL → pandas DataFrame."""
        conn = self._get_conn()
        try:
            return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()
```

**Why fresh connections?** Streamlit runs page rerenders in separate threads. SQLite connections are not thread-safe by default. Creating a fresh connection per query avoids `ProgrammingError: SQLite objects created in a thread can only be used in that same thread`.

### Dynamic Filtering

```python
@dataclass
class QueryFilters:
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    user_email: Optional[str] = None
    practice: Optional[str] = None
    level: Optional[str] = None
    model: Optional[str] = None
    tool_name: Optional[str] = None
```

Every query method accepts `QueryFilters` and dynamically builds WHERE clauses. This avoids duplicating filter logic across 13 methods.

### Query Methods (13 total)

| # | Method | Returns |
|---|--------|---------|
| 1 | `get_overview_stats()` | KPI dict (cost, sessions, users, events) |
| 2 | `get_daily_trends()` | Daily time series (cost, requests, tokens) |
| 3 | `get_cost_by_model()` | Cost breakdown by Claude model |
| 4 | `get_cost_by_practice()` | Cost by team/department |
| 5 | `get_cost_by_level()` | Cost by seniority level |
| 6 | `get_tool_usage()` | Tool frequency, success rate |
| 7 | `get_tool_decisions_summary()` | Allow/deny counts by source |
| 8 | `get_hourly_heatmap()` | Activity by day-of-week × hour |
| 9 | `get_error_breakdown()` | Errors by type, status, model |
| 10 | `get_user_rankings()` | User leaderboard by cost/sessions |
| 11 | `get_session_stats()` | Per-session metrics (5,000 rows) |
| 12 | `get_model_usage_over_time()` | Model trends over time |
| 13 | `execute_safe_query()` | AI-generated SQL (read-only) |

### Session Summary View

```sql
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

## 5. Analytics Engine

### Computed Metrics

The analytics layer adds business logic on top of raw SQL data:

| Metric | Formula | Example Value |
|--------|---------|--------------|
| Cache Hit Ratio | `cache_read_tokens / (input + cache_read)` | 98.95% |
| Token Efficiency | `output_tokens / input_tokens` | 0.66 |
| Tool Reliability | `0.7 × success_rate + 0.3 × speed_score` | 0-100 |
| Cost 7-Day Avg | `rolling mean of daily cost (window=7)` | $100.02 |
| WoW Comparison | `(this_week - last_week) / last_week × 100` | ±X% |
| Error Severity | Category mapping: overloaded→HIGH, rate_limit→MEDIUM | - |

### Method Count: 12 analytics methods

Each method calls one or more `DatabaseManager` queries and applies transformations:

```python
class AnalyticsEngine:
    def get_kpi_cards(filters) → dict       # Overview KPIs + computed ratios
    def get_cost_trend(filters) → DataFrame  # 7-day moving avg + cumulative
    def get_token_analysis(filters) → dict   # Cache ratio, efficiency
    def get_cost_by_model/practice/level()   # Breakdowns with percentages
    def get_tool_performance(filters)        # Reliability score
    def get_session_analysis(filters)        # Duration stats, quartiles
    def get_user_performance(filters)        # Rankings + metrics
    def get_activity_heatmap(filters)        # Pivot to day×hour matrix
    def get_error_analysis(filters)          # Severity categorization
    def get_model_comparison(filters)        # Cross-model analysis
    def get_wow_comparison(filters)          # Week-over-week changes
    def get_ai_context_summary(filters)      # Text summary for LLM
```

---

## 6. Dashboard (Streamlit)

### Page Structure

| Page | Charts | Interactive Elements |
|------|--------|---------------------|
| 1. Overview | 2 KPI rows, 1 timeline | Date filter |
| 2. Cost & Tokens | 3 breakdowns, 1 stacked | Model, practice filters |
| 3. Tool Usage | 3 charts + table | Tool filter |
| 4. User Behavior | Heatmap + rankings | User, practice filters |
| 5. Errors | Donut + bars | Model filter |
| 6. AI Insights | Chat interface | API key, model selector |
| 7. ML Anomalies | 4 tabs, 10+ charts | ML parameter sliders |

### Caching Strategy

```python
@st.cache_resource
def get_engine() -> AnalyticsEngine:
    """Singleton — one instance across all reruns."""

@st.cache_data
def get_filter_options(engine):
    """Cached filter dropdowns (practices, levels, models)."""

@st.cache_resource
def get_ml_engine() -> MLEngine:
    """Singleton ML engine (features loaded once)."""
```

### Threading Fix

Streamlit runs in multi-threaded mode. SQLite connections are thread-bound. Solution: create a **fresh connection per query** via `_get_conn()` instead of a shared connection.

---

## 7. AI Insights (LLM Integration)

### Pipeline Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ User Question│────▶│ generate_sql │────▶│ execute_sql  │
│  (natural    │     │ (LLM call 1) │     │ (safety      │
│   language)  │     │              │     │  checked)    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌──────────────┐     ┌───────▼──────┐
                     │    Answer    │◀────│interpret_sql │
                     │  (markdown) │     │ (LLM call 2) │
                     └──────────────┘     └──────────────┘
```

### System Prompt Design

The system prompt injects the **full database schema** so the LLM knows all table structures:

```python
DB_SCHEMA = """
Tables:
  api_requests: session_id, user_email, timestamp, model, cost_usd,
                duration_ms, input_tokens, output_tokens, ...
  tool_decisions: session_id, user_email, timestamp, tool_name, decision, ...
  ...
  employees: email, full_name, practice, level, location

Views:
  session_summary: (pre-aggregated session metrics)
"""
```

### Safety Measures

1. **SQL validation**: Only `SELECT` statements allowed
2. **Keyword blocking**: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE blocked
3. **Error handling**: Malformed SQL caught gracefully, user sees friendly error
4. **Read-only execution**: `execute_safe_query()` in database.py

### API Configuration

- **Provider**: OpenRouter (OpenAI-compatible API)
- **SDK**: `openai` Python SDK with custom `base_url`
- **Default model**: `google/gemini-2.0-flash-001` (fast, cheap, good SQL generation)
- **Available models**: 7 options including Gemini, Claude, GPT-4o, Llama

---

## 8. ML & Anomaly Detection

### Feature Engineering Pipeline

```
session_summary (SQL view)
        │
        ▼
get_session_stats() → 5,000 rows × 16 columns
        │
        ▼
build_session_features() → Add 3 engineered features
        │
        ▼
Feature Matrix: 5,000 rows × 12 numeric features
```

### Model 1: IsolationForest (Anomaly Detection)

**How it works:**
1. Build 100 random binary trees
2. Each tree: randomly pick a feature, randomly pick a split value
3. Normal points need many splits to isolate (deep paths)
4. Anomalous points need few splits (short paths)
5. Anomaly score = average path length (lower = more anomalous)

```python
IsolationForest(
    n_estimators=100,        # 100 random trees
    contamination=0.05,      # expect 5% anomalies
    random_state=42,         # reproducible
    n_jobs=-1,               # parallel
)
```

**Results**: 250 / 5,000 sessions flagged as anomalous.

### Model 2: DecisionTreeClassifier (Practice Classification)

**Goal**: Can we predict which team (practice) a session belongs to from usage patterns?

```python
DecisionTreeClassifier(
    max_depth=5,             # readable tree
    class_weight="balanced", # handle class imbalance
    random_state=42,
)
```

**Evaluation**: 5-fold cross-validation → ~17% accuracy (with synthetic data, teams have similar distributions; this is expected — real data would show more differentiation).

**Output**: Feature importance ranking + exportable tree rules in plain text.

### Model 3: KMeans (Session Clustering)

**Goal**: Group sessions into behavioral clusters.

```python
KMeans(n_clusters=4, random_state=42, n_init=10)
```

**Pipeline**: StandardScaler → KMeans → PCA (2D projection for visualization)

**Results**: 4 clusters of varying sizes, visualized via PCA scatter plot colored by cluster.

### Model 4: LinearRegression (Cost Forecasting)

**Goal**: Predict future daily cost trend.

```python
X = day_number (0, 1, 2, ..., 59)
y = daily_cost
LinearRegression().fit(X, y) → slope, R²
→ Extrapolate 30 days into future
```

**Results**: R² = 0.003 (flat — synthetic data is uniformly random), slope = -$0.057/day.

### Training Performance

| Model | Training Time | Memory |
|-------|:---:|:---:|
| IsolationForest | 0.20s | ~10 MB |
| DecisionTree | 0.12s | ~5 MB |
| KMeans | 1.81s | ~15 MB |
| LinearRegression | 0.13s | ~2 MB |
| **Total** | **~2.7s** | **~30 MB** |

All CPU-only. No GPU required.

---

## 9. Design Decisions & Trade-offs

### SQLite vs PostgreSQL
- **Chose SQLite**: Zero setup, single file, perfect for assessment scope
- **Trade-off**: No concurrent writes, limited to ~1 GB comfortably
- **Mitigation**: Read-only workload after ETL, 115 MB DB well within limits

### Streaming vs In-Memory Parsing
- **Chose streaming**: Line-by-line JSONL processing
- **Why**: 521 MB file would consume ~2 GB RAM if loaded entirely
- **Trade-off**: Slightly slower than bulk numpy/arrow parsing
- **Result**: ~50 MB memory usage, 24s parse time

### Fresh Connection per Query vs Connection Pool
- **Chose fresh connections**: `_get_conn()` creates new connection each time
- **Why**: SQLite connections are cheap (<1ms), Streamlit multi-threading creates issues
- **Trade-off**: Tiny overhead per query (~0.5ms)
- **Alternative**: `check_same_thread=False` alone is insufficient for write safety

### OpenRouter vs Direct Anthropic API
- **Chose OpenRouter**: Access to multiple models (Gemini, Claude, GPT, Llama)
- **Why**: User can choose the best model for their budget/needs
- **Trade-off**: Extra hop → slight latency increase
- **Benefit**: Model flexibility, single API key for all providers

### IsolationForest vs Other Anomaly Methods
- **Chose IsolationForest**: Unsupervised, fast, handles mixed features
- **Why**: We have no labeled anomalies (no ground truth)
- **Alternative**: DBSCAN (needs epsilon tuning), Z-score (univariate only)
- **Trade-off**: `contamination` parameter must be set manually

---

## 10. Performance Metrics

### ETL Pipeline
- **Throughput**: ~19,000 events/second
- **Memory**: ~50 MB peak (streaming)
- **Error rate**: 0 / 454,428 (0.000%)

### Database Queries
- **Average query time**: 5-20ms (with indexes)
- **Session stats (5,000 rows)**: ~120ms
- **Overview stats (aggregation)**: ~80ms

### Dashboard
- **Initial page load**: ~2s (engine + filter options)
- **Page switch**: <500ms
- **ML page (first load)**: ~3s (trains models)
- **ML page (cached)**: <200ms

### ML Models
- **Total training**: 2.7s on CPU
- **Feature building**: 0.42s (SQL + engineering)
- **Prediction (single session)**: <1ms

---

*Last updated: March 5, 2026*
