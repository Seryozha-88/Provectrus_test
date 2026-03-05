# LLM Usage Log

> **Claude Code Analytics Platform — AI Tool Usage Documentation**
> Provectus Python & Gen AI Internship, 2026

This document logs how LLM (Large Language Model) tools were used throughout the development of this project, as required by the assessment.

---

## Table of Contents

1. [Development Assistant Usage](#1-development-assistant-usage)
2. [AI Insights Feature (In-App LLM)](#2-ai-insights-feature-in-app-llm)
3. [Prompt Engineering Decisions](#3-prompt-engineering-decisions)
4. [Model Selection Rationale](#4-model-selection-rationale)
5. [Ethical Considerations](#5-ethical-considerations)

---

## 1. Development Assistant Usage

### Tool: GitHub Copilot (Claude Opus 4.6)

**Used for**: Pair programming throughout all 7 development phases.

| Phase | What the LLM Helped With |
|-------|--------------------------|
| Phase 0 | Project structure, config files, dependency list |
| Phase 1 | Pydantic model design with field aliases and type casting |
| Phase 2 | Streaming JSONL parser with double json.loads() logic |
| Phase 3 | SQL query design, dynamic WHERE clause builder |
| Phase 4 | Analytics computation (cache ratio, token efficiency, WoW) |
| Phase 5 | Streamlit dashboard layout, Plotly chart configuration |
| Phase 6 | OpenRouter integration, system prompt with DB schema injection |
| Phase 7 | scikit-learn model selection, feature engineering, PCA visualization |
| Phase 9 | README, technical documentation, this usage log |

### How It Was Used

1. **Architecture discussion** — Explained requirements, got implementation plan
2. **Code generation** — Generated modules with detailed docstrings and comments
3. **Debugging** — Fixed SQLite threading issues, PowerShell escaping problems
4. **Concept explanation** — Learned Pydantic, IsolationForest, KMeans, PCA concepts
5. **Code review** — Validated approach before implementation

### What Was NOT Delegated to LLM

- **Requirements analysis** — Manually read and understood the assessment PDF
- **Data generation** — Used the provided `generate_fake_data.py` script as-is
- **Testing** — Manually ran and verified each pipeline step
- **Git management** — Manual commits with descriptive messages

---

## 2. AI Insights Feature (In-App LLM)

### Architecture

The dashboard's Page 6 uses an LLM pipeline for natural language database queries:

```
User Question (natural language)
        │
        ▼
┌─────────────────────┐
│  LLM Call #1        │  System prompt: full DB schema + SQL rules
│  "Generate SQL"     │  User prompt: the question
│  Model: gemini-2.0  │  Output: pure SQL SELECT statement
│  via OpenRouter     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Execute SQL        │  Safety: only SELECT, blocked write keywords
│  (SQLite)           │  Result: pandas DataFrame (max 100 rows)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  LLM Call #2        │  System prompt: "interpret this data"
│  "Interpret Results"│  User prompt: question + SQL + results table
│  Model: gemini-2.0  │  Output: markdown answer with insights
│  via OpenRouter     │
└─────────┬───────────┘
          │
          ▼
    Final Answer (displayed in chat UI)
```

### API Details

| Parameter | Value |
|-----------|-------|
| Provider | OpenRouter |
| SDK | `openai` Python package (OpenAI-compatible) |
| Base URL | `https://openrouter.ai/api/v1` |
| Default Model | `google/gemini-2.0-flash-001` |
| Temperature (SQL) | 0.0 (deterministic) |
| Temperature (Interpretation) | 0.3 (slight creativity) |
| Max Tokens (SQL) | 500 |
| Max Tokens (Interpretation) | 1000 |

### Available Models

| Model | Provider | Best For |
|-------|----------|----------|
| google/gemini-2.0-flash-001 | Google | Fast SQL generation (default) |
| google/gemini-2.0-pro-exp-02-05 | Google | Complex queries |
| anthropic/claude-3.5-sonnet | Anthropic | Nuanced interpretation |
| anthropic/claude-3.5-haiku | Anthropic | Budget queries |
| openai/gpt-4o-mini | OpenAI | Balanced cost/quality |
| meta-llama/llama-3.1-70b-instruct | Meta | Open-source option |
| meta-llama/llama-3.1-8b-instruct | Meta | Cheapest option |

### Safety Measures

```python
# Blocked SQL keywords
BLOCKED = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", 
           "ALTER", "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA"]

# Only SELECT allowed
if not sql.strip().upper().startswith("SELECT"):
    raise ValueError("Only SELECT queries are allowed.")
```

---

## 3. Prompt Engineering Decisions

### SQL Generation Prompt

**Strategy**: Schema-first approach — inject the complete database schema into the system message so the LLM knows exact table/column names.

```
System: You are a SQL expert. Given this database schema:
[full schema with tables, columns, types, views]
Generate a SQLite-compatible SELECT query for the user's question.
Rules:
- Output ONLY the SQL query, no explanation
- Use proper JOINs with the employees table for team/level filters
- Use the session_summary view for session-level queries
- Always include ORDER BY and LIMIT for large result sets
```

**Why this approach**:
- Reduces hallucination (LLM knows exact column names)
- No need for few-shot examples (schema is self-documenting)
- Works across different LLM providers consistently

### Result Interpretation Prompt

```
System: You are a data analyst. Interpret SQL query results for a non-technical audience.
- Use markdown formatting
- Highlight key numbers and trends
- Be concise (2-4 paragraphs)
- If results are empty, explain what that means
```

### Key Prompt Engineering Lessons

1. **Low temperature for SQL** — Deterministic output avoids syntax variations
2. **Schema in system prompt** — More reliable than few-shot examples
3. **Separate generation and interpretation** — Two focused LLM calls > one complex call
4. **Safety as code, not prompt** — SQL validation in Python, not LLM instructions

---

## 4. Model Selection Rationale

### For Development (GitHub Copilot)

- **Model**: Claude Opus 4.6
- **Why**: Best at complex multi-file Python architecture, understands context deeply
- **Alternative considered**: GPT-4o — good but less consistent for large codebases

### For In-App AI Insights

- **Default**: `google/gemini-2.0-flash-001`
- **Why chosen**:
  - Fast response time (~1-2s)
  - Low cost per query
  - Excellent SQL generation accuracy
  - Good at structured output
- **When to switch**:
  - Complex analytical questions → Gemini Pro or Claude Sonnet
  - Budget constraints → Llama 3.1 8B or GPT-4o-mini
  - Maximum accuracy → Claude 3.5 Sonnet

### For ML Models (No LLM Needed)

- **Tools**: scikit-learn (traditional ML, not LLM-based)
- **Why not LLM**: Anomaly detection, clustering, and classification work better with structured numeric features than language models
- **Models**: IsolationForest, DecisionTree, KMeans, LinearRegression
- **All CPU-only**: No GPU, no API calls, <3 seconds total

---

## 5. Ethical Considerations

### Data Privacy
- All data is **synthetic** (generated by `generate_fake_data.py`)
- No real user data was processed
- Employee names and emails are randomly generated

### LLM Safety
- SQL injection prevention: Only SELECT queries executed
- No user data sent to external LLMs (only schema + aggregated results)
- API keys stored in session state, not committed to git

### Transparency
- This log documents all LLM usage honestly
- Generated code was reviewed and understood before committing
- Concepts explained by LLM were verified against documentation

### Reproducibility
- All models use `random_state=42` for deterministic results
- Data generation uses fixed seed logic
- Git history captures every step

---

## Usage Summary

| Category | Tool | Model | Purpose |
|----------|------|-------|---------|
| Development | GitHub Copilot | Claude Opus 4.6 | Code generation, debugging, learning |
| In-App Feature | OpenRouter API | Gemini 2.0 Flash | NL → SQL → answer (user-facing) |
| ML Models | scikit-learn | N/A (traditional ML) | Anomaly detection, clustering, etc. |

**Total LLM API calls during development**: ~100+ interactions across 7 phases
**Total in-app LLM calls per user query**: 2 (generate SQL + interpret results)

---

*Last updated: March 5, 2026*
