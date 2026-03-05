"""
AI Insights: Natural Language → SQL → Answer pipeline.

This module provides an LLM-powered interface that lets users
ask questions about the telemetry data in plain English.

PIPELINE:
    1. User types: "Which team spent the most last week?"
    2. System prompt gives LLM the full database schema
    3. LLM generates: SELECT e.practice, SUM(cost_usd)...
    4. We execute the SQL (read-only, with safety checks)
    5. LLM interprets the results into a human-readable answer
    6. Dashboard shows: answer + table + SQL used

USES OPENROUTER API:
    OpenRouter (openrouter.ai) provides a unified API that supports
    100+ models through an OpenAI-compatible interface. We use the
    `openai` Python SDK pointed at OpenRouter's base URL.

    Default model: google/gemini-2.0-flash-001 (fast, cheap, good at SQL)
    Configurable from the dashboard sidebar.

USAGE:
    from src.ai_insights import AIInsights

    ai = AIInsights(api_key="sk-or-...", db_path="data/analytics.db")
    result = ai.ask("Which model is the most expensive?")
    print(result["answer"])      # Human-readable answer
    print(result["sql"])         # The SQL query used
    print(result["data"])        # DataFrame of results
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from openai import OpenAI

from src.database import DatabaseManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DATABASE SCHEMA — injected into the system prompt so the LLM knows
# what tables and columns exist. This is the most critical part:
# the better the schema description, the better the SQL generation.
# ---------------------------------------------------------------------------

DB_SCHEMA = """
DATABASE SCHEMA (SQLite):

TABLE: employees
  - email TEXT PRIMARY KEY          -- e.g. "alex.brown@example.com"
  - full_name TEXT                  -- e.g. "Alex Brown"
  - practice TEXT                   -- team: "ML Engineering", "Frontend Engineering", "Data Engineering", "Backend Engineering", "Platform Engineering"
  - level TEXT                      -- seniority: "L1" through "L10" (L1=junior, L10=principal)
  - location TEXT                   -- e.g. "New York", "London", "Berlin"

TABLE: api_requests
  - id INTEGER PRIMARY KEY
  - session_id TEXT                 -- unique session identifier (UUID)
  - user_email TEXT                 -- FK → employees.email
  - timestamp TEXT                  -- ISO 8601 format: "2026-01-15T14:30:00.000Z"
  - model TEXT                      -- "claude-opus-4-5-20251101", "claude-opus-4-6", "claude-sonnet-4-5-20250929", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
  - cost_usd REAL                   -- cost in US dollars (e.g. 0.05)
  - duration_ms INTEGER             -- API call duration in milliseconds
  - input_tokens INTEGER            -- tokens sent to the model
  - output_tokens INTEGER           -- tokens received from the model
  - cache_read_tokens INTEGER       -- tokens served from cache (saves cost)
  - cache_creation_tokens INTEGER   -- tokens written to cache
  - terminal_type TEXT              -- "vscode", "jetbrains", "terminal", etc.
  - org_id TEXT                     -- organization identifier
  - scope_version TEXT              -- Claude Code version

TABLE: tool_decisions
  - id INTEGER PRIMARY KEY
  - session_id TEXT
  - user_email TEXT                 -- FK → employees.email
  - timestamp TEXT
  - tool_name TEXT                  -- e.g. "Edit", "Write", "Bash", "Glob", "Read", etc.
  - decision TEXT                   -- "approve" or "deny"
  - source TEXT                     -- "config", "user_temp", "user_perm", "auto"
  - terminal_type TEXT
  - org_id TEXT
  - scope_version TEXT

TABLE: tool_results
  - id INTEGER PRIMARY KEY
  - session_id TEXT
  - user_email TEXT                 -- FK → employees.email
  - timestamp TEXT
  - tool_name TEXT                  -- same tool names as tool_decisions
  - success TEXT                    -- "true" or "false" (as text)
  - duration_ms INTEGER             -- tool execution duration
  - decision_source TEXT            -- how the tool was approved
  - decision_type TEXT              -- type of approval decision
  - tool_result_size_bytes INTEGER  -- size of tool output (nullable)
  - terminal_type TEXT
  - org_id TEXT
  - scope_version TEXT

TABLE: user_prompts
  - id INTEGER PRIMARY KEY
  - session_id TEXT
  - user_email TEXT                 -- FK → employees.email
  - timestamp TEXT
  - prompt_length INTEGER           -- character count of the user's prompt
  - terminal_type TEXT
  - org_id TEXT
  - scope_version TEXT

TABLE: api_errors
  - id INTEGER PRIMARY KEY
  - session_id TEXT
  - user_email TEXT                 -- FK → employees.email
  - timestamp TEXT
  - model TEXT                      -- which model caused the error
  - error TEXT                      -- error message text
  - status_code TEXT                -- HTTP status: "429", "500", "400", "401", "undefined"
  - attempt INTEGER                 -- retry attempt number (1, 2, 3...)
  - duration_ms INTEGER
  - terminal_type TEXT
  - org_id TEXT
  - scope_version TEXT

VIEW: session_summary (pre-aggregated per session)
  - session_id, user_email, practice, level, location
  - start_time, end_time
  - api_call_count, total_cost
  - total_input_tokens, total_output_tokens, total_cache_read_tokens

RELATIONSHIPS:
  - All tables join on: table.user_email = employees.email
  - All event tables share: session_id (groups events in a session)
  - Date range: 2025-12-03 to 2026-01-31

KEY FACTS:
  - 100 employees, 5,000 sessions, 454,428 total events
  - Total cost: $6,001.43
  - 5 models, 5 practices, 10 levels (L1-L10), 17 tools
  - success column in tool_results is TEXT ('true'/'false'), not boolean
  - Timestamps are ISO 8601 strings — use DATE() or strftime() for date operations
"""

# ---------------------------------------------------------------------------
# SYSTEM PROMPTS — two prompts for the two-step pipeline
# ---------------------------------------------------------------------------

SQL_GENERATION_PROMPT = f"""You are a SQL expert assistant. You have access to a SQLite database containing Claude Code telemetry data from engineering teams.

{DB_SCHEMA}

RULES:
1. Generate ONLY a valid SQLite SELECT query. No explanations, no markdown, no code fences.
2. NEVER use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, or any write operation.
3. Use proper JOINs when querying across tables (e.g., JOIN employees e ON t.user_email = e.email).
4. Use DATE(timestamp) for date grouping, strftime('%H', timestamp) for hours.
5. Limit results to 50 rows max unless the user asks for more.
6. Use table aliases for readability (e.g., ar for api_requests, e for employees).
7. For "last week" or "this week", calculate relative to the max date in the data (2026-01-31).
8. Remember: success in tool_results is TEXT ('true'/'false'), compare with = 'true'.
9. Use ROUND() for decimal results, especially costs.
10. When asked about "cost", use the cost_usd column from api_requests.
11. Return ONLY the raw SQL query text. No markdown formatting."""

INTERPRETATION_PROMPT = """You are a data analyst assistant presenting query results to a non-technical user.

Given the user's original question and the SQL query results, provide:
1. A clear, concise answer in 2-3 sentences
2. Key insights or notable patterns in the data
3. If relevant, mention any caveats or limitations

Keep the tone professional but accessible. Use specific numbers from the results.
Format currency as $X,XXX.XX and large numbers with commas.
Do NOT include SQL queries or technical database terminology in your answer."""

# ---------------------------------------------------------------------------
# AVAILABLE MODELS — for the dashboard sidebar dropdown
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash-preview",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat-v3-0324",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o-mini",
]

DEFAULT_MODEL = "google/gemini-2.0-flash-001"


# ---------------------------------------------------------------------------
# AI INSIGHTS CLASS
# ---------------------------------------------------------------------------

class AIInsights:
    """LLM-powered natural language query interface for telemetry data.

    Uses OpenRouter API (OpenAI-compatible) to:
    1. Convert natural language questions to SQL
    2. Execute SQL safely against the database
    3. Interpret results into human-readable answers

    Usage:
        ai = AIInsights(api_key="sk-or-...", db_path="data/analytics.db")
        result = ai.ask("Which team spent the most?")
        # result = {"answer": "...", "sql": "SELECT ...", "data": DataFrame, "error": None}
    """

    def __init__(
        self,
        api_key: str,
        db_path: str,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Initialize the AI insights pipeline.

        Args:
            api_key: OpenRouter API key (starts with sk-or-...)
            db_path: Path to the analytics.db SQLite file
            model: OpenRouter model identifier (default: gemini-2.0-flash)
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.db = DatabaseManager(db_path)
        self.model = model

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Call the LLM via OpenRouter and return the response text.

        Args:
            system_prompt: The system/instruction prompt
            user_message: The user's message

        Returns:
            The LLM's response text

        Raises:
            Exception: If the API call fails
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,  # Deterministic for SQL generation
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def generate_sql(self, question: str) -> str:
        """Convert a natural language question to a SQL query.

        Args:
            question: The user's question in plain English

        Returns:
            A SQL SELECT query string
        """
        raw_sql = self._call_llm(SQL_GENERATION_PROMPT, question)

        # Clean up: remove markdown code fences if the model adds them
        sql = raw_sql.strip()
        if sql.startswith("```"):
            # Remove opening fence (```sql or ```)
            sql = sql.split("\n", 1)[-1] if "\n" in sql else sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()

        # Remove trailing semicolons (pandas doesn't like them)
        if sql.endswith(";"):
            sql = sql[:-1].strip()

        return sql

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query safely and return results as DataFrame.

        Uses DatabaseManager.execute_safe_query() which blocks
        all write operations (INSERT, DROP, etc.)

        Args:
            sql: A SQL SELECT query

        Returns:
            DataFrame with query results

        Raises:
            ValueError: If the query contains write operations
            Exception: If the query fails to execute
        """
        return self.db.execute_safe_query(sql)

    def interpret_results(
        self, question: str, sql: str, data: pd.DataFrame
    ) -> str:
        """Have the LLM interpret query results into a human answer.

        Args:
            question: The original user question
            sql: The SQL query that was executed
            data: The DataFrame of results

        Returns:
            A human-readable interpretation of the results
        """
        # Format the data as a readable string for the LLM
        if data.empty:
            data_str = "(No results returned)"
        elif len(data) > 20:
            data_str = data.head(20).to_string(index=False) + f"\n... ({len(data)} total rows)"
        else:
            data_str = data.to_string(index=False)

        user_msg = (
            f"Original question: {question}\n\n"
            f"SQL query used:\n{sql}\n\n"
            f"Results:\n{data_str}"
        )

        return self._call_llm(INTERPRETATION_PROMPT, user_msg)

    def ask(self, question: str) -> dict:
        """Full pipeline: question → SQL → execute → interpret.

        This is the main entry point. It runs the complete pipeline
        and returns all intermediate results for display.

        Args:
            question: A natural language question about the data

        Returns:
            Dict with keys:
                "answer" (str): Human-readable answer
                "sql" (str): The SQL query generated
                "data" (DataFrame): Raw query results
                "error" (str|None): Error message if something failed
        """
        result = {
            "answer": "",
            "sql": "",
            "data": pd.DataFrame(),
            "error": None,
        }

        try:
            # Step 1: Generate SQL
            logger.info(f"Generating SQL for: {question}")
            sql = self.generate_sql(question)
            result["sql"] = sql
            logger.info(f"Generated SQL: {sql}")

            # Step 2: Execute SQL
            data = self.execute_sql(sql)
            result["data"] = data
            logger.info(f"Query returned {len(data)} rows")

            # Step 3: Interpret results
            answer = self.interpret_results(question, sql, data)
            result["answer"] = answer

        except ValueError as e:
            # Safety check blocked the query
            result["error"] = f"Safety check: {e}"
            logger.warning(f"Blocked unsafe query: {e}")

        except Exception as e:
            # API error, SQL error, etc.
            result["error"] = f"Error: {str(e)}"
            logger.error(f"AI pipeline error: {e}")

        return result


# ---------------------------------------------------------------------------
# EXAMPLE QUESTIONS — shown in the dashboard as quick-start suggestions
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "Which team spent the most money overall?",
    "What is the average cost per session for each seniority level?",
    "Which model has the highest error rate?",
    "What are the top 5 most used tools and their success rates?",
    "Show me the daily cost trend for the last 2 weeks",
    "Which users have the highest cost efficiency (output tokens per dollar)?",
    "What percentage of tool decisions are auto-approved vs manual?",
    "Compare the average session cost between ML and Frontend engineering",
    "What are the most common error types and which models cause them?",
    "How does cache hit ratio vary across different models?",
]
