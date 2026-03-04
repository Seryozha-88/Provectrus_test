"""
Database query layer: SQL queries that return raw data from SQLite.

This module provides a DatabaseManager class that wraps all SQL queries
needed by the analytics and dashboard layers. It handles:
- Connection management (context manager pattern)
- Parameterized filters (date range, user, practice, model)
- Returning pandas DataFrames ready for analysis
- Safe read-only query execution for the AI chat feature

ARCHITECTURE:
    ┌──────────────────┐
    │  Streamlit UI    │
    │  (dashboard.py)  │
    └───────┬──────────┘
            │ calls
    ┌───────▼──────────┐
    │  Analytics Layer  │  ← computed metrics (cache ratio, efficiency)
    │  (analytics.py)   │
    └───────┬──────────┘
            │ calls
    ┌───────▼──────────┐
    │  Database Layer   │  ← THIS FILE: raw SQL queries → DataFrames
    │  (database.py)    │
    └───────┬──────────┘
            │ reads
    ┌───────▼──────────┐
    │  SQLite Database  │
    │  (analytics.db)   │
    └──────────────────┘

WHY SEPARATE FROM analytics.py?
    - Single Responsibility: database.py knows SQL, analytics.py knows math
    - Testability: database queries can be tested with a tiny test DB
    - Reusability: the same queries can back the dashboard, API, or CLI
    - Security: execute_safe_query() is isolated here with read-only checks

USAGE:
    from src.database import DatabaseManager

    db = DatabaseManager("data/analytics.db")
    stats = db.get_overview_stats()
    daily = db.get_daily_trends(date_from="2026-01-01", model="claude-sonnet")
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# QUERY FILTERS
# ---------------------------------------------------------------------------
# A dataclass that holds all possible filter parameters.
# Every query method accepts these filters and dynamically builds
# the WHERE clause. This avoids repeating filter logic in every method.
#
# WHY A DATACLASS?
# - Clean, typed structure (better than passing a dict)
# - Default values (None = no filter)
# - Easy to create from Streamlit sidebar widgets
# ---------------------------------------------------------------------------

@dataclass
class QueryFilters:
    """Filters that can be applied to any database query.

    All fields default to None (= no filter applied).
    The database methods build WHERE clauses dynamically
    based on which fields are set.

    Attributes:
        date_from: Start date (inclusive), e.g. "2026-01-01"
        date_to: End date (inclusive), e.g. "2026-01-31"
        user_email: Filter by specific user email
        practice: Filter by team/department name
        level: Filter by seniority level (Junior/Middle/Senior/Lead)
        model: Filter by Claude model name
        tool_name: Filter by specific tool name
    """
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    user_email: Optional[str] = None
    practice: Optional[str] = None
    level: Optional[str] = None
    model: Optional[str] = None
    tool_name: Optional[str] = None


# ---------------------------------------------------------------------------
# FILTER BUILDER HELPERS
# ---------------------------------------------------------------------------
# These utility functions generate SQL WHERE clause fragments and
# parameter lists from a QueryFilters object.
# ---------------------------------------------------------------------------

def _build_date_filter(
    filters: QueryFilters,
    timestamp_col: str = "timestamp",
) -> tuple[list[str], list[Any]]:
    """Build date range WHERE clauses.

    Returns:
        Tuple of (conditions list, params list).
    """
    conditions: list[str] = []
    params: list[Any] = []

    if filters.date_from:
        conditions.append(f"{timestamp_col} >= ?")
        params.append(filters.date_from)
    if filters.date_to:
        # Add time to make date_to inclusive for the whole day
        conditions.append(f"{timestamp_col} < date(?, '+1 day')")
        params.append(filters.date_to)

    return conditions, params


def _build_employee_join_filter(
    filters: QueryFilters,
    user_col: str = "t.user_email",
) -> tuple[str, list[str], list[Any]]:
    """Build JOIN + WHERE for employee-related filters.

    Returns:
        Tuple of (join_clause, conditions list, params list).
        join_clause is empty string if no employee filters are needed.
    """
    join = ""
    conditions: list[str] = []
    params: list[Any] = []

    needs_join = filters.practice or filters.level or filters.user_email

    if needs_join:
        join = f"JOIN employees e ON {user_col} = e.email"

    if filters.user_email:
        conditions.append(f"{user_col} = ?")
        params.append(filters.user_email)
    if filters.practice:
        conditions.append("e.practice = ?")
        params.append(filters.practice)
    if filters.level:
        conditions.append("e.level = ?")
        params.append(filters.level)

    return join, conditions, params


def _combine_where(conditions: list[str]) -> str:
    """Join conditions with AND and prepend WHERE if non-empty."""
    if not conditions:
        return ""
    return "WHERE " + " AND ".join(conditions)


# ---------------------------------------------------------------------------
# DATABASE MANAGER
# ---------------------------------------------------------------------------

class DatabaseManager:
    """Manages SQLite connection and provides query methods.

    All query methods return pandas DataFrames. This keeps the
    database layer focused on data retrieval, while analytics.py
    handles computed metrics and transformations.

    Usage:
        db = DatabaseManager("data/analytics.db")
        stats = db.get_overview_stats()
        daily = db.get_daily_trends(QueryFilters(date_from="2026-01-01"))
    """

    def __init__(self, db_path: str) -> None:
        """Initialize with path to SQLite database.

        Args:
            db_path: Path to the analytics.db file.
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy connection — opens on first use, reuses after."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _query_df(self, sql: str, params: list | tuple = ()) -> pd.DataFrame:
        """Execute a SQL query and return result as a DataFrame.

        This is the core helper used by all query methods.
        Using pandas.read_sql_query() automatically maps column names.
        """
        return pd.read_sql_query(sql, self.conn, params=params)

    # -------------------------------------------------------------------
    # 1. OVERVIEW STATS — single dict of KPI numbers
    # -------------------------------------------------------------------

    def get_overview_stats(self, filters: Optional[QueryFilters] = None) -> dict:
        """Get high-level KPI stats for the dashboard header.

        Returns dict with keys:
            total_cost, total_sessions, total_users, total_events,
            total_api_requests, total_tool_decisions, total_tool_results,
            total_user_prompts, total_api_errors, error_rate,
            avg_cost_per_session, total_input_tokens, total_output_tokens
        """
        f = filters or QueryFilters()
        stats: dict[str, Any] = {}

        # --- API requests aggregates ---
        date_conds, date_params = _build_date_filter(f)
        emp_join, emp_conds, emp_params = _build_employee_join_filter(f, "t.user_email")

        conds = date_conds + emp_conds
        if f.model:
            conds.append("t.model = ?")
            emp_params.append(f.model)

        where = _combine_where(conds)
        params = date_params + emp_params

        # Build timestamp column reference for api_requests
        timestamp_where = where.replace("timestamp", "t.timestamp") if where else ""

        sql = f"""
            SELECT
                COALESCE(SUM(t.cost_usd), 0) as total_cost,
                COUNT(DISTINCT t.session_id) as total_sessions,
                COUNT(DISTINCT t.user_email) as total_users,
                COUNT(*) as total_api_requests,
                COALESCE(SUM(t.input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(t.output_tokens), 0) as total_output_tokens,
                COALESCE(SUM(t.cache_read_tokens), 0) as total_cache_read_tokens,
                COALESCE(AVG(t.cost_usd), 0) as avg_cost_per_request
            FROM api_requests t
            {emp_join}
            {timestamp_where}
        """
        row = self._query_df(sql, params).iloc[0]
        stats.update(row.to_dict())

        # --- Event counts from other tables (with date filter only) ---
        for table, key in [
            ("tool_decisions", "total_tool_decisions"),
            ("tool_results", "total_tool_results"),
            ("user_prompts", "total_user_prompts"),
            ("api_errors", "total_api_errors"),
        ]:
            conds_t, params_t = _build_date_filter(f)
            where_t = _combine_where(conds_t)
            count_sql = f"SELECT COUNT(*) as cnt FROM {table} {where_t}"
            stats[key] = int(self._query_df(count_sql, params_t).iloc[0]["cnt"])

        # --- Computed ---
        stats["total_events"] = (
            int(stats["total_api_requests"])
            + stats["total_tool_decisions"]
            + stats["total_tool_results"]
            + stats["total_user_prompts"]
            + stats["total_api_errors"]
        )
        api_count = int(stats["total_api_requests"])
        stats["error_rate"] = (
            stats["total_api_errors"] / api_count if api_count > 0 else 0.0
        )
        sessions = int(stats["total_sessions"])
        stats["avg_cost_per_session"] = (
            float(stats["total_cost"]) / sessions if sessions > 0 else 0.0
        )

        return stats

    # -------------------------------------------------------------------
    # 2. DAILY TRENDS — time series of costs, events, tokens
    # -------------------------------------------------------------------

    def get_daily_trends(self, filters: Optional[QueryFilters] = None) -> pd.DataFrame:
        """Daily aggregated metrics for time series charts.

        Returns DataFrame with columns:
            date, daily_cost, daily_requests, daily_input_tokens,
            daily_output_tokens, daily_cache_read_tokens
        """
        f = filters or QueryFilters()
        date_conds, date_params = _build_date_filter(f, "t.timestamp")
        emp_join, emp_conds, emp_params = _build_employee_join_filter(f, "t.user_email")
        conds = date_conds + emp_conds
        if f.model:
            conds.append("t.model = ?")
            emp_params.append(f.model)
        where = _combine_where(conds)
        params = date_params + emp_params

        sql = f"""
            SELECT
                DATE(t.timestamp) as date,
                SUM(t.cost_usd) as daily_cost,
                COUNT(*) as daily_requests,
                SUM(t.input_tokens) as daily_input_tokens,
                SUM(t.output_tokens) as daily_output_tokens,
                SUM(t.cache_read_tokens) as daily_cache_read_tokens
            FROM api_requests t
            {emp_join}
            {where}
            GROUP BY DATE(t.timestamp)
            ORDER BY date
        """
        return self._query_df(sql, params)

    # -------------------------------------------------------------------
    # 3. COST BREAKDOWN BY MODEL
    # -------------------------------------------------------------------

    def get_cost_by_model(self, filters: Optional[QueryFilters] = None) -> pd.DataFrame:
        """Cost breakdown by Claude model.

        Returns DataFrame with columns: model, total_cost, request_count
        """
        f = filters or QueryFilters()
        date_conds, date_params = _build_date_filter(f, "t.timestamp")
        emp_join, emp_conds, emp_params = _build_employee_join_filter(f, "t.user_email")
        conds = date_conds + emp_conds
        where = _combine_where(conds)
        params = date_params + emp_params

        sql = f"""
            SELECT
                t.model,
                SUM(t.cost_usd) as total_cost,
                COUNT(*) as request_count,
                AVG(t.cost_usd) as avg_cost
            FROM api_requests t
            {emp_join}
            {where}
            GROUP BY t.model
            ORDER BY total_cost DESC
        """
        return self._query_df(sql, params)

    # -------------------------------------------------------------------
    # 4. COST BY PRACTICE (team/department)
    # -------------------------------------------------------------------

    def get_cost_by_practice(self, filters: Optional[QueryFilters] = None) -> pd.DataFrame:
        """Cost breakdown by team/practice.

        Returns DataFrame with columns: practice, total_cost, user_count, request_count
        """
        f = filters or QueryFilters()
        date_conds, date_params = _build_date_filter(f, "t.timestamp")
        conds = date_conds[:]
        params = date_params[:]

        if f.level:
            conds.append("e.level = ?")
            params.append(f.level)
        if f.model:
            conds.append("t.model = ?")
            params.append(f.model)
        where = _combine_where(conds)

        sql = f"""
            SELECT
                e.practice,
                SUM(t.cost_usd) as total_cost,
                COUNT(DISTINCT t.user_email) as user_count,
                COUNT(*) as request_count,
                AVG(t.cost_usd) as avg_cost_per_request
            FROM api_requests t
            JOIN employees e ON t.user_email = e.email
            {where}
            GROUP BY e.practice
            ORDER BY total_cost DESC
        """
        return self._query_df(sql, params)

    # -------------------------------------------------------------------
    # 5. COST BY LEVEL (seniority)
    # -------------------------------------------------------------------

    def get_cost_by_level(self, filters: Optional[QueryFilters] = None) -> pd.DataFrame:
        """Cost breakdown by seniority level.

        Returns DataFrame with columns: level, total_cost, user_count, avg_cost_per_user
        """
        f = filters or QueryFilters()
        date_conds, date_params = _build_date_filter(f, "t.timestamp")
        conds = date_conds[:]
        params = date_params[:]

        if f.practice:
            conds.append("e.practice = ?")
            params.append(f.practice)
        if f.model:
            conds.append("t.model = ?")
            params.append(f.model)
        where = _combine_where(conds)

        sql = f"""
            SELECT
                e.level,
                SUM(t.cost_usd) as total_cost,
                COUNT(DISTINCT t.user_email) as user_count,
                SUM(t.cost_usd) / COUNT(DISTINCT t.user_email) as avg_cost_per_user,
                COUNT(*) as request_count
            FROM api_requests t
            JOIN employees e ON t.user_email = e.email
            {where}
            GROUP BY e.level
            ORDER BY total_cost DESC
        """
        return self._query_df(sql, params)

    # -------------------------------------------------------------------
    # 6. TOOL USAGE — frequency and types
    # -------------------------------------------------------------------

    def get_tool_usage(self, filters: Optional[QueryFilters] = None) -> pd.DataFrame:
        """Tool usage frequency from tool_results.

        Returns DataFrame with columns:
            tool_name, use_count, avg_duration_ms, success_rate
        """
        f = filters or QueryFilters()
        date_conds, date_params = _build_date_filter(f, "t.timestamp")
        emp_join, emp_conds, emp_params = _build_employee_join_filter(f, "t.user_email")
        conds = date_conds + emp_conds
        if f.tool_name:
            conds.append("t.tool_name = ?")
            emp_params.append(f.tool_name)
        where = _combine_where(conds)
        params = date_params + emp_params

        sql = f"""
            SELECT
                t.tool_name,
                COUNT(*) as use_count,
                AVG(t.duration_ms) as avg_duration_ms,
                ROUND(100.0 * SUM(CASE WHEN t.success = 'true' THEN 1 ELSE 0 END)
                      / COUNT(*), 1) as success_rate,
                SUM(CASE WHEN t.success = 'true' THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN t.success = 'false' THEN 1 ELSE 0 END) as failure_count
            FROM tool_results t
            {emp_join}
            {where}
            GROUP BY t.tool_name
            ORDER BY use_count DESC
        """
        return self._query_df(sql, params)

    # -------------------------------------------------------------------
    # 7. TOOL DECISION SOURCES
    # -------------------------------------------------------------------

    def get_tool_decisions_summary(
        self, filters: Optional[QueryFilters] = None
    ) -> pd.DataFrame:
        """Tool decision breakdown by source and type.

        Returns DataFrame with columns:
            tool_name, decision, source, count
        """
        f = filters or QueryFilters()
        date_conds, date_params = _build_date_filter(f, "t.timestamp")
        emp_join, emp_conds, emp_params = _build_employee_join_filter(f, "t.user_email")
        conds = date_conds + emp_conds
        if f.tool_name:
            conds.append("t.tool_name = ?")
            emp_params.append(f.tool_name)
        where = _combine_where(conds)
        params = date_params + emp_params

        sql = f"""
            SELECT
                t.tool_name,
                t.decision,
                t.source,
                COUNT(*) as count
            FROM tool_decisions t
            {emp_join}
            {where}
            GROUP BY t.tool_name, t.decision, t.source
            ORDER BY count DESC
        """
        return self._query_df(sql, params)

    # -------------------------------------------------------------------
    # 8. HOURLY ACTIVITY HEATMAP
    # -------------------------------------------------------------------

    def get_hourly_heatmap(self, filters: Optional[QueryFilters] = None) -> pd.DataFrame:
        """Activity count by day-of-week and hour for heatmap.

        Returns DataFrame with columns:
            day_of_week (0=Sunday), hour, event_count
        """
        f = filters or QueryFilters()
        date_conds, date_params = _build_date_filter(f, "t.timestamp")
        emp_join, emp_conds, emp_params = _build_employee_join_filter(f, "t.user_email")
        conds = date_conds + emp_conds
        where = _combine_where(conds)
        params = date_params + emp_params

        sql = f"""
            SELECT
                CAST(strftime('%w', t.timestamp) AS INTEGER) as day_of_week,
                CAST(strftime('%H', t.timestamp) AS INTEGER) as hour,
                COUNT(*) as event_count
            FROM api_requests t
            {emp_join}
            {where}
            GROUP BY day_of_week, hour
            ORDER BY day_of_week, hour
        """
        return self._query_df(sql, params)

    # -------------------------------------------------------------------
    # 9. ERROR BREAKDOWN
    # -------------------------------------------------------------------

    def get_error_breakdown(self, filters: Optional[QueryFilters] = None) -> pd.DataFrame:
        """Error breakdown by error type and status code.

        Returns DataFrame with columns:
            error, status_code, model, count
        """
        f = filters or QueryFilters()
        date_conds, date_params = _build_date_filter(f, "t.timestamp")
        emp_join, emp_conds, emp_params = _build_employee_join_filter(f, "t.user_email")
        conds = date_conds + emp_conds
        if f.model:
            conds.append("t.model = ?")
            emp_params.append(f.model)
        where = _combine_where(conds)
        params = date_params + emp_params

        sql = f"""
            SELECT
                t.error,
                t.status_code,
                t.model,
                COUNT(*) as count,
                AVG(t.duration_ms) as avg_duration_ms,
                AVG(t.attempt) as avg_retry_attempts
            FROM api_errors t
            {emp_join}
            {where}
            GROUP BY t.error, t.status_code, t.model
            ORDER BY count DESC
        """
        return self._query_df(sql, params)

    # -------------------------------------------------------------------
    # 10. USER RANKINGS
    # -------------------------------------------------------------------

    def get_user_rankings(self, filters: Optional[QueryFilters] = None) -> pd.DataFrame:
        """User rankings by cost, sessions, and activity.

        Returns DataFrame with columns:
            user_email, full_name, practice, level, location,
            total_cost, session_count, request_count, avg_cost_per_session
        """
        f = filters or QueryFilters()
        date_conds, date_params = _build_date_filter(f, "t.timestamp")
        conds = date_conds[:]
        params = date_params[:]

        if f.practice:
            conds.append("e.practice = ?")
            params.append(f.practice)
        if f.level:
            conds.append("e.level = ?")
            params.append(f.level)
        if f.model:
            conds.append("t.model = ?")
            params.append(f.model)
        where = _combine_where(conds)

        sql = f"""
            SELECT
                t.user_email,
                e.full_name,
                e.practice,
                e.level,
                e.location,
                SUM(t.cost_usd) as total_cost,
                COUNT(DISTINCT t.session_id) as session_count,
                COUNT(*) as request_count,
                SUM(t.cost_usd) / COUNT(DISTINCT t.session_id) as avg_cost_per_session,
                SUM(t.input_tokens) as total_input_tokens,
                SUM(t.output_tokens) as total_output_tokens
            FROM api_requests t
            JOIN employees e ON t.user_email = e.email
            {where}
            GROUP BY t.user_email
            ORDER BY total_cost DESC
        """
        return self._query_df(sql, params)

    # -------------------------------------------------------------------
    # 11. SESSION STATS
    # -------------------------------------------------------------------

    def get_session_stats(self, filters: Optional[QueryFilters] = None) -> pd.DataFrame:
        """Per-session statistics.

        Returns DataFrame with columns:
            session_id, user_email, practice, level,
            start_time, end_time, duration_minutes,
            api_call_count, total_cost, total_input_tokens,
            total_output_tokens, prompt_count, tool_use_count
        """
        f = filters or QueryFilters()
        date_conds, date_params = _build_date_filter(f, "ss.start_time")
        conds = date_conds[:]
        params = date_params[:]

        if f.user_email:
            conds.append("ss.user_email = ?")
            params.append(f.user_email)
        if f.practice:
            conds.append("ss.practice = ?")
            params.append(f.practice)
        if f.level:
            conds.append("ss.level = ?")
            params.append(f.level)
        where = _combine_where(conds)

        # Use the session_summary view + LEFT JOINs for prompt/tool counts
        sql = f"""
            SELECT
                ss.session_id,
                ss.user_email,
                ss.practice,
                ss.level,
                ss.location,
                ss.start_time,
                ss.end_time,
                ROUND(
                    (julianday(ss.end_time) - julianday(ss.start_time)) * 1440, 1
                ) as duration_minutes,
                ss.api_call_count,
                ss.total_cost,
                ss.total_input_tokens,
                ss.total_output_tokens,
                ss.total_cache_read_tokens,
                COALESCE(up.prompt_count, 0) as prompt_count,
                COALESCE(tr.tool_use_count, 0) as tool_use_count,
                COALESCE(tr.tool_success_count, 0) as tool_success_count
            FROM session_summary ss
            LEFT JOIN (
                SELECT session_id, COUNT(*) as prompt_count
                FROM user_prompts
                GROUP BY session_id
            ) up ON ss.session_id = up.session_id
            LEFT JOIN (
                SELECT
                    session_id,
                    COUNT(*) as tool_use_count,
                    SUM(CASE WHEN success = 'true' THEN 1 ELSE 0 END) as tool_success_count
                FROM tool_results
                GROUP BY session_id
            ) tr ON ss.session_id = tr.session_id
            {where}
            ORDER BY ss.start_time DESC
        """
        return self._query_df(sql, params)

    # -------------------------------------------------------------------
    # 12. MODEL USAGE OVER TIME
    # -------------------------------------------------------------------

    def get_model_usage_over_time(
        self, filters: Optional[QueryFilters] = None
    ) -> pd.DataFrame:
        """Model usage trends over time.

        Returns DataFrame with columns: date, model, request_count, total_cost
        """
        f = filters or QueryFilters()
        date_conds, date_params = _build_date_filter(f, "t.timestamp")
        emp_join, emp_conds, emp_params = _build_employee_join_filter(f, "t.user_email")
        conds = date_conds + emp_conds
        where = _combine_where(conds)
        params = date_params + emp_params

        sql = f"""
            SELECT
                DATE(t.timestamp) as date,
                t.model,
                COUNT(*) as request_count,
                SUM(t.cost_usd) as total_cost
            FROM api_requests t
            {emp_join}
            {where}
            GROUP BY DATE(t.timestamp), t.model
            ORDER BY date, model
        """
        return self._query_df(sql, params)

    # -------------------------------------------------------------------
    # 13. SAFE QUERY EXECUTION — for AI chat feature
    # -------------------------------------------------------------------

    def execute_safe_query(self, sql: str) -> pd.DataFrame:
        """Execute a read-only SQL query (for the AI chat feature).

        Security:
        - Only SELECT statements are allowed
        - Common write keywords are blocked
        - Uses a fresh read-only approach

        Args:
            sql: A SQL SELECT statement to execute.

        Returns:
            DataFrame with query results.

        Raises:
            ValueError: If the query contains write operations.
        """
        # Normalize for safety check
        sql_upper = sql.strip().upper()

        # Block anything that isn't a SELECT
        blocked_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
            "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA",
        ]
        for keyword in blocked_keywords:
            if keyword in sql_upper:
                raise ValueError(
                    f"Write operations are not allowed. "
                    f"Blocked keyword: {keyword}"
                )

        if not sql_upper.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")

        return self._query_df(sql)

    # -------------------------------------------------------------------
    # HELPER: Get filter options for Streamlit sidebar
    # -------------------------------------------------------------------

    def get_filter_options(self) -> dict[str, list[str]]:
        """Get unique values for each filterable field.

        Returns dict with keys: practices, levels, models, users, tools
        Used by Streamlit sidebar to populate dropdown menus.
        """
        options: dict[str, list[str]] = {}

        options["practices"] = (
            self._query_df("SELECT DISTINCT practice FROM employees ORDER BY practice")
            ["practice"].tolist()
        )
        options["levels"] = (
            self._query_df("SELECT DISTINCT level FROM employees ORDER BY level")
            ["level"].tolist()
        )
        options["models"] = (
            self._query_df("SELECT DISTINCT model FROM api_requests ORDER BY model")
            ["model"].tolist()
        )
        options["users"] = (
            self._query_df(
                "SELECT DISTINCT email FROM employees ORDER BY email"
            )["email"].tolist()
        )
        options["tools"] = (
            self._query_df(
                "SELECT DISTINCT tool_name FROM tool_results ORDER BY tool_name"
            )["tool_name"].tolist()
        )

        # Date range
        date_range = self._query_df(
            "SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date "
            "FROM api_requests"
        ).iloc[0]
        options["min_date"] = date_range["min_date"]
        options["max_date"] = date_range["max_date"]

        return options
