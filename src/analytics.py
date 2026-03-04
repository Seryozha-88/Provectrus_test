"""
Analytics computation layer: business logic and derived metrics.

This module sits BETWEEN the database layer and the dashboard.
It calls DatabaseManager methods to get raw data, then computes
derived metrics that aren't simple SQL queries:

    - Cache hit ratio
    - Token efficiency
    - Cost anomaly flags
    - Session productivity scores
    - Trend comparisons (week-over-week)
    - Statistical summaries

WHY SEPARATE FROM database.py?
    database.py knows SQL. This file knows MATH.
    - Database layer: "give me SUM(cost) GROUP BY date"
    - Analytics layer: "compute week-over-week cost change %"

USAGE:
    from src.analytics import AnalyticsEngine

    engine = AnalyticsEngine("data/analytics.db")
    kpis = engine.get_kpi_cards()            # For dashboard header
    trends = engine.get_cost_trend()         # For line chart
    tools = engine.get_tool_performance()    # For tool analysis page
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np

from src.database import DatabaseManager, QueryFilters


class AnalyticsEngine:
    """High-level analytics that combine raw queries with computed metrics.

    This class wraps DatabaseManager and adds business logic on top.
    Every method returns either a dict (for KPIs) or a DataFrame
    (for charts), ready for Streamlit/Plotly.

    Usage:
        engine = AnalyticsEngine("data/analytics.db")
        kpis = engine.get_kpi_cards()
    """

    def __init__(self, db_path: str) -> None:
        """Initialize with path to SQLite database.

        Args:
            db_path: Path to the analytics.db file.
        """
        self.db = DatabaseManager(db_path)

    def close(self) -> None:
        """Close underlying database connection."""
        self.db.close()

    # -------------------------------------------------------------------
    # 1. KPI CARDS — headline numbers for the dashboard
    # -------------------------------------------------------------------

    def get_kpi_cards(self, filters: Optional[QueryFilters] = None) -> dict:
        """Compute KPI metrics for the dashboard header cards.

        Returns dict with:
            - total_cost, avg_cost_per_session
            - total_sessions, total_users
            - total_events, error_rate (%)
            - cache_hit_ratio (%)
            - token_efficiency (output/input ratio)
            - avg_tokens_per_request
        """
        stats = self.db.get_overview_stats(filters)

        # --- Cache hit ratio ---
        # How much of the input was served from cache vs fresh computation.
        # Higher = better (cheaper, faster).
        # Formula: cache_read / (cache_read + input_tokens) * 100
        cache_read = float(stats.get("total_cache_read_tokens", 0))
        input_tokens = float(stats.get("total_input_tokens", 0))
        total_input = cache_read + input_tokens
        stats["cache_hit_ratio"] = (
            (cache_read / total_input * 100) if total_input > 0 else 0.0
        )

        # --- Token efficiency ---
        # Ratio of output tokens to input tokens.
        # Lower = model produces concise responses relative to input.
        output_tokens = float(stats.get("total_output_tokens", 0))
        stats["token_efficiency"] = (
            (output_tokens / input_tokens) if input_tokens > 0 else 0.0
        )

        # --- Avg tokens per request ---
        api_count = int(stats.get("total_api_requests", 0))
        stats["avg_tokens_per_request"] = (
            int((input_tokens + output_tokens) / api_count) if api_count > 0 else 0
        )

        # --- Error rate as percentage ---
        stats["error_rate_pct"] = float(stats.get("error_rate", 0)) * 100

        return stats

    # -------------------------------------------------------------------
    # 2. COST TREND — daily cost with moving average
    # -------------------------------------------------------------------

    def get_cost_trend(self, filters: Optional[QueryFilters] = None) -> pd.DataFrame:
        """Daily cost trend with 7-day moving average.

        Returns DataFrame with columns:
            date, daily_cost, daily_requests, cost_7d_avg, requests_7d_avg
        """
        df = self.db.get_daily_trends(filters)

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])

        # 7-day rolling average for smoother trend lines
        df["cost_7d_avg"] = df["daily_cost"].rolling(window=7, min_periods=1).mean()
        df["requests_7d_avg"] = (
            df["daily_requests"].rolling(window=7, min_periods=1).mean()
        )

        # Cumulative cost (useful for "burn rate" chart)
        df["cumulative_cost"] = df["daily_cost"].cumsum()

        return df

    # -------------------------------------------------------------------
    # 3. TOKEN ANALYSIS — input/output/cache breakdown
    # -------------------------------------------------------------------

    def get_token_analysis(self, filters: Optional[QueryFilters] = None) -> pd.DataFrame:
        """Daily token usage with cache efficiency metrics.

        Returns DataFrame with columns:
            date, daily_input_tokens, daily_output_tokens,
            daily_cache_read_tokens, cache_hit_ratio, token_efficiency
        """
        df = self.db.get_daily_trends(filters)

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])

        # Cache hit ratio per day
        total_input = df["daily_cache_read_tokens"] + df["daily_input_tokens"]
        df["cache_hit_ratio"] = np.where(
            total_input > 0,
            df["daily_cache_read_tokens"] / total_input * 100,
            0.0,
        )

        # Token efficiency per day (output / input)
        df["token_efficiency"] = np.where(
            df["daily_input_tokens"] > 0,
            df["daily_output_tokens"] / df["daily_input_tokens"],
            0.0,
        )

        return df

    # -------------------------------------------------------------------
    # 4. COST DISTRIBUTION — by model, practice, level
    # -------------------------------------------------------------------

    def get_cost_by_model(self, filters: Optional[QueryFilters] = None) -> pd.DataFrame:
        """Cost breakdown by model with percentage share.

        Returns DataFrame with columns:
            model, total_cost, request_count, avg_cost, cost_share_pct
        """
        df = self.db.get_cost_by_model(filters)

        if df.empty:
            return df

        total = df["total_cost"].sum()
        df["cost_share_pct"] = (df["total_cost"] / total * 100).round(1)

        return df

    def get_cost_by_practice(
        self, filters: Optional[QueryFilters] = None
    ) -> pd.DataFrame:
        """Cost breakdown by practice with percentage share.

        Returns DataFrame with columns:
            practice, total_cost, user_count, request_count,
            avg_cost_per_request, cost_share_pct
        """
        df = self.db.get_cost_by_practice(filters)

        if df.empty:
            return df

        total = df["total_cost"].sum()
        df["cost_share_pct"] = (df["total_cost"] / total * 100).round(1)

        return df

    def get_cost_by_level(
        self, filters: Optional[QueryFilters] = None
    ) -> pd.DataFrame:
        """Cost breakdown by seniority level with percentage share."""
        df = self.db.get_cost_by_level(filters)

        if df.empty:
            return df

        total = df["total_cost"].sum()
        df["cost_share_pct"] = (df["total_cost"] / total * 100).round(1)

        return df

    # -------------------------------------------------------------------
    # 5. TOOL PERFORMANCE — usage, success, duration
    # -------------------------------------------------------------------

    def get_tool_performance(
        self, filters: Optional[QueryFilters] = None
    ) -> pd.DataFrame:
        """Tool performance metrics with reliability ranking.

        Returns DataFrame with columns:
            tool_name, use_count, avg_duration_ms, success_rate,
            success_count, failure_count, usage_share_pct, reliability_rank
        """
        df = self.db.get_tool_usage(filters)

        if df.empty:
            return df

        total_uses = df["use_count"].sum()
        df["usage_share_pct"] = (df["use_count"] / total_uses * 100).round(1)

        # Reliability rank: combine success rate (weight 70%) and speed (weight 30%)
        # Normalize both to 0-1 range
        if len(df) > 1:
            # Higher success rate = better
            sr_norm = (df["success_rate"] - df["success_rate"].min()) / (
                df["success_rate"].max() - df["success_rate"].min() + 1e-9
            )
            # Lower duration = better (invert)
            dur_norm = 1 - (df["avg_duration_ms"] - df["avg_duration_ms"].min()) / (
                df["avg_duration_ms"].max() - df["avg_duration_ms"].min() + 1e-9
            )
            df["reliability_score"] = (0.7 * sr_norm + 0.3 * dur_norm).round(3)
        else:
            df["reliability_score"] = 1.0

        df = df.sort_values("reliability_score", ascending=False)
        df["reliability_rank"] = range(1, len(df) + 1)

        return df

    # -------------------------------------------------------------------
    # 6. SESSION ANALYSIS — duration, productivity, cost
    # -------------------------------------------------------------------

    def get_session_analysis(
        self, filters: Optional[QueryFilters] = None
    ) -> pd.DataFrame:
        """Enriched session statistics with productivity metrics.

        Returns DataFrame with all session_stats columns plus:
            cost_per_minute, tools_per_prompt, tool_success_rate
        """
        df = self.db.get_session_stats(filters)

        if df.empty:
            return df

        # Cost per minute of session
        df["cost_per_minute"] = np.where(
            df["duration_minutes"] > 0,
            df["total_cost"] / df["duration_minutes"],
            0.0,
        )

        # Tools used per prompt (how much automation per human interaction)
        df["tools_per_prompt"] = np.where(
            df["prompt_count"] > 0,
            df["tool_use_count"] / df["prompt_count"],
            0.0,
        )

        # Tool success rate per session
        df["tool_success_rate"] = np.where(
            df["tool_use_count"] > 0,
            df["tool_success_count"] / df["tool_use_count"] * 100,
            0.0,
        )

        return df

    # -------------------------------------------------------------------
    # 7. USER PERFORMANCE — rankings with computed metrics
    # -------------------------------------------------------------------

    def get_user_performance(
        self, filters: Optional[QueryFilters] = None
    ) -> pd.DataFrame:
        """User performance rankings with efficiency metrics.

        Returns DataFrame with user_rankings columns plus:
            cost_efficiency (output_tokens / cost), tokens_per_session
        """
        df = self.db.get_user_rankings(filters)

        if df.empty:
            return df

        # Cost efficiency: output tokens per dollar spent
        df["cost_efficiency"] = np.where(
            df["total_cost"] > 0,
            df["total_output_tokens"] / df["total_cost"],
            0.0,
        )

        # Tokens per session
        df["tokens_per_session"] = np.where(
            df["session_count"] > 0,
            (df["total_input_tokens"] + df["total_output_tokens"]) / df["session_count"],
            0,
        )

        return df

    # -------------------------------------------------------------------
    # 8. HOURLY HEATMAP — restructured for Plotly
    # -------------------------------------------------------------------

    def get_activity_heatmap(
        self, filters: Optional[QueryFilters] = None
    ) -> pd.DataFrame:
        """Pivot hourly data into heatmap format for Plotly.

        Returns DataFrame with:
            - Index: day names (Monday-Sunday)
            - Columns: hours (0-23)
            - Values: event_count
        """
        df = self.db.get_hourly_heatmap(filters)

        if df.empty:
            return df

        # Map day numbers to names
        day_names = {
            0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
            4: "Thursday", 5: "Friday", 6: "Saturday",
        }
        df["day_name"] = df["day_of_week"].map(day_names)

        # Pivot into heatmap format: rows=days, columns=hours
        heatmap = df.pivot_table(
            index="day_name",
            columns="hour",
            values="event_count",
            fill_value=0,
        )

        # Reorder days Monday → Sunday
        day_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        # Only include days that exist in the data
        day_order = [d for d in day_order if d in heatmap.index]
        heatmap = heatmap.reindex(day_order)

        return heatmap

    # -------------------------------------------------------------------
    # 9. ERROR ANALYSIS — enriched with derived metrics
    # -------------------------------------------------------------------

    def get_error_analysis(
        self, filters: Optional[QueryFilters] = None
    ) -> pd.DataFrame:
        """Error analysis with categorization.

        Returns DataFrame with error_breakdown columns plus:
            error_category, severity
        """
        df = self.db.get_error_breakdown(filters)

        if df.empty:
            return df

        # Categorize errors by status code
        def categorize_error(status_code: str) -> str:
            code = str(status_code)
            if code.startswith("4"):
                return "Client Error"
            elif code.startswith("5"):
                return "Server Error"
            elif code == "timeout":
                return "Timeout"
            else:
                return "Other"

        df["error_category"] = df["status_code"].apply(categorize_error)

        # Severity based on retry attempts
        df["severity"] = pd.cut(
            df["avg_retry_attempts"],
            bins=[0, 1, 2, float("inf")],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )

        return df

    # -------------------------------------------------------------------
    # 10. MODEL COMPARISON — multi-model performance metrics
    # -------------------------------------------------------------------

    def get_model_comparison(
        self, filters: Optional[QueryFilters] = None
    ) -> pd.DataFrame:
        """Compare model performance across multiple dimensions.

        Returns DataFrame with columns:
            model, total_cost, request_count, avg_cost, cost_share_pct,
            avg_tokens (computed from daily data)
        """
        cost_df = self.db.get_cost_by_model(filters)

        if cost_df.empty:
            return cost_df

        # Get model usage over time for trend data
        time_df = self.db.get_model_usage_over_time(filters)

        if not time_df.empty:
            # Add avg daily requests per model
            daily_avg = (
                time_df.groupby("model")["request_count"]
                .mean()
                .reset_index()
                .rename(columns={"request_count": "avg_daily_requests"})
            )
            cost_df = cost_df.merge(daily_avg, on="model", how="left")

        total_cost = cost_df["total_cost"].sum()
        cost_df["cost_share_pct"] = (cost_df["total_cost"] / total_cost * 100).round(1)

        return cost_df

    # -------------------------------------------------------------------
    # 11. WEEK-OVER-WEEK COMPARISON
    # -------------------------------------------------------------------

    def get_wow_comparison(self, filters: Optional[QueryFilters] = None) -> dict:
        """Calculate week-over-week changes for key metrics.

        Returns dict with:
            cost_change_pct, request_change_pct,
            current_week_cost, previous_week_cost,
            current_week_requests, previous_week_requests
        """
        df = self.db.get_daily_trends(filters)

        if df.empty or len(df) < 14:
            return {
                "cost_change_pct": 0.0,
                "request_change_pct": 0.0,
                "current_week_cost": 0.0,
                "previous_week_cost": 0.0,
                "current_week_requests": 0,
                "previous_week_requests": 0,
            }

        df["date"] = pd.to_datetime(df["date"])

        # Last 7 days vs previous 7 days
        current_week = df.tail(7)
        previous_week = df.iloc[-14:-7]

        cw_cost = current_week["daily_cost"].sum()
        pw_cost = previous_week["daily_cost"].sum()
        cw_reqs = current_week["daily_requests"].sum()
        pw_reqs = previous_week["daily_requests"].sum()

        return {
            "cost_change_pct": (
                ((cw_cost - pw_cost) / pw_cost * 100) if pw_cost > 0 else 0.0
            ),
            "request_change_pct": (
                ((cw_reqs - pw_reqs) / pw_reqs * 100) if pw_reqs > 0 else 0.0
            ),
            "current_week_cost": cw_cost,
            "previous_week_cost": pw_cost,
            "current_week_requests": int(cw_reqs),
            "previous_week_requests": int(pw_reqs),
        }

    # -------------------------------------------------------------------
    # 12. SUMMARY FOR AI — structured data for LLM context
    # -------------------------------------------------------------------

    def get_ai_context_summary(
        self, filters: Optional[QueryFilters] = None
    ) -> str:
        """Generate a text summary of key metrics for LLM context.

        This is fed into the AI chat feature's system prompt so the
        LLM can answer questions about the data without running SQL.

        Returns:
            Multi-line string summary of key metrics.
        """
        kpis = self.get_kpi_cards(filters)
        cost_model = self.db.get_cost_by_model(filters)
        cost_practice = self.db.get_cost_by_practice(filters)
        tools = self.db.get_tool_usage(filters)

        lines = [
            "=== Claude Code Analytics Summary ===",
            f"Total Cost: ${kpis['total_cost']:,.2f}",
            f"Total Sessions: {kpis['total_sessions']:,}",
            f"Total Users: {kpis['total_users']}",
            f"Total Events: {kpis['total_events']:,}",
            f"Error Rate: {kpis['error_rate_pct']:.1f}%",
            f"Cache Hit Ratio: {kpis['cache_hit_ratio']:.1f}%",
            f"Avg Cost/Session: ${kpis['avg_cost_per_session']:.2f}",
            "",
            "--- Cost by Model ---",
        ]

        if not cost_model.empty:
            for _, row in cost_model.iterrows():
                lines.append(
                    f"  {row['model']}: ${row['total_cost']:,.2f} "
                    f"({row['request_count']:,} requests)"
                )
        lines.append("")
        lines.append("--- Cost by Practice ---")

        if not cost_practice.empty:
            for _, row in cost_practice.iterrows():
                lines.append(
                    f"  {row['practice']}: ${row['total_cost']:,.2f} "
                    f"({row['user_count']} users)"
                )
        lines.append("")
        lines.append("--- Top 10 Tools by Usage ---")

        if not tools.empty:
            for _, row in tools.head(10).iterrows():
                lines.append(
                    f"  {row['tool_name']}: {row['use_count']:,} uses "
                    f"({row['success_rate']}% success)"
                )

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # PASSTHROUGH: filter options and safe query
    # -------------------------------------------------------------------

    def get_filter_options(self) -> dict:
        """Get filter dropdown options. Delegates to DatabaseManager."""
        return self.db.get_filter_options()

    def execute_safe_query(self, sql: str) -> pd.DataFrame:
        """Execute a read-only SQL query. Delegates to DatabaseManager."""
        return self.db.execute_safe_query(sql)
