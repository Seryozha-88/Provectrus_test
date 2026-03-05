"""
Streamlit Dashboard: Interactive analytics for Claude Code telemetry.

This is the main entry point for the web application.
Run with: streamlit run src/dashboard.py

PAGES:
    1. Overview      — KPI cards + daily activity timeline
    2. Cost & Tokens — Cost breakdowns, token analysis, cache efficiency
    3. Tool Usage    — Tool frequency, success rates, reliability
    4. User Behavior — Heatmap, user rankings, session analysis
    5. Errors        — Error types, status codes, retry analysis
    6. AI Insights   — Natural language → SQL → answer (OpenRouter LLM)
    7. ML Anomalies  — Anomaly detection, clustering, classification, forecasting

ARCHITECTURE:
    dashboard.py  →  AnalyticsEngine  →  DatabaseManager  →  SQLite
    (this file)      (analytics.py)      (database.py)       (analytics.db)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so imports work when running
# `streamlit run src/dashboard.py` from the project root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import AnalyticsEngine
from src.database import QueryFilters
from src.ai_insights import AIInsights, AVAILABLE_MODELS, DEFAULT_MODEL, EXAMPLE_QUESTIONS
from src.ml_anomaly import MLEngine

# ---------------------------------------------------------------------------
# PAGE CONFIG — must be the FIRST Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Claude Code Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CACHED ENGINE — reuse across reruns (Streamlit reruns the whole script
# on every interaction, so we cache the engine to avoid reconnecting)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_engine() -> AnalyticsEngine:
    """Create and cache the analytics engine (singleton)."""
    db_path = PROJECT_ROOT / "data" / "analytics.db"
    return AnalyticsEngine(str(db_path))


@st.cache_data(ttl=300)
def get_filter_options() -> dict:
    """Cache filter options for 5 minutes."""
    return get_engine().get_filter_options()


# ---------------------------------------------------------------------------
# SIDEBAR — global filters that affect all pages
# ---------------------------------------------------------------------------

def render_sidebar() -> QueryFilters:
    """Render the sidebar with filter controls and return selected filters."""
    st.sidebar.title("📊 Claude Code Analytics")
    st.sidebar.markdown("---")

    options = get_filter_options()

    # ---- Date range ----
    st.sidebar.subheader("📅 Date Range")
    min_date = pd.to_datetime(options["min_date"]).date()
    max_date = pd.to_datetime(options["max_date"]).date()

    col1, col2 = st.sidebar.columns(2)
    date_from = col1.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
    date_to = col2.date_input("To", value=max_date, min_value=min_date, max_value=max_date)

    # ---- Practice ----
    st.sidebar.subheader("🏢 Practice")
    practices = ["All"] + options["practices"]
    selected_practice = st.sidebar.selectbox("Team", practices, index=0)

    # ---- Level ----
    st.sidebar.subheader("📈 Seniority Level")
    levels = ["All"] + options["levels"]
    selected_level = st.sidebar.selectbox("Level", levels, index=0)

    # ---- Model ----
    st.sidebar.subheader("🤖 Model")
    models = ["All"] + options["models"]
    selected_model = st.sidebar.selectbox("Model", models, index=0)

    # ---- Build QueryFilters ----
    filters = QueryFilters(
        date_from=str(date_from),
        date_to=str(date_to),
        practice=selected_practice if selected_practice != "All" else None,
        level=selected_level if selected_level != "All" else None,
        model=selected_model if selected_model != "All" else None,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Data: {min_date} → {max_date}\n\n"
        f"Practices: {len(options['practices'])} | "
        f"Users: {len(options['users'])} | "
        f"Models: {len(options['models'])}"
    )

    return filters


# ===================================================================
# PAGE 1: OVERVIEW
# ===================================================================

def render_overview_page(engine: AnalyticsEngine, filters: QueryFilters) -> None:
    """Main overview page with KPI cards and daily activity."""
    st.header("📊 Overview Dashboard")

    # ---- KPI Cards ----
    kpis = engine.get_kpi_cards(filters)
    wow = engine.get_wow_comparison(filters)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric(
        "Total Cost",
        f"${kpis['total_cost']:,.2f}",
        delta=f"{wow['cost_change_pct']:+.1f}% WoW",
    )
    col2.metric(
        "Sessions",
        f"{int(kpis['total_sessions']):,}",
    )
    col3.metric(
        "Active Users",
        f"{int(kpis['total_users'])}",
    )
    col4.metric(
        "Avg Cost/Session",
        f"${kpis['avg_cost_per_session']:.2f}",
    )
    col5.metric(
        "Error Rate",
        f"{kpis['error_rate_pct']:.1f}%",
    )
    col6.metric(
        "Cache Hit Ratio",
        f"{kpis['cache_hit_ratio']:.1f}%",
    )

    st.markdown("---")

    # ---- Daily Activity Timeline ----
    st.subheader("📈 Daily Activity Timeline")

    trend = engine.get_cost_trend(filters)

    if trend.empty:
        st.warning("No data available for selected filters.")
        return

    # Dual-axis chart: cost (left) + requests (right)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["daily_cost"],
        name="Daily Cost ($)",
        line=dict(color="#636EFA", width=1),
        opacity=0.4,
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["cost_7d_avg"],
        name="7-Day Avg Cost ($)",
        line=dict(color="#636EFA", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["daily_requests"],
        name="Daily Requests",
        line=dict(color="#EF553B", width=1, dash="dot"),
        yaxis="y2",
        opacity=0.6,
    ))

    fig.update_layout(
        yaxis=dict(title="Cost ($)", side="left"),
        yaxis2=dict(title="Requests", side="right", overlaying="y"),
        hovermode="x unified",
        height=400,
        margin=dict(t=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Cumulative Cost ----
    st.subheader("💰 Cumulative Cost Burn")
    fig_cum = px.area(
        trend, x="date", y="cumulative_cost",
        labels={"cumulative_cost": "Cumulative Cost ($)", "date": "Date"},
        color_discrete_sequence=["#636EFA"],
    )
    fig_cum.update_layout(height=300, margin=dict(t=20))
    st.plotly_chart(fig_cum, use_container_width=True)

    # ---- Quick Stats Table ----
    st.subheader("📋 Quick Stats")
    stats_data = {
        "Metric": [
            "Total Events", "API Requests", "Tool Decisions",
            "Tool Results", "User Prompts", "API Errors",
            "Total Input Tokens", "Total Output Tokens",
            "Token Efficiency (out/in)", "Avg Tokens/Request",
        ],
        "Value": [
            f"{int(kpis['total_events']):,}",
            f"{int(kpis['total_api_requests']):,}",
            f"{kpis['total_tool_decisions']:,}",
            f"{kpis['total_tool_results']:,}",
            f"{kpis['total_user_prompts']:,}",
            f"{kpis['total_api_errors']:,}",
            f"{int(kpis['total_input_tokens']):,}",
            f"{int(kpis['total_output_tokens']):,}",
            f"{kpis['token_efficiency']:.2f}",
            f"{kpis['avg_tokens_per_request']:,}",
        ],
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)


# ===================================================================
# PAGE 2: COST & TOKENS
# ===================================================================

def render_cost_page(engine: AnalyticsEngine, filters: QueryFilters) -> None:
    """Cost analysis and token usage page."""
    st.header("💰 Cost & Token Analysis")

    # ---- Row 1: Cost by Model + Cost by Practice ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cost by Model")
        cost_model = engine.get_cost_by_model(filters)
        if not cost_model.empty:
            fig = px.pie(
                cost_model, values="total_cost", names="model",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=350, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data.")

    with col2:
        st.subheader("Cost by Practice")
        cost_practice = engine.get_cost_by_practice(filters)
        if not cost_practice.empty:
            fig = px.bar(
                cost_practice, x="total_cost", y="practice",
                orientation="h",
                text=cost_practice["total_cost"].apply(lambda x: f"${x:,.0f}"),
                color="practice",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_layout(
                height=350, margin=dict(t=20),
                showlegend=False, xaxis_title="Total Cost ($)",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data.")

    st.markdown("---")

    # ---- Row 2: Cost by Level + Model Over Time ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cost by Seniority Level")
        cost_level = engine.get_cost_by_level(filters)
        if not cost_level.empty:
            fig = px.bar(
                cost_level, x="level", y="avg_cost_per_user",
                text=cost_level["avg_cost_per_user"].apply(lambda x: f"${x:,.0f}"),
                color="level",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig.update_layout(
                height=350, margin=dict(t=20),
                showlegend=False,
                xaxis_title="Level", yaxis_title="Avg Cost per User ($)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data.")

    with col2:
        st.subheader("Model Usage Over Time")
        model_time = engine.db.get_model_usage_over_time(filters)
        if not model_time.empty:
            model_time["date"] = pd.to_datetime(model_time["date"])
            fig = px.line(
                model_time, x="date", y="total_cost", color="model",
                labels={"total_cost": "Daily Cost ($)", "date": "Date"},
            )
            fig.update_layout(height=350, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data.")

    st.markdown("---")

    # ---- Row 3: Token Analysis ----
    st.subheader("🔤 Token Usage Over Time")
    token_df = engine.get_token_analysis(filters)
    if not token_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Stacked area: input vs output vs cache tokens
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=token_df["date"], y=token_df["daily_cache_read_tokens"],
                name="Cache Read", stackgroup="one",
                line=dict(color="#00CC96"),
            ))
            fig.add_trace(go.Scatter(
                x=token_df["date"], y=token_df["daily_input_tokens"],
                name="Input", stackgroup="one",
                line=dict(color="#636EFA"),
            ))
            fig.add_trace(go.Scatter(
                x=token_df["date"], y=token_df["daily_output_tokens"],
                name="Output", stackgroup="one",
                line=dict(color="#EF553B"),
            ))
            fig.update_layout(
                height=350, margin=dict(t=20),
                yaxis_title="Tokens",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Cache hit ratio trend
            fig = px.line(
                token_df, x="date", y="cache_hit_ratio",
                labels={"cache_hit_ratio": "Cache Hit Ratio (%)", "date": "Date"},
                color_discrete_sequence=["#00CC96"],
            )
            fig.update_layout(height=350, margin=dict(t=20))
            fig.add_hline(
                y=token_df["cache_hit_ratio"].mean(),
                line_dash="dash", line_color="gray",
                annotation_text=f"Avg: {token_df['cache_hit_ratio'].mean():.1f}%",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data.")

    # ---- Cost details table ----
    st.subheader("📊 Model Cost Details")
    cost_model = engine.get_cost_by_model(filters)
    if not cost_model.empty:
        display_df = cost_model.copy()
        display_df["total_cost"] = display_df["total_cost"].apply(lambda x: f"${x:,.2f}")
        display_df["avg_cost"] = display_df["avg_cost"].apply(lambda x: f"${x:,.4f}")
        display_df.columns = ["Model", "Total Cost", "Requests", "Avg Cost/Request", "Share %"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ===================================================================
# PAGE 3: TOOL USAGE
# ===================================================================

def render_tools_page(engine: AnalyticsEngine, filters: QueryFilters) -> None:
    """Tool usage analysis page."""
    st.header("🔧 Tool Usage Analysis")

    tools = engine.get_tool_performance(filters)

    if tools.empty:
        st.warning("No tool data available for selected filters.")
        return

    # ---- Row 1: Tool Frequency + Success Rate ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tool Frequency (Top 17)")
        fig = px.bar(
            tools.head(17), x="use_count", y="tool_name",
            orientation="h",
            text="use_count",
            color="usage_share_pct",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            height=500, margin=dict(t=20),
            yaxis=dict(categoryorder="total ascending", title=""),
            xaxis_title="Usage Count",
            coloraxis_colorbar_title="Share %",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Success Rate by Tool")
        fig = px.bar(
            tools.head(17), x="success_rate", y="tool_name",
            orientation="h",
            text=tools.head(17)["success_rate"].apply(lambda x: f"{x}%"),
            color="success_rate",
            color_continuous_scale="RdYlGn",
            range_color=[80, 100],
        )
        fig.update_layout(
            height=500, margin=dict(t=20),
            yaxis=dict(categoryorder="total ascending", title=""),
            xaxis_title="Success Rate (%)",
            coloraxis_colorbar_title="%",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- Row 2: Duration + Reliability ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Avg Duration by Tool (ms)")
        fig = px.bar(
            tools.head(17), x="avg_duration_ms", y="tool_name",
            orientation="h",
            text=tools.head(17)["avg_duration_ms"].apply(lambda x: f"{x:,.0f}ms"),
            color="avg_duration_ms",
            color_continuous_scale="OrRd",
        )
        fig.update_layout(
            height=500, margin=dict(t=20),
            yaxis=dict(categoryorder="total ascending", title=""),
            xaxis_title="Avg Duration (ms)",
            xaxis_type="log",
            coloraxis_colorbar_title="ms",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏆 Tool Reliability Ranking")
        ranking = tools[["tool_name", "reliability_rank", "reliability_score",
                         "use_count", "success_rate", "avg_duration_ms"]].copy()
        ranking["avg_duration_ms"] = ranking["avg_duration_ms"].apply(lambda x: f"{x:,.0f}")
        ranking["reliability_score"] = ranking["reliability_score"].apply(lambda x: f"{x:.3f}")
        ranking["success_rate"] = ranking["success_rate"].apply(lambda x: f"{x}%")
        ranking.columns = ["Tool", "Rank", "Score", "Uses", "Success %", "Avg Duration (ms)"]
        st.dataframe(ranking, use_container_width=True, hide_index=True, height=480)

    st.markdown("---")

    # ---- Row 3: Decision Source Breakdown ----
    st.subheader("🎯 Tool Decision Sources")
    decisions = engine.db.get_tool_decisions_summary(filters)
    if not decisions.empty:
        # Aggregate by source
        source_agg = decisions.groupby("source")["count"].sum().reset_index()
        source_agg = source_agg.sort_values("count", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                source_agg, values="count", names="source",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=350, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Decision type breakdown per top 10 tools
            top_tools = decisions.groupby("tool_name")["count"].sum().nlargest(10).index
            top_decisions = decisions[decisions["tool_name"].isin(top_tools)]
            fig = px.bar(
                top_decisions, x="count", y="tool_name", color="decision",
                orientation="h",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                height=350, margin=dict(t=20),
                yaxis=dict(categoryorder="total ascending", title=""),
                xaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE 4: USER BEHAVIOR
# ===================================================================

def render_users_page(engine: AnalyticsEngine, filters: QueryFilters) -> None:
    """User behavior and activity patterns page."""
    st.header("👥 User Behavior Analysis")

    # ---- Row 1: Activity Heatmap ----
    st.subheader("🕐 Activity Heatmap (Hour × Day)")
    heatmap = engine.get_activity_heatmap(filters)

    if not heatmap.empty:
        fig = px.imshow(
            heatmap,
            labels=dict(x="Hour of Day", y="Day of Week", color="Events"),
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        fig.update_layout(height=300, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No heatmap data.")

    st.markdown("---")

    # ---- Row 2: User Rankings + Session Distribution ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏅 Top 15 Users by Cost")
        users = engine.get_user_performance(filters)
        if not users.empty:
            top15 = users.head(15)
            fig = px.bar(
                top15, x="total_cost", y="full_name",
                orientation="h",
                text=top15["total_cost"].apply(lambda x: f"${x:,.0f}"),
                color="practice",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                height=500, margin=dict(t=20),
                yaxis=dict(categoryorder="total ascending", title=""),
                xaxis_title="Total Cost ($)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user data.")

    with col2:
        st.subheader("📊 Sessions per User Distribution")
        users = engine.get_user_performance(filters)
        if not users.empty:
            fig = px.histogram(
                users, x="session_count", nbins=20,
                labels={"session_count": "Sessions per User", "count": "Number of Users"},
                color_discrete_sequence=["#636EFA"],
            )
            fig.update_layout(
                height=500, margin=dict(t=20),
                yaxis_title="Number of Users",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user data.")

    st.markdown("---")

    # ---- Row 3: Cost Efficiency + Practice Comparison ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚡ Cost Efficiency (Tokens per $)")
        users = engine.get_user_performance(filters)
        if not users.empty:
            top20 = users.nlargest(20, "cost_efficiency")
            fig = px.bar(
                top20, x="cost_efficiency", y="full_name",
                orientation="h",
                text=top20["cost_efficiency"].apply(lambda x: f"{x:,.0f}"),
                color="level",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig.update_layout(
                height=500, margin=dict(t=20),
                yaxis=dict(categoryorder="total ascending", title=""),
                xaxis_title="Output Tokens per Dollar",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏢 Practice Comparison")
        cost_practice = engine.get_cost_by_practice(filters)
        if not cost_practice.empty:
            fig = px.bar(
                cost_practice, x="practice", y=["total_cost", "request_count"],
                barmode="group",
                labels={"value": "Value", "variable": "Metric"},
                color_discrete_sequence=["#636EFA", "#EF553B"],
            )
            fig.update_layout(height=500, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- Row 4: Full User Table ----
    st.subheader("📋 Full User Rankings Table")
    users = engine.get_user_performance(filters)
    if not users.empty:
        display = users[[
            "full_name", "practice", "level", "total_cost",
            "session_count", "request_count", "avg_cost_per_session",
            "cost_efficiency",
        ]].copy()
        display["total_cost"] = display["total_cost"].apply(lambda x: f"${x:,.2f}")
        display["avg_cost_per_session"] = display["avg_cost_per_session"].apply(
            lambda x: f"${x:,.2f}"
        )
        display["cost_efficiency"] = display["cost_efficiency"].apply(
            lambda x: f"{x:,.0f}"
        )
        display.columns = [
            "Name", "Practice", "Level", "Total Cost",
            "Sessions", "Requests", "Avg $/Session", "Tokens/$",
        ]
        st.dataframe(display, use_container_width=True, hide_index=True)


# ===================================================================
# PAGE 5: ERRORS
# ===================================================================

def render_errors_page(engine: AnalyticsEngine, filters: QueryFilters) -> None:
    """Error analysis page."""
    st.header("⚠️ Error Analysis")

    errors = engine.get_error_analysis(filters)

    if errors.empty:
        st.success("🎉 No errors found for the selected filters!")
        return

    # ---- KPI Cards ----
    kpis = engine.get_kpi_cards(filters)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Errors", f"{kpis['total_api_errors']:,}")
    col2.metric("Error Rate", f"{kpis['error_rate_pct']:.1f}%")
    col3.metric(
        "Unique Error Types",
        f"{errors['error'].nunique()}",
    )
    col4.metric(
        "Avg Retry Attempts",
        f"{errors['avg_retry_attempts'].mean():.1f}",
    )

    st.markdown("---")

    # ---- Row 1: Error Type Donut + Error Category ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Error Types")
        error_type_agg = errors.groupby("error")["count"].sum().reset_index()
        error_type_agg = error_type_agg.sort_values("count", ascending=False)
        # Truncate long error names for display
        error_type_agg["error_short"] = error_type_agg["error"].apply(
            lambda x: x[:50] + "..." if len(x) > 50 else x
        )
        fig = px.pie(
            error_type_agg, values="count", names="error_short",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        fig.update_layout(height=400, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Error Categories")
        cat_agg = errors.groupby("error_category")["count"].sum().reset_index()
        fig = px.bar(
            cat_agg, x="error_category", y="count",
            text="count",
            color="error_category",
            color_discrete_map={
                "Client Error": "#FFA15A",
                "Server Error": "#EF553B",
                "Timeout": "#AB63FA",
                "Other": "#B6B6B6",
            },
        )
        fig.update_layout(
            height=400, margin=dict(t=20),
            showlegend=False,
            xaxis_title="", yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- Row 2: Status Codes + Errors by Model ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Status Code Distribution")
        status_agg = errors.groupby("status_code")["count"].sum().reset_index()
        status_agg = status_agg.sort_values("count", ascending=False)
        fig = px.bar(
            status_agg, x="status_code", y="count",
            text="count",
            color="status_code",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(
            height=400, margin=dict(t=20),
            showlegend=False,
            xaxis_title="HTTP Status Code", yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Errors by Model")
        model_agg = errors.groupby(["model", "error_category"])["count"].sum().reset_index()
        fig = px.bar(
            model_agg, x="model", y="count", color="error_category",
            barmode="stack",
            color_discrete_map={
                "Client Error": "#FFA15A",
                "Server Error": "#EF553B",
                "Timeout": "#AB63FA",
                "Other": "#B6B6B6",
            },
        )
        fig.update_layout(
            height=400, margin=dict(t=20),
            xaxis_title="Model", yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- Error Details Table ----
    st.subheader("📋 Error Details")
    display = errors[["error", "status_code", "model", "count",
                       "error_category", "severity", "avg_retry_attempts"]].copy()
    display["avg_retry_attempts"] = display["avg_retry_attempts"].apply(
        lambda x: f"{x:.1f}"
    )
    display.columns = [
        "Error Message", "Status", "Model", "Count",
        "Category", "Severity", "Avg Retries",
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)


# ===================================================================
# PAGE 6: AI INSIGHTS
# ===================================================================

def render_ai_page() -> None:
    """AI-powered natural language query interface.

    Users type a question in English → LLM generates SQL →
    executes against database → LLM interprets results →
    shows answer + table + SQL used.
    """
    st.header("🤖 AI Insights")
    st.caption(
        "Ask questions about the telemetry data in plain English. "
        "The AI will generate SQL, run it, and explain the results."
    )

    # ---- Sidebar: API Key + Model selection ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 AI Settings")

    api_key = st.sidebar.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-v1-...",
        help="Get your key at https://openrouter.ai/keys",
    )

    selected_model = st.sidebar.selectbox(
        "LLM Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
        help="Model used for SQL generation and interpretation",
    )

    if not api_key:
        st.warning(
            "⚠️ Please enter your **OpenRouter API key** in the sidebar to use AI Insights.\n\n"
            "Get one free at [openrouter.ai/keys](https://openrouter.ai/keys)"
        )
        # Show example questions even without API key
        st.markdown("### 💡 Example Questions")
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            st.markdown(f"{i+1}. {q}")
        return

    # ---- Initialize AI engine ----
    db_path = str(PROJECT_ROOT / "data" / "analytics.db")
    ai = AIInsights(api_key=api_key, db_path=db_path, model=selected_model)

    # ---- Chat interface ----
    # Initialize chat history in session state
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []

    # Display chat history
    for msg in st.session_state.ai_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sql" in msg:
                with st.expander("📝 SQL Query Used"):
                    st.code(msg["sql"], language="sql")
            if "data" in msg and msg["data"] is not None and len(msg["data"]) > 0:
                with st.expander(f"📊 Results Table ({len(msg['data'])} rows)"):
                    st.dataframe(msg["data"], use_container_width=True, hide_index=True)

    # ---- Quick question buttons ----
    if not st.session_state.ai_messages:
        st.markdown("### 💡 Try a question:")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS[:6]):
            col = cols[i % 2]
            if col.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.ai_pending_question = q
                st.rerun()

    # ---- Handle pending question from button click ----
    pending = st.session_state.pop("ai_pending_question", None)

    # ---- Chat input ----
    user_input = st.chat_input("Ask a question about the telemetry data...")
    question = pending or user_input

    if question:
        # Display user message
        st.session_state.ai_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking... (using {selected_model.split('/')[-1]})"):
                result = ai.ask(question)

            if result["error"]:
                st.error(f"❌ {result['error']}")
                assistant_msg = {
                    "role": "assistant",
                    "content": f"❌ {result['error']}",
                }
            else:
                # Show answer
                st.markdown(result["answer"])

                # Show SQL in expander
                if result["sql"]:
                    with st.expander("📝 SQL Query Used"):
                        st.code(result["sql"], language="sql")

                # Show results table in expander
                if not result["data"].empty:
                    with st.expander(f"📊 Results Table ({len(result['data'])} rows)"):
                        st.dataframe(
                            result["data"],
                            use_container_width=True,
                            hide_index=True,
                        )

                assistant_msg = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sql": result["sql"],
                    "data": result["data"],
                }

            st.session_state.ai_messages.append(assistant_msg)

    # ---- Clear chat button ----
    if st.session_state.ai_messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.ai_messages = []
            st.rerun()


# ===================================================================
# PAGE 7: ML & ANOMALY DETECTION
# ===================================================================

@st.cache_resource
def get_ml_engine() -> MLEngine:
    """Create and cache the ML engine (singleton, trains models once)."""
    return MLEngine(str(PROJECT_ROOT / "data" / "analytics.db"))


def render_ml_page() -> None:
    """ML & Anomaly Detection page with 4 model sections."""
    st.header("🔬 ML & Anomaly Detection")
    st.caption(
        "Four scikit-learn models trained on session-level features "
        "(~5,000 sessions × 12 features). All CPU-only, trains in < 3 seconds."
    )

    ml = get_ml_engine()

    # ---- Sidebar ML parameters ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 ML Parameters")
    contamination = st.sidebar.slider(
        "Anomaly contamination",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01,
        help="Expected fraction of anomalous sessions (IsolationForest)",
    )
    n_clusters = st.sidebar.slider(
        "Number of clusters",
        min_value=2, max_value=8, value=4, step=1,
        help="KMeans cluster count",
    )
    max_depth = st.sidebar.slider(
        "Decision tree depth",
        min_value=2, max_value=10, value=5, step=1,
        help="Max depth for DecisionTree classifier",
    )
    days_ahead = st.sidebar.slider(
        "Forecast horizon (days)",
        min_value=7, max_value=90, value=30, step=7,
        help="Number of days to forecast ahead",
    )

    # ---- Run all models (cached by Streamlit on param change) ----
    with st.spinner("Training ML models..."):
        anomalies = ml.detect_anomalies(contamination)
        classification = ml.classify_practice(max_depth)
        clusters = ml.cluster_sessions(n_clusters)
        forecast = ml.forecast_cost(days_ahead)

    # ---- Tabs for each model ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Anomaly Detection",
        "🌳 Practice Classification",
        "🔵 Session Clustering",
        "📈 Cost Forecast",
    ])

    # ==== TAB 1: Anomaly Detection ====
    with tab1:
        _render_anomaly_tab(anomalies)

    # ==== TAB 2: Practice Classification ====
    with tab2:
        _render_classification_tab(classification)

    # ==== TAB 3: Session Clustering ====
    with tab3:
        _render_cluster_tab(clusters)

    # ==== TAB 4: Cost Forecast ====
    with tab4:
        _render_forecast_tab(forecast)


# ---- Anomaly Tab ----
def _render_anomaly_tab(anom) -> None:
    """Render anomaly detection results."""
    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sessions", f"{anom.total_sessions:,}")
    c2.metric("Anomalies Found", f"{anom.anomaly_count:,}")
    c3.metric("Anomaly Rate", f"{anom.anomaly_count / anom.total_sessions:.1%}")
    c4.metric("Contamination", f"{anom.contamination:.0%}")

    st.markdown("---")

    # PCA scatter with anomalies highlighted
    st.subheader("Session Map (PCA 2D Projection)")
    df = anom.features.copy()
    # Quick PCA for visualization
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X = df[[
        "api_call_count", "total_cost", "total_input_tokens",
        "total_output_tokens", "total_cache_read_tokens", "prompt_count",
        "tool_use_count", "tool_success_count", "duration_minutes",
        "cost_per_api_call", "tokens_per_api_call", "tool_success_rate",
    ]].values
    X_scaled = StandardScaler().fit_transform(X)
    pca_coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    df["PC1"] = pca_coords[:, 0]
    df["PC2"] = pca_coords[:, 1]
    df["Status"] = df["is_anomaly"].map({True: "Anomaly", False: "Normal"})

    fig = px.scatter(
        df, x="PC1", y="PC2",
        color="Status",
        color_discrete_map={"Normal": "#636EFA", "Anomaly": "#EF553B"},
        hover_data=["session_id", "practice", "total_cost", "api_call_count"],
        title="Sessions in 2D (PCA) — Anomalies in Red",
        opacity=0.6,
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly score distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Anomaly Score Distribution")
        fig_hist = px.histogram(
            df, x="anomaly_score", color="Status",
            color_discrete_map={"Normal": "#636EFA", "Anomaly": "#EF553B"},
            nbins=50,
            title="Distribution of Anomaly Scores (lower = more anomalous)",
            barmode="overlay",
            opacity=0.7,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Anomalies by Practice")
        anom_by_practice = (
            df[df["is_anomaly"]]
            .groupby("practice")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        if not anom_by_practice.empty:
            fig_bar = px.bar(
                anom_by_practice, x="practice", y="count",
                color="practice",
                title="Anomalous Sessions per Team",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No anomalies detected.")

    # Top anomalies table
    st.subheader("Top 20 Most Anomalous Sessions")
    display_cols = [
        "session_id", "practice", "level", "total_cost",
        "api_call_count", "tool_use_count", "duration_minutes",
        "anomaly_score",
    ]
    available_cols = [c for c in display_cols if c in anom.top_anomalies.columns]
    if not anom.top_anomalies.empty:
        st.dataframe(
            anom.top_anomalies[available_cols].reset_index(drop=True),
            use_container_width=True,
            height=400,
        )
    else:
        st.info("No anomalies to display.")


# ---- Classification Tab ----
def _render_classification_tab(clf) -> None:
    """Render practice classification results."""
    # KPI row
    c1, c2, c3 = st.columns(3)
    c1.metric("CV Accuracy (5-fold)", f"{clf.accuracy:.1%}")
    c2.metric("Classes", f"{len(clf.classes)}")
    c3.metric(
        "Best / Worst Fold",
        f"{clf.cv_scores.max():.1%} / {clf.cv_scores.min():.1%}",
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance")
        fig_imp = px.bar(
            clf.feature_importances,
            x="importance", y="feature",
            orientation="h",
            title="Which features distinguish teams?",
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig_imp.update_layout(height=450, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        st.subheader("Cross-Validation Scores")
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(clf.cv_scores))],
            "Accuracy": clf.cv_scores,
        })
        fig_cv = px.bar(
            cv_df, x="Fold", y="Accuracy",
            title="Accuracy per CV Fold",
            text=cv_df["Accuracy"].apply(lambda x: f"{x:.1%}"),
            color="Accuracy",
            color_continuous_scale="RdYlGn",
        )
        fig_cv.update_layout(height=450)
        st.plotly_chart(fig_cv, use_container_width=True)

    # Classification report
    st.subheader("Classification Report")
    st.code(clf.class_report, language="text")

    # Decision tree rules (collapsible)
    with st.expander("🌳 Decision Tree Rules (click to expand)"):
        st.code(clf.tree_rules, language="text")


# ---- Clustering Tab ----
def _render_cluster_tab(clust) -> None:
    """Render session clustering results."""
    # KPI row
    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters", f"{clust.n_clusters}")
    c2.metric("Sessions", f"{sum(clust.cluster_sizes.values()):,}")
    c3.metric("Inertia", f"{clust.inertia:,.0f}")

    st.markdown("---")

    # PCA scatter colored by cluster
    st.subheader("Session Clusters (PCA 2D)")
    pca_df = clust.pca_2d.copy()
    pca_df["cluster"] = pca_df["cluster"].astype(str)
    fig_scatter = px.scatter(
        pca_df, x="PC1", y="PC2",
        color="cluster",
        hover_data=["session_id", "practice", "total_cost"],
        title="Sessions Projected to 2D — Colored by Cluster",
        opacity=0.6,
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cluster Sizes")
        sizes_df = pd.DataFrame([
            {"Cluster": f"Cluster {k}", "Sessions": v}
            for k, v in sorted(clust.cluster_sizes.items())
        ])
        fig_sizes = px.pie(
            sizes_df, values="Sessions", names="Cluster",
            title="Sessions per Cluster",
        )
        st.plotly_chart(fig_sizes, use_container_width=True)

    with col2:
        st.subheader("Cluster × Practice")
        # Cross-tabulation of cluster vs practice
        cross = pd.crosstab(
            pca_df["cluster"], pca_df["practice"],
        ).reset_index()
        cross_melted = cross.melt(
            id_vars="cluster", var_name="practice", value_name="count"
        )
        fig_cross = px.bar(
            cross_melted, x="cluster", y="count", color="practice",
            title="Practice Distribution per Cluster",
            barmode="group",
        )
        st.plotly_chart(fig_cross, use_container_width=True)

    # Cluster profiles heatmap
    st.subheader("Cluster Profiles (Mean Feature Values)")
    profiles = clust.cluster_profiles.set_index("cluster")
    # Normalize for heatmap (0-1 scale per feature)
    profiles_norm = (profiles - profiles.min()) / (profiles.max() - profiles.min() + 1e-9)
    fig_heat = px.imshow(
        profiles_norm.T,
        labels=dict(x="Cluster", y="Feature", color="Normalized Value"),
        x=[f"Cluster {i}" for i in profiles.index],
        y=profiles.columns.tolist(),
        color_continuous_scale="YlOrRd",
        title="Cluster Profiles (normalized 0–1)",
        aspect="auto",
    )
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Raw profiles table
    with st.expander("📊 Raw Cluster Profiles (click to expand)"):
        st.dataframe(clust.cluster_profiles, use_container_width=True)


# ---- Forecast Tab ----
def _render_forecast_tab(fc) -> None:
    """Render cost forecast results."""
    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{fc.r2_score:.4f}")
    c2.metric("Trend", f"{fc.slope_per_day:+.4f} $/day")
    c3.metric("Avg Daily Cost", f"${fc.avg_daily_cost:.2f}")
    if not fc.forecast.empty:
        c4.metric(
            f"{len(fc.forecast)}-Day Forecast Total",
            f"${fc.forecast['predicted_cost'].sum():,.2f}",
        )
    else:
        c4.metric("Forecast", "N/A")

    st.markdown("---")

    if fc.historical.empty:
        st.warning("No historical data available for forecasting.")
        return

    # Combined chart: actual + fitted + forecast
    st.subheader("Daily Cost: Actual vs Forecast")

    fig = go.Figure()

    # Actual daily costs
    fig.add_trace(go.Scatter(
        x=fc.historical["date"],
        y=fc.historical["daily_cost"],
        mode="lines",
        name="Actual Cost",
        line=dict(color="#636EFA", width=1.5),
        opacity=0.7,
    ))

    # Fitted trend line
    fig.add_trace(go.Scatter(
        x=fc.historical["date"],
        y=fc.historical["fitted_cost"],
        mode="lines",
        name="Trend (fitted)",
        line=dict(color="#00CC96", width=2, dash="dash"),
    ))

    # Forecast line
    if not fc.forecast.empty:
        fig.add_trace(go.Scatter(
            x=fc.forecast["date"],
            y=fc.forecast["predicted_cost"],
            mode="lines",
            name="Forecast",
            line=dict(color="#EF553B", width=2, dash="dot"),
        ))

        # Shade forecast region
        fig.add_vrect(
            x0=fc.forecast["date"].iloc[0],
            x1=fc.forecast["date"].iloc[-1],
            fillcolor="rgba(239, 85, 59, 0.08)",
            line_width=0,
            annotation_text="Forecast",
            annotation_position="top left",
        )

    fig.update_layout(
        title="Historical Daily Cost + Linear Forecast",
        xaxis_title="Date",
        yaxis_title="Daily Cost ($)",
        height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    direction = "increasing" if fc.slope_per_day > 0 else "decreasing"
    st.info(
        f"**Trend interpretation:** Daily cost is **{direction}** at "
        f"**${abs(fc.slope_per_day):.4f}/day** "
        f"(R² = {fc.r2_score:.4f}). "
        f"{'A low R² means costs are volatile — the linear trend captures only a small portion of variance.' if fc.r2_score < 0.3 else 'The trend is a reasonable fit to the data.'}"
    )

    # Forecast table (collapsible)
    with st.expander("📋 Forecast Table (click to expand)"):
        st.dataframe(
            fc.forecast.assign(
                predicted_cost=fc.forecast["predicted_cost"].round(2)
            )[['date', 'predicted_cost']],
            use_container_width=True,
            height=300,
        )


# ===================================================================
# MAIN: Navigation + Page Routing
# ===================================================================

def main() -> None:
    """Main application entry point with page navigation."""
    engine = get_engine()
    filters = render_sidebar()

    # ---- Page navigation ----
    pages = {
        "📊 Overview": "overview",
        "💰 Cost & Tokens": "cost",
        "🔧 Tool Usage": "tools",
        "👥 User Behavior": "users",
        "⚠️ Errors": "errors",
        "🤖 AI Insights": "ai",
        "🔬 ML Anomalies": "ml",
    }

    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Pages")
    selected_page = st.sidebar.radio(
        "Navigate to",
        list(pages.keys()),
        label_visibility="collapsed",
    )

    page_key = pages[selected_page]

    # ---- Route to the selected page ----
    if page_key == "overview":
        render_overview_page(engine, filters)
    elif page_key == "cost":
        render_cost_page(engine, filters)
    elif page_key == "tools":
        render_tools_page(engine, filters)
    elif page_key == "users":
        render_users_page(engine, filters)
    elif page_key == "errors":
        render_errors_page(engine, filters)
    elif page_key == "ai":
        render_ai_page()
    elif page_key == "ml":
        render_ml_page()


if __name__ == "__main__":
    main()
