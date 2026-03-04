"""
ETL Pipeline: Extract, Transform, Load telemetry data.

This module reads raw data files (JSONL + CSV), parses them through
our Pydantic models, and loads the clean data into a SQLite database.

PIPELINE OVERVIEW:
    ┌────────────────────┐
    │  1. EXTRACT        │  Read raw files from disk
    │  - JSONL (521 MB)  │  Line-by-line streaming (memory-efficient)
    │  - CSV (100 rows)  │  All at once (tiny file)
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │  2. TRANSFORM      │  Parse + validate + type-cast
    │  - json.loads()x2  │  First for batch, then for message
    │  - parse_event()   │  Route to correct Pydantic model
    │  - Skip bad records│  Log errors, don't crash
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │  3. LOAD           │  Insert into SQLite database
    │  - 5 event tables  │  One table per event type
    │  - 1 employee table│  From CSV
    └────────────────────┘

WHY STREAM LINE-BY-LINE?
    The JSONL file is ~521 MB. Loading it all into memory would use >1 GB.
    Instead, we read one line at a time, parse it, extract events, and
    discard the raw line. This keeps memory usage constant (~50 MB).

THE DOUBLE json.loads() TRICK:
    Each JSONL line is a batch with logEvents[]. Each logEvent has a
    "message" field that is ITSELF a JSON string. So we need two levels
    of parsing:
        1. json.loads(line)           → batch dict with logEvents[]
        2. json.loads(event.message)  → the actual event data

USAGE:
    # From command line:
    python -m src.ingest --data-dir data --db-path data/analytics.db

    # From Python:
    from src.ingest import run_pipeline
    run_pipeline("data", "data/analytics.db")
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

from src.models import (
    ApiErrorEvent,
    ApiRequestEvent,
    Employee,
    ToolDecisionEvent,
    ToolResultEvent,
    UserPromptEvent,
    parse_event,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# We use Python's built-in logging to track progress and errors.
# This is better than print() because:
# - It includes timestamps
# - Errors go to stderr, info goes to stdout
# - It can be easily redirected to a file
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STEP 1: EXTRACT — Parse employees.csv
# ---------------------------------------------------------------------------
# This is the simple part. The CSV has 5 columns, ~100 rows.
# We read each row, validate it with our Employee Pydantic model,
# and return a list.
#
# Why validate with Pydantic?
# - Ensures all 5 fields are present and are strings
# - If the CSV has a bad row, we catch it here instead of during DB insert
# ---------------------------------------------------------------------------

def parse_employees(csv_path: str) -> list[Employee]:
    """Parse employees.csv into validated Employee models.

    Args:
        csv_path: Path to employees.csv file.

    Returns:
        List of Employee objects, one per CSV row.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Employee CSV not found: {csv_path}")

    employees: list[Employee] = []
    errors = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=1):
            try:
                emp = Employee.model_validate(row)
                employees.append(emp)
            except Exception as e:
                errors += 1
                logger.warning(f"Bad employee row {row_num}: {e}")

    logger.info(f"Parsed {len(employees)} employees ({errors} errors)")
    return employees


# ---------------------------------------------------------------------------
# STEP 2: EXTRACT + TRANSFORM — Parse telemetry_logs.jsonl
# ---------------------------------------------------------------------------
# This is the complex part. The file is ~521 MB with ~82,000 lines.
#
# Each line is a JSON batch containing 1-10 logEvents.
# Each logEvent.message is ITSELF a JSON string that must be parsed again.
#
# We stream line-by-line to keep memory low, and use parse_event()
# from models.py to route each event to the correct Pydantic model.
#
# The output is a dict with 5 lists — one per event type:
#   {
#       "api_requests": [ApiRequestEvent, ...],       ~118,000 events
#       "tool_decisions": [ToolDecisionEvent, ...],   ~151,000 events
#       "tool_results": [ToolResultEvent, ...],       ~148,000 events
#       "user_prompts": [UserPromptEvent, ...],       ~35,000 events
#       "api_errors": [ApiErrorEvent, ...],           ~1,400 events
#   }
# ---------------------------------------------------------------------------

def parse_telemetry(jsonl_path: str) -> dict[str, list]:
    """Parse telemetry_logs.jsonl into categorized event lists.

    Reads the file line by line (streaming), parses the double-nested
    JSON structure, validates with Pydantic, and sorts events by type.

    Args:
        jsonl_path: Path to telemetry_logs.jsonl file.

    Returns:
        Dict mapping event type name → list of Pydantic model instances:
            "api_requests"   → list[ApiRequestEvent]
            "tool_decisions" → list[ToolDecisionEvent]
            "tool_results"   → list[ToolResultEvent]
            "user_prompts"   → list[UserPromptEvent]
            "api_errors"     → list[ApiErrorEvent]

    Raises:
        FileNotFoundError: If the JSONL file doesn't exist.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry JSONL not found: {jsonl_path}")

    # Initialize buckets for each event type
    events: dict[str, list] = {
        "api_requests": [],
        "tool_decisions": [],
        "tool_results": [],
        "user_prompts": [],
        "api_errors": [],
    }

    # Map Pydantic model class → bucket name
    type_to_key = {
        ApiRequestEvent: "api_requests",
        ToolDecisionEvent: "tool_decisions",
        ToolResultEvent: "tool_results",
        UserPromptEvent: "user_prompts",
        ApiErrorEvent: "api_errors",
    }

    # Counters for logging
    total_batches = 0
    total_events = 0
    parse_errors = 0
    start_time = time.time()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            # ----- Level 1: Parse the JSONL line into a batch -----
            try:
                batch = json.loads(line)
            except json.JSONDecodeError as e:
                parse_errors += 1
                logger.warning(f"Bad JSON at line {line_num}: {e}")
                continue

            total_batches += 1
            log_events = batch.get("logEvents", [])

            # ----- Level 2: Parse each logEvent.message -----
            for log_event in log_events:
                message_str = log_event.get("message", "")

                try:
                    raw = json.loads(message_str)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue

                # ----- Level 3: Validate with Pydantic model -----
                event = parse_event(raw)

                if event is None:
                    parse_errors += 1
                    continue

                # Route to the correct bucket
                key = type_to_key.get(type(event))
                if key:
                    events[key].append(event)
                    total_events += 1

            # Progress logging every 10,000 batches
            if line_num % 10000 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"  Processed {line_num:,} batches, "
                    f"{total_events:,} events ({elapsed:.1f}s)"
                )

    elapsed = time.time() - start_time
    logger.info(
        f"Parsing complete: {total_batches:,} batches, "
        f"{total_events:,} events, {parse_errors} errors "
        f"in {elapsed:.1f}s"
    )
    for key, lst in events.items():
        logger.info(f"  {key}: {len(lst):,}")

    return events


# ---------------------------------------------------------------------------
# STEP 3: LOAD — Insert into SQLite database
# ---------------------------------------------------------------------------
# We create 6 tables (1 employee + 5 event types) and bulk-insert
# the parsed data using executemany() for speed.
#
# WHY SQLite?
# - Zero setup (comes with Python)
# - Single file (easy to share/deploy)
# - Fast enough for ~450K rows
# - SQL queries for our analytics layer
#
# WHY executemany()?
# - Inserting one row at a time would take minutes
# - executemany() batches inserts into transactions → ~10x faster
# ---------------------------------------------------------------------------

def create_tables(conn: sqlite3.Connection) -> None:
    """Create all database tables, indexes, and views.

    Drops existing tables first (idempotent — safe to re-run).
    """
    cursor = conn.cursor()

    # Drop existing tables for clean re-load
    for table in [
        "api_requests", "tool_decisions", "tool_results",
        "user_prompts", "api_errors", "employees",
    ]:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    cursor.execute("DROP VIEW IF EXISTS session_summary")

    # ---- Employees table ----
    cursor.execute("""
        CREATE TABLE employees (
            email TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            practice TEXT NOT NULL,
            level TEXT NOT NULL,
            location TEXT NOT NULL
        )
    """)

    # ---- API Requests table ----
    cursor.execute("""
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
        )
    """)

    # ---- Tool Decisions table ----
    cursor.execute("""
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
        )
    """)

    # ---- Tool Results table ----
    cursor.execute("""
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
        )
    """)

    # ---- User Prompts table ----
    cursor.execute("""
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
        )
    """)

    # ---- API Errors table ----
    cursor.execute("""
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
        )
    """)

    # ---- Indexes for query performance ----
    # These speed up the most common queries in our analytics layer
    cursor.execute("CREATE INDEX idx_api_requests_session ON api_requests(session_id)")
    cursor.execute("CREATE INDEX idx_api_requests_user ON api_requests(user_email)")
    cursor.execute("CREATE INDEX idx_api_requests_timestamp ON api_requests(timestamp)")
    cursor.execute("CREATE INDEX idx_api_requests_model ON api_requests(model)")
    cursor.execute("CREATE INDEX idx_tool_decisions_session ON tool_decisions(session_id)")
    cursor.execute("CREATE INDEX idx_tool_results_tool ON tool_results(tool_name)")
    cursor.execute("CREATE INDEX idx_tool_results_session ON tool_results(session_id)")
    cursor.execute("CREATE INDEX idx_user_prompts_session ON user_prompts(session_id)")
    cursor.execute("CREATE INDEX idx_api_errors_session ON api_errors(session_id)")

    # ---- Session summary view ----
    # Pre-built SQL view that aggregates api_requests by session
    # Joins with employees to get practice/level/location
    cursor.execute("""
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
        GROUP BY ar.session_id
    """)

    conn.commit()
    logger.info("Database tables, indexes, and views created")


def load_employees(conn: sqlite3.Connection, employees: list[Employee]) -> None:
    """Bulk insert employees into the database."""
    rows = [
        (e.email, e.full_name, e.practice, e.level, e.location)
        for e in employees
    ]
    conn.executemany(
        "INSERT INTO employees (email, full_name, practice, level, location) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info(f"Loaded {len(rows)} employees")


def load_api_requests(conn: sqlite3.Connection, events: list[ApiRequestEvent]) -> None:
    """Bulk insert API request events into the database."""
    rows = [
        (
            e.session_id, e.user_email, e.timestamp, e.model,
            e.cost_usd, e.duration_ms, e.input_tokens, e.output_tokens,
            e.cache_read_tokens, e.cache_creation_tokens,
            e.terminal_type, e.org_id, e.scope.version,
        )
        for e in events
    ]
    conn.executemany(
        "INSERT INTO api_requests "
        "(session_id, user_email, timestamp, model, cost_usd, duration_ms, "
        "input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, "
        "terminal_type, org_id, scope_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info(f"Loaded {len(rows):,} api_requests")


def load_tool_decisions(conn: sqlite3.Connection, events: list[ToolDecisionEvent]) -> None:
    """Bulk insert tool decision events into the database."""
    rows = [
        (
            e.session_id, e.user_email, e.timestamp, e.tool_name,
            e.decision, e.source, e.terminal_type, e.org_id, e.scope.version,
        )
        for e in events
    ]
    conn.executemany(
        "INSERT INTO tool_decisions "
        "(session_id, user_email, timestamp, tool_name, decision, source, "
        "terminal_type, org_id, scope_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info(f"Loaded {len(rows):,} tool_decisions")


def load_tool_results(conn: sqlite3.Connection, events: list[ToolResultEvent]) -> None:
    """Bulk insert tool result events into the database."""
    rows = [
        (
            e.session_id, e.user_email, e.timestamp, e.tool_name,
            e.success, e.duration_ms, e.decision_source, e.decision_type,
            e.tool_result_size_bytes, e.terminal_type, e.org_id, e.scope.version,
        )
        for e in events
    ]
    conn.executemany(
        "INSERT INTO tool_results "
        "(session_id, user_email, timestamp, tool_name, success, duration_ms, "
        "decision_source, decision_type, tool_result_size_bytes, "
        "terminal_type, org_id, scope_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info(f"Loaded {len(rows):,} tool_results")


def load_user_prompts(conn: sqlite3.Connection, events: list[UserPromptEvent]) -> None:
    """Bulk insert user prompt events into the database."""
    rows = [
        (
            e.session_id, e.user_email, e.timestamp, e.prompt_length,
            e.terminal_type, e.org_id, e.scope.version,
        )
        for e in events
    ]
    conn.executemany(
        "INSERT INTO user_prompts "
        "(session_id, user_email, timestamp, prompt_length, "
        "terminal_type, org_id, scope_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info(f"Loaded {len(rows):,} user_prompts")


def load_api_errors(conn: sqlite3.Connection, events: list[ApiErrorEvent]) -> None:
    """Bulk insert API error events into the database."""
    rows = [
        (
            e.session_id, e.user_email, e.timestamp, e.model,
            e.error, e.status_code, e.attempt, e.duration_ms,
            e.terminal_type, e.org_id, e.scope.version,
        )
        for e in events
    ]
    conn.executemany(
        "INSERT INTO api_errors "
        "(session_id, user_email, timestamp, model, error, status_code, "
        "attempt, duration_ms, terminal_type, org_id, scope_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info(f"Loaded {len(rows):,} api_errors")


# ---------------------------------------------------------------------------
# MAIN PIPELINE: run_pipeline()
# ---------------------------------------------------------------------------
# Orchestrates the full ETL: parse → create DB → load data.
# This is the single entry point for the entire ingestion process.
# ---------------------------------------------------------------------------

def run_pipeline(data_dir: str, db_path: str) -> None:
    """Run the full ETL pipeline: parse raw files → load into SQLite.

    Args:
        data_dir: Directory containing telemetry_logs.jsonl and employees.csv.
        db_path: Path where the SQLite database will be created.

    Steps:
        1. Parse employees.csv → list of Employee models
        2. Parse telemetry_logs.jsonl → dict of event lists (streaming)
        3. Create SQLite database with tables + indexes
        4. Bulk insert all data
        5. Print summary
    """
    data_dir = Path(data_dir)
    db_path = Path(db_path)

    logger.info("=" * 60)
    logger.info("STARTING ETL PIPELINE")
    logger.info("=" * 60)

    pipeline_start = time.time()

    # ---- Step 1: Parse employees ----
    logger.info("Step 1: Parsing employees.csv ...")
    employees = parse_employees(str(data_dir / "employees.csv"))

    # ---- Step 2: Parse telemetry ----
    logger.info("Step 2: Parsing telemetry_logs.jsonl ...")
    events = parse_telemetry(str(data_dir / "telemetry_logs.jsonl"))

    # ---- Step 3: Create database ----
    logger.info(f"Step 3: Creating database at {db_path} ...")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    # Enable WAL mode for better write performance
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    create_tables(conn)

    # ---- Step 4: Load data ----
    logger.info("Step 4: Loading data into database ...")
    load_employees(conn, employees)
    load_api_requests(conn, events["api_requests"])
    load_tool_decisions(conn, events["tool_decisions"])
    load_tool_results(conn, events["tool_results"])
    load_user_prompts(conn, events["user_prompts"])
    load_api_errors(conn, events["api_errors"])

    # ---- Step 5: Verify ----
    logger.info("Step 5: Verifying loaded data ...")
    cursor = conn.cursor()

    tables = [
        "employees", "api_requests", "tool_decisions",
        "tool_results", "user_prompts", "api_errors",
    ]
    total_rows = 0
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        total_rows += count
        logger.info(f"  {table}: {count:,} rows")

    # Quick sanity check on session_summary view
    cursor.execute("SELECT COUNT(*) FROM session_summary")
    session_count = cursor.fetchone()[0]
    logger.info(f"  session_summary view: {session_count:,} sessions")

    cursor.execute("SELECT SUM(total_cost) FROM session_summary")
    total_cost = cursor.fetchone()[0]
    logger.info(f"  Total cost: ${total_cost:,.2f}")

    conn.close()

    elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info(
        f"PIPELINE COMPLETE: {total_rows:,} total rows loaded "
        f"in {elapsed:.1f}s"
    )
    logger.info(f"Database: {db_path} ({db_path.stat().st_size / 1024 / 1024:.1f} MB)")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
# Allows running the pipeline from the command line:
#   python -m src.ingest --data-dir data --db-path data/analytics.db
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the telemetry ETL pipeline")
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing telemetry_logs.jsonl and employees.csv"
    )
    parser.add_argument(
        "--db-path", default="data/analytics.db",
        help="Path for the output SQLite database"
    )
    args = parser.parse_args()

    run_pipeline(args.data_dir, args.db_path)
