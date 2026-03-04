"""
Pydantic data models for Claude Code telemetry data.

This module defines typed, validated models for every data structure
in the pipeline — from raw JSONL batches to individual event types.

WHY PYDANTIC?
- Raw JSON has numeric values stored as strings ("0.071", "10230")
  → Pydantic auto-casts them to float/int
- Some fields are optional (tool_result_size_bytes exists ~30% of time)
  → Pydantic handles None gracefully
- JSON keys have dots ("user.email", "host.arch") which aren't valid Python names
  → Pydantic aliases map them to clean Python attributes
- If data is malformed, Pydantic raises a clear ValidationError
  → We catch it and skip bad records instead of crashing

MODELS OVERVIEW:
    Employee            — One row from employees.csv (5 fields)
    EventScope          — The "scope" block in each event (name + version)
    EventResource       — The "resource" block (host/OS/user info)
    ApiRequestEvent     — claude_code.api_request (LLM API call with cost/tokens)
    ToolDecisionEvent   — claude_code.tool_decision (tool approved or rejected)
    ToolResultEvent     — claude_code.tool_result (tool execution outcome)
    UserPromptEvent     — claude_code.user_prompt (user typed something)
    ApiErrorEvent       — claude_code.api_error (API call failed)
    parse_event()       — Factory function: raw dict → correct event model
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# 1. EMPLOYEE MODEL
# ---------------------------------------------------------------------------
# Maps directly to employees.csv — one row = one Employee.
#
# CSV columns:  email, full_name, practice, level, location
# Example row:  alex.chen@example.com, Alex Chen, ML Engineering, L5, United States
#
# This is the simplest model — no aliases or type casting needed.
# ---------------------------------------------------------------------------

class Employee(BaseModel):
    """Employee from the employees.csv directory.

    Attributes:
        email: Unique identifier, also the join key to telemetry events.
        full_name: Display name (e.g., "Alex Chen").
        practice: Engineering team (e.g., "ML Engineering", "Backend Engineering").
        level: Seniority from L1 (junior) to L10 (senior).
        location: Country (e.g., "United States", "Germany").
    """

    email: str
    full_name: str
    practice: str
    level: str
    location: str


# ---------------------------------------------------------------------------
# 2. EVENT SCOPE MODEL
# ---------------------------------------------------------------------------
# Every telemetry event has a "scope" block that tells us which version
# of Claude Code generated the event.
#
# Raw JSON:
#   "scope": {
#       "name": "com.anthropic.claude_code.events",
#       "version": "2.1.50"
#   }
#
# We need this to track version adoption across users.
# ---------------------------------------------------------------------------

class EventScope(BaseModel):
    """Instrumentation scope metadata.

    Attributes:
        name: Always "com.anthropic.claude_code.events".
        version: Claude Code version string (e.g., "2.1.50").
    """

    name: str
    version: str


# ---------------------------------------------------------------------------
# 3. EVENT RESOURCE MODEL
# ---------------------------------------------------------------------------
# Every event has a "resource" block with host/machine/user info.
#
# Raw JSON keys have DOTS in them (e.g., "host.arch", "os.type").
# Python doesn't allow dots in variable names, so we use Pydantic's
# Field(alias=...) to map "host.arch" → host_arch.
#
# IMPORTANT: resource.user.email is always "" (empty string).
# The real email is in attributes.user.email — we handle that in the
# event models below, not here.
#
# Raw JSON:
#   "resource": {
#       "host.arch": "arm64",
#       "host.name": "Alexs-MacBook-Pro.local",
#       "os.type": "darwin",
#       "os.version": "24.6.0",
#       "service.name": "claude-code-None",
#       "service.version": "2.1.50",
#       "user.email": "",
#       "user.practice": "ML Engineering",
#       "user.profile": "alex.chen",
#       "user.serial": "ABC1234567"
#   }
# ---------------------------------------------------------------------------

class EventResource(BaseModel):
    """Host and user environment information.

    Uses Field(alias=...) because the raw JSON keys contain dots
    which aren't valid Python attribute names.

    Attributes:
        host_arch: CPU architecture ("arm64" or "x86_64").
        host_name: Machine hostname.
        os_type: Operating system ("darwin", "linux", "windows").
        os_version: OS version string.
        service_name: Always "claude-code-None".
        service_version: Same as scope.version.
        user_practice: Engineering practice (also in employees.csv).
        user_profile: Username pattern on the machine.
        user_serial: 10-character alphanumeric device serial.
    """

    # model_config: allow creating model with either alias or field name
    model_config = {"populate_by_name": True}

    # Each Field(alias=...) maps a dotted JSON key to a clean Python name
    host_arch: str = Field(alias="host.arch")
    host_name: str = Field(alias="host.name")
    os_type: str = Field(alias="os.type")
    os_version: str = Field(alias="os.version")
    service_name: str = Field(alias="service.name")
    service_version: str = Field(alias="service.version")
    user_practice: str = Field(alias="user.practice")
    user_profile: str = Field(alias="user.profile")
    user_serial: str = Field(alias="user.serial")


# ---------------------------------------------------------------------------
# 4. EVENT MODELS (one per telemetry event type)
# ---------------------------------------------------------------------------
# All 5 event types share COMMON fields from "attributes":
#   - event.timestamp    → timestamp (str, ISO 8601)
#   - session.id         → session_id (str, UUID)
#   - user.email         → user_email (str, the REAL email)
#   - user.id            → user_id (str, SHA-256 hash)
#   - user.account_uuid  → account_uuid (str, UUID)
#   - organization.id    → org_id (str, UUID)
#   - terminal.type      → terminal_type (str, e.g., "vscode")
#
# Then each event type adds its OWN specific fields.
#
# WHY NOT USE INHERITANCE?
# We could make a BaseEvent class and inherit from it. But for clarity
# and simplicity (evaluators can read each model independently), we
# repeat the common fields in each model. Pydantic makes this painless.
#
# TYPE CASTING:
# The raw JSON stores ALL numbers as strings: "0.071", "10230", "263".
# Pydantic auto-casts these because we declare the fields as float/int.
# Example: cost_usd: float → "0.071" automatically becomes 0.071
# ---------------------------------------------------------------------------


class ApiRequestEvent(BaseModel):
    """claude_code.api_request — An LLM API call.

    This is the most important event for cost analysis.
    Each record = one call to a Claude model with token counts and cost.

    Type casting examples:
        "0.071"   → cost_usd: float  → 0.071
        "10230"   → duration_ms: int → 10230
        "263"     → input_tokens: int → 263
    """

    model_config = {"populate_by_name": True}

    # --- Common fields (from attributes) ---
    timestamp: str = Field(alias="event.timestamp")
    session_id: str = Field(alias="session.id")
    user_email: str = Field(alias="user.email")
    user_id: str = Field(alias="user.id")
    account_uuid: str = Field(alias="user.account_uuid")
    org_id: str = Field(alias="organization.id")
    terminal_type: str = Field(alias="terminal.type")

    # --- API Request specific fields ---
    model: str                      # e.g., "claude-opus-4-6"
    cost_usd: float                 # Auto-cast from "0.071" → 0.071
    duration_ms: int                # Auto-cast from "10230" → 10230
    input_tokens: int               # Tokens sent to the model
    output_tokens: int              # Tokens generated by the model
    cache_read_tokens: int          # Tokens read from cache
    cache_creation_tokens: int      # Tokens used to create cache

    # --- Nested objects (scope + resource) ---
    scope: EventScope
    resource: EventResource

    # Validator: ensure cost is non-negative
    @field_validator("cost_usd")
    @classmethod
    def cost_must_be_non_negative(cls, v: float) -> float:
        """Cost should never be negative."""
        if v < 0:
            raise ValueError("cost_usd must be >= 0")
        return v


class ToolDecisionEvent(BaseModel):
    """claude_code.tool_decision — Whether a tool use was approved.

    When Claude wants to use a tool (Read, Bash, Edit, etc.), this event
    records whether it was accepted or rejected, and by what mechanism.

    Fields:
        decision: "accept" or "reject"
        source: Who decided — "config" (80%), "user_temporary" (15%),
                "user_permanent" (3%), "user_reject" (2%)
        tool_name: Which tool (e.g., "Read", "Bash", "Edit")
    """

    model_config = {"populate_by_name": True}

    # --- Common fields ---
    timestamp: str = Field(alias="event.timestamp")
    session_id: str = Field(alias="session.id")
    user_email: str = Field(alias="user.email")
    user_id: str = Field(alias="user.id")
    account_uuid: str = Field(alias="user.account_uuid")
    org_id: str = Field(alias="organization.id")
    terminal_type: str = Field(alias="terminal.type")

    # --- Tool Decision specific fields ---
    tool_name: str
    decision: str                   # "accept" or "reject"
    source: str                     # "config", "user_temporary", etc.

    # --- Nested objects ---
    scope: EventScope
    resource: EventResource


class ToolResultEvent(BaseModel):
    """claude_code.tool_result — The outcome of running a tool.

    After a tool is approved and executed, this records whether it
    succeeded, how long it took, and optionally how big the result was.

    IMPORTANT: tool_result_size_bytes is OPTIONAL — only present ~30%
    of the time. We use Optional[int] = None so missing values don't crash.
    """

    model_config = {"populate_by_name": True}

    # --- Common fields ---
    timestamp: str = Field(alias="event.timestamp")
    session_id: str = Field(alias="session.id")
    user_email: str = Field(alias="user.email")
    user_id: str = Field(alias="user.id")
    account_uuid: str = Field(alias="user.account_uuid")
    org_id: str = Field(alias="organization.id")
    terminal_type: str = Field(alias="terminal.type")

    # --- Tool Result specific fields ---
    tool_name: str
    success: str                    # "true" or "false" (kept as string for DB)
    duration_ms: int                # Auto-cast from string
    decision_source: str
    decision_type: str
    tool_result_size_bytes: Optional[int] = None  # Missing ~70% of the time

    # --- Nested objects ---
    scope: EventScope
    resource: EventResource


class UserPromptEvent(BaseModel):
    """claude_code.user_prompt — User typed a prompt.

    Each time a user sends a message to Claude Code, this event fires.
    The actual prompt text is always "<REDACTED>" for privacy.
    We only get the length.
    """

    model_config = {"populate_by_name": True}

    # --- Common fields ---
    timestamp: str = Field(alias="event.timestamp")
    session_id: str = Field(alias="session.id")
    user_email: str = Field(alias="user.email")
    user_id: str = Field(alias="user.id")
    account_uuid: str = Field(alias="user.account_uuid")
    org_id: str = Field(alias="organization.id")
    terminal_type: str = Field(alias="terminal.type")

    # --- User Prompt specific fields ---
    prompt_length: int              # Auto-cast from string

    # --- Nested objects ---
    scope: EventScope
    resource: EventResource


class ApiErrorEvent(BaseModel):
    """claude_code.api_error — An API call that failed.

    Records what went wrong, which model was called, the HTTP status,
    and which retry attempt this was.

    Common errors:
        "Request was aborted."                              (52% of errors)
        "This request would exceed your account's rate limit..." (23%)
        "Internal server error"                             (5%)
    """

    model_config = {"populate_by_name": True}

    # --- Common fields ---
    timestamp: str = Field(alias="event.timestamp")
    session_id: str = Field(alias="session.id")
    user_email: str = Field(alias="user.email")
    user_id: str = Field(alias="user.id")
    account_uuid: str = Field(alias="user.account_uuid")
    org_id: str = Field(alias="organization.id")
    terminal_type: str = Field(alias="terminal.type")

    # --- API Error specific fields ---
    model: str                      # Which model failed
    error: str                      # Error message text
    status_code: str                # HTTP status ("429", "500", "undefined")
    attempt: int                    # Retry attempt number (1, 2, or 3)
    duration_ms: int                # How long before the error

    # --- Nested objects ---
    scope: EventScope
    resource: EventResource


# ---------------------------------------------------------------------------
# 5. EVENT TYPE ROUTING MAP
# ---------------------------------------------------------------------------
# The raw JSON "body" field tells us which event type it is.
# This dict maps body → the correct Pydantic model class.
# ---------------------------------------------------------------------------

EVENT_TYPE_MAP: dict[str, type[BaseModel]] = {
    "claude_code.api_request": ApiRequestEvent,
    "claude_code.tool_decision": ToolDecisionEvent,
    "claude_code.tool_result": ToolResultEvent,
    "claude_code.user_prompt": UserPromptEvent,
    "claude_code.api_error": ApiErrorEvent,
}


# ---------------------------------------------------------------------------
# 6. FACTORY FUNCTION: parse_event()
# ---------------------------------------------------------------------------
# Takes a raw event dict (after json.loads(message)) and returns
# the correct typed Pydantic model.
#
# How it works:
#   1. Read the "body" field to determine event type
#   2. Look up the correct model class in EVENT_TYPE_MAP
#   3. Merge "attributes" with "scope" and "resource" into one flat dict
#   4. Pass to the model class — Pydantic validates + casts types
#   5. Return the validated model instance
#
# If the event type is unknown or validation fails, returns None
# so the caller can skip it and keep going.
# ---------------------------------------------------------------------------

def parse_event(raw: dict) -> Optional[BaseModel]:
    """Parse a raw event dict into the correct typed Pydantic model.

    Args:
        raw: A dict from json.loads(logEvent["message"]) containing:
            - "body": event type string (e.g., "claude_code.api_request")
            - "attributes": dict of event-specific + common fields
            - "scope": dict with name and version
            - "resource": dict with host/OS info

    Returns:
        A validated Pydantic model instance (ApiRequestEvent, etc.),
        or None if the event type is unknown or validation fails.

    Example:
        >>> raw = json.loads(log_event["message"])
        >>> event = parse_event(raw)
        >>> if event:
        ...     print(type(event))  # <class 'ApiRequestEvent'>
        ...     print(event.cost_usd)  # 0.071 (float, auto-cast from "0.071")
    """
    body = raw.get("body")
    model_class = EVENT_TYPE_MAP.get(body)

    if model_class is None:
        # Unknown event type — skip it
        return None

    # Build the input dict for Pydantic:
    # Start with all attributes (contains both common + specific fields)
    # Then add scope and resource as nested objects
    attributes = raw.get("attributes", {})
    data = {
        **attributes,
        "scope": raw.get("scope", {}),
        "resource": raw.get("resource", {}),
    }

    try:
        return model_class.model_validate(data)
    except Exception:
        # Validation failed (malformed data) — skip this event
        return None
