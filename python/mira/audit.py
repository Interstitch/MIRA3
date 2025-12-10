"""
MIRA3 Audit Logging Module

Tracks all significant operations for compliance, debugging, and analytics.
Audit logs are append-only and stored in both local SQLite and central Postgres.
"""

import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .utils import log, get_mira_path
from .db_manager import get_db_manager

AUDIT_DB = "audit.db"

AUDIT_SCHEMA = """
-- Audit log table - append-only record of all operations
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    user_id TEXT,
    machine_id TEXT,
    parameters TEXT,  -- JSON
    result_summary TEXT,  -- JSON
    status TEXT NOT NULL,  -- success, failure, error
    duration_ms INTEGER,
    error_message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_log(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_log(status);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
"""

_initialized = False
_machine_id = None


def _get_machine_id() -> str:
    """Get or generate a unique machine identifier."""
    global _machine_id
    if _machine_id:
        return _machine_id

    mira_path = get_mira_path()
    machine_file = mira_path / ".machine_id"

    if machine_file.exists():
        _machine_id = machine_file.read_text().strip()
    else:
        _machine_id = str(uuid.uuid4())[:8]
        try:
            machine_file.write_text(_machine_id)
        except Exception:
            pass

    return _machine_id


def _get_user_id() -> str:
    """Get user identifier from git config or environment."""
    # Try git config
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'config', 'user.email'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    # Fall back to environment
    return os.environ.get('USER', os.environ.get('USERNAME', 'unknown'))


def init_audit_db():
    """Initialize the audit database."""
    global _initialized
    if _initialized:
        return

    db = get_db_manager()
    db.init_schema(AUDIT_DB, AUDIT_SCHEMA)
    _initialized = True


class AuditContext:
    """Context manager for tracking operation duration and status."""

    def __init__(
        self,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.action = action
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.parameters = parameters or {}
        self.event_id = str(uuid.uuid4())
        self.start_time = None
        self.status = "success"
        self.error_message = None
        self.result_summary = {}

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = int((time.time() - self.start_time) * 1000)

        if exc_type is not None:
            self.status = "error"
            self.error_message = str(exc_val)

        audit_log(
            action=self.action,
            resource_type=self.resource_type,
            resource_id=self.resource_id,
            parameters=self.parameters,
            result_summary=self.result_summary,
            status=self.status,
            duration_ms=duration_ms,
            error_message=self.error_message,
            event_id=self.event_id,
        )

        # Don't suppress exceptions
        return False

    def set_result(self, **kwargs):
        """Set result summary fields."""
        self.result_summary.update(kwargs)

    def set_failure(self, message: str):
        """Mark operation as failed."""
        self.status = "failure"
        self.error_message = message


def audit_log(
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    result_summary: Optional[Dict[str, Any]] = None,
    status: str = "success",
    duration_ms: Optional[int] = None,
    error_message: Optional[str] = None,
    event_id: Optional[str] = None,
):
    """
    Log an audit event.

    Args:
        action: The action performed (search, ingest, init, etc.)
        resource_type: Type of resource (session, archive, etc.)
        resource_id: Specific resource identifier
        parameters: Input parameters (sanitized)
        result_summary: Summary of results
        status: success, failure, or error
        duration_ms: Operation duration in milliseconds
        error_message: Error details if failed
        event_id: Unique event identifier
    """
    init_audit_db()

    if event_id is None:
        event_id = str(uuid.uuid4())

    timestamp = datetime.utcnow().isoformat() + "Z"

    # Sanitize parameters - remove sensitive data
    safe_params = _sanitize_params(parameters or {})

    db = get_db_manager()
    try:
        db.execute_write(
            AUDIT_DB,
            """INSERT INTO audit_log (
                event_id, timestamp, action, resource_type, resource_id,
                user_id, machine_id, parameters, result_summary,
                status, duration_ms, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                timestamp,
                action,
                resource_type,
                resource_id,
                _get_user_id(),
                _get_machine_id(),
                json.dumps(safe_params),
                json.dumps(result_summary or {}),
                status,
                duration_ms,
                error_message,
            )
        )
    except Exception as e:
        # Audit logging should never break the main operation
        log(f"Audit log failed: {e}")


def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive data from parameters."""
    sensitive_keys = {'password', 'secret', 'token', 'key', 'credential'}
    result = {}

    for k, v in params.items():
        k_lower = k.lower()
        if any(s in k_lower for s in sensitive_keys):
            result[k] = "[REDACTED]"
        elif isinstance(v, str) and len(v) > 500:
            result[k] = v[:500] + f"... [truncated, {len(v)} chars]"
        elif isinstance(v, dict):
            result[k] = _sanitize_params(v)
        elif isinstance(v, list) and len(v) > 20:
            result[k] = v[:20] + [f"... [{len(v)} items]"]
        else:
            result[k] = v

    return result


def get_recent_audit_logs(
    limit: int = 100,
    action: Optional[str] = None,
    status: Optional[str] = None,
    since: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query recent audit logs.

    Args:
        limit: Maximum number of records to return
        action: Filter by action type
        status: Filter by status (success, failure, error)
        since: ISO timestamp to filter from

    Returns:
        List of audit log entries
    """
    init_audit_db()
    db = get_db_manager()

    conditions = []
    params = []

    if action:
        conditions.append("action = ?")
        params.append(action)

    if status:
        conditions.append("status = ?")
        params.append(status)

    if since:
        conditions.append("timestamp >= ?")
        params.append(since)

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    params.append(limit)

    rows = db.execute_read(
        AUDIT_DB,
        f"""SELECT * FROM audit_log
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?""",
        tuple(params)
    )

    result = []
    for row in rows:
        entry = dict(row)
        # Parse JSON fields
        if entry.get('parameters'):
            try:
                entry['parameters'] = json.loads(entry['parameters'])
            except json.JSONDecodeError:
                pass
        if entry.get('result_summary'):
            try:
                entry['result_summary'] = json.loads(entry['result_summary'])
            except json.JSONDecodeError:
                pass
        result.append(entry)

    return result


def get_audit_stats(days: int = 7) -> Dict[str, Any]:
    """
    Get audit statistics for the specified period.

    Returns:
        Dict with action counts, error rates, etc.
    """
    init_audit_db()
    db = get_db_manager()

    since = datetime.utcnow().isoformat()[:10]  # Simplified - just use recent data

    # Get action counts
    rows = db.execute_read(
        AUDIT_DB,
        """SELECT action, status, COUNT(*) as count
           FROM audit_log
           GROUP BY action, status
           ORDER BY count DESC""",
        ()
    )

    action_counts = {}
    status_counts = {"success": 0, "failure": 0, "error": 0}

    for row in rows:
        action = row['action']
        status = row['status']
        count = row['count']

        if action not in action_counts:
            action_counts[action] = {"total": 0, "success": 0, "failure": 0, "error": 0}

        action_counts[action]["total"] += count
        action_counts[action][status] += count
        status_counts[status] += count

    # Get total count
    total_row = db.execute_read_one(
        AUDIT_DB,
        "SELECT COUNT(*) as total FROM audit_log",
        ()
    )

    # Get average duration by action
    duration_rows = db.execute_read(
        AUDIT_DB,
        """SELECT action, AVG(duration_ms) as avg_duration, MAX(duration_ms) as max_duration
           FROM audit_log
           WHERE duration_ms IS NOT NULL
           GROUP BY action""",
        ()
    )

    durations = {}
    for row in duration_rows:
        durations[row['action']] = {
            "avg_ms": int(row['avg_duration'] or 0),
            "max_ms": int(row['max_duration'] or 0),
        }

    return {
        "total_events": total_row['total'] if total_row else 0,
        "by_status": status_counts,
        "by_action": action_counts,
        "durations": durations,
    }
