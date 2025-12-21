"""
MIRA Recent Sessions Tool

Get recent conversation sessions with summaries.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

from mira.core import get_mira_path, log


def _format_project_path(encoded_path: str) -> str:
    """Convert encoded project path to readable format."""
    if not encoded_path:
        return "unknown"
    # Convert -workspaces-MIRA3 to /workspaces/MIRA3
    readable = encoded_path.replace('-', '/')
    if not readable.startswith('/'):
        readable = '/' + readable
    return readable


def handle_recent(params: dict, storage=None) -> dict:
    """Get recent conversation sessions.

    Tries central Postgres first, falls back to local SQLite, then local metadata files.

    Args:
        params: dict with 'limit' (int) and optional 'days' (int) to filter by time
        storage: Storage instance
    """
    limit = params.get("limit", 10)
    days = params.get("days")  # Optional: filter to last N days

    # Calculate cutoff time if days specified
    cutoff_time = None
    if days is not None and days > 0:
        cutoff_time = datetime.now() - timedelta(days=days)

    # Try storage (central or local SQLite via storage abstraction)
    if storage:
        try:
            sessions = storage.get_recent_sessions(limit=limit, since=cutoff_time)
            if sessions:
                # Group by project
                projects = {}
                for session in sessions:
                    project = session.get("project_path", "unknown")
                    if project not in projects:
                        projects[project] = []
                    projects[project].append({
                        "session_id": session.get("session_id", ""),
                        "summary": session.get("summary", ""),
                        "project_path": project,
                        "timestamp": str(session.get("started_at", "")),
                        "accomplishments": session.get("accomplishments", []),
                    })

                source = "central" if storage.using_central else "local_sqlite"
                result = {
                    "projects": [{"path": k, "sessions": v} for k, v in projects.items()],
                    "total": len(sessions),
                    "source": source
                }
                if days:
                    result["filtered_to_days"] = days
                return result
        except Exception as e:
            log(f"Storage recent query failed: {e}")

    # Fallback to local metadata files
    mira_path = get_mira_path()
    metadata_path = mira_path / "metadata"

    sessions = []
    if metadata_path.exists():
        for meta_file in sorted(metadata_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            # Check time filter first (before limit)
            if cutoff_time:
                file_mtime = datetime.fromtimestamp(meta_file.stat().st_mtime)
                if file_mtime < cutoff_time:
                    continue  # Skip files older than cutoff

            if len(sessions) >= limit:
                break

            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                raw_path = meta.get("project_path", "")
                sessions.append({
                    "session_id": meta_file.stem,
                    "summary": meta.get("summary", ""),
                    "project_path": _format_project_path(raw_path),
                    "timestamp": meta.get("extracted_at", ""),
                    "accomplishments": meta.get("accomplishments", []),
                })
            except (json.JSONDecodeError, IOError, OSError):
                pass

    # Group by project
    projects = {}
    for session in sessions:
        project = session.get("project_path", "unknown")
        if project not in projects:
            projects[project] = []
        projects[project].append(session)

    result = {
        "projects": [{"path": k, "sessions": v} for k, v in projects.items()],
        "total": len(sessions),
        "source": "local"
    }
    if days:
        result["filtered_to_days"] = days
    return result
