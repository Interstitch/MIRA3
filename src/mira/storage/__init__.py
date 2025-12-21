"""
MIRA Storage Package

Local-first storage abstraction with optional central sync.

All writes go to local SQLite first, then sync to central in background.
"""

from .storage import Storage, StorageError, get_storage, reset_storage
from .local_store import (
    init_local_db,
    get_session_count,
    get_or_create_project,
    upsert_session,
    get_recent_sessions,
)

__all__ = [
    # Main storage
    "Storage",
    "StorageError",
    "get_storage",
    "reset_storage",
    # Local store
    "init_local_db",
    "get_session_count",
    "get_or_create_project",
    "upsert_session",
    "get_recent_sessions",
]
