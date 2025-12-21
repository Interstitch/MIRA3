"""
MIRA Ingestion Package

Handles the full pipeline of parsing, extracting, archiving, and indexing conversations.
"""

from .core import (
    ingest_conversation,
    sync_active_session,
    discover_conversations,
    get_active_ingestions,
)

from .batch import (
    run_full_ingestion,
)

__all__ = [
    # Core ingestion
    "ingest_conversation",
    "sync_active_session",
    "discover_conversations",
    "get_active_ingestions",
    # Batch ingestion
    "run_full_ingestion",
]
