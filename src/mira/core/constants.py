"""
MIRA Constants and Configuration Values

Central location for all magic numbers, paths, and configuration constants.
"""

from pathlib import Path

# Version
VERSION = "0.3.3"

# Approximate chars per token (for text length estimation)
CHARS_PER_TOKEN = 4

# Time gap threshold for session breaks (in seconds)
# 2 hours = likely went away and came back
TIME_GAP_THRESHOLD = 2 * 60 * 60  # 2 hours in seconds

# File watcher debounce time (seconds)
WATCHER_DEBOUNCE_SECONDS = 5

# Active session sync interval (seconds)
# How often to check and sync the active session to remote storage
ACTIVE_SESSION_SYNC_INTERVAL = 10

# Default MIRA storage path
def get_mira_path() -> Path:
    """Get the .mira storage directory path."""
    return Path.cwd() / ".mira"

# Shortcut for common use
MIRA_PATH = get_mira_path()

# Core dependencies for venv bootstrap
DEPENDENCIES = [
    "mcp>=1.25.0",
    "watchdog>=3.0.0",
    "psycopg2-binary>=2.9",
    "qdrant-client",
]

# Optional semantic search dependencies
DEPENDENCIES_SEMANTIC = [
    "fastembed",
    "sqlite-vec",
]

# Local semantic search configuration
LOCAL_SEMANTIC_ENABLED = True
LOCAL_SEMANTIC_INDEX_INTERVAL = 30  # Seconds between indexing queue checks
LOCAL_SEMANTIC_BATCH_SIZE = 5  # Sessions to index per batch
LOCAL_SEMANTIC_PROACTIVE = True  # Download model & index proactively (not just on remote failure)
LOCAL_SEMANTIC_STARTUP_DELAY = 30  # Seconds to wait before proactive download (avoid slowing startup)

# Database names
DB_LOCAL_STORE = "local_store.db"
DB_ARTIFACTS = "artifacts.db"
DB_CUSTODIAN = "custodian.db"
DB_INSIGHTS = "insights.db"
DB_CONCEPTS = "concepts.db"
DB_CODE_HISTORY = "code_history.db"
DB_SYNC_QUEUE = "sync_queue.db"
DB_MIGRATIONS = "migrations.db"
DB_LOCAL_VECTORS = "local_vectors.db"
