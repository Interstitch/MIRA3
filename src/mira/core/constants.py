"""
MIRA Constants and Configuration Values

Central location for all magic numbers, paths, and configuration constants.

Storage Architecture (Hybrid Model):
=====================================

GLOBAL (~/.mira/):
  - .venv/           Python virtualenv (shared, ~600MB)
  - custodian.db     User preferences (follow you everywhere)
  - insights.db      Error patterns & decisions (useful across projects)
  - config.json      Global install state (venv version, etc.)

PROJECT (<cwd>/.mira/):
  - local_store.db   Conversation index for THIS project
  - artifacts.db     Code artifacts from THIS project
  - concepts.db      Codebase concepts ("this project uses React")
  - sync_queue.db    Sync state for this project
  - local_vectors.db Local semantic search vectors
  - server.json      Remote storage credentials (can be project-specific)
  - archives/        Conversation file copies
  - metadata/        Extracted session metadata
  - mira.log         Project-specific log
"""

from pathlib import Path
from typing import Set

# Version - imported from single source of truth
from mira._version import __version__ as VERSION

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


# =============================================================================
# Path Functions (Cross-Platform)
# =============================================================================

def get_global_mira_path() -> Path:
    """
    Get the global MIRA directory (~/.mira/).

    Contains:
    - Shared Python virtualenv (.venv/)
    - User-global databases (custodian.db, insights.db)
    - Global configuration (config.json)

    This path is the same regardless of which project you're in.
    Works on both Windows (C:\\Users\\name\\.mira) and Unix (~/.mira).
    """
    return Path.home() / ".mira"


def get_project_mira_path() -> Path:
    """
    Get the project-local MIRA directory (<cwd>/.mira/).

    Contains:
    - Project-specific databases (local_store.db, artifacts.db, etc.)
    - Conversation archives and metadata
    - Project-specific logs

    This path changes based on your current working directory.
    """
    return Path.cwd() / ".mira"


# Backwards compatibility alias
def get_mira_path() -> Path:
    """
    Get the project-local MIRA directory.

    DEPRECATED: Use get_project_mira_path() for clarity.
    This function exists for backwards compatibility.
    """
    return get_project_mira_path()


# Shortcut for common use (project path)
MIRA_PATH = get_project_mira_path()

# Core dependencies for venv bootstrap
# NOTE: claude-mira3 is NOT installed in venv - runs from global install
# This avoids file locking issues on Windows when upgrading
DEPENDENCIES = [
    "mcp>=1.25.0",
    "watchdog>=3.0.0",
    "psycopg2-binary>=2.9",
    "qdrant-client",
    "httpx",  # Required by mcp
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

# =============================================================================
# Database Names and Locations
# =============================================================================

# Database file names
DB_LOCAL_STORE = "local_store.db"
DB_ARTIFACTS = "artifacts.db"
DB_CUSTODIAN = "custodian.db"
DB_INSIGHTS = "insights.db"
DB_CONCEPTS = "concepts.db"
DB_CODE_HISTORY = "code_history.db"
DB_SYNC_QUEUE = "sync_queue.db"
DB_MIGRATIONS = "migrations.db"
DB_LOCAL_VECTORS = "local_vectors.db"

# Databases that live in GLOBAL path (~/.mira/)
# These contain user-wide data that should be shared across all projects
GLOBAL_DATABASES: Set[str] = {
    DB_CUSTODIAN,   # User preferences (name, workflow, rules)
    DB_INSIGHTS,    # Error patterns & decisions (useful everywhere)
}

# Databases that live in PROJECT path (<cwd>/.mira/)
# These contain project-specific data
PROJECT_DATABASES: Set[str] = {
    DB_LOCAL_STORE,   # Conversation index for this project
    DB_ARTIFACTS,     # Code artifacts from this project
    DB_CONCEPTS,      # Codebase concepts for this project
    DB_CODE_HISTORY,  # Code change history for this project
    DB_SYNC_QUEUE,    # Sync queue for this project
    DB_MIGRATIONS,    # Migration state for this project
    DB_LOCAL_VECTORS, # Local semantic vectors for this project
}


def get_db_path(db_name: str) -> Path:
    """
    Get the full path for a database file.

    Automatically routes to global or project path based on the database type.
    Works on both Windows and Unix.

    Args:
        db_name: Database filename (e.g., "custodian.db")

    Returns:
        Full path to the database file
    """
    if db_name in GLOBAL_DATABASES:
        return get_global_mira_path() / db_name
    else:
        return get_project_mira_path() / db_name
