"""
MIRA3 Constants and Configuration
"""

# Embedding model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
# all-MiniLM-L6-v2 was trained on sequences up to 256 tokens
# Text beyond this is truncated and lost - we need to handle this properly
EMBEDDING_MAX_TOKENS = 256
# Approximate chars per token (for estimation)
CHARS_PER_TOKEN = 4

# Time gap threshold for session breaks (in seconds)
# 2 hours = likely went away and came back
TIME_GAP_THRESHOLD = 2 * 60 * 60  # 2 hours in seconds

# File watcher debounce time (seconds)
WATCHER_DEBOUNCE_SECONDS = 5

# Active session sync interval (seconds)
# How often to check and sync the active session to remote storage
ACTIVE_SESSION_SYNC_INTERVAL = 10

# Dependencies to install in venv
# Core dependencies (always installed)
# NOTE: ChromaDB removed - using remote Qdrant + Postgres exclusively
DEPENDENCIES = [
    "sentence-transformers",  # For local embedding generation
    "watchdog",               # File watching for auto-ingestion
    "qdrant-client",          # Vector database client
    "psycopg2-binary",        # PostgreSQL client
]
