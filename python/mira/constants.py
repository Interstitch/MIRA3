"""
MIRA3 Constants and Configuration
"""

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

# Dependencies to install in venv
# Lightweight local dependencies only (~50MB venv)
# No PyTorch or sentence-transformers - embeddings computed by remote service
DEPENDENCIES = [
    "watchdog",               # File watching for auto-ingestion
    "psycopg2-binary",        # PostgreSQL client
]
