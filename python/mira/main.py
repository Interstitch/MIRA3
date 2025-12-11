"""
MIRA3 Main Entry Point

Initializes all components and runs the main JSON-RPC loop.
Supports both central storage (Qdrant + Postgres) and local fallback (SQLite).
"""

import sys
import json
import os
import signal
import fcntl
import threading
from pathlib import Path

from .utils import log, get_mira_path
from .bootstrap import ensure_venv_and_deps, reexec_in_venv
from .config import get_config
from .storage import get_storage, Storage
from .embedding import get_embedding_function
from .handlers import handle_rpc_request
from .watcher import run_file_watcher
from .ingestion import run_full_ingestion
from .migrations import ensure_schema_current, check_migrations_needed


# Global lock file handle - must stay open for duration of process
_lock_file = None


def acquire_singleton_lock(mira_path: Path) -> bool:
    """
    Acquire exclusive lock to ensure only one MIRA instance runs.

    Uses flock() which automatically releases on process death.
    If another instance holds the lock, kills it and takes over.

    Returns True if lock acquired, False on failure.
    """
    global _lock_file

    lock_path = mira_path / "mira.lock"
    pid_path = mira_path / "mira.pid"

    # Open lock file (create if doesn't exist)
    _lock_file = open(lock_path, "w")

    try:
        # Try non-blocking exclusive lock
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Got the lock - write our PID
        pid_path.write_text(str(os.getpid()))
        log(f"Acquired singleton lock (PID {os.getpid()})")
        return True

    except BlockingIOError:
        # Another process holds the lock - try to kill it
        log("Another MIRA instance detected, attempting takeover...")

        old_pid = None
        if pid_path.exists():
            try:
                old_pid = int(pid_path.read_text().strip())
            except (ValueError, OSError):
                pass

        if old_pid:
            try:
                # Send SIGTERM first (graceful)
                os.kill(old_pid, signal.SIGTERM)
                log(f"Sent SIGTERM to old MIRA process {old_pid}")

                # Wait briefly for it to die
                import time
                for _ in range(10):  # 1 second max
                    time.sleep(0.1)
                    try:
                        # Check if still alive (signal 0 = just check)
                        os.kill(old_pid, 0)
                    except ProcessLookupError:
                        break  # Dead
                else:
                    # Still alive - force kill
                    try:
                        os.kill(old_pid, signal.SIGKILL)
                        log(f"Sent SIGKILL to stubborn process {old_pid}")
                    except ProcessLookupError:
                        pass

            except ProcessLookupError:
                log(f"Old process {old_pid} already dead")
            except PermissionError:
                log(f"Cannot kill process {old_pid} - permission denied")
                return False

        # Try to acquire lock again
        try:
            fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pid_path.write_text(str(os.getpid()))
            log(f"Acquired singleton lock after takeover (PID {os.getpid()})")
            return True
        except BlockingIOError:
            log("Failed to acquire lock even after kill attempt")
            return False


def send_notification(method: str, params: dict):
    """Send a JSON-RPC notification (no id, no response expected)."""
    notification = {"jsonrpc": "2.0", "method": method, "params": params}
    print(json.dumps(notification), flush=True)


def run_backend():
    """Main backend loop - handles JSON-RPC requests."""

    # Initialize storage paths
    mira_path = get_mira_path()
    archives_path = mira_path / "archives"
    metadata_path = mira_path / "metadata"

    # Ensure directories exist
    for path in [archives_path, metadata_path]:
        path.mkdir(parents=True, exist_ok=True)

    # Acquire singleton lock - kills any existing MIRA instance
    if not acquire_singleton_lock(mira_path):
        log("FATAL: Could not acquire singleton lock")
        sys.exit(1)

    # Run schema migrations before anything else
    log("Checking database schema...")
    try:
        ensure_schema_current()
    except Exception as e:
        log(f"Schema migration warning: {e}")

    # Load configuration and initialize storage
    config = get_config()
    log("Initializing storage...")
    storage = get_storage()

    # Health check and mode reporting
    health = storage.health_check()
    if storage.using_central:
        log(f"Storage: CENTRAL (Qdrant={health['qdrant_healthy']}, Postgres={health['postgres_healthy']})")
    else:
        log("Storage: LOCAL (SQLite with FTS - keyword search only)")
        log("To enable semantic search, configure ~/.mira/server.json")

    # Initialize embedding function (local - generates vectors to send to Qdrant)
    log("Initializing embedding model...")
    embed_fn = get_embedding_function()
    # Force model load now
    embed_fn._ensure_model()

    # Run initial ingestion in background
    def initial_ingestion():
        try:
            # Pass storage explicitly (collection param is deprecated/ignored)
            stats = run_full_ingestion(collection=None, mira_path=mira_path, storage=storage)
            log(f"Initial ingestion: {stats['ingested']} new conversations indexed")
        except Exception as e:
            log(f"Initial ingestion error: {e}")

    ingestion_thread = threading.Thread(target=initial_ingestion, daemon=True)
    ingestion_thread.start()

    # Send ready signal to Node.js
    send_notification("ready", {})

    # Start file watcher in background thread
    # Pass None for collection (deprecated), storage for central Qdrant + Postgres
    watcher_thread = threading.Thread(
        target=run_file_watcher,
        kwargs={"collection": None, "mira_path": mira_path, "storage": storage},
        daemon=True
    )
    watcher_thread.start()

    # Start sync worker - flushes local queue to central storage
    sync_worker = None
    try:
        from .sync_worker import start_sync_worker
        sync_worker = start_sync_worker(storage)
        log("Sync worker started")
    except Exception as e:
        log(f"Failed to start sync worker: {e}")

    # Main JSON-RPC loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = None
        try:
            request = json.loads(line)
            response = handle_rpc_request(request, None, storage)
            print(json.dumps(response), flush=True)

            # Handle shutdown
            if request.get("method") == "shutdown":
                if sync_worker:
                    sync_worker.stop()
                storage.close()
                break

        except json.JSONDecodeError as e:
            log(f"Invalid JSON: {e}")
        except Exception as e:
            log(f"Error handling request: {e}")
            if request is not None:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {"code": -32603, "message": "Internal error processing request"}
                }
                print(json.dumps(error_response), flush=True)


def main():
    """Main entry point with bootstrap."""
    try:
        # Bootstrap: ensure venv and deps
        if ensure_venv_and_deps():
            # Need to re-exec in venv
            log("Re-executing in virtualenv...")
            reexec_in_venv()

        # Now running in venv with all deps available
        run_backend()
    except KeyboardInterrupt:
        log("Shutting down...")
    except Exception as e:
        log(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
