"""
MIRA3 Main Entry Point

Initializes all components and runs the main JSON-RPC loop.
Supports both central storage (Qdrant + Postgres) and local fallback (SQLite).
"""

import sys
import json
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
