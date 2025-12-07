"""
MIRA3 Main Entry Point

Initializes all components and runs the main JSON-RPC loop.
"""

import sys
import json
import threading
from pathlib import Path

from .utils import log, get_mira_path
from .bootstrap import ensure_venv_and_deps, reexec_in_venv
from .db_manager import get_db_manager, shutdown_db_manager
from .artifacts import init_artifact_db
from .custodian import init_custodian_db
from .insights import init_insights_db
from .concepts import init_concepts_db
from .embedding import MiraEmbeddingFunction
from .handlers import handle_rpc_request
from .watcher import run_file_watcher
from .ingestion import run_full_ingestion
from .constants import EMBEDDING_DIMENSIONS


def send_notification(method: str, params: dict):
    """Send a JSON-RPC notification (no id, no response expected)."""
    notification = {"jsonrpc": "2.0", "method": method, "params": params}
    print(json.dumps(notification), flush=True)


def run_backend():
    """Main backend loop - handles JSON-RPC requests."""

    # Import dependencies (now available in venv)
    try:
        import chromadb
    except ImportError as e:
        log(f"Failed to import chromadb: {e}")
        sys.exit(1)

    # Initialize storage paths
    mira_path = get_mira_path()
    chroma_path = mira_path / "chroma"
    archives_path = mira_path / "archives"
    metadata_path = mira_path / "metadata"

    # Ensure directories exist
    for path in [chroma_path, archives_path, metadata_path]:
        path.mkdir(parents=True, exist_ok=True)

    # Initialize centralized database manager (handles WAL mode and write queue)
    db_manager = get_db_manager()
    log("Database manager initialized")

    # Initialize all SQLite databases (through db_manager for thread safety)
    init_artifact_db()
    init_custodian_db()
    init_insights_db()
    init_concepts_db()

    # Initialize embedding function
    embedding_fn = MiraEmbeddingFunction()

    # Initialize ChromaDB
    log("Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))

    # Try to get existing collection, create if not exists
    try:
        collection = chroma_client.get_collection(
            name="conversations",
            embedding_function=embedding_fn
        )
        log(f"ChromaDB ready. Collection has {collection.count()} documents.")
    except Exception:
        # Collection doesn't exist, create it with proper metadata
        log("Creating new ChromaDB collection...")
        collection = chroma_client.create_collection(
            name="conversations",
            metadata={
                "description": "MIRA3 conversation embeddings",
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
                "hnsw:space": "cosine",
                "hnsw:M": 16,
                "hnsw:construction_ef": 100,
                "hnsw:search_ef": 50
            },
            embedding_function=embedding_fn
        )
        log("ChromaDB collection created.")

    # Run initial ingestion in background
    def initial_ingestion():
        try:
            stats = run_full_ingestion(collection, mira_path)
            log(f"Initial ingestion: {stats['ingested']} new conversations indexed")
        except Exception as e:
            log(f"Initial ingestion error: {e}")

    ingestion_thread = threading.Thread(target=initial_ingestion, daemon=True)
    ingestion_thread.start()

    # Send ready signal to Node.js
    send_notification("ready", {})

    # Start file watcher in background thread
    watcher_thread = threading.Thread(
        target=run_file_watcher,
        args=(collection, mira_path),
        daemon=True
    )
    watcher_thread.start()

    # Main JSON-RPC loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = None
        try:
            request = json.loads(line)
            response = handle_rpc_request(request, collection)
            print(json.dumps(response), flush=True)

            # Handle shutdown
            if request.get("method") == "shutdown":
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
