"""
MIRA3 File Watcher Module

Watches for new/modified conversations and triggers ingestion.

Uses central Qdrant + Postgres storage exclusively.
"""

import threading
import time
from datetime import datetime
from pathlib import Path

from .utils import log
from .constants import WATCHER_DEBOUNCE_SECONDS
from .ingestion import ingest_conversation


class ConversationWatcher:
    """
    File watcher that monitors Claude Code conversations and triggers ingestion.

    Features:
    - Debounces rapid file changes (waits for file to stabilize)
    - Handles both new files and modifications
    - Thread-safe queuing
    """

    def __init__(self, collection, mira_path: Path, storage=None):
        """
        Initialize watcher.

        Args:
            collection: Deprecated - kept for API compatibility, ignored
            mira_path: Path to .mira directory
            storage: Storage instance for central Qdrant + Postgres
        """
        self.storage = storage
        self.mira_path = mira_path
        self.pending_files = {}  # file_path -> timestamp
        self.lock = threading.Lock()
        self.running = False
        self.debounce_thread = None

    def queue_file(self, file_path: str):
        """Queue a file for ingestion after debounce period."""
        with self.lock:
            self.pending_files[file_path] = time.time()

    def _debounce_worker(self):
        """Background thread that processes debounced files."""
        while self.running:
            time.sleep(1)  # Check every second

            files_to_process = []
            current_time = time.time()

            with self.lock:
                for file_path, queued_time in list(self.pending_files.items()):
                    # If file hasn't been modified in DEBOUNCE seconds, process it
                    if current_time - queued_time >= WATCHER_DEBOUNCE_SECONDS:
                        files_to_process.append(file_path)
                        del self.pending_files[file_path]

            for file_path in files_to_process:
                self._process_file(file_path)

    def _process_file(self, file_path: str):
        """Process a single file for ingestion."""
        path = Path(file_path)

        # Skip agent files
        if path.name.startswith("agent-"):
            return

        # Build file_info
        session_id = path.stem
        project_dir = path.parent.name

        try:
            mtime = path.stat().st_mtime
            last_modified = datetime.fromtimestamp(mtime).isoformat()
        except (OSError, ValueError):
            last_modified = ""

        file_info = {
            'session_id': session_id,
            'file_path': str(path),
            'project_path': project_dir,
            'last_modified': last_modified
        }

        try:
            # Pass None for collection (deprecated), use storage
            if ingest_conversation(file_info, None, self.mira_path, self.storage):
                log(f"Ingested: {session_id}")
        except Exception as e:
            log(f"Failed to ingest {session_id}: {e}")

    def start(self):
        """Start the debounce worker thread."""
        self.running = True
        self.debounce_thread = threading.Thread(target=self._debounce_worker, daemon=True)
        self.debounce_thread.start()

    def stop(self):
        """Stop the watcher."""
        self.running = False
        if self.debounce_thread:
            self.debounce_thread.join(timeout=2)


def run_file_watcher(collection, mira_path: Path = None, storage=None):
    """
    Background thread that watches for new conversations.

    Args:
        collection: Deprecated - kept for API compatibility, ignored
        mira_path: Path to .mira directory
        storage: Storage instance for central Qdrant + Postgres
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        log("watchdog not available, file watching disabled")
        return

    if mira_path is None:
        from .utils import get_mira_path
        mira_path = get_mira_path()

    claude_path = Path.home() / ".claude" / "projects"
    if not claude_path.exists():
        log(f"Claude projects path not found: {claude_path}")
        return

    # Create conversation watcher with debouncing
    conv_watcher = ConversationWatcher(None, mira_path, storage)
    conv_watcher.start()

    class ConversationHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            if event.src_path.endswith(".jsonl"):
                log(f"New conversation detected: {event.src_path}")
                conv_watcher.queue_file(event.src_path)

        def on_modified(self, event):
            if event.is_directory:
                return
            if event.src_path.endswith(".jsonl"):
                log(f"Conversation updated: {event.src_path}")
                conv_watcher.queue_file(event.src_path)

    observer = Observer()
    observer.schedule(ConversationHandler(), str(claude_path), recursive=True)
    observer.start()
    log(f"File watcher started on {claude_path} (debounce: {WATCHER_DEBOUNCE_SECONDS}s)")

    # Keep thread alive
    try:
        while True:
            threading.Event().wait(1)
    except (KeyboardInterrupt, SystemExit):
        observer.stop()
        conv_watcher.stop()
    observer.join()
