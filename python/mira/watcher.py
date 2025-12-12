"""
MIRA3 File Watcher Module

Watches for new/modified conversations and triggers ingestion.

Uses central Qdrant + Postgres if available, falls back to local SQLite.
"""

import threading
import time
from datetime import datetime
from pathlib import Path

from .utils import log
from .constants import WATCHER_DEBOUNCE_SECONDS, ACTIVE_SESSION_SYNC_INTERVAL
from .ingestion import ingest_conversation, sync_active_session, _mark_ingestion_active, _mark_ingestion_done

# TTL for pending files - entries older than this are cleaned up
# This prevents memory leaks if files are queued but never processed
PENDING_FILE_TTL_SECONDS = 3600  # 1 hour


class ConversationWatcher:
    """
    File watcher that monitors Claude Code conversations and triggers ingestion.

    Features:
    - Debounces rapid file changes (waits for file to stabilize)
    - Handles both new files and modifications
    - Thread-safe queuing
    - TTL cleanup to prevent memory leaks from long-running sessions
    - Active session tracking with periodic sync to remote storage
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
        self.active_sync_thread = None
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # Run cleanup every 5 minutes

        # Active session tracking - sync most recently modified file periodically
        self.active_session_path = None  # Path to most recently modified session
        self.active_session_mtime = 0.0  # mtime when last synced
        self.active_session_last_sync = 0.0  # timestamp of last sync attempt

    def queue_file(self, file_path: str):
        """Queue a file for ingestion after debounce period."""
        with self.lock:
            self.pending_files[file_path] = time.time()
            # Track as active session (most recently modified, excluding agent files)
            if not Path(file_path).name.startswith("agent-"):
                self.active_session_path = file_path

    def _debounce_worker(self):
        """Background thread that processes debounced files."""
        while self.running:
            time.sleep(1)  # Check every second

            files_to_process = []
            current_time = time.time()

            with self.lock:
                # Process files that have stabilized (debounce complete)
                for file_path, queued_time in list(self.pending_files.items()):
                    # If file hasn't been modified in DEBOUNCE seconds, process it
                    if current_time - queued_time >= WATCHER_DEBOUNCE_SECONDS:
                        files_to_process.append(file_path)
                        del self.pending_files[file_path]

                # Periodic TTL cleanup to prevent memory leaks
                if current_time - self.last_cleanup >= self.cleanup_interval:
                    self._cleanup_stale_entries(current_time)
                    self.last_cleanup = current_time

            for file_path in files_to_process:
                self._process_file(file_path)

    def _cleanup_stale_entries(self, current_time: float):
        """
        Remove entries older than TTL from pending_files.

        Called periodically to prevent unbounded memory growth.
        Must be called while holding self.lock.
        """
        stale_count = 0
        for file_path, queued_time in list(self.pending_files.items()):
            age = current_time - queued_time
            if age > PENDING_FILE_TTL_SECONDS:
                del self.pending_files[file_path]
                stale_count += 1

        if stale_count > 0:
            log(f"Watcher cleanup: removed {stale_count} stale entries")

    def _process_file(self, file_path: str):
        """Process a single file for ingestion."""
        path = Path(file_path)

        # Build file_info
        session_id = path.stem
        project_dir = path.parent.name
        is_agent_file = path.name.startswith("agent-")

        try:
            mtime = path.stat().st_mtime
            last_modified = datetime.fromtimestamp(mtime).isoformat()
        except (OSError, ValueError):
            last_modified = ""

        file_info = {
            'session_id': session_id,
            'file_path': str(path),
            'project_path': project_dir,
            'last_modified': last_modified,
            'is_agent': is_agent_file,
        }

        # Track as active ingestion
        _mark_ingestion_active(session_id, file_path, project_dir, "Watcher")
        try:
            # Pass None for collection (deprecated), use storage
            if ingest_conversation(file_info, None, self.mira_path, self.storage):
                log(f"Ingested: {session_id}")
        except Exception as e:
            log(f"Failed to ingest {session_id}: {e}")
        finally:
            _mark_ingestion_done(session_id)

    def _active_sync_worker(self):
        """
        Background thread that periodically syncs the active session.

        Runs every ACTIVE_SESSION_SYNC_INTERVAL seconds and checks if the
        active session file has been modified since the last sync. If so,
        triggers a sync to remote storage.

        This enables near real-time archival of the current conversation
        without waiting for the debounce period to complete.
        """
        while self.running:
            time.sleep(ACTIVE_SESSION_SYNC_INTERVAL)

            # Get current active session atomically
            with self.lock:
                active_path = self.active_session_path

            if not active_path:
                continue

            try:
                path = Path(active_path)
                if not path.exists():
                    continue

                current_mtime = path.stat().st_mtime

                # Check if file has changed since last sync
                if current_mtime > self.active_session_mtime:
                    session_id = path.stem
                    project_dir = path.parent.name

                    log(f"[active-sync] Syncing active session: {session_id[:12]}...")

                    # Sync to remote storage
                    success = sync_active_session(
                        file_path=active_path,
                        session_id=session_id,
                        project_path=project_dir,
                        mira_path=self.mira_path,
                        storage=self.storage
                    )

                    if success:
                        self.active_session_mtime = current_mtime
                        self.active_session_last_sync = time.time()
                        log(f"[active-sync] Synced: {session_id[:12]}")
                    else:
                        log(f"[active-sync] Sync skipped (no changes): {session_id[:12]}")

            except Exception as e:
                log(f"[active-sync] Error syncing active session: {e}")

    def start(self):
        """Start the debounce and active sync worker threads."""
        self.running = True
        self.debounce_thread = threading.Thread(target=self._debounce_worker, daemon=True)
        self.debounce_thread.start()

        # Start active session sync worker
        self.active_sync_thread = threading.Thread(target=self._active_sync_worker, daemon=True)
        self.active_sync_thread.start()
        log(f"[active-sync] Started (interval: {ACTIVE_SESSION_SYNC_INTERVAL}s)")

    def stop(self):
        """Stop the watcher."""
        self.running = False
        if self.debounce_thread:
            self.debounce_thread.join(timeout=2)
        if self.active_sync_thread:
            self.active_sync_thread.join(timeout=2)

    def get_stats(self) -> dict:
        """Get watcher statistics for monitoring."""
        with self.lock:
            return {
                "pending_count": len(self.pending_files),
                "running": self.running,
                "last_cleanup": self.last_cleanup,
                "active_session": self.active_session_path,
                "active_session_last_sync": self.active_session_last_sync,
            }


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
