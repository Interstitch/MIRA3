"""
MIRA File Watcher Module

Watches for new/modified conversations and triggers ingestion.
Supports graceful shutdown via stop_file_watcher().
"""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from mira.core import log, get_mira_path, get_claude_projects_path, get_project_filter
from mira.core.constants import WATCHER_DEBOUNCE_SECONDS, ACTIVE_SESSION_SYNC_INTERVAL
from mira.ingestion.core import (
    ingest_conversation,
    sync_active_session,
    _mark_ingestion_active,
    _mark_ingestion_done,
)

# TTL for pending files - entries older than this are cleaned up
PENDING_FILE_TTL_SECONDS = 3600  # 1 hour

# Global references for graceful shutdown
_watcher_observer = None
_watcher_conv = None
_watcher_shutdown_event = threading.Event()


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
        self._shutdown_event = threading.Event()

        # Active session tracking
        self.active_session_path = None
        self.active_session_mtime = 0.0
        self.active_session_last_sync = 0.0

    def queue_file(self, file_path: str):
        """Queue a file for ingestion after debounce period."""
        with self.lock:
            self.pending_files[file_path] = time.time()
            # Track as active session (most recently modified, excluding agent files)
            if not Path(file_path).name.startswith("agent-"):
                self.active_session_path = file_path

    def _debounce_worker(self):
        """Background thread that processes debounced files."""
        while self.running and not self._shutdown_event.is_set():
            # Use interruptible sleep
            if self._shutdown_event.wait(timeout=1):
                break

            files_to_process = []
            current_time = time.time()

            with self.lock:
                for file_path, queued_time in list(self.pending_files.items()):
                    if current_time - queued_time >= WATCHER_DEBOUNCE_SECONDS:
                        files_to_process.append(file_path)
                        del self.pending_files[file_path]

                if current_time - self.last_cleanup >= self.cleanup_interval:
                    self._cleanup_stale_entries(current_time)
                    self.last_cleanup = current_time

            # Process remaining files even during shutdown
            for file_path in files_to_process:
                self._process_file(file_path)

    def _cleanup_stale_entries(self, current_time: float):
        """Remove entries older than TTL from pending_files."""
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

        session_id = path.stem
        project_dir = path.parent.name
        is_agent_file = path.name.startswith("agent-")

        # Skip agent sub-conversations
        if is_agent_file:
            log(f"Skipping agent file: {session_id}")
            return

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

        _mark_ingestion_active(session_id, file_path, project_dir, "Watcher")
        try:
            if ingest_conversation(file_info, None, self.mira_path, self.storage):
                log(f"Ingested: {session_id}")
        except Exception as e:
            log(f"Failed to ingest {session_id}: {e}")
        finally:
            _mark_ingestion_done(session_id)

    def _active_sync_worker(self):
        """Background thread that periodically syncs the active session."""
        while self.running and not self._shutdown_event.is_set():
            # Use interruptible sleep
            if self._shutdown_event.wait(timeout=ACTIVE_SESSION_SYNC_INTERVAL):
                break

            with self.lock:
                active_path = self.active_session_path

            if not active_path:
                continue

            try:
                path = Path(active_path)
                if not path.exists():
                    continue

                current_mtime = path.stat().st_mtime

                if current_mtime > self.active_session_mtime:
                    session_id = path.stem
                    project_dir = path.parent.name

                    log(f"[active-sync] Syncing active session: {session_id[:12]}...")

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
                        log(f"[active-sync] Sync skipped: {session_id[:12]}")

            except Exception as e:
                log(f"[active-sync] Error syncing active session: {e}")

    def start(self):
        """Start the debounce and active sync worker threads."""
        self.running = True
        self.debounce_thread = threading.Thread(target=self._debounce_worker, daemon=True)
        self.debounce_thread.start()

        self.active_sync_thread = threading.Thread(target=self._active_sync_worker, daemon=True)
        self.active_sync_thread.start()
        log(f"[active-sync] Started (interval: {ACTIVE_SESSION_SYNC_INTERVAL}s)")

    def stop(self):
        """Stop the watcher gracefully."""
        self.running = False
        self._shutdown_event.set()  # Signal workers to stop waiting
        if self.debounce_thread:
            self.debounce_thread.join(timeout=5)
        if self.active_sync_thread:
            self.active_sync_thread.join(timeout=5)
        log("ConversationWatcher stopped")

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
        storage: Storage instance
    """
    global _watcher_observer, _watcher_conv, _watcher_shutdown_event

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        log("watchdog not available, file watching disabled")
        return

    if mira_path is None:
        mira_path = get_mira_path()

    claude_path = get_claude_projects_path()
    if not claude_path.exists():
        log(f"Claude projects path not found: {claude_path}")
        return

    project_filter = get_project_filter()
    if project_filter:
        watch_path = claude_path / project_filter
        if not watch_path.exists():
            log(f"Filtered project path not found: {watch_path}")
            log(f"Falling back to watching all projects")
            watch_path = claude_path
            project_filter = None
    else:
        watch_path = claude_path

    conv_watcher = ConversationWatcher(None, mira_path, storage)
    conv_watcher.start()
    _watcher_conv = conv_watcher

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
    observer.schedule(ConversationHandler(), str(watch_path), recursive=True)
    observer.start()
    _watcher_observer = observer

    if project_filter:
        log(f"File watcher started on {watch_path} (filtered, debounce: {WATCHER_DEBOUNCE_SECONDS}s)")
    else:
        log(f"File watcher started on {watch_path} (all projects, debounce: {WATCHER_DEBOUNCE_SECONDS}s)")

    # Wait for shutdown signal (interruptible)
    try:
        while not _watcher_shutdown_event.is_set():
            _watcher_shutdown_event.wait(timeout=1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        observer.stop()
        conv_watcher.stop()
        observer.join(timeout=5)
        log("File watcher stopped")


def stop_file_watcher():
    """
    Stop the file watcher gracefully.

    Called from server shutdown to cleanly terminate the watcher thread.
    """
    global _watcher_observer, _watcher_conv, _watcher_shutdown_event

    log("Stopping file watcher...")
    _watcher_shutdown_event.set()

    if _watcher_conv:
        _watcher_conv.stop()
        _watcher_conv = None

    if _watcher_observer:
        _watcher_observer.stop()
        _watcher_observer.join(timeout=5)
        _watcher_observer = None
