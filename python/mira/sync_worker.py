"""
MIRA3 Sync Worker Module

Background worker that flushes the local sync queue to central storage.
Runs as a daemon thread, checking the queue periodically.

Features:
- Batched sync operations for efficiency
- Exponential backoff on failures
- Per-data-type sync handlers
- Connectivity checks before sync attempts
"""

import threading
import time
from typing import Any, Dict, List, Optional

from .utils import log
from .sync_queue import get_sync_queue, SyncQueue
from .constants import ACTIVE_SESSION_SYNC_INTERVAL

# Sync configuration
SYNC_CHECK_INTERVAL = 30  # Check queue every 30 seconds
SYNC_BATCH_SIZE = 50  # Sync up to 50 items per batch
MAX_RETRIES = 5  # Max retry attempts before giving up
BACKOFF_BASE = 2  # Exponential backoff base (seconds)
BACKOFF_MAX = 300  # Max backoff (5 minutes)


class SyncWorker:
    """
    Background worker that syncs queued data to central storage.

    Usage:
        worker = SyncWorker(storage)
        worker.start()
        # ... later ...
        worker.stop()
    """

    def __init__(self, storage=None):
        self.storage = storage
        self.queue = get_sync_queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._backoff_until = 0  # Timestamp when backoff expires
        self._consecutive_failures = 0

    def start(self):
        """Start the sync worker thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        log("Sync worker started")

    def stop(self):
        """Stop the sync worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        log("Sync worker stopped")

    def _worker_loop(self):
        """Main worker loop - periodically checks and syncs queue."""
        while self.running:
            try:
                # Check if we're in backoff period
                now = time.time()
                if now < self._backoff_until:
                    time.sleep(min(SYNC_CHECK_INTERVAL, self._backoff_until - now))
                    continue

                # Check central connectivity
                if not self._check_central_available():
                    self._apply_backoff()
                    time.sleep(SYNC_CHECK_INTERVAL)
                    continue

                # Process queue
                synced = self._sync_batch()

                if synced > 0:
                    self._consecutive_failures = 0
                    log(f"Sync worker: synced {synced} items")

            except Exception as e:
                log(f"Sync worker error: {e}")
                self._apply_backoff()

            time.sleep(SYNC_CHECK_INTERVAL)

    def _check_central_available(self) -> bool:
        """Check if central storage is available."""
        if self.storage is None:
            try:
                from .storage import get_storage
                self.storage = get_storage()
            except ImportError:
                return False

        return self.storage.using_central and self.storage.postgres is not None

    def _apply_backoff(self):
        """Apply exponential backoff after failure."""
        self._consecutive_failures += 1
        backoff = min(BACKOFF_BASE ** self._consecutive_failures, BACKOFF_MAX)
        self._backoff_until = time.time() + backoff
        log(f"Sync worker backing off for {backoff}s (failures: {self._consecutive_failures})")

    def _sync_batch(self) -> int:
        """Sync a batch of items from the queue. Returns count synced."""
        total_synced = 0

        # Sync each data type
        for data_type in ["artifact", "session", "error", "decision", "custodian"]:
            items = self.queue.get_pending(data_type, limit=SYNC_BATCH_SIZE)
            if not items:
                continue

            synced, failed = self._sync_items(data_type, items)
            total_synced += synced

            # Record stats
            self.queue.record_sync_stats(
                data_type=data_type,
                items_synced=synced,
                items_failed=failed,
                duration_ms=0,  # TODO: track actual duration
            )

        return total_synced

    def _sync_items(self, data_type: str, items: List[Dict[str, Any]]) -> tuple:
        """
        Sync a list of items to central storage.

        Returns:
            (synced_count, failed_count)
        """
        if not items:
            return 0, 0

        # Mark items as syncing
        item_ids = [item["id"] for item in items]
        self.queue.mark_syncing(item_ids)

        synced_ids = []
        failed_ids = []
        error_msg = None

        for item in items:
            try:
                success = self._sync_single_item(data_type, item["payload"])
                if success:
                    synced_ids.append(item["id"])
                else:
                    failed_ids.append(item["id"])
            except Exception as e:
                error_msg = str(e)
                failed_ids.append(item["id"])

        # Update queue status
        if synced_ids:
            self.queue.mark_synced(synced_ids)
        if failed_ids:
            self.queue.mark_failed(failed_ids, error_msg or "Unknown error")

        return len(synced_ids), len(failed_ids)

    def _sync_single_item(self, data_type: str, payload: Dict[str, Any]) -> bool:
        """
        Sync a single item to central storage.

        Args:
            data_type: Type of data (artifact, session, etc.)
            payload: The data payload to sync

        Returns:
            True if synced successfully
        """
        if data_type == "artifact":
            return self._sync_artifact(payload)
        elif data_type == "session":
            return self._sync_session(payload)
        elif data_type == "error":
            return self._sync_error(payload)
        elif data_type == "decision":
            return self._sync_decision(payload)
        elif data_type == "custodian":
            return self._sync_custodian(payload)
        else:
            log(f"Unknown data type for sync: {data_type}")
            return False

    def _sync_artifact(self, payload: Dict[str, Any]) -> bool:
        """Sync an artifact to central Postgres."""
        if not self.storage or not self.storage.postgres:
            return False

        try:
            self.storage.postgres.insert_artifact(
                session_id=payload.get("postgres_session_id"),
                artifact_type=payload.get("artifact_type"),
                content=payload.get("content"),
                language=payload.get("language"),
                line_count=payload.get("line_count"),
                metadata=payload.get("metadata"),
            )
            return True
        except Exception as e:
            # Duplicate is OK - already synced
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                return True
            raise

    def _sync_session(self, payload: Dict[str, Any]) -> bool:
        """Sync a session to central storage."""
        if not self.storage:
            return False

        try:
            self.storage.upsert_session(
                project_path=payload.get("project_path"),
                session_id=payload.get("session_id"),
                summary=payload.get("summary"),
                keywords=payload.get("keywords"),
                facts=payload.get("facts"),
                task_description=payload.get("task_description"),
                git_branch=payload.get("git_branch"),
                models_used=payload.get("models_used"),
                tools_used=payload.get("tools_used"),
                files_touched=payload.get("files_touched"),
                message_count=payload.get("message_count", 0),
            )
            return True
        except Exception as e:
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                return True
            raise

    def _sync_error(self, payload: Dict[str, Any]) -> bool:
        """Sync an error pattern to central storage."""
        if not self.storage or not self.storage.postgres:
            return False

        try:
            project_id = payload.get("project_id")
            if not project_id and payload.get("project_path"):
                project_id = self.storage.postgres.get_or_create_project(
                    payload.get("project_path")
                )

            self.storage.postgres.upsert_error_pattern(
                project_id=project_id,
                signature=payload.get("signature"),
                error_type=payload.get("error_type"),
                error_text=payload.get("error_text"),
                solution=payload.get("solution"),
                file_path=payload.get("file_path"),
            )
            return True
        except Exception as e:
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                return True
            raise

    def _sync_decision(self, payload: Dict[str, Any]) -> bool:
        """Sync a decision to central storage."""
        if not self.storage or not self.storage.postgres:
            return False

        try:
            project_id = payload.get("project_id")
            if not project_id and payload.get("project_path"):
                project_id = self.storage.postgres.get_or_create_project(
                    payload.get("project_path")
                )

            self.storage.postgres.insert_decision(
                project_id=project_id,
                decision=payload.get("decision"),
                category=payload.get("category"),
                reasoning=payload.get("reasoning"),
                alternatives=payload.get("alternatives"),
                session_id=payload.get("session_id"),
                confidence=payload.get("confidence", 0.5),
            )
            return True
        except Exception as e:
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                return True
            raise

    def _sync_custodian(self, payload: Dict[str, Any]) -> bool:
        """Sync a custodian preference to central storage."""
        if not self.storage or not self.storage.postgres:
            return False

        try:
            self.storage.postgres.upsert_custodian(
                key=payload.get("key"),
                value=payload.get("value"),
                category=payload.get("category"),
                confidence=payload.get("confidence", 0.5),
                source_session=payload.get("source_session"),
            )
            return True
        except Exception as e:
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                return True
            raise

    def force_sync(self) -> Dict[str, Any]:
        """
        Force an immediate sync of all queued items.

        Returns stats about the sync operation.
        """
        if not self._check_central_available():
            return {"error": "Central storage not available", "synced": 0}

        start_time = time.time()
        total_synced = 0
        total_failed = 0
        by_type = {}

        for data_type in ["artifact", "session", "error", "decision", "custodian"]:
            items = self.queue.get_pending(data_type, limit=1000)
            if items:
                synced, failed = self._sync_items(data_type, items)
                total_synced += synced
                total_failed += failed
                by_type[data_type] = {"synced": synced, "failed": failed}

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            "synced": total_synced,
            "failed": total_failed,
            "by_type": by_type,
            "duration_ms": duration_ms,
        }


# Global sync worker instance
_worker: Optional[SyncWorker] = None


def get_sync_worker(storage=None) -> SyncWorker:
    """Get the global sync worker instance."""
    global _worker
    if _worker is None:
        _worker = SyncWorker(storage)
    return _worker


def start_sync_worker(storage=None):
    """Start the global sync worker."""
    worker = get_sync_worker(storage)
    worker.start()
    return worker


def stop_sync_worker():
    """Stop the global sync worker."""
    global _worker
    if _worker:
        _worker.stop()
        _worker = None
