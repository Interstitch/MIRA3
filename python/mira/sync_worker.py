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

        # Sync each data type (in dependency order: projects first)
        data_types = [
            "project",           # Projects must sync first (sessions depend on them)
            "session",           # Sessions depend on projects
            "archive",           # Archives depend on sessions
            "archive_update",    # Archive updates depend on archives
            "session_metadata",  # Metadata updates depend on sessions
            "artifact",          # Artifacts depend on sessions
            "file_operation",    # File operations depend on sessions
            "error",             # Errors depend on projects
            "decision",          # Decisions depend on projects
            "custodian",         # Custodian is global
        ]

        for data_type in data_types:
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

        # For batch-optimized types, process all items together
        if data_type == "file_operation":
            return self._sync_file_operations_batch(items)
        if data_type == "artifact":
            return self._sync_artifacts_batch(items)

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

        IMPORTANT: Calls postgres backend directly, NOT through storage.py
        to avoid infinite recursion (storage.py queues for sync).

        Args:
            data_type: Type of data (artifact, session, etc.)
            payload: The data payload to sync

        Returns:
            True if synced successfully
        """
        if data_type == "project":
            return self._sync_project(payload)
        elif data_type == "session":
            return self._sync_session(payload)
        elif data_type == "archive":
            return self._sync_archive(payload)
        elif data_type == "archive_update":
            return self._sync_archive_update(payload)
        elif data_type == "session_metadata":
            return self._sync_session_metadata(payload)
        elif data_type == "artifact":
            return self._sync_artifact(payload)
        elif data_type == "file_operation":
            return self._sync_file_operation(payload)
        elif data_type == "error":
            return self._sync_error(payload)
        elif data_type == "decision":
            return self._sync_decision(payload)
        elif data_type == "custodian":
            return self._sync_custodian(payload)
        else:
            log(f"Unknown data type for sync: {data_type}")
            return False

    def _sync_project(self, payload: Dict[str, Any]) -> bool:
        """Sync a project to central Postgres."""
        if not self.storage or not self.storage.postgres:
            return False

        try:
            self.storage.postgres.get_or_create_project(
                path=payload.get("path"),
                slug=payload.get("slug"),
                git_remote=payload.get("git_remote"),
            )
            return True
        except Exception as e:
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                return True
            raise

    def _sync_session(self, payload: Dict[str, Any]) -> bool:
        """Sync a session to central Postgres (directly, not through storage)."""
        if not self.storage or not self.storage.postgres:
            return False

        try:
            # Get or create project in central first
            project_id = self.storage.postgres.get_or_create_project(
                path=payload.get("project_path"),
                git_remote=payload.get("git_remote"),
            )

            # Upsert session directly to postgres
            self.storage.postgres.upsert_session(
                project_id=project_id,
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

    def _sync_archive(self, payload: Dict[str, Any]) -> bool:
        """Sync an archive to central Postgres."""
        if not self.storage or not self.storage.postgres:
            return False

        try:
            import hashlib
            session_id = payload.get("session_id")  # UUID
            content = payload.get("content")
            content_hash = payload.get("content_hash") or hashlib.sha256(content.encode()).hexdigest()

            if not session_id:
                raise ValueError("No session_id in archive payload")

            # Look up the central session ID by UUID
            with self.storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM sessions WHERE session_id = %s",
                        (session_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        return False  # Session not synced yet, will retry

                    postgres_session_id = row[0]

                    # Insert/update archive with the correct remote session ID
                    cur.execute(
                        """
                        INSERT INTO archives (session_id, content, content_hash, size_bytes, line_count)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            content_hash = EXCLUDED.content_hash,
                            size_bytes = EXCLUDED.size_bytes,
                            line_count = EXCLUDED.line_count,
                            updated_at = NOW()
                        """,
                        (postgres_session_id, content, content_hash,
                         len(content.encode('utf-8')), content.count('\n'))
                    )
                    conn.commit()
            return True
        except Exception as e:
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                return True
            raise

    def _sync_archive_update(self, payload: Dict[str, Any]) -> bool:
        """Sync an archive update to central Postgres."""
        if not self.storage or not self.storage.postgres:
            return False

        try:
            import hashlib
            session_id = payload.get("session_id")
            content = payload.get("content")
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            with self.storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    # Get the internal session ID
                    cur.execute(
                        "SELECT id FROM sessions WHERE session_id = %s",
                        (session_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        return False  # Session not synced yet, will retry

                    postgres_session_id = row[0]

                    # Update archive
                    cur.execute(
                        """
                        INSERT INTO archives (session_id, content, content_hash, size_bytes, line_count)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            content_hash = EXCLUDED.content_hash,
                            size_bytes = EXCLUDED.size_bytes,
                            line_count = EXCLUDED.line_count,
                            updated_at = NOW()
                        """,
                        (postgres_session_id, content, content_hash,
                         payload.get("size_bytes", 0), payload.get("line_count", 0))
                    )
                    conn.commit()
            return True
        except Exception as e:
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                return True
            raise

    def _sync_session_metadata(self, payload: Dict[str, Any]) -> bool:
        """Sync session metadata update to central Postgres."""
        if not self.storage or not self.storage.postgres:
            return False

        try:
            session_id = payload.get("session_id")
            summary = payload.get("summary")
            keywords = payload.get("keywords")

            with self.storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE sessions
                        SET summary = %s, keywords = %s
                        WHERE session_id = %s
                        """,
                        (summary, keywords, session_id)
                    )
                    conn.commit()
            return True
        except Exception as e:
            raise

    def _sync_artifact(self, payload: Dict[str, Any]) -> bool:
        """Sync an artifact to central Postgres."""
        if not self.storage or not self.storage.postgres:
            return False

        try:
            # Look up central session ID from session_uuid
            # Support both new format (_local_session_id=UUID) and old format (session_id=int)
            session_uuid = payload.get("_local_session_id")
            if not session_uuid:
                # Fallback: look up UUID from local session_id (int)
                local_session_id = payload.get("session_id")
                if local_session_id and isinstance(local_session_id, int):
                    from .local_store import local_store
                    rows = local_store.execute_read(
                        "SELECT session_id FROM sessions WHERE id = ?",
                        (local_session_id,)
                    )
                    if rows:
                        session_uuid = rows[0]["session_id"]

            if not session_uuid:
                return False

            with self.storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM sessions WHERE session_id = %s",
                        (session_uuid,)
                    )
                    row = cur.fetchone()
                    if not row:
                        return False  # Session not synced yet, will retry

                    postgres_session_id = row[0]

            self.storage.postgres.insert_artifact(
                session_id=postgres_session_id,
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

    def _sync_file_operation(self, payload: Dict[str, Any]) -> bool:
        """Sync a file operation to central Postgres."""
        if not self.storage or not self.storage.postgres:
            return False

        try:
            # Look up central session ID from local session ID
            local_session_id = payload.get("_local_session_id")  # UUID string
            if not local_session_id:
                return False

            with self.storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM sessions WHERE session_id = %s",
                        (local_session_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        return False  # Session not synced yet, will retry

                    postgres_session_id = row[0]

            # Update payload with correct session ID
            payload['session_id'] = postgres_session_id

            self.storage.postgres.batch_insert_file_operations([payload])
            return True
        except Exception as e:
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                return True
            raise

    def _sync_file_operations_batch(self, items: List[Dict[str, Any]]) -> tuple:
        """
        Batch sync file operations to central Postgres.

        Returns:
            (synced_count, failed_count)
        """
        if not self.storage or not self.storage.postgres:
            self.queue.mark_failed([item["id"] for item in items], "No postgres connection")
            return 0, len(items)

        item_ids = [item["id"] for item in items]

        try:
            # Look up all session mappings in one query
            session_uuids = list(set(
                item["payload"].get("_local_session_id")
                for item in items
                if item["payload"].get("_local_session_id")
            ))

            if not session_uuids:
                self.queue.mark_failed(item_ids, "No session IDs")
                return 0, len(items)

            # Batch lookup session mappings
            session_map = {}  # uuid -> postgres_id
            with self.storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    placeholders = ",".join(["%s"] * len(session_uuids))
                    cur.execute(
                        f"SELECT session_id, id FROM sessions WHERE session_id IN ({placeholders})",
                        tuple(session_uuids)
                    )
                    for row in cur.fetchall():
                        session_map[row[0]] = row[1]

            # Separate items into ready (session synced) and not ready
            ready_items = []
            not_ready_ids = []

            for item in items:
                uuid = item["payload"].get("_local_session_id")
                if uuid in session_map:
                    # Update payload with postgres session ID
                    payload = item["payload"].copy()
                    payload["session_id"] = session_map[uuid]
                    ready_items.append((item["id"], payload))
                else:
                    not_ready_ids.append(item["id"])

            # Batch insert all ready items
            synced_ids = []
            if ready_items:
                payloads = [p for _, p in ready_items]
                self.storage.postgres.batch_insert_file_operations(payloads)
                synced_ids = [id for id, _ in ready_items]

            # Mark statuses
            if synced_ids:
                self.queue.mark_synced(synced_ids)
            if not_ready_ids:
                self.queue.mark_failed(not_ready_ids, "Session not synced yet")

            return len(synced_ids), len(not_ready_ids)

        except Exception as e:
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                self.queue.mark_synced(item_ids)
                return len(items), 0
            self.queue.mark_failed(item_ids, str(e))
            return 0, len(items)

    def _sync_artifacts_batch(self, items: List[Dict[str, Any]]) -> tuple:
        """
        Batch sync artifacts to central Postgres.

        Returns:
            (synced_count, failed_count)
        """
        if not self.storage or not self.storage.postgres:
            self.queue.mark_failed([item["id"] for item in items], "No postgres connection")
            return 0, len(items)

        item_ids = [item["id"] for item in items]

        try:
            # Look up all session mappings in one query
            # Support both new format (_local_session_id=UUID) and old format (session_id=int)
            session_uuids = list(set(
                item["payload"].get("_local_session_id")
                for item in items
                if item["payload"].get("_local_session_id")
            ))

            # Fallback: if no UUIDs, try to look up from local session_id (int)
            if not session_uuids:
                local_session_ids = list(set(
                    item["payload"].get("session_id")
                    for item in items
                    if item["payload"].get("session_id") and isinstance(item["payload"].get("session_id"), int)
                ))
                if local_session_ids:
                    # Look up UUIDs from local database
                    from .local_store import local_store
                    uuid_map = {}  # local_id -> uuid
                    for local_id in local_session_ids:
                        rows = local_store.execute_read(
                            "SELECT session_id FROM sessions WHERE id = ?",
                            (local_id,)
                        )
                        if rows:
                            uuid_map[local_id] = rows[0]["session_id"]
                    # Update items with UUIDs
                    for item in items:
                        local_id = item["payload"].get("session_id")
                        if local_id in uuid_map:
                            item["payload"]["_local_session_id"] = uuid_map[local_id]
                            session_uuids.append(uuid_map[local_id])
                    session_uuids = list(set(session_uuids))

            if not session_uuids:
                self.queue.mark_failed(item_ids, "No session IDs")
                return 0, len(items)

            # Batch lookup session mappings
            session_map = {}  # uuid -> postgres_id
            with self.storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    placeholders = ",".join(["%s"] * len(session_uuids))
                    cur.execute(
                        f"SELECT session_id, id FROM sessions WHERE session_id IN ({placeholders})",
                        tuple(session_uuids)
                    )
                    for row in cur.fetchall():
                        session_map[row[0]] = row[1]

            # Separate items into ready (session synced) and not ready
            ready_items = []
            not_ready_ids = []

            for item in items:
                uuid = item["payload"].get("_local_session_id")
                if uuid in session_map:
                    ready_items.append((item["id"], item["payload"], session_map[uuid]))
                else:
                    not_ready_ids.append(item["id"])

            # Insert all ready items
            synced_ids = []
            for item_id, payload, postgres_session_id in ready_items:
                try:
                    self.storage.postgres.insert_artifact(
                        session_id=postgres_session_id,
                        artifact_type=payload.get("artifact_type"),
                        content=payload.get("content"),
                        language=payload.get("language"),
                        line_count=payload.get("line_count"),
                        metadata=payload.get("metadata"),
                    )
                    synced_ids.append(item_id)
                except Exception as e:
                    if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                        synced_ids.append(item_id)  # Already exists, count as success
                    else:
                        not_ready_ids.append(item_id)

            # Mark statuses
            if synced_ids:
                self.queue.mark_synced(synced_ids)
            if not_ready_ids:
                self.queue.mark_failed(not_ready_ids, "Session not synced yet or insert failed")

            return len(synced_ids), len(not_ready_ids)

        except Exception as e:
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                self.queue.mark_synced(item_ids)
                return len(items), 0
            self.queue.mark_failed(item_ids, str(e))
            return 0, len(items)

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
