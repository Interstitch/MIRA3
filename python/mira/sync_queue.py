"""
MIRA3 Sync Queue Module

Provides a durable local queue for data that needs to be synced to central storage.
All data is written locally first, then flushed to central when available.

Architecture:
- Local SQLite acts as a QUEUE, not a cache
- Data enters queue when written (regardless of central availability)
- Sync worker flushes queue to central periodically
- Items removed from queue only after successful central write
- Retry logic with exponential backoff for failures

Data types supported:
- artifacts: Code blocks, lists, tables, errors, configs, urls
- sessions: Conversation metadata
- errors: Error patterns and solutions
- decisions: Architectural decisions
- custodian: User preferences and patterns
"""

import json
import time
import threading
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .utils import log, get_mira_path
from .db_manager import get_db_manager

SYNC_QUEUE_DB = "sync_queue.db"

# Sync queue schema
SYNC_QUEUE_SCHEMA = """
-- Sync queue table - holds all pending items for central sync
CREATE TABLE IF NOT EXISTS sync_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Item identification
    data_type TEXT NOT NULL,           -- artifact, session, error, decision, custodian
    item_hash TEXT NOT NULL,           -- SHA256 hash for deduplication

    -- Payload
    payload TEXT NOT NULL,             -- JSON-serialized data

    -- Sync state
    status TEXT DEFAULT 'pending',     -- pending, syncing, synced, failed
    retry_count INTEGER DEFAULT 0,
    last_error TEXT,

    -- Timestamps
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    synced_at TEXT,

    -- Prevent duplicates
    UNIQUE(data_type, item_hash)
);

-- Indexes for efficient queue operations
CREATE INDEX IF NOT EXISTS idx_sync_queue_status ON sync_queue(status);
CREATE INDEX IF NOT EXISTS idx_sync_queue_type_status ON sync_queue(data_type, status);
CREATE INDEX IF NOT EXISTS idx_sync_queue_created ON sync_queue(created_at);

-- Sync stats table - tracks sync history
CREATE TABLE IF NOT EXISTS sync_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data_type TEXT NOT NULL,
    items_synced INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    duration_ms INTEGER,
    error_message TEXT,
    synced_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sync_stats_type ON sync_stats(data_type);
CREATE INDEX IF NOT EXISTS idx_sync_stats_time ON sync_stats(synced_at);
"""


class SyncStatus(Enum):
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"


class SyncQueue:
    """
    Durable queue for syncing local data to central storage.

    Usage:
        queue = SyncQueue()

        # Add item to queue
        queue.enqueue("artifact", item_hash, payload_dict)

        # Get pending items for sync
        items = queue.get_pending("artifact", limit=100)

        # Mark items as synced after successful central write
        queue.mark_synced([item_id1, item_id2])

        # Mark items as failed (will be retried)
        queue.mark_failed([item_id3], "Connection timeout")
    """

    _initialized = False
    _lock = threading.Lock()

    def __init__(self):
        self._ensure_initialized()

    def _ensure_initialized(self):
        """Initialize the sync queue database."""
        with self._lock:
            if SyncQueue._initialized:
                return

            db = get_db_manager()
            db.init_schema(SYNC_QUEUE_DB, SYNC_QUEUE_SCHEMA)
            SyncQueue._initialized = True
            log("Sync queue database initialized")

    def enqueue(
        self,
        data_type: str,
        item_hash: str,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Add an item to the sync queue.

        Args:
            data_type: Type of data (artifact, session, error, decision, custodian)
            item_hash: Unique hash for deduplication
            payload: Data to sync (will be JSON serialized)

        Returns:
            True if enqueued (new item), False if duplicate exists
        """
        db = get_db_manager()

        try:
            payload_json = json.dumps(payload, default=str)

            db.execute_write(
                SYNC_QUEUE_DB,
                """
                INSERT INTO sync_queue (data_type, item_hash, payload, status)
                VALUES (?, ?, ?, 'pending')
                ON CONFLICT (data_type, item_hash) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    updated_at = CURRENT_TIMESTAMP
                    WHERE sync_queue.status = 'failed'
                """,
                (data_type, item_hash, payload_json)
            )
            return True
        except Exception as e:
            log(f"Failed to enqueue {data_type}: {e}")
            return False

    def get_pending(
        self,
        data_type: Optional[str] = None,
        limit: int = 100,
        max_retries: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get pending items from the queue.

        Args:
            data_type: Filter by data type (None for all types)
            limit: Maximum items to return
            max_retries: Exclude items with more retries than this

        Returns:
            List of queue items with id, data_type, item_hash, payload
        """
        db = get_db_manager()

        if data_type:
            rows = db.execute_read(
                SYNC_QUEUE_DB,
                """
                SELECT id, data_type, item_hash, payload, retry_count
                FROM sync_queue
                WHERE data_type = ?
                  AND status IN ('pending', 'failed')
                  AND retry_count < ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (data_type, max_retries, limit)
            )
        else:
            rows = db.execute_read(
                SYNC_QUEUE_DB,
                """
                SELECT id, data_type, item_hash, payload, retry_count
                FROM sync_queue
                WHERE status IN ('pending', 'failed')
                  AND retry_count < ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (max_retries, limit)
            )

        items = []
        for row in rows:
            try:
                items.append({
                    "id": row["id"],
                    "data_type": row["data_type"],
                    "item_hash": row["item_hash"],
                    "payload": json.loads(row["payload"]),
                    "retry_count": row["retry_count"],
                })
            except json.JSONDecodeError:
                log(f"Invalid JSON in queue item {row['id']}")

        return items

    def mark_syncing(self, item_ids: List[int]):
        """Mark items as currently syncing (prevents double-processing)."""
        if not item_ids:
            return

        db = get_db_manager()
        placeholders = ",".join("?" * len(item_ids))

        db.execute_write(
            SYNC_QUEUE_DB,
            f"""
            UPDATE sync_queue
            SET status = 'syncing', updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
            """,
            tuple(item_ids)
        )

    def mark_synced(self, item_ids: List[int]):
        """Mark items as successfully synced (removes from queue)."""
        if not item_ids:
            return

        db = get_db_manager()
        placeholders = ",".join("?" * len(item_ids))

        # Delete synced items - they're now in central storage
        db.execute_write(
            SYNC_QUEUE_DB,
            f"DELETE FROM sync_queue WHERE id IN ({placeholders})",
            tuple(item_ids)
        )

    def mark_failed(self, item_ids: List[int], error: str):
        """Mark items as failed (will be retried later)."""
        if not item_ids:
            return

        db = get_db_manager()
        placeholders = ",".join("?" * len(item_ids))

        db.execute_write(
            SYNC_QUEUE_DB,
            f"""
            UPDATE sync_queue
            SET status = 'failed',
                retry_count = retry_count + 1,
                last_error = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
            """,
            (error,) + tuple(item_ids)
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        db = get_db_manager()

        # Count by status and type
        rows = db.execute_read(
            SYNC_QUEUE_DB,
            """
            SELECT data_type, status, COUNT(*) as count
            FROM sync_queue
            GROUP BY data_type, status
            """
        )

        stats = {
            "pending": {},
            "syncing": {},
            "failed": {},
            "total_pending": 0,
            "total_failed": 0,
        }

        for row in rows:
            dtype = row["data_type"]
            status = row["status"]
            count = row["count"]

            if status not in stats:
                stats[status] = {}
            stats[status][dtype] = count

            if status in ("pending", "syncing"):
                stats["total_pending"] += count
            elif status == "failed":
                stats["total_failed"] += count

        # Get oldest pending item age
        row = db.execute_read_one(
            SYNC_QUEUE_DB,
            """
            SELECT MIN(created_at) as oldest
            FROM sync_queue
            WHERE status IN ('pending', 'failed')
            """
        )
        if row and row["oldest"]:
            stats["oldest_pending"] = row["oldest"]

        return stats

    def record_sync_stats(
        self,
        data_type: str,
        items_synced: int,
        items_failed: int,
        duration_ms: int,
        error_message: Optional[str] = None,
    ):
        """Record sync operation statistics."""
        db = get_db_manager()

        db.execute_write(
            SYNC_QUEUE_DB,
            """
            INSERT INTO sync_stats (data_type, items_synced, items_failed, duration_ms, error_message)
            VALUES (?, ?, ?, ?, ?)
            """,
            (data_type, items_synced, items_failed, duration_ms, error_message)
        )

    def get_recent_sync_stats(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sync operation stats."""
        db = get_db_manager()

        rows = db.execute_read(
            SYNC_QUEUE_DB,
            """
            SELECT data_type, items_synced, items_failed, duration_ms, error_message, synced_at
            FROM sync_stats
            ORDER BY synced_at DESC
            LIMIT ?
            """,
            (limit,)
        )

        return [dict(row) for row in rows]

    def purge_old_failures(self, max_retries: int = 10):
        """Remove items that have exceeded max retry attempts."""
        db = get_db_manager()

        result = db.execute_write(
            SYNC_QUEUE_DB,
            "DELETE FROM sync_queue WHERE status = 'failed' AND retry_count >= ?",
            (max_retries,)
        )

        if result:
            log(f"Purged {result} items that exceeded max retries")

        return result


# Global queue instance
_queue: Optional[SyncQueue] = None


def get_sync_queue() -> SyncQueue:
    """Get the global sync queue instance."""
    global _queue
    if _queue is None:
        _queue = SyncQueue()
    return _queue
