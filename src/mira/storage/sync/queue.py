"""
MIRA Sync Queue

SQLite-based queue for items pending sync to central storage.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mira.core import get_mira_path

log = logging.getLogger(__name__)

# Global queue instance
_queue: Optional["SyncQueue"] = None
_queue_lock = threading.Lock()


class SyncQueue:
    """
    Persistent queue for sync operations.

    Items are stored in SQLite and processed by the sync worker.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = get_mira_path() / "sync_queue.db"
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the queue database."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_type TEXT NOT NULL,
                    item_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    attempts INTEGER DEFAULT 0,
                    UNIQUE(data_type, item_hash)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sync_queue_status
                ON sync_queue(status)
            """)
            conn.commit()
        finally:
            conn.close()

    def enqueue(self, data_type: str, item_hash: str, payload: Dict[str, Any]):
        """
        Add an item to the sync queue.

        If an item with the same hash exists, updates it.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            payload_json = json.dumps(payload, default=str)
            conn.execute("""
                INSERT INTO sync_queue (data_type, item_hash, payload, status)
                VALUES (?, ?, ?, 'pending')
                ON CONFLICT(data_type, item_hash) DO UPDATE SET
                    payload = excluded.payload,
                    status = 'pending',
                    updated_at = CURRENT_TIMESTAMP,
                    error_message = NULL
            """, (data_type, item_hash, payload_json))
            conn.commit()
        except Exception as e:
            log.error(f"Failed to enqueue sync item: {e}")
        finally:
            conn.close()

    def get_pending(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending items from the queue."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT id, data_type, item_hash, payload, attempts
                FROM sync_queue
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
            """, (limit,))
            rows = cur.fetchall()
            return [
                {
                    'id': row['id'],
                    'data_type': row['data_type'],
                    'item_hash': row['item_hash'],
                    'payload': json.loads(row['payload']),
                    'attempts': row['attempts'],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def mark_completed(self, item_id: int):
        """Mark an item as successfully synced."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                UPDATE sync_queue
                SET status = 'completed', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (item_id,))
            conn.commit()
        finally:
            conn.close()

    def mark_failed(self, item_id: int, error: str):
        """Mark an item as failed."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                UPDATE sync_queue
                SET status = 'failed',
                    error_message = ?,
                    attempts = attempts + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (error, item_id))
            conn.commit()
        finally:
            conn.close()

    def retry_failed(self):
        """Reset failed items to pending for retry."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                UPDATE sync_queue
                SET status = 'pending', updated_at = CURRENT_TIMESTAMP
                WHERE status = 'failed' AND attempts < 3
            """)
            conn.commit()
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM sync_queue
                GROUP BY status
            """)
            stats = {row[0]: row[1] for row in cur.fetchall()}
            return {
                'total_pending': stats.get('pending', 0),
                'total_completed': stats.get('completed', 0),
                'total_failed': stats.get('failed', 0),
            }
        finally:
            conn.close()

    def cleanup_completed(self, older_than_days: int = 7):
        """Remove old completed items."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                DELETE FROM sync_queue
                WHERE status = 'completed'
                AND updated_at < datetime('now', ?)
            """, (f'-{older_than_days} days',))
            conn.commit()
        finally:
            conn.close()


def get_sync_queue() -> SyncQueue:
    """Get the global sync queue instance."""
    global _queue
    with _queue_lock:
        if _queue is None:
            _queue = SyncQueue()
        return _queue


def reset_sync_queue():
    """Reset the global sync queue (for testing)."""
    global _queue
    with _queue_lock:
        _queue = None
