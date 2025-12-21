"""
MIRA Database Manager

Centralized SQLite database management with:
- WAL mode for better concurrency
- Thread-safe write queue to prevent "database is locked" errors
- Single writer thread for serialized writes
- Per-thread read connections for concurrent reads
"""

import queue
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional

from .constants import get_mira_path
from .utils import log


class DatabaseManager:
    """
    Manages SQLite database connections with thread-safe writes.

    All writes go through a queue processed by a single writer thread,
    preventing "database is locked" errors during parallel ingestion.
    Reads use thread-local connections for concurrent access.
    """

    def __init__(self):
        self._write_connections: dict[str, sqlite3.Connection] = {}
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._write_lock = threading.Lock()
        self._initialized = False
        self._initialized_dbs: set[str] = set()
        self._local = threading.local()

    def start(self):
        """Start the background writer thread."""
        if self._writer_thread is not None:
            return

        self._shutdown.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        self._initialized = True

    def stop(self):
        """Stop the writer thread and close all connections."""
        if self._writer_thread is None:
            return

        self._shutdown.set()
        self._write_queue.put(None)
        self._writer_thread.join(timeout=5.0)
        self._writer_thread = None

        with self._write_lock:
            for conn in self._write_connections.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._write_connections.clear()
            self._initialized_dbs.clear()
        self._initialized = False

    def _get_write_connection(self, db_name: str) -> sqlite3.Connection:
        """Get or create a write connection for the given database."""
        if db_name not in self._write_connections:
            db_path = get_mira_path() / db_name
            conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row

            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")

            self._write_connections[db_name] = conn
            self._initialized_dbs.add(db_name)

        return self._write_connections[db_name]

    def _get_read_connection(self, db_name: str) -> sqlite3.Connection:
        """Get or create a thread-local read connection."""
        if not hasattr(self._local, 'connections'):
            self._local.connections = {}

        if db_name not in self._local.connections:
            db_path = get_mira_path() / db_name
            conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.connections[db_name] = conn

        return self._local.connections[db_name]

    def _writer_loop(self):
        """Background thread that processes all write operations."""
        while not self._shutdown.is_set():
            try:
                item = self._write_queue.get(timeout=1.0)
                if item is None:
                    continue

                db_name, operation, args, result_event, result_holder = item

                try:
                    conn = self._get_write_connection(db_name)
                    cursor = conn.cursor()

                    if callable(operation):
                        result_holder['result'] = operation(cursor, *args)
                    else:
                        cursor.execute(operation, args)
                        result_holder['result'] = cursor.lastrowid

                    conn.commit()
                    result_holder['success'] = True

                except Exception as e:
                    result_holder['error'] = e
                    result_holder['success'] = False
                    try:
                        if db_name in self._write_connections:
                            self._write_connections[db_name].rollback()
                    except Exception:
                        pass

                finally:
                    result_event.set()

            except queue.Empty:
                continue

    def execute_write(self, db_name: str, sql: str, params: tuple = ()) -> int:
        """Execute a write operation (INSERT, UPDATE, DELETE)."""
        if not self._initialized:
            self.start()

        result_event = threading.Event()
        result_holder = {'result': None, 'success': False, 'error': None}

        self._write_queue.put((db_name, sql, params, result_event, result_holder))
        result_event.wait()

        if not result_holder['success']:
            raise result_holder['error']

        return result_holder['result']

    def execute_write_many(self, db_name: str, sql: str, params_list: List[tuple]) -> int:
        """Execute multiple write operations in a single transaction."""
        if not self._initialized:
            self.start()

        def batch_operation(cursor, sql, params_list):
            cursor.executemany(sql, params_list)
            return cursor.rowcount

        result_event = threading.Event()
        result_holder = {'result': None, 'success': False, 'error': None}

        self._write_queue.put((db_name, batch_operation, (sql, params_list), result_event, result_holder))
        result_event.wait()

        if not result_holder['success']:
            raise result_holder['error']

        return result_holder['result']

    def execute_write_func(self, db_name: str, func: Callable[[sqlite3.Cursor], Any]) -> Any:
        """Execute a custom write function."""
        if not self._initialized:
            self.start()

        result_event = threading.Event()
        result_holder = {'result': None, 'success': False, 'error': None}

        self._write_queue.put((db_name, func, (), result_event, result_holder))
        result_event.wait()

        if not result_holder['success']:
            raise result_holder['error']

        return result_holder['result']

    def execute_read(self, db_name: str, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a read operation (SELECT)."""
        if not self._initialized:
            self.start()

        conn = self._get_read_connection(db_name)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchall()

    def execute_read_one(self, db_name: str, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Execute a read operation and return a single row."""
        if not self._initialized:
            self.start()

        conn = self._get_read_connection(db_name)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchone()

    def init_schema(self, db_name: str, schema_sql: str):
        """Initialize database schema (tables, indexes, etc.)."""
        if not self._initialized:
            self.start()

        def init_op(cursor):
            cursor.executescript(schema_sql)
            return True

        return self.execute_write_func(db_name, init_op)


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None
_db_manager_lock = threading.Lock()


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    with _db_manager_lock:
        if _db_manager is None:
            _db_manager = DatabaseManager()
            _db_manager.start()
        return _db_manager


def shutdown_db_manager():
    """Shutdown the global database manager."""
    global _db_manager
    with _db_manager_lock:
        if _db_manager is not None:
            _db_manager.stop()
            _db_manager = None
