"""
MIRA Code History Module

Tracks file operations (Read/Write/Edit) across conversation sessions.
Enables mira_code_history tool.
"""

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from mira.core import log, DB_CODE_HISTORY
from mira.core.database import get_db_manager


CODE_HISTORY_SCHEMA = """
-- File operations
CREATE TABLE IF NOT EXISTS file_operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('read', 'write', 'edit')),
    timestamp TEXT NOT NULL,
    content_hash TEXT,
    project_path TEXT,
    message_index INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- File snapshots (full content)
CREATE TABLE IF NOT EXISTS file_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id INTEGER NOT NULL UNIQUE,
    content TEXT NOT NULL,
    line_count INTEGER,
    byte_size INTEGER,
    compressed INTEGER DEFAULT 0,
    FOREIGN KEY (operation_id) REFERENCES file_operations(id) ON DELETE CASCADE
);

-- Edit operations
CREATE TABLE IF NOT EXISTS file_edits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id INTEGER NOT NULL UNIQUE,
    old_string TEXT NOT NULL,
    new_string TEXT NOT NULL,
    line_hint INTEGER,
    FOREIGN KEY (operation_id) REFERENCES file_operations(id) ON DELETE CASCADE
);

-- Symbol definitions
CREATE TABLE IF NOT EXISTS symbol_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_type TEXT NOT NULL,
    line_number INTEGER,
    language TEXT,
    signature TEXT,
    FOREIGN KEY (snapshot_id) REFERENCES file_snapshots(id) ON DELETE CASCADE
);

-- Processing state
CREATE TABLE IF NOT EXISTS processing_state (
    session_id TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL,
    operations_found INTEGER DEFAULT 0
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_file_ops_path ON file_operations(file_path);
CREATE INDEX IF NOT EXISTS idx_file_ops_session ON file_operations(session_id);
CREATE INDEX IF NOT EXISTS idx_file_ops_timestamp ON file_operations(timestamp);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbol_definitions(symbol_name);
"""


def init_code_history_db():
    """Initialize the code history database."""
    db = get_db_manager()
    db.init_schema(DB_CODE_HISTORY, CODE_HISTORY_SCHEMA)


@dataclass
class FileOperation:
    """Represents a file operation."""
    session_id: str
    file_path: str
    operation: str  # 'read', 'write', 'edit'
    timestamp: str
    content_hash: Optional[str] = None
    project_path: Optional[str] = None
    message_index: int = 0
    content: Optional[str] = None
    old_string: Optional[str] = None
    new_string: Optional[str] = None


@dataclass
class ReconstructionResult:
    """Result of reconstructing a file."""
    file_path: str
    target_date: str
    content: Optional[str]
    confidence: float
    source_snapshot_date: Optional[str] = None
    edits_applied: int = 0
    edits_failed: int = 0
    gaps: List[str] = field(default_factory=list)


def get_file_timeline(
    file_path: str,
    symbol: Optional[str] = None,
    limit: int = 50,
) -> List[Dict]:
    """Get timeline of changes to a file."""
    init_code_history_db()
    db = get_db_manager()

    sql = """
        SELECT fo.*, fs.line_count, fs.byte_size
        FROM file_operations fo
        LEFT JOIN file_snapshots fs ON fo.id = fs.operation_id
        WHERE fo.file_path LIKE ?
        ORDER BY fo.timestamp DESC
        LIMIT ?
    """

    rows = db.execute_read(DB_CODE_HISTORY, sql, (file_path, limit))

    results = []
    for row in rows:
        results.append({
            "session_id": row['session_id'],
            "operation": row['operation'],
            "timestamp": row['timestamp'],
            "line_count": row.get('line_count'),
            "byte_size": row.get('byte_size'),
        })
    return results


def get_symbol_history(symbol: str, limit: int = 20) -> List[Dict]:
    """Get history of a symbol across sessions."""
    init_code_history_db()
    db = get_db_manager()

    sql = """
        SELECT sd.*, fs.operation_id, fo.file_path, fo.session_id, fo.timestamp
        FROM symbol_definitions sd
        JOIN file_snapshots fs ON sd.snapshot_id = fs.id
        JOIN file_operations fo ON fs.operation_id = fo.id
        WHERE sd.symbol_name LIKE ?
        ORDER BY fo.timestamp DESC
        LIMIT ?
    """

    rows = db.execute_read(DB_CODE_HISTORY, sql, (f"%{symbol}%", limit))

    results = []
    for row in rows:
        results.append({
            "symbol_name": row['symbol_name'],
            "symbol_type": row['symbol_type'],
            "file_path": row['file_path'],
            "line_number": row['line_number'],
            "session_id": row['session_id'],
            "timestamp": row['timestamp'],
        })
    return results


def get_file_snapshot_at_date(file_path: str, target_date: str) -> Optional[Dict]:
    """Get the most recent snapshot before a date."""
    init_code_history_db()
    db = get_db_manager()

    sql = """
        SELECT fo.*, fs.content, fs.compressed
        FROM file_operations fo
        JOIN file_snapshots fs ON fo.id = fs.operation_id
        WHERE fo.file_path LIKE ? AND fo.timestamp <= ?
        ORDER BY fo.timestamp DESC
        LIMIT 1
    """

    row = db.execute_read_one(DB_CODE_HISTORY, sql, (file_path, target_date))

    if row:
        content = row['content']
        if row.get('compressed'):
            content = gzip.decompress(content.encode()).decode()
        return {
            "file_path": row['file_path'],
            "content": content,
            "timestamp": row['timestamp'],
            "session_id": row['session_id'],
        }
    return None


def get_edits_between(file_path: str, start_date: str, end_date: str) -> List[Dict]:
    """Get edits to a file between two dates."""
    init_code_history_db()
    db = get_db_manager()

    sql = """
        SELECT fo.*, fe.old_string, fe.new_string
        FROM file_operations fo
        JOIN file_edits fe ON fo.id = fe.operation_id
        WHERE fo.file_path LIKE ? AND fo.timestamp > ? AND fo.timestamp <= ?
        ORDER BY fo.timestamp ASC
    """

    rows = db.execute_read(DB_CODE_HISTORY, sql, (file_path, start_date, end_date))

    results = []
    for row in rows:
        results.append({
            "session_id": row['session_id'],
            "timestamp": row['timestamp'],
            "old_string": row['old_string'],
            "new_string": row['new_string'],
        })
    return results


def reconstruct_file_at_date(file_path: str, target_date: str) -> ReconstructionResult:
    """Reconstruct file content at a specific date."""
    # Find the most recent snapshot before target date
    snapshot = get_file_snapshot_at_date(file_path, target_date)

    if not snapshot:
        return ReconstructionResult(
            file_path=file_path,
            target_date=target_date,
            content=None,
            confidence=0.0,
            gaps=["No snapshot found before target date"],
        )

    content = snapshot['content']
    snapshot_date = snapshot['timestamp']

    # Apply edits between snapshot and target date
    edits = get_edits_between(file_path, snapshot_date, target_date)

    edits_applied = 0
    edits_failed = 0

    for edit in edits:
        old_str = edit['old_string']
        new_str = edit['new_string']

        if old_str in content:
            content = content.replace(old_str, new_str, 1)
            edits_applied += 1
        else:
            edits_failed += 1

    # Calculate confidence
    total_edits = edits_applied + edits_failed
    if total_edits == 0:
        confidence = 0.95  # Just snapshot, high confidence
    else:
        confidence = 0.95 * (edits_applied / total_edits)

    return ReconstructionResult(
        file_path=file_path,
        target_date=target_date,
        content=content,
        confidence=confidence,
        source_snapshot_date=snapshot_date,
        edits_applied=edits_applied,
        edits_failed=edits_failed,
    )


def get_code_history_stats() -> Dict:
    """Get statistics about code history."""
    init_code_history_db()
    db = get_db_manager()

    try:
        ops = db.execute_read_one(DB_CODE_HISTORY, "SELECT COUNT(*) as cnt FROM file_operations", ())
        snapshots = db.execute_read_one(DB_CODE_HISTORY, "SELECT COUNT(*) as cnt FROM file_snapshots", ())
        edits = db.execute_read_one(DB_CODE_HISTORY, "SELECT COUNT(*) as cnt FROM file_edits", ())
        unique_files = db.execute_read_one(
            DB_CODE_HISTORY,
            "SELECT COUNT(DISTINCT file_path) as cnt FROM file_operations",
            ()
        )

        return {
            "total_operations": ops['cnt'] if ops else 0,
            "snapshots": snapshots['cnt'] if snapshots else 0,
            "edits": edits['cnt'] if edits else 0,
            "unique_files": unique_files['cnt'] if unique_files else 0,
        }
    except Exception as e:
        log(f"Code history stats failed: {e}")
        return {"total_operations": 0, "snapshots": 0, "edits": 0, "unique_files": 0}
