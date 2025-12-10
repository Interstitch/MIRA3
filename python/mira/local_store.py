"""
MIRA3 Local Storage Module

SQLite-based local storage for sessions and archives when central storage is unavailable.
Uses FTS5 for keyword search (no vector/semantic search in local mode).

This provides a degraded but functional experience for users without central storage.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from .utils import log, get_mira_path
from .db_manager import get_db_manager

LOCAL_DB = "local_store.db"

LOCAL_SCHEMA = """
-- Projects table (mirrors central)
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    slug TEXT,
    git_remote TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_projects_path ON projects(path);
CREATE INDEX IF NOT EXISTS idx_projects_git_remote ON projects(git_remote);

-- Sessions table (mirrors central)
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    summary TEXT,
    keywords TEXT,  -- JSON array
    facts TEXT,     -- JSON array
    task_description TEXT,
    git_branch TEXT,
    models_used TEXT,    -- JSON array
    tools_used TEXT,     -- JSON array
    files_touched TEXT,  -- JSON array
    message_count INTEGER DEFAULT 0,
    started_at TEXT,
    ended_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);

-- FTS for sessions (keyword search)
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    summary, task_description, keywords, facts,
    content='sessions', content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
    INSERT INTO sessions_fts(rowid, summary, task_description, keywords, facts)
    VALUES (new.id, new.summary, new.task_description, new.keywords, new.facts);
END;

CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, rowid, summary, task_description, keywords, facts)
    VALUES('delete', old.id, old.summary, old.task_description, old.keywords, old.facts);
END;

CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, rowid, summary, task_description, keywords, facts)
    VALUES('delete', old.id, old.summary, old.task_description, old.keywords, old.facts);
    INSERT INTO sessions_fts(rowid, summary, task_description, keywords, facts)
    VALUES (new.id, new.summary, new.task_description, new.keywords, new.facts);
END;

-- Archives table (mirrors central)
CREATE TABLE IF NOT EXISTS archives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER,
    line_count INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id)
);
CREATE INDEX IF NOT EXISTS idx_archives_session ON archives(session_id);

-- FTS for archive content
CREATE VIRTUAL TABLE IF NOT EXISTS archives_fts USING fts5(
    content,
    content='archives', content_rowid='id'
);

-- Triggers for archive FTS
CREATE TRIGGER IF NOT EXISTS archives_ai AFTER INSERT ON archives BEGIN
    INSERT INTO archives_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS archives_ad AFTER DELETE ON archives BEGIN
    INSERT INTO archives_fts(archives_fts, rowid, content)
    VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS archives_au AFTER UPDATE ON archives BEGIN
    INSERT INTO archives_fts(archives_fts, rowid, content)
    VALUES('delete', old.id, old.content);
    INSERT INTO archives_fts(rowid, content) VALUES (new.id, new.content);
END;

-- Custodian preferences (local copy)
CREATE TABLE IF NOT EXISTS custodian (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    category TEXT,
    confidence REAL DEFAULT 0.5,
    frequency INTEGER DEFAULT 1,
    source_sessions TEXT,  -- JSON array
    first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_custodian_category ON custodian(category);

-- Error patterns (local copy)
CREATE TABLE IF NOT EXISTS error_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    signature TEXT NOT NULL,
    error_type TEXT,
    error_text TEXT NOT NULL,
    solution TEXT,
    file_path TEXT,
    occurrences INTEGER DEFAULT 1,
    first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, signature)
);
CREATE INDEX IF NOT EXISTS idx_errors_project ON error_patterns(project_id);

-- Decisions (local copy)
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    category TEXT,
    decision TEXT NOT NULL,
    reasoning TEXT,
    alternatives TEXT,  -- JSON array
    confidence REAL DEFAULT 0.5,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project_id);
CREATE INDEX IF NOT EXISTS idx_decisions_category ON decisions(category);
"""

_initialized = False


def init_local_db():
    """Initialize the local storage database."""
    global _initialized
    if _initialized:
        return

    db = get_db_manager()
    db.init_schema(LOCAL_DB, LOCAL_SCHEMA)
    _initialized = True
    log("Local storage database initialized")


def get_or_create_project(
    path: str,
    slug: Optional[str] = None,
    git_remote: Optional[str] = None
) -> int:
    """Get or create a project, returns project ID."""
    init_local_db()
    db = get_db_manager()

    # First try to find by git_remote (canonical identifier)
    if git_remote:
        row = db.execute_read_one(
            LOCAL_DB,
            "SELECT id FROM projects WHERE git_remote = ?",
            (git_remote,)
        )
        if row:
            return row['id']

    # Then try by path
    row = db.execute_read_one(
        LOCAL_DB,
        "SELECT id FROM projects WHERE path = ?",
        (path,)
    )
    if row:
        # Update git_remote if we have it now
        if git_remote:
            db.execute_write(
                LOCAL_DB,
                "UPDATE projects SET git_remote = ? WHERE id = ?",
                (git_remote, row['id'])
            )
        return row['id']

    # Create new project
    return db.execute_write(
        LOCAL_DB,
        "INSERT INTO projects (path, slug, git_remote) VALUES (?, ?, ?)",
        (path, slug, git_remote)
    )


def upsert_session(
    project_id: int,
    session_id: str,
    summary: str = "",
    keywords: List[str] = None,
    facts: List[str] = None,
    task_description: str = "",
    git_branch: str = None,
    models_used: List[str] = None,
    tools_used: List[str] = None,
    files_touched: List[str] = None,
    message_count: int = 0,
    started_at: str = None,
    ended_at: str = None,
) -> int:
    """Create or update a session, returns session ID."""
    init_local_db()
    db = get_db_manager()

    keywords_json = json.dumps(keywords or [])
    facts_json = json.dumps(facts or [])
    models_json = json.dumps(models_used or [])
    tools_json = json.dumps(tools_used or [])
    files_json = json.dumps(files_touched or [])

    # Check if exists
    row = db.execute_read_one(
        LOCAL_DB,
        "SELECT id FROM sessions WHERE project_id = ? AND session_id = ?",
        (project_id, session_id)
    )

    if row:
        # Update
        db.execute_write(
            LOCAL_DB,
            """UPDATE sessions SET
                summary = ?, keywords = ?, facts = ?, task_description = ?,
                git_branch = ?, models_used = ?, tools_used = ?, files_touched = ?,
                message_count = ?, started_at = ?, ended_at = ?
            WHERE id = ?""",
            (summary, keywords_json, facts_json, task_description,
             git_branch, models_json, tools_json, files_json,
             message_count, started_at, ended_at, row['id'])
        )
        return row['id']
    else:
        # Insert
        return db.execute_write(
            LOCAL_DB,
            """INSERT INTO sessions (
                project_id, session_id, summary, keywords, facts, task_description,
                git_branch, models_used, tools_used, files_touched,
                message_count, started_at, ended_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, session_id, summary, keywords_json, facts_json, task_description,
             git_branch, models_json, tools_json, files_json,
             message_count, started_at, ended_at)
        )


def upsert_archive(
    session_db_id: int,
    content: str,
    content_hash: str,
) -> int:
    """Store or update a conversation archive, returns archive ID."""
    init_local_db()
    db = get_db_manager()

    size_bytes = len(content.encode('utf-8'))
    line_count = content.count('\n') + 1 if content else 0

    # Check if exists
    row = db.execute_read_one(
        LOCAL_DB,
        "SELECT id FROM archives WHERE session_id = ?",
        (session_db_id,)
    )

    if row:
        # Update
        db.execute_write(
            LOCAL_DB,
            """UPDATE archives SET
                content = ?, content_hash = ?, size_bytes = ?, line_count = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?""",
            (content, content_hash, size_bytes, line_count, row['id'])
        )
        return row['id']
    else:
        # Insert
        return db.execute_write(
            LOCAL_DB,
            """INSERT INTO archives (session_id, content, content_hash, size_bytes, line_count)
            VALUES (?, ?, ?, ?, ?)""",
            (session_db_id, content, content_hash, size_bytes, line_count)
        )


def get_archive(session_uuid: str) -> Optional[str]:
    """Get archive content by session UUID (the filename)."""
    init_local_db()
    db = get_db_manager()

    row = db.execute_read_one(
        LOCAL_DB,
        """SELECT a.content FROM archives a
           JOIN sessions s ON a.session_id = s.id
           WHERE s.session_id = ?""",
        (session_uuid,)
    )
    return row['content'] if row else None


def get_recent_sessions(
    project_id: Optional[int] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Get recent sessions, optionally filtered by project."""
    init_local_db()
    db = get_db_manager()

    if project_id:
        rows = db.execute_read(
            LOCAL_DB,
            """SELECT s.*, p.path as project_path FROM sessions s
               JOIN projects p ON s.project_id = p.id
               WHERE s.project_id = ?
               ORDER BY s.started_at DESC LIMIT ?""",
            (project_id, limit)
        )
    else:
        rows = db.execute_read(
            LOCAL_DB,
            """SELECT s.*, p.path as project_path FROM sessions s
               JOIN projects p ON s.project_id = p.id
               ORDER BY s.started_at DESC LIMIT ?""",
            (limit,)
        )

    return [_session_row_to_dict(row) for row in rows]


def search_sessions_fts(
    query: str,
    project_id: Optional[int] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Full-text search on sessions."""
    init_local_db()
    db = get_db_manager()

    # Escape FTS special characters
    safe_query = query.replace('"', '""')

    if project_id:
        rows = db.execute_read(
            LOCAL_DB,
            """SELECT s.*, p.path as project_path,
                      bm25(sessions_fts) as rank
               FROM sessions s
               JOIN projects p ON s.project_id = p.id
               JOIN sessions_fts ON sessions_fts.rowid = s.id
               WHERE sessions_fts MATCH ?
                 AND s.project_id = ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, project_id, limit)
        )
    else:
        rows = db.execute_read(
            LOCAL_DB,
            """SELECT s.*, p.path as project_path,
                      bm25(sessions_fts) as rank
               FROM sessions s
               JOIN projects p ON s.project_id = p.id
               JOIN sessions_fts ON sessions_fts.rowid = s.id
               WHERE sessions_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, limit)
        )

    return [_session_row_to_dict(row) for row in rows]


def search_archives_fts(
    query: str,
    project_id: Optional[int] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Full-text search on archive content."""
    init_local_db()
    db = get_db_manager()

    safe_query = query.replace('"', '""')

    if project_id:
        rows = db.execute_read(
            LOCAL_DB,
            """SELECT a.content, s.session_id, s.summary, p.path as project_path,
                      bm25(archives_fts) as rank
               FROM archives a
               JOIN sessions s ON a.session_id = s.id
               JOIN projects p ON s.project_id = p.id
               JOIN archives_fts ON archives_fts.rowid = a.id
               WHERE archives_fts MATCH ?
                 AND s.project_id = ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, project_id, limit)
        )
    else:
        rows = db.execute_read(
            LOCAL_DB,
            """SELECT a.content, s.session_id, s.summary, p.path as project_path,
                      bm25(archives_fts) as rank
               FROM archives a
               JOIN sessions s ON a.session_id = s.id
               JOIN projects p ON s.project_id = p.id
               JOIN archives_fts ON archives_fts.rowid = a.id
               WHERE archives_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, limit)
        )

    return [dict(row) for row in rows]


def get_session_count() -> int:
    """Get total number of sessions."""
    init_local_db()
    db = get_db_manager()
    row = db.execute_read_one(LOCAL_DB, "SELECT COUNT(*) as count FROM sessions", ())
    return row['count'] if row else 0


def get_archive_stats() -> Dict[str, Any]:
    """Get statistics about stored archives."""
    init_local_db()
    db = get_db_manager()
    row = db.execute_read_one(
        LOCAL_DB,
        """SELECT COUNT(*) as total,
                  COALESCE(SUM(size_bytes), 0) as bytes,
                  COALESCE(SUM(line_count), 0) as lines
           FROM archives""",
        ()
    )
    if row:
        return {
            "total_archives": row['total'],
            "total_bytes": row['bytes'],
            "total_lines": row['lines'],
            "avg_bytes": row['bytes'] // row['total'] if row['total'] > 0 else 0
        }
    return {"total_archives": 0, "total_bytes": 0, "total_lines": 0, "avg_bytes": 0}


# ==================== Custodian Operations ====================

def upsert_custodian(
    key: str,
    value: str,
    category: Optional[str] = None,
    confidence: float = 0.5,
    source_session: Optional[str] = None
):
    """Store or update a custodian preference."""
    init_local_db()
    db = get_db_manager()

    row = db.execute_read_one(
        LOCAL_DB,
        "SELECT id, source_sessions, frequency FROM custodian WHERE key = ?",
        (key,)
    )

    if row:
        sources = json.loads(row['source_sessions'] or '[]')
        if source_session and source_session not in sources:
            sources.append(source_session)
        db.execute_write(
            LOCAL_DB,
            """UPDATE custodian SET
                value = ?, category = ?, confidence = ?,
                frequency = frequency + 1, source_sessions = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE key = ?""",
            (value, category, confidence, json.dumps(sources), key)
        )
    else:
        sources = [source_session] if source_session else []
        db.execute_write(
            LOCAL_DB,
            """INSERT INTO custodian (key, value, category, confidence, source_sessions)
            VALUES (?, ?, ?, ?, ?)""",
            (key, value, category, confidence, json.dumps(sources))
        )


def get_custodian_all() -> List[Dict[str, Any]]:
    """Get all custodian preferences."""
    init_local_db()
    db = get_db_manager()
    rows = db.execute_read(
        LOCAL_DB,
        "SELECT * FROM custodian ORDER BY confidence DESC",
        ()
    )
    result = []
    for row in rows:
        d = dict(row)
        d['source_sessions'] = json.loads(d.get('source_sessions') or '[]')
        result.append(d)
    return result


# ==================== Error Pattern Operations ====================

def upsert_error_pattern(
    project_id: int,
    signature: str,
    error_text: str,
    error_type: Optional[str] = None,
    solution: Optional[str] = None,
    file_path: Optional[str] = None
):
    """Store or update an error pattern."""
    init_local_db()
    db = get_db_manager()

    row = db.execute_read_one(
        LOCAL_DB,
        "SELECT id FROM error_patterns WHERE project_id = ? AND signature = ?",
        (project_id, signature)
    )

    if row:
        db.execute_write(
            LOCAL_DB,
            """UPDATE error_patterns SET
                error_text = ?, error_type = ?, solution = ?, file_path = ?,
                occurrences = occurrences + 1, last_seen = CURRENT_TIMESTAMP
            WHERE id = ?""",
            (error_text, error_type, solution, file_path, row['id'])
        )
    else:
        db.execute_write(
            LOCAL_DB,
            """INSERT INTO error_patterns (
                project_id, signature, error_text, error_type, solution, file_path
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            (project_id, signature, error_text, error_type, solution, file_path)
        )


def search_error_patterns(
    query: str,
    project_id: Optional[int] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Search error patterns."""
    init_local_db()
    db = get_db_manager()

    # Simple LIKE-based search (no FTS for errors)
    like_query = f"%{query}%"

    if project_id:
        rows = db.execute_read(
            LOCAL_DB,
            """SELECT * FROM error_patterns
               WHERE project_id = ? AND (error_text LIKE ? OR solution LIKE ?)
               ORDER BY occurrences DESC, last_seen DESC
               LIMIT ?""",
            (project_id, like_query, like_query, limit)
        )
    else:
        rows = db.execute_read(
            LOCAL_DB,
            """SELECT * FROM error_patterns
               WHERE error_text LIKE ? OR solution LIKE ?
               ORDER BY occurrences DESC, last_seen DESC
               LIMIT ?""",
            (like_query, like_query, limit)
        )

    return [dict(row) for row in rows]


# ==================== Decision Operations ====================

def upsert_decision(
    project_id: int,
    decision: str,
    category: Optional[str] = None,
    reasoning: Optional[str] = None,
    alternatives: List[str] = None,
    confidence: float = 0.5,
    session_db_id: Optional[int] = None
):
    """Store a decision."""
    init_local_db()
    db = get_db_manager()

    db.execute_write(
        LOCAL_DB,
        """INSERT INTO decisions (
            project_id, session_id, category, decision, reasoning, alternatives, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (project_id, session_db_id, category, decision, reasoning,
         json.dumps(alternatives or []), confidence)
    )


def search_decisions(
    query: str,
    project_id: Optional[int] = None,
    category: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search decisions."""
    init_local_db()
    db = get_db_manager()

    like_query = f"%{query}%"
    params = [like_query, like_query]
    sql = """SELECT * FROM decisions WHERE (decision LIKE ? OR reasoning LIKE ?)"""

    if project_id:
        sql += " AND project_id = ?"
        params.append(project_id)

    if category:
        sql += " AND category = ?"
        params.append(category)

    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = db.execute_read(LOCAL_DB, sql, tuple(params))

    result = []
    for row in rows:
        d = dict(row)
        d['alternatives'] = json.loads(d.get('alternatives') or '[]')
        result.append(d)
    return result


# ==================== Helpers ====================

def _session_row_to_dict(row) -> Dict[str, Any]:
    """Convert a session row to a dict with parsed JSON fields."""
    d = dict(row)
    for field in ['keywords', 'facts', 'models_used', 'tools_used', 'files_touched']:
        if field in d and d[field]:
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                d[field] = []
        else:
            d[field] = []
    return d
