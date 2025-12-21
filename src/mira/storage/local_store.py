"""
MIRA Local Storage Module

SQLite-based local storage for sessions and archives.
Uses FTS5 for keyword search (no vector/semantic search in local mode).
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from mira.core import log, get_mira_path, DB_LOCAL_STORE
from mira.core.database import get_db_manager


LOCAL_SCHEMA = """
-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    slug TEXT,
    git_remote TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_projects_path ON projects(path);
CREATE INDEX IF NOT EXISTS idx_projects_git_remote ON projects(git_remote);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    summary TEXT,
    keywords TEXT,  -- JSON array
    facts TEXT,     -- JSON array
    task_description TEXT,
    git_branch TEXT,
    models_used TEXT,
    tools_used TEXT,
    files_touched TEXT,
    message_count INTEGER DEFAULT 0,
    started_at TEXT,
    ended_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);

-- FTS for sessions
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

-- Archives table
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
"""

_initialized = False


def init_local_db():
    """Initialize the local SQLite database."""
    global _initialized
    if _initialized:
        return

    db = get_db_manager()
    db.init_schema(DB_LOCAL_STORE, LOCAL_SCHEMA)
    _initialized = True


def _ensure_db():
    """Ensure database is initialized."""
    if not _initialized:
        init_local_db()


def get_session_count() -> int:
    """Get total number of sessions."""
    _ensure_db()
    db = get_db_manager()
    row = db.execute_read_one(DB_LOCAL_STORE, "SELECT COUNT(*) as cnt FROM sessions", ())
    return row['cnt'] if row else 0


def get_project_id(path: str) -> Optional[int]:
    """Get project ID by path."""
    _ensure_db()
    db = get_db_manager()
    row = db.execute_read_one(DB_LOCAL_STORE, "SELECT id FROM projects WHERE path = ?", (path,))
    return row['id'] if row else None


def get_or_create_project(path: str, slug: Optional[str] = None, git_remote: Optional[str] = None) -> int:
    """Get or create a project, return its ID."""
    _ensure_db()
    db = get_db_manager()

    # Try to get existing
    row = db.execute_read_one(DB_LOCAL_STORE, "SELECT id FROM projects WHERE path = ?", (path,))
    if row:
        return row['id']

    # Create new
    return db.execute_write(
        DB_LOCAL_STORE,
        "INSERT INTO projects (path, slug, git_remote) VALUES (?, ?, ?)",
        (path, slug, git_remote)
    )


def upsert_session(
    project_id: int,
    session_id: str,
    summary: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    facts: Optional[List[str]] = None,
    task_description: Optional[str] = None,
    git_branch: Optional[str] = None,
    models_used: Optional[List[str]] = None,
    tools_used: Optional[List[str]] = None,
    files_touched: Optional[List[str]] = None,
    message_count: int = 0,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
) -> int:
    """Insert or update a session."""
    _ensure_db()
    db = get_db_manager()

    # Check if exists
    row = db.execute_read_one(
        DB_LOCAL_STORE,
        "SELECT id FROM sessions WHERE project_id = ? AND session_id = ?",
        (project_id, session_id)
    )

    keywords_json = json.dumps(keywords) if keywords else None
    facts_json = json.dumps(facts) if facts else None
    models_json = json.dumps(models_used) if models_used else None
    tools_json = json.dumps(tools_used) if tools_used else None
    files_json = json.dumps(files_touched) if files_touched else None

    if row:
        # Update
        db.execute_write(
            DB_LOCAL_STORE,
            """UPDATE sessions SET
               summary = COALESCE(?, summary),
               keywords = COALESCE(?, keywords),
               facts = COALESCE(?, facts),
               task_description = COALESCE(?, task_description),
               git_branch = COALESCE(?, git_branch),
               models_used = COALESCE(?, models_used),
               tools_used = COALESCE(?, tools_used),
               files_touched = COALESCE(?, files_touched),
               message_count = ?,
               started_at = COALESCE(?, started_at),
               ended_at = COALESCE(?, ended_at)
            WHERE id = ?""",
            (summary, keywords_json, facts_json, task_description, git_branch,
             models_json, tools_json, files_json, message_count, started_at, ended_at, row['id'])
        )
        return row['id']
    else:
        # Insert
        return db.execute_write(
            DB_LOCAL_STORE,
            """INSERT INTO sessions
               (project_id, session_id, summary, keywords, facts, task_description,
                git_branch, models_used, tools_used, files_touched, message_count,
                started_at, ended_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, session_id, summary, keywords_json, facts_json, task_description,
             git_branch, models_json, tools_json, files_json, message_count, started_at, ended_at)
        )


def get_recent_sessions(
    project_id: Optional[int] = None,
    limit: int = 10,
    since: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Get recent sessions."""
    _ensure_db()
    db = get_db_manager()

    query = """
        SELECT s.*, p.path as project_path
        FROM sessions s
        JOIN projects p ON s.project_id = p.id
        WHERE 1=1
    """
    params = []

    if project_id:
        query += " AND s.project_id = ?"
        params.append(project_id)

    if since:
        query += " AND s.started_at >= ?"
        params.append(since.isoformat())

    query += " ORDER BY s.started_at DESC LIMIT ?"
    params.append(limit)

    rows = db.execute_read(DB_LOCAL_STORE, query, tuple(params))

    results = []
    for row in rows:
        result = dict(row)
        # Parse JSON fields
        for field in ['keywords', 'facts', 'models_used', 'tools_used', 'files_touched']:
            if result.get(field):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    result[field] = []
        results.append(result)

    return results


def search_sessions_fts(query: str, project_id: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Full-text search on sessions."""
    _ensure_db()
    db = get_db_manager()

    sql = """
        SELECT s.*, p.path as project_path
        FROM sessions s
        JOIN projects p ON s.project_id = p.id
        JOIN sessions_fts fts ON s.id = fts.rowid
        WHERE sessions_fts MATCH ?
    """
    params = [query]

    if project_id:
        sql += " AND s.project_id = ?"
        params.append(project_id)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    rows = db.execute_read(DB_LOCAL_STORE, sql, tuple(params))

    results = []
    for row in rows:
        result = dict(row)
        for field in ['keywords', 'facts', 'models_used', 'tools_used', 'files_touched']:
            if result.get(field):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    result[field] = []
        results.append(result)

    return results
