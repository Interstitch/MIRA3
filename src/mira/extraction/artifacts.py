"""
MIRA Artifact Storage Module

Hybrid storage for artifacts:
- File operations (Write/Edit tracking): Local SQLite
- Artifact content (code blocks, lists, etc.): Central Postgres + local fallback
"""

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

from mira.core import log, MIRA_PATH
from mira.core.utils import extract_text_content, extract_query_terms
from mira.core.database import get_db_manager
from mira.core.constants import DB_ARTIFACTS

# Pre-compiled regex patterns
RE_CODE_BLOCK = re.compile(r'```(\w*)\n([\s\S]*?)```')
RE_INDENTED_CODE = re.compile(r'(?:^[ ]{4,}[^\s].*\n?)+', re.MULTILINE)
RE_NUMBERED_LIST = re.compile(r'(?:^\d+[.)]\s+.+\n?)+', re.MULTILINE)
RE_BULLET_LIST = re.compile(r'(?:^[\-\*\+]\s+.+\n?)+', re.MULTILINE)
RE_TABLE = re.compile(r'(?:^\|.+\|.*\n?)+', re.MULTILINE)
RE_JSON_BLOCK = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
RE_ERROR_PATTERNS = [
    re.compile(r'(?:error|exception|traceback|failed|failure)[\s:]+.+', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*File ".+", line \d+', re.MULTILINE),
    re.compile(r'^\s*\w+Error:', re.MULTILINE),
]
RE_SHELL_COMMAND = re.compile(r'^(?:\$|>|#)\s+.+', re.MULTILINE)
RE_URL = re.compile(r'https?://[^\s<>\[\]()\'\"]+[^\s<>\[\]()\'\".,;:!?]')

# Schema for artifacts database
ARTIFACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    content TEXT,
    old_string TEXT,
    new_string TEXT,
    replace_all INTEGER DEFAULT 0,
    sequence_num INTEGER,
    timestamp TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    language TEXT,
    title TEXT,
    line_count INTEGER,
    char_count INTEGER,
    role TEXT,
    message_index INTEGER,
    timestamp TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_file_ops_session ON file_operations(session_id);
CREATE INDEX IF NOT EXISTS idx_file_ops_path ON file_operations(file_path);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
    content, title, content='artifacts', content_rowid='id'
);
"""

# Language detection patterns
LANGUAGE_PATTERNS = {
    'python': [r'\bdef\s+\w+\s*\(', r'\bimport\s+\w+', r'\bclass\s+\w+:', r'^\s*@\w+'],
    'javascript': [r'\bfunction\s+\w+', r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*=', r'=>\s*[{(]'],
    'typescript': [r':\s*(string|number|boolean|any)\b', r'\binterface\s+\w+', r'<\w+>'],
    'bash': [r'^#!/bin/(ba)?sh', r'\$\{?\w+\}?', r'\becho\s+', r'\bif\s+\[\[?'],
    'sql': [r'\bSELECT\s+', r'\bFROM\s+', r'\bWHERE\s+', r'\bINSERT\s+INTO\b'],
    'json': [r'^\s*\{[\s\S]*\}\s*$', r'^\s*\[[\s\S]*\]\s*$'],
    'yaml': [r'^\w+:\s*$', r'^\s+-\s+\w+:', r'^\w+:\s+\w+'],
    'html': [r'<\w+[^>]*>', r'</\w+>', r'<!DOCTYPE'],
    'css': [r'\.\w+\s*\{', r'#\w+\s*\{', r'@media\s+'],
    'go': [r'\bfunc\s+\w+', r'\bpackage\s+\w+', r'\btype\s+\w+\s+struct'],
    'rust': [r'\bfn\s+\w+', r'\blet\s+mut\s+', r'\bimpl\s+\w+'],
}


def init_artifact_db():
    """Initialize the SQLite database for artifact storage."""
    db = get_db_manager()
    db.init_schema(DB_ARTIFACTS, ARTIFACTS_SCHEMA)
    log("Artifact database initialized")


def store_file_operation(
    session_id: str,
    op_type: str,
    file_path: str,
    content: Optional[str] = None,
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    replace_all: bool = False,
    sequence_num: int = 0,
    timestamp: Optional[str] = None
):
    """Store a file Write or Edit operation for later reconstruction."""
    db = get_db_manager()
    db.execute_write(
        DB_ARTIFACTS,
        '''INSERT INTO file_operations (
            session_id, operation_type, file_path, content,
            old_string, new_string, replace_all, sequence_num, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (session_id, op_type, file_path, content,
         old_string, new_string, 1 if replace_all else 0,
         sequence_num, timestamp)
    )


def get_file_operations(
    file_path: Optional[str] = None,
    session_id: Optional[str] = None
) -> List[Dict]:
    """Get file operations for reconstruction."""
    db = get_db_manager()

    sql = 'SELECT * FROM file_operations WHERE 1=1'
    params = []

    if file_path:
        sql += ' AND file_path = ?'
        params.append(file_path)

    if session_id:
        sql += ' AND session_id = ?'
        params.append(session_id)

    sql += ' ORDER BY session_id, sequence_num'

    rows = db.execute_read(DB_ARTIFACTS, sql, tuple(params))
    return [dict(row) for row in rows]


def reconstruct_file(file_path: str) -> Optional[str]:
    """Reconstruct a file from stored Write and Edit operations."""
    db = get_db_manager()

    write_op = db.execute_read_one(
        DB_ARTIFACTS,
        '''SELECT * FROM file_operations
           WHERE file_path = ? AND operation_type = 'write'
           ORDER BY created_at DESC LIMIT 1''',
        (file_path,)
    )

    if not write_op:
        return None

    content = write_op['content']
    write_id = write_op['id']

    edits = db.execute_read(
        DB_ARTIFACTS,
        '''SELECT * FROM file_operations
           WHERE file_path = ? AND operation_type = 'edit' AND id > ?
           ORDER BY id, sequence_num''',
        (file_path, write_id)
    )

    for edit in edits:
        old = edit['old_string']
        new = edit['new_string']
        replace_all = edit['replace_all']

        if old and old in content:
            if replace_all:
                content = content.replace(old, new)
            else:
                content = content.replace(old, new, 1)

    return content


def detect_language(content: str) -> Optional[str]:
    """Detect the programming language of a code block."""
    scores = {}
    for lang, patterns in LANGUAGE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                score += 1
        if score > 0:
            scores[lang] = score

    if scores:
        return max(scores, key=scores.get)
    return None


def store_artifact(
    session_id: str,
    artifact_type: str,
    content: str,
    language: Optional[str] = None,
    title: Optional[str] = None,
    role: Optional[str] = None,
    message_index: Optional[int] = None,
    timestamp: Optional[str] = None,
    postgres_session_id: Optional[int] = None,
    storage=None,
    project_path: Optional[str] = None
) -> bool:
    """Store a detected artifact - local first, queue for central sync."""
    line_count = content.count('\n') + 1
    char_count = len(content)

    hash_input = f"{session_id}:{artifact_type}:{content[:500]}"
    content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:32]

    # Store locally first
    local_stored = False
    try:
        db_path = MIRA_PATH / "artifacts.db"
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    language TEXT,
                    title TEXT,
                    line_count INTEGER,
                    char_count INTEGER,
                    role TEXT,
                    message_index INTEGER,
                    timestamp TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, content_hash)
                )
            """)
            cur.execute("""
                INSERT OR IGNORE INTO artifacts
                (session_id, artifact_type, content, content_hash, language, title,
                 line_count, char_count, role, message_index, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, artifact_type, content, content_hash, language, title,
                  line_count, char_count, role, message_index, timestamp))
            conn.commit()
            local_stored = True
        finally:
            conn.close()
    except Exception as e:
        log(f"Local artifact storage failed: {e}")

    # Queue for central sync if configured
    if storage is None:
        try:
            from mira.storage import get_storage
            storage = get_storage()
        except ImportError:
            storage = None

    if storage and storage.central_configured:
        try:
            from mira.storage.sync_queue import get_sync_queue
            queue = get_sync_queue()

            metadata = {
                'role': role,
                'message_index': message_index,
                'timestamp': timestamp,
                'title': title,
            }

            payload = {
                'session_id': session_id,
                'postgres_session_id': postgres_session_id,
                'artifact_type': artifact_type,
                'content': content,
                'language': language,
                'line_count': line_count,
                'metadata': metadata,
                'project_path': project_path,
            }

            queue.enqueue("artifact", content_hash, payload)
        except Exception as e:
            log(f"Failed to queue artifact for sync: {e}")

    return local_stored


def collect_artifacts_from_content(
    content: str,
    session_id: str,
    role: Optional[str] = None,
    message_index: Optional[int] = None,
    timestamp: Optional[str] = None,
    postgres_session_id: Optional[int] = None
) -> List[Dict]:
    """Collect artifacts from message content WITHOUT storing them."""
    artifacts = []

    def add_artifact(artifact_type: str, artifact_content: str,
                     language: Optional[str] = None, title: Optional[str] = None):
        hash_input = f"{session_id}:{artifact_type}:{artifact_content[:500]}"
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:32]

        artifacts.append({
            "session_id": postgres_session_id,
            "_local_session_id": session_id,
            "artifact_type": artifact_type,
            "content": artifact_content,
            "language": language,
            "line_count": artifact_content.count('\n') + 1,
            "metadata": {
                "role": role,
                "message_index": message_index,
                "timestamp": timestamp,
                "title": title,
                "content_hash": content_hash,
            }
        })

    # Fenced code blocks
    for match in RE_CODE_BLOCK.finditer(content):
        lang_hint = match.group(1).lower() if match.group(1) else None
        code = match.group(2).strip()
        if len(code) >= 20:
            language = lang_hint or detect_language(code)
            add_artifact('code_block', code, language, f"{language or 'code'} block")

    # Indented code blocks
    for match in RE_INDENTED_CODE.finditer(content):
        code = match.group(0)
        if code.count('\n') >= 2 and len(code) >= 50:
            language = detect_language(code)
            add_artifact('code_block', code.strip(), language, f"indented {language or 'code'} block")

    # Numbered lists
    for match in RE_NUMBERED_LIST.finditer(content):
        list_content = match.group(0).strip()
        items = [line.strip() for line in list_content.split('\n') if line.strip()]
        if len(items) >= 3:
            add_artifact('list', list_content, None, f"numbered list ({len(items)} items)")

    # Bullet lists
    for match in RE_BULLET_LIST.finditer(content):
        list_content = match.group(0).strip()
        items = [line.strip() for line in list_content.split('\n') if line.strip()]
        if len(items) >= 3:
            add_artifact('list', list_content, None, f"bullet list ({len(items)} items)")

    # Markdown tables
    for match in RE_TABLE.finditer(content):
        table = match.group(0).strip()
        rows = [r for r in table.split('\n') if r.strip()]
        if len(rows) >= 3 and any('---' in r for r in rows):
            add_artifact('table', table, None, f"markdown table ({len(rows)} rows)")

    # JSON blocks
    for match in RE_JSON_BLOCK.finditer(content):
        json_content = match.group(0)
        try:
            parsed = json.loads(json_content)
            if isinstance(parsed, dict) and len(parsed) >= 2:
                add_artifact('config', json_content, 'json', 'JSON configuration')
        except (json.JSONDecodeError, ValueError):
            pass

    # Error messages
    for pattern in RE_ERROR_PATTERNS:
        for match in pattern.finditer(content):
            error_content = match.group(0).strip()
            if len(error_content) >= 50:
                add_artifact('error', error_content, None, 'error/stack trace')

    # URLs
    urls = list(set(RE_URL.findall(content)))
    significant_urls = [u for u in urls if len(u) > 20 and 'example.com' not in u]
    if len(significant_urls) >= 2:
        add_artifact('url', '\n'.join(significant_urls), None, f"URLs ({len(significant_urls)})")

    return artifacts


def extract_file_operations_from_messages(
    messages: list,
    session_id: str,
    postgres_session_id: Optional[int] = None,
    storage=None
) -> int:
    """Extract Write and Edit tool operations from conversation messages."""
    operations = []
    sequence_num = 0

    for msg in messages:
        content = msg.get('message', {}).get('content', [])
        timestamp = msg.get('timestamp', '')

        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict) or item.get('type') != 'tool_use':
                continue

            name = item.get('name', '')
            inp = item.get('input', {})

            if not isinstance(inp, dict):
                continue

            file_path = inp.get('file_path', '')
            if not file_path:
                continue

            if name == 'Write':
                file_content = inp.get('content', '')
                if file_content:
                    hash_input = f"{session_id}:write:{file_path}:{file_content[:500]}"
                    op_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:32]

                    operations.append({
                        'session_id': postgres_session_id,
                        'operation_type': 'write',
                        'file_path': file_path,
                        'content': file_content,
                        'old_string': None,
                        'new_string': None,
                        'replace_all': False,
                        'sequence_num': sequence_num,
                        'timestamp': timestamp,
                        'operation_hash': op_hash,
                        '_local_session_id': session_id,
                    })
                    sequence_num += 1

            elif name == 'Edit':
                old_string = inp.get('old_string', '')
                new_string = inp.get('new_string', '')
                replace_all = inp.get('replace_all', False)

                if old_string:
                    hash_input = f"{session_id}:edit:{file_path}:{old_string[:200]}:{new_string[:200]}"
                    op_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:32]

                    operations.append({
                        'session_id': postgres_session_id,
                        'operation_type': 'edit',
                        'file_path': file_path,
                        'content': None,
                        'old_string': old_string,
                        'new_string': new_string,
                        'replace_all': replace_all,
                        'sequence_num': sequence_num,
                        'timestamp': timestamp,
                        'operation_hash': op_hash,
                        '_local_session_id': session_id,
                    })
                    sequence_num += 1

    if not operations:
        return 0

    ops_stored = 0
    for op in operations:
        try:
            store_file_operation(
                session_id=op['_local_session_id'],
                op_type=op['operation_type'],
                file_path=op['file_path'],
                content=op.get('content'),
                old_string=op.get('old_string'),
                new_string=op.get('new_string'),
                replace_all=op.get('replace_all', False),
                sequence_num=op['sequence_num'],
                timestamp=op.get('timestamp')
            )
            ops_stored += 1
        except Exception as e:
            log(f"Local file_op store failed: {e}")

    if storage and storage.central_configured:
        try:
            from mira.storage.sync_queue import get_sync_queue
            queue = get_sync_queue()
            for op in operations:
                queue.enqueue("file_operation", op.get('operation_hash', ''), op)
        except Exception as e:
            log(f"Failed to queue file_ops for sync: {e}")

    return ops_stored


def extract_artifacts_from_messages(
    messages: list,
    session_id: str,
    postgres_session_id: Optional[int] = None,
    storage=None,
    message_start_index: int = 0
) -> int:
    """Extract artifacts from all messages using batch insertion."""
    all_artifacts = []

    for idx, msg in enumerate(messages):
        actual_message_index = message_start_index + idx
        msg_type = msg.get('type', '')
        role = msg_type if msg_type in ('user', 'assistant') else None

        if not role:
            continue

        message = msg.get('message', {})
        content = extract_text_content(message)
        timestamp = msg.get('timestamp', '')

        if content:
            artifacts = collect_artifacts_from_content(
                content=content,
                session_id=session_id,
                role=role,
                message_index=actual_message_index,
                timestamp=timestamp,
                postgres_session_id=postgres_session_id,
            )
            all_artifacts.extend(artifacts)

    if not all_artifacts:
        return 0

    stored_count = 0

    # Batch insert to local SQLite
    try:
        db_path = MIRA_PATH / "artifacts.db"
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                language TEXT,
                title TEXT,
                line_count INTEGER,
                char_count INTEGER,
                role TEXT,
                message_index INTEGER,
                timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, content_hash)
            )
        """)

        for artifact in all_artifacts:
            meta = artifact.get('metadata', {})
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO artifacts
                    (session_id, artifact_type, content, content_hash, language, title,
                     line_count, char_count, role, message_index, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    artifact.get('session_id'),
                    artifact.get('artifact_type'),
                    artifact.get('content'),
                    meta.get('content_hash'),
                    artifact.get('language'),
                    meta.get('title'),
                    meta.get('line_count'),
                    meta.get('char_count'),
                    meta.get('role'),
                    meta.get('message_index'),
                    meta.get('timestamp'),
                ))
                if cur.rowcount > 0:
                    stored_count += 1
            except Exception:
                pass

        conn.commit()
        conn.close()
    except Exception as e:
        log(f"Local artifact batch storage failed: {e}")

    # Queue for central sync
    try:
        from mira.storage.sync_queue import get_sync_queue
        queue = get_sync_queue()

        batch_items = []
        for artifact in all_artifacts:
            hash_input = f"artifact:{artifact.get('metadata', {}).get('content_hash', '')}"
            item_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:32]
            batch_items.append(("artifact", item_hash, artifact))

        queue.batch_enqueue(batch_items)
    except Exception as e:
        log(f"Failed to queue artifacts for sync: {e}")

    return stored_count or len(all_artifacts)


def get_artifact_stats(project_path: Optional[str] = None) -> Dict[str, Any]:
    """Get statistics about stored artifacts."""
    db = get_db_manager()

    def _get_stats(session_ids: Optional[List] = None) -> Dict[str, Any]:
        stats = {'total': 0, 'by_type': {}, 'by_language': {}, 'file_operations': 0}

        if session_ids is not None and len(session_ids) == 0:
            return stats

        if session_ids:
            placeholders = ','.join('?' * len(session_ids))
            session_filter = f" WHERE session_id IN ({placeholders})"
            session_params = tuple(session_ids)
        else:
            session_filter = ""
            session_params = ()

        row = db.execute_read_one(
            DB_ARTIFACTS,
            f'SELECT COUNT(*) as cnt FROM artifacts{session_filter}',
            session_params
        )
        stats['total'] = row['cnt'] if row else 0

        rows = db.execute_read(
            DB_ARTIFACTS,
            f'SELECT artifact_type, COUNT(*) as cnt FROM artifacts{session_filter} GROUP BY artifact_type',
            session_params
        )
        for row in rows:
            stats['by_type'][row['artifact_type']] = row['cnt']

        rows = db.execute_read(
            DB_ARTIFACTS,
            f'SELECT language, COUNT(*) as cnt FROM artifacts WHERE language IS NOT NULL{" AND session_id IN (" + placeholders + ")" if session_ids else ""} GROUP BY language',
            session_params
        )
        for row in rows:
            stats['by_language'][row['language']] = row['cnt']

        row = db.execute_read_one(
            DB_ARTIFACTS,
            f'SELECT COUNT(*) as cnt FROM file_operations{session_filter}',
            session_params
        )
        stats['file_operations'] = row['cnt'] if row else 0

        return stats

    global_stats = _get_stats()

    if not project_path:
        return global_stats

    from mira.storage.local_store import get_project_id
    from mira.core.constants import DB_LOCAL_STORE

    project_id = get_project_id(project_path)
    if not project_id:
        return {
            'global': global_stats,
            'project': {'total': 0, 'by_type': {}, 'by_language': {}, 'file_operations': 0}
        }

    rows = db.execute_read(
        DB_LOCAL_STORE,
        "SELECT session_id FROM sessions WHERE project_id = ?",
        (project_id,)
    )
    session_ids = [row['session_id'] for row in rows]

    project_stats = _get_stats(session_ids)

    return {
        'global': global_stats,
        'project': project_stats
    }


def get_journey_stats() -> Dict[str, Any]:
    """Get journey statistics from file operations."""
    db = get_db_manager()

    stats = {
        'files_created': 0,
        'files_modified': 0,
        'total_edits': 0,
        'unique_files': 0,
        'lines_written': 0,
        'most_active_files': [],
        'recent_files': [],
    }

    try:
        row = db.execute_read_one(DB_ARTIFACTS, "SELECT COUNT(*) as cnt FROM file_operations WHERE operation_type = 'write'")
        stats['files_created'] = row['cnt'] if row else 0

        row = db.execute_read_one(DB_ARTIFACTS, "SELECT COUNT(*) as cnt FROM file_operations WHERE operation_type = 'edit'")
        stats['total_edits'] = row['cnt'] if row else 0

        row = db.execute_read_one(DB_ARTIFACTS, "SELECT COUNT(DISTINCT file_path) as cnt FROM file_operations")
        stats['unique_files'] = row['cnt'] if row else 0

        row = db.execute_read_one(DB_ARTIFACTS, """
            SELECT COUNT(DISTINCT file_path) as cnt FROM file_operations
            WHERE operation_type = 'edit'
        """)
        stats['files_modified'] = row['cnt'] if row else 0

        row = db.execute_read_one(DB_ARTIFACTS, """
            SELECT SUM(LENGTH(content)) as total FROM file_operations
            WHERE operation_type = 'write' AND content IS NOT NULL
        """)
        total_chars = row['total'] if row and row['total'] else 0
        stats['lines_written'] = total_chars // 40

        row = db.execute_read_one(DB_ARTIFACTS, """
            SELECT SUM(LENGTH(new_string)) as total FROM file_operations
            WHERE operation_type = 'edit' AND new_string IS NOT NULL
        """)
        edit_chars = row['total'] if row and row['total'] else 0
        stats['lines_written'] += edit_chars // 40

        rows = db.execute_read(DB_ARTIFACTS, """
            SELECT file_path, COUNT(*) as ops,
                   SUM(CASE WHEN operation_type = 'write' THEN 1 ELSE 0 END) as writes,
                   SUM(CASE WHEN operation_type = 'edit' THEN 1 ELSE 0 END) as edits
            FROM file_operations
            GROUP BY file_path
            ORDER BY ops DESC
            LIMIT 25
        """)
        for row in rows:
            full_path = row['file_path']
            if not Path(full_path).exists():
                continue
            filename = full_path.split('/')[-1] if '/' in full_path else full_path
            stats['most_active_files'].append({
                'file': filename,
                'full_path': full_path,
                'total_ops': row['ops'],
                'writes': row['writes'],
                'edits': row['edits']
            })
            if len(stats['most_active_files']) >= 10:
                break

        rows = db.execute_read(DB_ARTIFACTS, """
            SELECT DISTINCT file_path FROM file_operations
            ORDER BY created_at DESC
            LIMIT 15
        """)
        for row in rows:
            full_path = row['file_path']
            if not Path(full_path).exists():
                continue
            filename = full_path.split('/')[-1] if '/' in full_path else full_path
            stats['recent_files'].append(filename)
            if len(stats['recent_files']) >= 5:
                break

    except Exception as e:
        log(f"Error getting journey stats: {e}")

    return stats


def search_artifacts_for_query(
    query: str,
    limit: int = 10,
    storage=None
) -> List[Dict]:
    """Search artifacts using full-text search."""
    if not query:
        return []

    if storage is None:
        try:
            from mira.storage import get_storage
            storage = get_storage()
        except ImportError:
            return _search_artifacts_local(query, limit)

    if not storage.using_central or not storage.postgres:
        return _search_artifacts_local(query, limit)

    try:
        results = storage.postgres.search_artifacts_fts(query, limit=limit)
        formatted = []
        for r in results:
            content = r.get('content', '')
            formatted.append({
                'session_id': r.get('session_id', ''),
                'artifact_type': r.get('artifact_type', ''),
                'title': r.get('metadata', {}).get('title') if isinstance(r.get('metadata'), dict) else None,
                'language': r.get('language'),
                'excerpt': content[:500] + ('...' if len(content) > 500 else ''),
                'project_path': r.get('project_path', ''),
            })
        return formatted
    except Exception as e:
        log(f"Central artifact search error: {e}")
        return _search_artifacts_local(query, limit)


def _search_artifacts_local(query: str, limit: int = 10) -> List[Dict]:
    """Fallback to local SQLite search for artifacts."""
    db = get_db_manager()

    terms = extract_query_terms(query, max_terms=5)
    if not terms:
        return []

    def _escape_fts_term(term: str) -> str:
        escaped = term.replace('"', '""')
        return f'"{escaped}"'

    escaped_terms = [_escape_fts_term(t) for t in terms]
    fts_query = ' OR '.join(escaped_terms)

    try:
        rows = db.execute_read(
            DB_ARTIFACTS,
            '''SELECT a.* FROM artifacts a
               JOIN artifacts_fts fts ON a.id = fts.rowid
               WHERE artifacts_fts MATCH ?
               ORDER BY rank LIMIT ?''',
            (fts_query, limit)
        )

        results = []
        for row in rows:
            results.append({
                'session_id': row['session_id'],
                'artifact_type': row['artifact_type'],
                'title': row['title'],
                'language': row['language'],
                'excerpt': row['content'][:500] + ('...' if len(row['content']) > 500 else ''),
            })
        return results
    except Exception as e:
        log(f"Local artifact search error: {e}")
        return []
