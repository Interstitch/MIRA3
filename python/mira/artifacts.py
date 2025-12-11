"""
MIRA3 Artifact Storage Module

Hybrid storage for artifacts:
- File operations (Write/Edit tracking): Local SQLite (only useful locally)
- Artifact content (code blocks, lists, etc.): Central Postgres (searchable across machines)

Uses centralized db_manager for local thread-safe writes.
"""

import json
import re

from .utils import log, extract_query_terms
from .db_manager import get_db_manager

# Database name for local file operations only
ARTIFACTS_DB = "artifacts.db"

# Pre-compiled regex patterns for artifact extraction (performance optimization)
# These are compiled once at module load, not per-function-call
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
-- File operations table - stores Write and Edit operations for file reconstruction
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

-- General artifacts table
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

-- Indexes
CREATE INDEX IF NOT EXISTS idx_file_ops_session ON file_operations(session_id);
CREATE INDEX IF NOT EXISTS idx_file_ops_path ON file_operations(file_path);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);

-- Full-text search for artifacts
CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
    content, title, content='artifacts', content_rowid='id'
);
"""


def init_artifact_db():
    """Initialize the SQLite database for artifact storage."""
    db = get_db_manager()
    db.init_schema(ARTIFACTS_DB, ARTIFACTS_SCHEMA)
    log("Artifact database initialized")


def store_file_operation(session_id: str, op_type: str, file_path: str,
                         content: str = None, old_string: str = None,
                         new_string: str = None, replace_all: bool = False,
                         sequence_num: int = 0, timestamp: str = None):
    """
    Store a file Write or Edit operation for later reconstruction.

    Args:
        session_id: The conversation session ID
        op_type: 'write' or 'edit'
        file_path: Path to the file being modified
        content: Full file content (for Write operations)
        old_string: Text being replaced (for Edit operations)
        new_string: Replacement text (for Edit operations)
        replace_all: Whether to replace all occurrences
        sequence_num: Order of operation within session
        timestamp: When the operation occurred
    """
    db = get_db_manager()
    db.execute_write(
        ARTIFACTS_DB,
        '''INSERT INTO file_operations (
            session_id, operation_type, file_path, content,
            old_string, new_string, replace_all, sequence_num, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (session_id, op_type, file_path, content,
         old_string, new_string, 1 if replace_all else 0,
         sequence_num, timestamp)
    )


def get_file_operations(file_path: str = None, session_id: str = None) -> list:
    """
    Get file operations for reconstruction.

    Args:
        file_path: Filter by file path
        session_id: Filter by session

    Returns list of operations in sequence order.
    """
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

    rows = db.execute_read(ARTIFACTS_DB, sql, tuple(params))
    return [dict(row) for row in rows]


def reconstruct_file(file_path: str) -> str:
    """
    Reconstruct a file from stored Write and Edit operations.

    Finds the most recent Write operation and applies subsequent Edits.

    Returns the reconstructed file content, or None if not found.
    """
    db = get_db_manager()

    # Find the most recent Write operation for this file
    write_op = db.execute_read_one(
        ARTIFACTS_DB,
        '''SELECT * FROM file_operations
           WHERE file_path = ? AND operation_type = 'write'
           ORDER BY created_at DESC LIMIT 1''',
        (file_path,)
    )

    if not write_op:
        return None

    content = write_op['content']
    write_id = write_op['id']

    # Get all Edit operations after this Write
    edits = db.execute_read(
        ARTIFACTS_DB,
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


def extract_file_operations_from_messages(
    messages: list,
    session_id: str,
    postgres_session_id: int = None,
    storage=None
) -> int:
    """
    Extract Write and Edit tool operations from conversation messages.

    Parses tool_use blocks to find file operations and stores them.
    Prefers central storage (Postgres) with local SQLite fallback.

    Args:
        messages: List of conversation messages
        session_id: UUID session ID (for local storage)
        postgres_session_id: Integer session ID (for central storage)
        storage: Storage instance for central Postgres

    Returns the number of operations stored.
    """
    import hashlib

    operations = []
    sequence_num = 0

    for msg in messages:
        # Check for tool_use in message content
        content = msg.get('message', {}).get('content', [])
        timestamp = msg.get('timestamp', '')

        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue

            if item.get('type') != 'tool_use':
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
                    # Create hash for deduplication
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
                        # For local fallback
                        '_local_session_id': session_id,
                    })
                    sequence_num += 1

            elif name == 'Edit':
                old_string = inp.get('old_string', '')
                new_string = inp.get('new_string', '')
                replace_all = inp.get('replace_all', False)

                if old_string:  # Edit must have old_string
                    # Create hash for deduplication
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
                        # For local fallback
                        '_local_session_id': session_id,
                    })
                    sequence_num += 1

    if not operations:
        return 0

    # Try central storage first
    if storage and storage.using_central and storage.postgres and postgres_session_id:
        try:
            count = storage.postgres.batch_insert_file_operations(operations)
            log(f"Batch inserted {count} file_ops to central storage")
            return count
        except Exception as e:
            log(f"Central file_ops insert failed, falling back to local: {e}")

    # Fallback to local SQLite
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

    return ops_stored


def get_artifact_stats() -> dict:
    """Get statistics about stored artifacts."""
    db = get_db_manager()

    stats = {'total': 0, 'by_type': {}, 'by_language': {}, 'file_operations': 0}

    # Count artifacts
    row = db.execute_read_one(ARTIFACTS_DB, 'SELECT COUNT(*) as cnt FROM artifacts')
    stats['total'] = row['cnt'] if row else 0

    rows = db.execute_read(ARTIFACTS_DB, 'SELECT artifact_type, COUNT(*) as cnt FROM artifacts GROUP BY artifact_type')
    for row in rows:
        stats['by_type'][row['artifact_type']] = row['cnt']

    rows = db.execute_read(ARTIFACTS_DB, 'SELECT language, COUNT(*) as cnt FROM artifacts WHERE language IS NOT NULL GROUP BY language')
    for row in rows:
        stats['by_language'][row['language']] = row['cnt']

    # Count file operations
    row = db.execute_read_one(ARTIFACTS_DB, 'SELECT COUNT(*) as cnt FROM file_operations')
    stats['file_operations'] = row['cnt'] if row else 0

    return stats


def get_journey_stats() -> dict:
    """
    Get journey statistics from file operations.

    Returns metrics about files created, edited, lines written, etc.
    This helps a fresh Claude understand the development effort and trajectory.

    Note: Filters out files that no longer exist on disk (stale references
    from refactored code).
    """
    from pathlib import Path

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
        # Count Write operations (new file creations)
        row = db.execute_read_one(ARTIFACTS_DB, "SELECT COUNT(*) as cnt FROM file_operations WHERE operation_type = 'write'")
        stats['files_created'] = row['cnt'] if row else 0

        # Count Edit operations (modifications)
        row = db.execute_read_one(ARTIFACTS_DB, "SELECT COUNT(*) as cnt FROM file_operations WHERE operation_type = 'edit'")
        stats['total_edits'] = row['cnt'] if row else 0

        # Count unique files touched
        row = db.execute_read_one(ARTIFACTS_DB, "SELECT COUNT(DISTINCT file_path) as cnt FROM file_operations")
        stats['unique_files'] = row['cnt'] if row else 0

        # Count files that were modified (have edit operations)
        row = db.execute_read_one(ARTIFACTS_DB, """
            SELECT COUNT(DISTINCT file_path) as cnt FROM file_operations
            WHERE operation_type = 'edit'
        """)
        stats['files_modified'] = row['cnt'] if row else 0

        # Estimate lines written from Write operations (content length / avg line length ~40)
        row = db.execute_read_one(ARTIFACTS_DB, """
            SELECT SUM(LENGTH(content)) as total FROM file_operations
            WHERE operation_type = 'write' AND content IS NOT NULL
        """)
        total_chars = row['total'] if row and row['total'] else 0
        stats['lines_written'] = total_chars // 40  # Rough estimate

        # Also count lines from edit new_string additions
        row = db.execute_read_one(ARTIFACTS_DB, """
            SELECT SUM(LENGTH(new_string)) as total FROM file_operations
            WHERE operation_type = 'edit' AND new_string IS NOT NULL
        """)
        edit_chars = row['total'] if row and row['total'] else 0
        stats['lines_written'] += edit_chars // 40

        # Most active files (by total operations)
        rows = db.execute_read(ARTIFACTS_DB, """
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
            # Skip files that no longer exist (stale references from refactored code)
            if not Path(full_path).exists():
                continue
            # Extract just the filename for readability
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

        # Recently touched files (last 5 unique files that still exist)
        rows = db.execute_read(ARTIFACTS_DB, """
            SELECT DISTINCT file_path FROM file_operations
            ORDER BY created_at DESC
            LIMIT 15
        """)
        for row in rows:
            full_path = row['file_path']
            # Skip files that no longer exist
            if not Path(full_path).exists():
                continue
            filename = full_path.split('/')[-1] if '/' in full_path else full_path
            stats['recent_files'].append(filename)
            if len(stats['recent_files']) >= 5:
                break

    except Exception as e:
        log(f"Error getting journey stats: {e}")

    return stats


def _escape_fts_term(term: str) -> str:
    """Escape special FTS5 characters in a search term."""
    # FTS5 special characters that need escaping: " - * ( ) : ^
    # Wrap term in quotes to treat as literal, escape any internal quotes
    escaped = term.replace('"', '""')
    return f'"{escaped}"'


def search_artifacts_for_query(query: str, limit: int = 10, storage=None) -> list:
    """
    Search artifacts using full-text search in central Postgres.

    Args:
        query: Search query
        limit: Maximum results
        storage: Storage instance for central Postgres

    Returns list of matching artifacts with excerpts.
    """
    if not query:
        return []

    # Get storage if not provided
    if storage is None:
        try:
            from .storage import get_storage
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


def _search_artifacts_local(query: str, limit: int = 10) -> list:
    """Fallback to local SQLite search for artifacts."""
    db = get_db_manager()

    terms = extract_query_terms(query, max_terms=5)
    if not terms:
        return []

    escaped_terms = [_escape_fts_term(t) for t in terms]
    fts_query = ' OR '.join(escaped_terms)

    try:
        rows = db.execute_read(
            ARTIFACTS_DB,
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


def store_artifact(session_id: str, artifact_type: str, content: str,
                   language: str = None, title: str = None, role: str = None,
                   message_index: int = None, timestamp: str = None,
                   postgres_session_id: int = None, storage=None,
                   project_path: str = None) -> bool:
    """
    Store a detected artifact - tries central first, queues locally if unavailable.

    Storage strategy:
    1. Try to store directly to central Postgres
    2. If central unavailable, queue to local sync queue
    3. Sync worker will flush queue to central when available

    Args:
        session_id: The conversation session ID (string)
        artifact_type: Type of artifact (code_block, list, table, config, error, url, command)
        content: The artifact content
        language: Programming language (for code blocks)
        title: Optional title or description
        role: user or assistant
        message_index: Index of the message in conversation
        timestamp: When the artifact was created
        postgres_session_id: The Postgres session ID (int) for foreign key
        storage: Storage instance for central Postgres
        project_path: Project path for queued items

    Returns:
        True if stored (central or queued), False if failed completely
    """
    import hashlib

    # Build payload for both central storage and queue
    line_count = content.count('\n') + 1
    metadata = {
        'role': role,
        'message_index': message_index,
        'timestamp': timestamp,
        'title': title,
    }

    # Generate hash for deduplication
    hash_input = f"{session_id}:{artifact_type}:{content[:500]}"
    item_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:32]

    # Get storage if not provided
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            storage = None

    # Try central storage first
    if storage and storage.using_central and storage.postgres:
        try:
            storage.postgres.insert_artifact(
                session_id=postgres_session_id,
                artifact_type=artifact_type,
                content=content,
                language=language,
                line_count=line_count,
                metadata=metadata,
            )
            return True  # Success - stored in central
        except Exception as e:
            # Duplicate is fine - already stored
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                return True
            # Other errors - fall through to queue
            log(f"Central artifact storage failed, queueing: {e}")

    # Central unavailable or failed - queue for later sync
    try:
        from .sync_queue import get_sync_queue
        queue = get_sync_queue()

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

        queued = queue.enqueue("artifact", item_hash, payload)
        if queued:
            log(f"Artifact queued for sync: {artifact_type}")
        return queued
    except Exception as e:
        log(f"Failed to queue artifact: {e}")
        return False


# Language detection patterns for code blocks
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


def detect_language(content: str) -> str:
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


def extract_artifacts_from_content(content: str, session_id: str, role: str = None,
                                   message_index: int = None, timestamp: str = None,
                                   postgres_session_id: int = None, storage=None) -> int:
    """
    Extract and store artifacts from message content.

    Detects:
    - Code blocks (fenced with ``` or indented)
    - Numbered and bullet lists (3+ items)
    - Markdown tables
    - Configuration blocks (JSON, YAML)
    - Error messages and stack traces
    - URLs
    - Shell commands

    Args:
        content: Message content to extract artifacts from
        session_id: String session ID
        role: user or assistant
        message_index: Index of message in conversation
        timestamp: Message timestamp
        postgres_session_id: Postgres session ID for foreign key (required for central storage)
        storage: Storage instance for central Postgres

    Returns the number of artifacts stored.
    """
    artifacts_stored = 0

    # 1. Fenced code blocks (```language ... ```)
    code_block_pattern = r'```(\w*)\n([\s\S]*?)```'
    for match in re.finditer(code_block_pattern, content):
        lang_hint = match.group(1).lower() if match.group(1) else None
        code = match.group(2).strip()

        if len(code) < 20:  # Skip trivial blocks
            continue

        # Detect language if not specified
        language = lang_hint or detect_language(code)

        if store_artifact(
            session_id=session_id,
            artifact_type='code_block',
            content=code,
            language=language,
            title=f"{language or 'code'} block",
            role=role,
            message_index=message_index,
            timestamp=timestamp,
            postgres_session_id=postgres_session_id,
            storage=storage
        ):
            artifacts_stored += 1

    # 2. Indented code blocks (4+ spaces, multiple lines)
    indented_pattern = r'(?:^[ ]{4,}[^\s].*\n?)+'
    for match in re.finditer(indented_pattern, content, re.MULTILINE):
        code = match.group(0)
        # Must have at least 3 lines to be significant
        if code.count('\n') >= 2 and len(code) >= 50:
            language = detect_language(code)
            if store_artifact(
                session_id=session_id,
                artifact_type='code_block',
                content=code.strip(),
                language=language,
                title=f"indented {language or 'code'} block",
                role=role,
                message_index=message_index,
                timestamp=timestamp,
                postgres_session_id=postgres_session_id,
                storage=storage
            ):
                artifacts_stored += 1

    # 3. Numbered lists (3+ items)
    numbered_list_pattern = r'(?:^\d+[.)]\s+.+\n?)+'
    for match in re.finditer(numbered_list_pattern, content, re.MULTILINE):
        list_content = match.group(0).strip()
        items = [line.strip() for line in list_content.split('\n') if line.strip()]
        if len(items) >= 3:
            if store_artifact(
                session_id=session_id,
                artifact_type='list',
                content=list_content,
                title=f"numbered list ({len(items)} items)",
                role=role,
                message_index=message_index,
                timestamp=timestamp,
                postgres_session_id=postgres_session_id,
                storage=storage
            ):
                artifacts_stored += 1

    # 4. Bullet lists (3+ items)
    bullet_list_pattern = r'(?:^[\-\*\+]\s+.+\n?)+'
    for match in re.finditer(bullet_list_pattern, content, re.MULTILINE):
        list_content = match.group(0).strip()
        items = [line.strip() for line in list_content.split('\n') if line.strip()]
        if len(items) >= 3:
            if store_artifact(
                session_id=session_id,
                artifact_type='list',
                content=list_content,
                title=f"bullet list ({len(items)} items)",
                role=role,
                message_index=message_index,
                timestamp=timestamp,
                postgres_session_id=postgres_session_id,
                storage=storage
            ):
                artifacts_stored += 1

    # 5. Markdown tables
    table_pattern = r'(?:^\|.+\|.*\n?)+'
    for match in re.finditer(table_pattern, content, re.MULTILINE):
        table = match.group(0).strip()
        rows = [r for r in table.split('\n') if r.strip()]
        # Must have header separator and at least 2 data rows
        if len(rows) >= 3 and any('---' in r or '---' in r for r in rows):
            if store_artifact(
                session_id=session_id,
                artifact_type='table',
                content=table,
                title=f"markdown table ({len(rows)} rows)",
                role=role,
                message_index=message_index,
                timestamp=timestamp,
                postgres_session_id=postgres_session_id,
                storage=storage
            ):
                artifacts_stored += 1

    # 6. JSON configuration blocks (standalone, not in code fences)
    json_pattern = r'(?<![`])(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})(?![`])'
    for match in re.finditer(json_pattern, content):
        json_content = match.group(1)
        try:
            import json as json_module
            parsed = json_module.loads(json_content)
            # Only store if it has multiple keys (actual config, not simple object)
            if isinstance(parsed, dict) and len(parsed) >= 2:
                if store_artifact(
                    session_id=session_id,
                    artifact_type='config',
                    content=json_content,
                    language='json',
                    title='JSON configuration',
                    role=role,
                    message_index=message_index,
                    timestamp=timestamp,
                    postgres_session_id=postgres_session_id,
                    storage=storage
                ):
                    artifacts_stored += 1
        except (json.JSONDecodeError, ValueError):
            pass

    # 7. Error messages and stack traces
    error_patterns = [
        r'(?:Error|Exception|Traceback)[:\s].*(?:\n\s+.*)*',
        r'(?:^|\n)(?:at\s+\w+.*\n)+',  # JavaScript stack trace
        r'File ".*", line \d+.*(?:\n.*)+',  # Python traceback
    ]
    for pattern in error_patterns:
        for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
            error_content = match.group(0).strip()
            if len(error_content) >= 50:
                if store_artifact(
                    session_id=session_id,
                    artifact_type='error',
                    content=error_content,
                    title='error/stack trace',
                    role=role,
                    message_index=message_index,
                    timestamp=timestamp,
                    postgres_session_id=postgres_session_id,
                    storage=storage
                ):
                    artifacts_stored += 1

    # 8. Shell commands (lines starting with $ or common command patterns)
    command_pattern = r'^\$\s+(.+)$'
    commands = []
    for match in re.finditer(command_pattern, content, re.MULTILINE):
        cmd = match.group(1).strip()
        if len(cmd) >= 10:  # Non-trivial command
            commands.append(cmd)

    if commands:
        if store_artifact(
            session_id=session_id,
            artifact_type='command',
            content='\n'.join(commands),
            language='bash',
            title=f"shell commands ({len(commands)})",
            role=role,
            message_index=message_index,
            timestamp=timestamp,
            postgres_session_id=postgres_session_id,
            storage=storage
        ):
            artifacts_stored += 1

    # 9. URLs (extract significant URLs, not inline links)
    url_pattern = r'https?://[^\s<>\"\')]+(?:/[^\s<>\"\')]*)?'
    urls = list(set(re.findall(url_pattern, content)))
    # Filter to significant URLs (not just example.com)
    significant_urls = [u for u in urls if len(u) > 20 and 'example.com' not in u]

    if len(significant_urls) >= 2:  # Multiple URLs worth storing
        if store_artifact(
            session_id=session_id,
            artifact_type='url',
            content='\n'.join(significant_urls),
            title=f"URLs ({len(significant_urls)})",
            role=role,
            message_index=message_index,
            timestamp=timestamp,
            postgres_session_id=postgres_session_id,
            storage=storage
        ):
            artifacts_stored += 1

    return artifacts_stored


def collect_artifacts_from_content(content: str, session_id: str, role: str = None,
                                    message_index: int = None, timestamp: str = None,
                                    postgres_session_id: int = None) -> list:
    """
    Collect artifacts from message content WITHOUT storing them.

    Returns a list of artifact dicts ready for batch insertion.
    This is much faster than storing one at a time.
    """
    import hashlib

    artifacts = []

    def add_artifact(artifact_type: str, artifact_content: str, language: str = None, title: str = None):
        """Helper to add artifact to collection with deduplication."""
        # Generate hash for deduplication
        hash_input = f"{session_id}:{artifact_type}:{artifact_content[:500]}"
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:32]

        artifacts.append({
            "session_id": postgres_session_id,
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

    # 1. Fenced code blocks (```language ... ```) - uses pre-compiled RE_CODE_BLOCK
    for match in RE_CODE_BLOCK.finditer(content):
        lang_hint = match.group(1).lower() if match.group(1) else None
        code = match.group(2).strip()
        if len(code) >= 20:
            language = lang_hint or detect_language(code)
            add_artifact('code_block', code, language, f"{language or 'code'} block")

    # 2. Indented code blocks (4+ spaces, multiple lines) - uses pre-compiled RE_INDENTED_CODE
    for match in RE_INDENTED_CODE.finditer(content):
        code = match.group(0)
        if code.count('\n') >= 2 and len(code) >= 50:
            language = detect_language(code)
            add_artifact('code_block', code.strip(), language, f"indented {language or 'code'} block")

    # 3. Numbered lists (3+ items) - uses pre-compiled RE_NUMBERED_LIST
    for match in RE_NUMBERED_LIST.finditer(content):
        list_content = match.group(0).strip()
        items = [line.strip() for line in list_content.split('\n') if line.strip()]
        if len(items) >= 3:
            add_artifact('list', list_content, None, f"numbered list ({len(items)} items)")

    # 4. Bullet lists (3+ items) - uses pre-compiled RE_BULLET_LIST
    for match in RE_BULLET_LIST.finditer(content):
        list_content = match.group(0).strip()
        items = [line.strip() for line in list_content.split('\n') if line.strip()]
        if len(items) >= 3:
            add_artifact('list', list_content, None, f"bullet list ({len(items)} items)")

    # 5. Markdown tables - uses pre-compiled RE_TABLE
    for match in RE_TABLE.finditer(content):
        table = match.group(0).strip()
        rows = [r for r in table.split('\n') if r.strip()]
        if len(rows) >= 3 and any('---' in r for r in rows):
            add_artifact('table', table, None, f"markdown table ({len(rows)} rows)")

    # 6. JSON configuration blocks - uses pre-compiled RE_JSON_BLOCK
    for match in RE_JSON_BLOCK.finditer(content):
        json_content = match.group(0)
        try:
            import json as json_module
            parsed = json_module.loads(json_content)
            if isinstance(parsed, dict) and len(parsed) >= 2:
                add_artifact('config', json_content, 'json', 'JSON configuration')
        except (json.JSONDecodeError, ValueError):
            pass

    # 7. Error messages and stack traces - uses pre-compiled RE_ERROR_PATTERNS
    for pattern in RE_ERROR_PATTERNS:
        for match in pattern.finditer(content):
            error_content = match.group(0).strip()
            if len(error_content) >= 50:
                add_artifact('error', error_content, None, 'error/stack trace')

    # 8. URLs - uses pre-compiled RE_URL
    urls = list(set(RE_URL.findall(content)))
    significant_urls = [u for u in urls if len(u) > 20 and 'example.com' not in u]
    if len(significant_urls) >= 2:
        add_artifact('url', '\n'.join(significant_urls), None, f"URLs ({len(significant_urls)})")

    return artifacts


def extract_artifacts_from_messages(messages: list, session_id: str,
                                     postgres_session_id: int = None,
                                     storage=None,
                                     message_start_index: int = 0) -> int:
    """
    Extract artifacts from all messages in a conversation.

    Uses BATCH insertion for performance - collects all artifacts first,
    then inserts in a single database operation.

    Supports incremental extraction via message_start_index - artifacts
    will have correct message_index relative to the full conversation.

    Args:
        messages: List of conversation messages (may be a slice for incremental)
        session_id: String session ID
        postgres_session_id: Postgres session ID for foreign key (required for central storage)
        storage: Storage instance for central Postgres
        message_start_index: Starting index in full conversation (for incremental processing)

    Returns the total number of artifacts stored.
    """
    from .utils import extract_text_content

    # Collect all artifacts first (fast - no DB calls)
    all_artifacts = []

    for idx, msg in enumerate(messages):
        # Calculate actual message index in full conversation
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
                message_index=actual_message_index,  # Use actual index for incremental support
                timestamp=timestamp,
                postgres_session_id=postgres_session_id,
            )
            all_artifacts.extend(artifacts)

    # Batch insert all artifacts in one operation
    if not all_artifacts:
        return 0

    # Get storage if not provided
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            storage = None

    # Try batch insert to central
    if storage and storage.using_central and storage.postgres:
        try:
            count = storage.postgres.batch_insert_artifacts(all_artifacts)
            log(f"Batch inserted {count} artifacts to central storage")
            return count
        except Exception as e:
            log(f"Batch insert failed, falling back to queue: {e}")

    # Fallback: queue for later sync
    try:
        from .sync_queue import get_sync_queue
        queue = get_sync_queue()
        import hashlib

        queued = 0
        for artifact in all_artifacts:
            hash_input = f"artifact:{artifact.get('metadata', {}).get('content_hash', '')}"
            item_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:32]
            if queue.enqueue("artifact", item_hash, artifact):
                queued += 1

        log(f"Queued {queued} artifacts for later sync")
        return queued
    except Exception as e:
        log(f"Failed to queue artifacts: {e}")
        return 0
