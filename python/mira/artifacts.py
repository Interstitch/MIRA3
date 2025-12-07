"""
MIRA3 Artifact Storage Module

SQLite-based storage for artifacts (code blocks, file operations, etc.)
"""

import json
import re

from .utils import get_artifact_db_path, log, extract_query_terms

# Global database connection
_artifact_db = None


def init_artifact_db():
    """Initialize the SQLite database for artifact storage."""
    global _artifact_db
    import sqlite3

    db_path = get_artifact_db_path()
    _artifact_db = sqlite3.connect(str(db_path), check_same_thread=False)
    _artifact_db.row_factory = sqlite3.Row

    cursor = _artifact_db.cursor()

    # File operations table - stores Write and Edit operations for file reconstruction
    cursor.execute('''
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
        )
    ''')

    # General artifacts table
    cursor.execute('''
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
    ''')

    # Indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_ops_session ON file_operations(session_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_ops_path ON file_operations(file_path)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id)')

    # Full-text search for artifacts
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
            content, title, content='artifacts', content_rowid='id'
        )
    ''')

    _artifact_db.commit()
    log("Artifact database initialized")
    return _artifact_db


def get_artifact_db():
    """Get the artifact database connection, initializing if needed."""
    global _artifact_db
    if _artifact_db is None:
        init_artifact_db()
    return _artifact_db


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
    db = get_artifact_db()
    cursor = db.cursor()

    cursor.execute('''
        INSERT INTO file_operations (
            session_id, operation_type, file_path, content,
            old_string, new_string, replace_all, sequence_num, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id, op_type, file_path, content,
        old_string, new_string, 1 if replace_all else 0,
        sequence_num, timestamp
    ))
    db.commit()


def get_file_operations(file_path: str = None, session_id: str = None) -> list:
    """
    Get file operations for reconstruction.

    Args:
        file_path: Filter by file path
        session_id: Filter by session

    Returns list of operations in sequence order.
    """
    db = get_artifact_db()
    cursor = db.cursor()

    sql = 'SELECT * FROM file_operations WHERE 1=1'
    params = []

    if file_path:
        sql += ' AND file_path = ?'
        params.append(file_path)

    if session_id:
        sql += ' AND session_id = ?'
        params.append(session_id)

    sql += ' ORDER BY session_id, sequence_num'

    cursor.execute(sql, params)
    return [dict(row) for row in cursor.fetchall()]


def reconstruct_file(file_path: str) -> str:
    """
    Reconstruct a file from stored Write and Edit operations.

    Finds the most recent Write operation and applies subsequent Edits.

    Returns the reconstructed file content, or None if not found.
    """
    db = get_artifact_db()
    cursor = db.cursor()

    # Find the most recent Write operation for this file
    cursor.execute('''
        SELECT * FROM file_operations
        WHERE file_path = ? AND operation_type = 'write'
        ORDER BY created_at DESC LIMIT 1
    ''', (file_path,))

    write_op = cursor.fetchone()
    if not write_op:
        return None

    content = write_op['content']
    write_time = write_op['created_at']

    # Get all Edit operations after or at the same time as this Write
    # Use >= to catch edits in the same second, filter by sequence_num > 0 or id > write_id
    write_id = write_op['id']
    cursor.execute('''
        SELECT * FROM file_operations
        WHERE file_path = ? AND operation_type = 'edit' AND id > ?
        ORDER BY id, sequence_num
    ''', (file_path, write_id))

    for edit in cursor.fetchall():
        old = edit['old_string']
        new = edit['new_string']
        replace_all = edit['replace_all']

        if old and old in content:
            if replace_all:
                content = content.replace(old, new)
            else:
                content = content.replace(old, new, 1)

    return content


def extract_file_operations_from_messages(messages: list, session_id: str) -> int:
    """
    Extract Write and Edit tool operations from conversation messages.

    Parses tool_use blocks to find file operations and stores them.

    Returns the number of operations stored.
    """
    ops_stored = 0
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
                    store_file_operation(
                        session_id=session_id,
                        op_type='write',
                        file_path=file_path,
                        content=file_content,
                        sequence_num=sequence_num,
                        timestamp=timestamp
                    )
                    ops_stored += 1
                    sequence_num += 1

            elif name == 'Edit':
                old_string = inp.get('old_string', '')
                new_string = inp.get('new_string', '')
                replace_all = inp.get('replace_all', False)

                if old_string:  # Edit must have old_string
                    store_file_operation(
                        session_id=session_id,
                        op_type='edit',
                        file_path=file_path,
                        old_string=old_string,
                        new_string=new_string,
                        replace_all=replace_all,
                        sequence_num=sequence_num,
                        timestamp=timestamp
                    )
                    ops_stored += 1
                    sequence_num += 1

    return ops_stored


def get_artifact_stats() -> dict:
    """Get statistics about stored artifacts."""
    db = get_artifact_db()
    cursor = db.cursor()

    stats = {'total': 0, 'by_type': {}, 'by_language': {}, 'file_operations': 0}

    # Count artifacts
    cursor.execute('SELECT COUNT(*) FROM artifacts')
    stats['total'] = cursor.fetchone()[0]

    cursor.execute('SELECT artifact_type, COUNT(*) FROM artifacts GROUP BY artifact_type')
    for row in cursor.fetchall():
        stats['by_type'][row[0]] = row[1]

    cursor.execute('SELECT language, COUNT(*) FROM artifacts WHERE language IS NOT NULL GROUP BY language')
    for row in cursor.fetchall():
        stats['by_language'][row[0]] = row[1]

    # Count file operations
    cursor.execute('SELECT COUNT(*) FROM file_operations')
    stats['file_operations'] = cursor.fetchone()[0]

    return stats


def get_journey_stats() -> dict:
    """
    Get journey statistics from file operations.

    Returns metrics about files created, edited, lines written, etc.
    This helps a fresh Claude understand the development effort and trajectory.
    """
    db = get_artifact_db()
    cursor = db.cursor()

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
        cursor.execute("SELECT COUNT(*) FROM file_operations WHERE operation_type = 'write'")
        stats['files_created'] = cursor.fetchone()[0]

        # Count Edit operations (modifications)
        cursor.execute("SELECT COUNT(*) FROM file_operations WHERE operation_type = 'edit'")
        stats['total_edits'] = cursor.fetchone()[0]

        # Count unique files touched
        cursor.execute("SELECT COUNT(DISTINCT file_path) FROM file_operations")
        stats['unique_files'] = cursor.fetchone()[0]

        # Count files that were modified (have edit operations)
        cursor.execute("""
            SELECT COUNT(DISTINCT file_path) FROM file_operations
            WHERE operation_type = 'edit'
        """)
        stats['files_modified'] = cursor.fetchone()[0]

        # Estimate lines written from Write operations (content length / avg line length ~40)
        cursor.execute("""
            SELECT SUM(LENGTH(content)) FROM file_operations
            WHERE operation_type = 'write' AND content IS NOT NULL
        """)
        total_chars = cursor.fetchone()[0] or 0
        stats['lines_written'] = total_chars // 40  # Rough estimate

        # Also count lines from edit new_string additions
        cursor.execute("""
            SELECT SUM(LENGTH(new_string)) FROM file_operations
            WHERE operation_type = 'edit' AND new_string IS NOT NULL
        """)
        edit_chars = cursor.fetchone()[0] or 0
        stats['lines_written'] += edit_chars // 40

        # Most active files (by total operations)
        cursor.execute("""
            SELECT file_path, COUNT(*) as ops,
                   SUM(CASE WHEN operation_type = 'write' THEN 1 ELSE 0 END) as writes,
                   SUM(CASE WHEN operation_type = 'edit' THEN 1 ELSE 0 END) as edits
            FROM file_operations
            GROUP BY file_path
            ORDER BY ops DESC
            LIMIT 10
        """)
        for row in cursor.fetchall():
            # Extract just the filename for readability
            full_path = row[0]
            filename = full_path.split('/')[-1] if '/' in full_path else full_path
            stats['most_active_files'].append({
                'file': filename,
                'full_path': full_path,
                'total_ops': row[1],
                'writes': row[2],
                'edits': row[3]
            })

        # Recently touched files (last 5 unique files)
        cursor.execute("""
            SELECT DISTINCT file_path FROM file_operations
            ORDER BY created_at DESC
            LIMIT 5
        """)
        for row in cursor.fetchall():
            full_path = row[0]
            filename = full_path.split('/')[-1] if '/' in full_path else full_path
            stats['recent_files'].append(filename)

    except Exception as e:
        log(f"Error getting journey stats: {e}")

    return stats


def _escape_fts_term(term: str) -> str:
    """Escape special FTS5 characters in a search term."""
    # FTS5 special characters that need escaping: " - * ( ) : ^
    # Wrap term in quotes to treat as literal, escape any internal quotes
    escaped = term.replace('"', '""')
    return f'"{escaped}"'


def search_artifacts_for_query(query: str, limit: int = 10) -> list:
    """Search artifacts using full-text search."""
    db = get_artifact_db()
    cursor = db.cursor()

    # Extract search terms using centralized function (alphanumeric, 3+ chars)
    terms = extract_query_terms(query, max_terms=5)
    if not terms:
        return []

    # Build FTS query with escaped terms to prevent FTS injection
    escaped_terms = [_escape_fts_term(t) for t in terms]
    fts_query = ' OR '.join(escaped_terms)

    try:
        cursor.execute('''
            SELECT a.* FROM artifacts a
            JOIN artifacts_fts fts ON a.id = fts.rowid
            WHERE artifacts_fts MATCH ?
            ORDER BY rank LIMIT ?
        ''', (fts_query, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'session_id': row['session_id'],
                'artifact_type': row['artifact_type'],
                'title': row['title'],
                'language': row['language'],
                'excerpt': row['content'][:500] + ('...' if len(row['content']) > 500 else ''),
            })
        return results
    except Exception as e:
        log(f"Artifact search error: {e}")
        return []


def store_artifact(session_id: str, artifact_type: str, content: str,
                   language: str = None, title: str = None, role: str = None,
                   message_index: int = None, timestamp: str = None) -> bool:
    """
    Store a detected artifact in the database.

    Args:
        session_id: The conversation session ID
        artifact_type: Type of artifact (code_block, list, table, config, error, url, command)
        content: The artifact content
        language: Programming language (for code blocks)
        title: Optional title or description
        role: user or assistant
        message_index: Index of the message in conversation
        timestamp: When the artifact was created

    Returns:
        True if stored successfully, False if duplicate
    """
    import hashlib

    db = get_artifact_db()
    cursor = db.cursor()

    # Create content hash for deduplication
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:32]

    try:
        cursor.execute('''
            INSERT INTO artifacts (
                session_id, artifact_type, content, content_hash,
                language, title, line_count, char_count,
                role, message_index, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, artifact_type, content, content_hash,
            language, title, content.count('\n') + 1, len(content),
            role, message_index, timestamp
        ))
        db.commit()

        # Update FTS index
        cursor.execute('''
            INSERT INTO artifacts_fts (rowid, content, title)
            VALUES (?, ?, ?)
        ''', (cursor.lastrowid, content, title or ''))
        db.commit()

        return True
    except Exception as e:
        # Likely duplicate (UNIQUE constraint)
        if 'UNIQUE' not in str(e):
            log(f"Error storing artifact: {e}")
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
                                   message_index: int = None, timestamp: str = None) -> int:
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
            timestamp=timestamp
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
                timestamp=timestamp
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
                timestamp=timestamp
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
                timestamp=timestamp
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
                timestamp=timestamp
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
                    timestamp=timestamp
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
                    timestamp=timestamp
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
            timestamp=timestamp
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
            timestamp=timestamp
        ):
            artifacts_stored += 1

    return artifacts_stored


def extract_artifacts_from_messages(messages: list, session_id: str) -> int:
    """
    Extract artifacts from all messages in a conversation.

    Returns the total number of artifacts stored.
    """
    from .utils import extract_text_content

    total_stored = 0

    for idx, msg in enumerate(messages):
        msg_type = msg.get('type', '')
        role = msg_type if msg_type in ('user', 'assistant') else None

        if not role:
            continue

        message = msg.get('message', {})
        content = extract_text_content(message)
        timestamp = msg.get('timestamp', '')

        if content:
            stored = extract_artifacts_from_content(
                content=content,
                session_id=session_id,
                role=role,
                message_index=idx,
                timestamp=timestamp
            )
            total_stored += stored

    return total_stored
