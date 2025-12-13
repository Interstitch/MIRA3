"""
MIRA3 Code History Module

Tracks file operations (Read/Write/Edit) across conversation sessions,
enabling:
- Timeline view of all changes to a file
- Symbol (function/class) tracking across sessions
- File reconstruction at any historical point

This is the "code archaeology" feature for MIRA.
"""

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .utils import log, get_mira_path
from .db_manager import get_db_manager

# Database name
CODE_HISTORY_DB = "code_history.db"

# Schema definition
CODE_HISTORY_SCHEMA = """
-- File operations (reads, writes, edits) across sessions
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

-- File snapshots (full content from Read/Write operations)
CREATE TABLE IF NOT EXISTS file_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id INTEGER NOT NULL UNIQUE,
    content TEXT NOT NULL,
    line_count INTEGER,
    byte_size INTEGER,
    compressed INTEGER DEFAULT 0,
    FOREIGN KEY (operation_id) REFERENCES file_operations(id) ON DELETE CASCADE
);

-- Edit operations with before/after strings
CREATE TABLE IF NOT EXISTS file_edits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id INTEGER NOT NULL UNIQUE,
    old_string TEXT NOT NULL,
    new_string TEXT NOT NULL,
    line_hint INTEGER,
    FOREIGN KEY (operation_id) REFERENCES file_operations(id) ON DELETE CASCADE
);

-- Symbol definitions extracted from snapshots
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

-- Track which sessions have been processed
CREATE TABLE IF NOT EXISTS processing_state (
    session_id TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL,
    operations_found INTEGER DEFAULT 0
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_file_ops_path ON file_operations(file_path);
CREATE INDEX IF NOT EXISTS idx_file_ops_session ON file_operations(session_id);
CREATE INDEX IF NOT EXISTS idx_file_ops_timestamp ON file_operations(timestamp);
CREATE INDEX IF NOT EXISTS idx_file_ops_project ON file_operations(project_path);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbol_definitions(symbol_name);
CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbol_definitions(symbol_type);
CREATE INDEX IF NOT EXISTS idx_snapshots_hash ON file_operations(content_hash);
"""

# Compression threshold (bytes) - compress content larger than this
COMPRESSION_THRESHOLD = 10000


def init_code_history_db():
    """Initialize the code history database."""
    db = get_db_manager()
    db.init_schema(CODE_HISTORY_DB, CODE_HISTORY_SCHEMA)


# ==================== Data Classes ====================

@dataclass
class FileOperation:
    """Represents a file operation extracted from a conversation."""
    session_id: str
    file_path: str
    operation: str  # 'read', 'write', 'edit'
    timestamp: str
    content_hash: Optional[str] = None
    project_path: Optional[str] = None
    message_index: int = 0
    # For snapshots (read/write)
    content: Optional[str] = None
    # For edits
    old_string: Optional[str] = None
    new_string: Optional[str] = None


@dataclass
class SymbolDef:
    """Represents a symbol definition extracted from code."""
    name: str
    symbol_type: str  # 'function', 'class', 'method', 'interface', etc.
    line_number: int
    language: str
    signature: Optional[str] = None


@dataclass
class TimelineEntry:
    """An entry in a file's change timeline."""
    session_id: str
    date: str
    operation: str
    summary: Optional[str] = None
    symbols_changed: List[str] = field(default_factory=list)


@dataclass
class ReconstructionResult:
    """Result of reconstructing a file at a point in time."""
    file_path: str
    target_date: str
    content: Optional[str]
    confidence: float
    source_snapshot_date: Optional[str] = None
    edits_applied: int = 0
    edits_failed: int = 0
    gaps: List[str] = field(default_factory=list)


# ==================== Symbol Extraction ====================

# Multi-language symbol patterns
SYMBOL_PATTERNS = {
    'python': [
        (r'^def\s+(\w+)\s*\(', 'function'),
        (r'^class\s+(\w+)', 'class'),
        (r'^\s+def\s+(\w+)\s*\(self', 'method'),
        (r'^\s+async\s+def\s+(\w+)\s*\(self', 'method'),
        (r'^async\s+def\s+(\w+)\s*\(', 'function'),
    ],
    'typescript': [
        (r'(?:export\s+)?(?:async\s+)?function\s+(\w+)', 'function'),
        (r'(?:export\s+)?class\s+(\w+)', 'class'),
        (r'(?:export\s+)?interface\s+(\w+)', 'interface'),
        (r'(?:export\s+)?type\s+(\w+)\s*=', 'type'),
        (r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(', 'function'),
        (r'(?:export\s+)?enum\s+(\w+)', 'enum'),
    ],
    'javascript': [
        (r'(?:export\s+)?(?:async\s+)?function\s+(\w+)', 'function'),
        (r'(?:export\s+)?class\s+(\w+)', 'class'),
        (r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(', 'function'),
    ],
    'rust': [
        (r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)', 'function'),
        (r'(?:pub\s+)?struct\s+(\w+)', 'struct'),
        (r'(?:pub\s+)?enum\s+(\w+)', 'enum'),
        (r'(?:pub\s+)?trait\s+(\w+)', 'trait'),
        (r'impl(?:<[^>]+>)?\s+(\w+)', 'impl'),
    ],
    'go': [
        (r'^func\s+(\w+)\s*\(', 'function'),
        (r'^func\s+\([^)]+\)\s+(\w+)\s*\(', 'method'),
        (r'^type\s+(\w+)\s+struct', 'struct'),
        (r'^type\s+(\w+)\s+interface', 'interface'),
    ],
    'java': [
        (r'(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+)?\s*\{', 'method'),
        (r'(?:public|private|protected)?\s*(?:abstract\s+)?class\s+(\w+)', 'class'),
        (r'(?:public|private|protected)?\s*interface\s+(\w+)', 'interface'),
        (r'(?:public|private|protected)?\s*enum\s+(\w+)', 'enum'),
    ],
    'ruby': [
        (r'^def\s+(\w+)', 'method'),
        (r'^class\s+(\w+)', 'class'),
        (r'^module\s+(\w+)', 'module'),
    ],
}

# File extension to language mapping
EXTENSION_TO_LANG = {
    '.py': 'python',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.rs': 'rust',
    '.go': 'go',
    '.java': 'java',
    '.rb': 'ruby',
}


def detect_language(file_path: str) -> Optional[str]:
    """Detect programming language from file extension."""
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANG.get(ext)


def extract_symbols(content: str, file_path: str) -> List[SymbolDef]:
    """
    Extract symbol definitions from file content.

    Args:
        content: The file content
        file_path: Path to determine language

    Returns:
        List of SymbolDef objects
    """
    language = detect_language(file_path)
    if not language or language not in SYMBOL_PATTERNS:
        return []

    symbols = []
    patterns = SYMBOL_PATTERNS[language]
    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        for pattern, symbol_type in patterns:
            match = re.search(pattern, line)
            if match:
                name = match.group(1)
                # Get the full signature (the whole matching line, cleaned up)
                signature = line.strip()
                symbols.append(SymbolDef(
                    name=name,
                    symbol_type=symbol_type,
                    line_number=line_num,
                    language=language,
                    signature=signature[:200]  # Truncate long signatures
                ))
                break  # Only match first pattern per line

    return symbols


# ==================== Content Handling ====================

def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def compress_content(content: str) -> Tuple[bytes, bool]:
    """
    Optionally compress content if it exceeds threshold.

    Returns:
        Tuple of (content_bytes, is_compressed)
    """
    content_bytes = content.encode('utf-8')
    if len(content_bytes) > COMPRESSION_THRESHOLD:
        compressed = gzip.compress(content_bytes)
        # Only use compression if it actually saves space
        if len(compressed) < len(content_bytes) * 0.8:
            return compressed, True
    return content_bytes, False


def decompress_content(content: bytes, compressed: bool) -> str:
    """Decompress content if it was compressed."""
    if compressed:
        return gzip.decompress(content).decode('utf-8')
    return content.decode('utf-8') if isinstance(content, bytes) else content


# ==================== Extraction from Archives ====================

def extract_file_operations_from_archive(archive_path: Path, session_id: str, project_path: str = "") -> List[FileOperation]:
    """
    Parse a conversation archive and extract all file operations.

    Args:
        archive_path: Path to the JSONL archive file
        session_id: Session ID (slug)
        project_path: Encoded project path

    Returns:
        List of FileOperation objects
    """
    operations = []

    if not archive_path.exists():
        return operations

    try:
        with open(archive_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract timestamp
                timestamp = message.get('timestamp', '')

                # Look for tool_use in assistant messages
                if message.get('role') == 'assistant':
                    content = message.get('content', [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'tool_use':
                                tool_name = item.get('name', '')
                                tool_input = item.get('input', {})

                                op = _extract_operation_from_tool_use(
                                    tool_name, tool_input, timestamp,
                                    session_id, project_path, line_num
                                )
                                if op:
                                    operations.append(op)

                # Look for tool_result in user messages (contains Read results)
                if message.get('role') == 'user':
                    content = message.get('content', [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'tool_result':
                                tool_use_id = item.get('tool_use_id', '')
                                result_content = item.get('content', '')

                                # We need to correlate this with the tool_use
                                # For now, we'll extract Read results from the content
                                if isinstance(result_content, str) and result_content.startswith('     1'):
                                    # This looks like Read output (line-numbered content)
                                    # We'd need to track the tool_use to get the file_path
                                    # This is handled in a second pass
                                    pass

    except Exception as e:
        log(f"Error extracting from archive {archive_path}: {e}")

    return operations


def _extract_operation_from_tool_use(
    tool_name: str,
    tool_input: Dict[str, Any],
    timestamp: str,
    session_id: str,
    project_path: str,
    message_index: int
) -> Optional[FileOperation]:
    """Extract a FileOperation from a tool_use block."""

    if tool_name == 'Read':
        file_path = tool_input.get('file_path', '')
        if file_path:
            return FileOperation(
                session_id=session_id,
                file_path=file_path,
                operation='read',
                timestamp=timestamp,
                project_path=project_path,
                message_index=message_index,
                # Content will be filled in from tool_result
            )

    elif tool_name == 'Write':
        file_path = tool_input.get('file_path', '')
        content = tool_input.get('content', '')
        if file_path and content:
            return FileOperation(
                session_id=session_id,
                file_path=file_path,
                operation='write',
                timestamp=timestamp,
                content_hash=compute_content_hash(content),
                project_path=project_path,
                message_index=message_index,
                content=content,
            )

    elif tool_name == 'Edit':
        file_path = tool_input.get('file_path', '')
        old_string = tool_input.get('old_string', '')
        new_string = tool_input.get('new_string', '')
        if file_path and (old_string or new_string):
            return FileOperation(
                session_id=session_id,
                file_path=file_path,
                operation='edit',
                timestamp=timestamp,
                project_path=project_path,
                message_index=message_index,
                old_string=old_string,
                new_string=new_string,
            )

    return None


def extract_operations_with_results(archive_path: Path, session_id: str, project_path: str = "") -> List[FileOperation]:
    """
    Extract file operations with full content from Read results.

    This does a two-pass extraction:
    1. Collect all tool_use blocks
    2. Match tool_results to get Read content

    Note: Claude Code stores messages in a nested format where the actual
    content is in message.content[].type == 'tool_use', not directly on the message.
    """
    operations = []
    tool_uses = {}  # tool_use_id -> (tool_name, tool_input, timestamp, message_index)

    if not archive_path.exists():
        return operations

    try:
        with open(archive_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue

            timestamp = message.get('timestamp', '')
            msg_type = message.get('type', '')

            # Pass 1: Collect tool_use blocks from assistant messages
            # Format: {"type": "assistant", "message": {"role": "assistant", "content": [...]}}
            if msg_type == 'assistant':
                inner_msg = message.get('message', {})
                content = inner_msg.get('content', [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'tool_use':
                            tool_id = item.get('id', '')
                            tool_name = item.get('name', '')
                            tool_input = item.get('input', {})
                            if tool_id and tool_name in ('Read', 'Write', 'Edit'):
                                tool_uses[tool_id] = (tool_name, tool_input, timestamp, line_num)

            # Pass 2: Match tool_results to tool_uses from user messages
            # Format: {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", ...}]}}
            if msg_type == 'user':
                inner_msg = message.get('message', {})
                content = inner_msg.get('content', [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'tool_result':
                            tool_use_id = item.get('tool_use_id', '')
                            result_content = item.get('content', '')

                            if tool_use_id in tool_uses:
                                tool_name, tool_input, ts, msg_idx = tool_uses[tool_use_id]

                                op = _create_operation_with_result(
                                    tool_name, tool_input, result_content,
                                    ts, session_id, project_path, msg_idx
                                )
                                if op:
                                    operations.append(op)

                                # Remove to avoid duplicate processing
                                del tool_uses[tool_use_id]

        # Handle any tool_uses without results (Write/Edit that succeeded)
        for tool_id, (tool_name, tool_input, ts, msg_idx) in tool_uses.items():
            if tool_name in ('Write', 'Edit'):
                op = _extract_operation_from_tool_use(
                    tool_name, tool_input, ts, session_id, project_path, msg_idx
                )
                if op:
                    operations.append(op)

    except Exception as e:
        log(f"Error extracting operations from {archive_path}: {e}")

    return operations


def _create_operation_with_result(
    tool_name: str,
    tool_input: Dict[str, Any],
    result_content: Any,
    timestamp: str,
    session_id: str,
    project_path: str,
    message_index: int
) -> Optional[FileOperation]:
    """Create a FileOperation with result content (for Read operations)."""

    if tool_name == 'Read':
        file_path = tool_input.get('file_path', '')
        if not file_path:
            return None

        # Parse the Read result content
        content = _parse_read_result(result_content)
        if content:
            return FileOperation(
                session_id=session_id,
                file_path=file_path,
                operation='read',
                timestamp=timestamp,
                content_hash=compute_content_hash(content),
                project_path=project_path,
                message_index=message_index,
                content=content,
            )

    elif tool_name == 'Write':
        file_path = tool_input.get('file_path', '')
        content = tool_input.get('content', '')
        if file_path and content:
            return FileOperation(
                session_id=session_id,
                file_path=file_path,
                operation='write',
                timestamp=timestamp,
                content_hash=compute_content_hash(content),
                project_path=project_path,
                message_index=message_index,
                content=content,
            )

    elif tool_name == 'Edit':
        file_path = tool_input.get('file_path', '')
        old_string = tool_input.get('old_string', '')
        new_string = tool_input.get('new_string', '')
        if file_path:
            return FileOperation(
                session_id=session_id,
                file_path=file_path,
                operation='edit',
                timestamp=timestamp,
                project_path=project_path,
                message_index=message_index,
                old_string=old_string,
                new_string=new_string,
            )

    return None


def _parse_read_result(result_content: Any) -> Optional[str]:
    """
    Parse Read tool result to extract actual file content.

    Read results are formatted with line numbers like:
         1	first line
         2	second line
    """
    if not result_content:
        return None

    if isinstance(result_content, list):
        # Handle list of content blocks
        text_parts = []
        for item in result_content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
            elif isinstance(item, str):
                text_parts.append(item)
        result_content = '\n'.join(text_parts)

    if not isinstance(result_content, str):
        return None

    # Check if it's line-numbered content (Read output format)
    # Pattern: whitespace, line number, tab or arrow, content
    lines = result_content.split('\n')
    content_lines = []

    # Check first few lines to see if this looks like Read output
    line_num_pattern = re.compile(r'^\s*\d+[\t→](.*)$')

    has_line_numbers = False
    for line in lines[:5]:
        if line_num_pattern.match(line):
            has_line_numbers = True
            break

    if has_line_numbers:
        for line in lines:
            match = line_num_pattern.match(line)
            if match:
                content_lines.append(match.group(1))
            elif line.strip() == '':
                content_lines.append('')
        return '\n'.join(content_lines)

    # Not line-numbered, return as-is (might be error message or other content)
    return None


# ==================== Storage Operations ====================

def store_file_operation(op: FileOperation) -> Optional[int]:
    """
    Store a file operation and its associated data.

    Returns:
        The operation ID, or None if failed
    """
    init_code_history_db()
    db = get_db_manager()

    try:
        # Insert the operation - execute_write returns lastrowid
        op_id = db.execute_write(
            CODE_HISTORY_DB,
            """INSERT INTO file_operations
               (session_id, file_path, operation, timestamp, content_hash, project_path, message_index)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (op.session_id, op.file_path, op.operation, op.timestamp,
             op.content_hash, op.project_path, op.message_index)
        )

        if not op_id:
            return None

        # Store snapshot for read/write operations
        if op.operation in ('read', 'write') and op.content:
            line_count = op.content.count('\n') + 1
            byte_size = len(op.content.encode('utf-8'))

            # Check for duplicate by hash
            existing = db.execute_read_one(
                CODE_HISTORY_DB,
                """SELECT s.id FROM file_snapshots s
                   JOIN file_operations o ON s.operation_id = o.id
                   WHERE o.file_path = ? AND o.content_hash = ?""",
                (op.file_path, op.content_hash)
            )

            if not existing:
                # Store the snapshot - execute_write returns lastrowid
                snap_id = db.execute_write(
                    CODE_HISTORY_DB,
                    """INSERT INTO file_snapshots
                       (operation_id, content, line_count, byte_size, compressed)
                       VALUES (?, ?, ?, ?, ?)""",
                    (op_id, op.content, line_count, byte_size, 0)
                )

                # Extract and store symbols
                if snap_id:
                    symbols = extract_symbols(op.content, op.file_path)
                    for sym in symbols:
                        db.execute_write(
                            CODE_HISTORY_DB,
                            """INSERT INTO symbol_definitions
                               (snapshot_id, symbol_name, symbol_type, line_number, language, signature)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (snap_id, sym.name, sym.symbol_type, sym.line_number,
                             sym.language, sym.signature)
                        )

        # Store edit data
        elif op.operation == 'edit' and (op.old_string or op.new_string):
            db.execute_write(
                CODE_HISTORY_DB,
                """INSERT INTO file_edits
                   (operation_id, old_string, new_string, line_hint)
                   VALUES (?, ?, ?, ?)""",
                (op_id, op.old_string or '', op.new_string or '', None)
            )

        return op_id

    except Exception as e:
        log(f"Error storing file operation: {e}")
        return None


def mark_session_processed(session_id: str, operations_found: int):
    """Mark a session as processed for code history."""
    init_code_history_db()
    db = get_db_manager()

    db.execute_write(
        CODE_HISTORY_DB,
        """INSERT OR REPLACE INTO processing_state
           (session_id, processed_at, operations_found)
           VALUES (?, ?, ?)""",
        (session_id, datetime.now(timezone.utc).isoformat(), operations_found)
    )


def is_session_processed(session_id: str) -> bool:
    """Check if a session has been processed for code history."""
    init_code_history_db()
    db = get_db_manager()

    row = db.execute_read_one(
        CODE_HISTORY_DB,
        "SELECT session_id FROM processing_state WHERE session_id = ?",
        (session_id,)
    )
    return row is not None


# ==================== Query Interface ====================

def get_file_timeline(
    file_path: str,
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Get timeline of changes for a file.

    Args:
        file_path: Exact path or pattern (use % for LIKE matching)
        symbol: Filter to sessions that touched this symbol
        start_date: Filter to changes after this date (ISO format)
        end_date: Filter to changes before this date (ISO format)
        limit: Maximum results

    Returns:
        List of timeline entries with session info
    """
    init_code_history_db()
    db = get_db_manager()

    # Build query
    query = """
        SELECT DISTINCT
            fo.session_id,
            fo.timestamp,
            fo.operation,
            fo.file_path,
            GROUP_CONCAT(DISTINCT sd.symbol_name) as symbols
        FROM file_operations fo
        LEFT JOIN file_snapshots fs ON fs.operation_id = fo.id
        LEFT JOIN symbol_definitions sd ON sd.snapshot_id = fs.id
        WHERE 1=1
    """
    params = []

    # File path filter
    if '%' in file_path:
        query += " AND fo.file_path LIKE ?"
    else:
        query += " AND fo.file_path = ?"
    params.append(file_path)

    # Symbol filter
    if symbol:
        query += " AND sd.symbol_name = ?"
        params.append(symbol)

    # Date filters
    if start_date:
        query += " AND fo.timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND fo.timestamp <= ?"
        params.append(end_date)

    query += """
        GROUP BY fo.id
        ORDER BY fo.timestamp DESC
        LIMIT ?
    """
    params.append(limit)

    rows = db.execute_read(CODE_HISTORY_DB, query, tuple(params))

    # Enrich with session summaries from local_store
    results = []
    for row in rows:
        entry = {
            'session_id': row['session_id'],
            'date': row['timestamp'],
            'operation': row['operation'],
            'file_path': row['file_path'],
            'symbols_changed': row['symbols'].split(',') if row['symbols'] else [],
        }

        # Try to get session summary
        try:
            from .local_store import LOCAL_DB
            summary_row = db.execute_read_one(
                LOCAL_DB,
                "SELECT summary FROM sessions WHERE slug = ?",
                (row['session_id'],)
            )
            if summary_row:
                entry['summary'] = summary_row['summary']
        except Exception:
            pass

        results.append(entry)

    return results


def get_symbol_history(
    symbol_name: str,
    symbol_type: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Get history of a symbol (function, class, etc.) across sessions.

    Args:
        symbol_name: Name of the symbol to track
        symbol_type: Filter by type ('function', 'class', etc.)
        limit: Maximum results

    Returns:
        List of appearances with file, line, and session info
    """
    init_code_history_db()
    db = get_db_manager()

    query = """
        SELECT
            sd.symbol_name,
            sd.symbol_type,
            sd.line_number,
            sd.signature,
            sd.language,
            fo.file_path,
            fo.session_id,
            fo.timestamp
        FROM symbol_definitions sd
        JOIN file_snapshots fs ON sd.snapshot_id = fs.id
        JOIN file_operations fo ON fs.operation_id = fo.id
        WHERE sd.symbol_name = ?
    """
    params = [symbol_name]

    if symbol_type:
        query += " AND sd.symbol_type = ?"
        params.append(symbol_type)

    query += " ORDER BY fo.timestamp DESC LIMIT ?"
    params.append(limit)

    rows = db.execute_read(CODE_HISTORY_DB, query, tuple(params))

    return [dict(row) for row in rows]


def get_file_snapshot_at_date(
    file_path: str,
    target_date: str
) -> Optional[Dict[str, Any]]:
    """
    Get the most recent snapshot of a file before a given date.

    Args:
        file_path: Path to the file (supports LIKE patterns with %)
        target_date: ISO format date

    Returns:
        Dict with content, timestamp, session info, or None
    """
    init_code_history_db()
    db = get_db_manager()

    # Use LIKE if pattern contains %, otherwise exact match
    if '%' in file_path:
        path_clause = "fo.file_path LIKE ?"
    else:
        path_clause = "fo.file_path = ?"

    row = db.execute_read_one(
        CODE_HISTORY_DB,
        f"""SELECT
            fs.content,
            fs.line_count,
            fo.timestamp,
            fo.session_id,
            fo.operation,
            fo.file_path
        FROM file_snapshots fs
        JOIN file_operations fo ON fs.operation_id = fo.id
        WHERE {path_clause} AND fo.timestamp <= ?
        ORDER BY fo.timestamp DESC
        LIMIT 1""",
        (file_path, target_date)
    )

    if row:
        return dict(row)
    return None


def get_edits_between(
    file_path: str,
    start_date: str,
    end_date: str
) -> List[Dict[str, Any]]:
    """
    Get all edit operations on a file between two dates.

    Args:
        file_path: Path to the file (supports LIKE patterns with %)
        start_date: Start of range (ISO format)
        end_date: End of range (ISO format)

    Returns:
        List of edit operations in chronological order
    """
    init_code_history_db()
    db = get_db_manager()

    # Use LIKE if pattern contains %, otherwise exact match
    if '%' in file_path:
        path_clause = "fo.file_path LIKE ?"
    else:
        path_clause = "fo.file_path = ?"

    rows = db.execute_read(
        CODE_HISTORY_DB,
        f"""SELECT
            fe.old_string,
            fe.new_string,
            fo.timestamp,
            fo.session_id,
            fo.file_path
        FROM file_edits fe
        JOIN file_operations fo ON fe.operation_id = fo.id
        WHERE {path_clause}
          AND fo.timestamp > ?
          AND fo.timestamp <= ?
        ORDER BY fo.timestamp ASC""",
        (file_path, start_date, end_date)
    )

    return [dict(row) for row in rows]


def reconstruct_file_at_date(file_path: str, target_date: str) -> ReconstructionResult:
    """
    Attempt to reconstruct a file's content at a specific date.

    Algorithm:
    1. Find most recent snapshot before target_date
    2. Apply all edits between snapshot and target_date
    3. Return result with confidence score

    Args:
        file_path: Path to reconstruct
        target_date: Target date (ISO format)

    Returns:
        ReconstructionResult with content and confidence
    """
    # Get base snapshot
    snapshot = get_file_snapshot_at_date(file_path, target_date)

    if not snapshot:
        return ReconstructionResult(
            file_path=file_path,
            target_date=target_date,
            content=None,
            confidence=0.0,
            gaps=["No snapshots found before target date"]
        )

    content = snapshot['content']
    snapshot_date = snapshot['timestamp']

    # Get edits to apply
    edits = get_edits_between(file_path, snapshot_date, target_date)

    if not edits:
        # No edits needed, snapshot is our reconstruction
        return ReconstructionResult(
            file_path=file_path,
            target_date=target_date,
            content=content,
            confidence=1.0,
            source_snapshot_date=snapshot_date,
            edits_applied=0,
            edits_failed=0
        )

    # Apply edits
    applied = 0
    failed = 0
    gaps = []

    for edit in edits:
        old_string = edit['old_string']
        new_string = edit['new_string']

        if old_string in content:
            content = content.replace(old_string, new_string, 1)
            applied += 1
        else:
            failed += 1
            gaps.append(f"Edit at {edit['timestamp']} could not be applied (content mismatch)")

    # Calculate confidence
    total_edits = applied + failed
    confidence = applied / total_edits if total_edits > 0 else 1.0

    return ReconstructionResult(
        file_path=file_path,
        target_date=target_date,
        content=content,
        confidence=confidence,
        source_snapshot_date=snapshot_date,
        edits_applied=applied,
        edits_failed=failed,
        gaps=gaps
    )


def backfill_code_history(max_sessions: int = 0, verbose: bool = False) -> Dict[str, Any]:
    """
    Process all existing conversation files to populate code history database.

    Reads directly from ~/.claude/projects/ to find all conversation files.
    This is idempotent - sessions already processed are skipped.

    Args:
        max_sessions: Maximum sessions to process (0 = unlimited)
        verbose: Log progress for each session

    Returns:
        Dict with backfill statistics
    """
    init_code_history_db()

    # Find all conversation files in Claude's projects directory
    claude_projects = Path.home() / ".claude" / "projects"

    if not claude_projects.exists():
        return {
            "status": "no_conversations",
            "message": "No Claude projects directory found",
            "processed": 0,
        }

    # Get all JSONL files (conversations), excluding agent-* files
    conversation_files = [
        f for f in claude_projects.rglob("*.jsonl")
        if not f.name.startswith("agent-")
    ]

    # Sort by modification time (oldest first for chronological order)
    conversation_files.sort(key=lambda p: p.stat().st_mtime)

    stats = {
        "status": "success",
        "total_conversations": len(conversation_files),
        "processed": 0,
        "skipped": 0,
        "operations_found": 0,
        "errors": [],
    }

    mira_path = get_mira_path()

    for conv_file in conversation_files:
        if max_sessions > 0 and stats["processed"] >= max_sessions:
            stats["status"] = "partial"
            stats["message"] = f"Stopped after {max_sessions} sessions (limit)"
            break

        session_id = conv_file.stem

        # Skip if already processed
        if is_session_processed(session_id):
            stats["skipped"] += 1
            continue

        try:
            # Extract project_path from file location
            # Path: ~/.claude/projects/{encoded-project-path}/{session}.jsonl
            project_path = conv_file.parent.name

            # Also check metadata for project_path (more reliable)
            metadata_file = mira_path / "metadata" / f"{session_id}.json"
            if metadata_file.exists():
                try:
                    meta = json.loads(metadata_file.read_text())
                    project_path = meta.get("project_path", project_path)
                except Exception:
                    pass

            # Extract operations from conversation file
            operations = extract_operations_with_results(conv_file, session_id, project_path)

            # Store each operation
            ops_stored = 0
            for op in operations:
                op_id = store_file_operation(op)
                if op_id:
                    ops_stored += 1

            # Mark session as processed
            mark_session_processed(session_id, ops_stored)

            stats["processed"] += 1
            stats["operations_found"] += ops_stored

            if verbose:
                log(f"[backfill] {session_id[:12]}: {ops_stored} operations")

        except Exception as e:
            stats["errors"].append({
                "session_id": session_id[:12],
                "error": str(e)
            })
            if verbose:
                log(f"[backfill] {session_id[:12]}: ERROR - {e}")

    if not stats["errors"]:
        del stats["errors"]

    return stats


def extract_and_store_from_session(session_id: str, archive_path: Path, project_path: str = "") -> int:
    """
    Extract and store file operations from a single session.

    Called during ingestion for new sessions.

    Args:
        session_id: Session ID (slug)
        archive_path: Path to the archive file
        project_path: Encoded project path

    Returns:
        Number of operations stored
    """
    init_code_history_db()

    # Skip if already processed
    if is_session_processed(session_id):
        return 0

    try:
        operations = extract_operations_with_results(archive_path, session_id, project_path)

        ops_stored = 0
        for op in operations:
            op_id = store_file_operation(op)
            if op_id:
                ops_stored += 1

        mark_session_processed(session_id, ops_stored)
        return ops_stored

    except Exception as e:
        log(f"Code history extraction failed for {session_id}: {e}")
        return 0


def repair_missing_snapshots(verbose: bool = False) -> Dict[str, Any]:
    """
    Repair missing snapshots from operations that were stored without content.

    This handles the case where the initial backfill stored operations but
    didn't properly extract content (due to message format parsing bug).

    Args:
        verbose: Log progress

    Returns:
        Dict with repair statistics
    """
    init_code_history_db()
    db = get_db_manager()

    stats = {
        "status": "success",
        "operations_checked": 0,
        "snapshots_added": 0,
        "edits_added": 0,
        "symbols_added": 0,
        "errors": [],
    }

    # Find operations without corresponding snapshots/edits
    orphan_ops = db.execute_read(
        CODE_HISTORY_DB,
        """SELECT fo.id, fo.session_id, fo.file_path, fo.operation, fo.timestamp, fo.message_index
           FROM file_operations fo
           LEFT JOIN file_snapshots fs ON fs.operation_id = fo.id
           LEFT JOIN file_edits fe ON fe.operation_id = fo.id
           WHERE (fo.operation IN ('read', 'write') AND fs.id IS NULL)
              OR (fo.operation = 'edit' AND fe.id IS NULL)
           ORDER BY fo.session_id""",
        ()
    )

    if not orphan_ops:
        stats["status"] = "no_repairs_needed"
        return stats

    stats["operations_checked"] = len(orphan_ops)

    # Group by session for efficient processing
    sessions = {}
    for op in orphan_ops:
        session_id = op['session_id']
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(op)

    # Process each session
    claude_projects = Path.home() / ".claude" / "projects"

    for session_id, ops in sessions.items():
        # Find the archive file for this session
        archive_files = list(claude_projects.rglob(f"{session_id}.jsonl"))
        if not archive_files:
            stats["errors"].append({"session": session_id[:12], "error": "archive not found"})
            continue

        archive_path = archive_files[0]

        try:
            # Re-extract operations with content
            fresh_ops = extract_operations_with_results(
                archive_path, session_id,
                archive_path.parent.name  # project_path
            )

            # Build lookup by (file_path, message_index, operation)
            fresh_lookup = {}
            for fop in fresh_ops:
                key = (fop.file_path, fop.message_index, fop.operation)
                fresh_lookup[key] = fop

            # Match and repair
            for orphan in ops:
                key = (orphan['file_path'], orphan['message_index'], orphan['operation'])
                fresh = fresh_lookup.get(key)

                if not fresh:
                    continue

                op_id = orphan['id']

                # Add missing snapshot
                if fresh.operation in ('read', 'write') and fresh.content:
                    line_count = fresh.content.count('\n') + 1
                    byte_size = len(fresh.content.encode('utf-8'))

                    # Check for existing snapshot (shouldn't exist, but safety check)
                    existing = db.execute_read_one(
                        CODE_HISTORY_DB,
                        "SELECT id FROM file_snapshots WHERE operation_id = ?",
                        (op_id,)
                    )

                    if not existing:
                        # Insert snapshot - execute_write returns lastrowid
                        snap_id = db.execute_write(
                            CODE_HISTORY_DB,
                            """INSERT INTO file_snapshots
                               (operation_id, content, line_count, byte_size, compressed)
                               VALUES (?, ?, ?, ?, ?)""",
                            (op_id, fresh.content, line_count, byte_size, 0)
                        )
                        stats["snapshots_added"] += 1

                        # Extract and store symbols
                        if snap_id:
                            symbols = extract_symbols(fresh.content, fresh.file_path)
                            for sym in symbols:
                                db.execute_write(
                                    CODE_HISTORY_DB,
                                    """INSERT INTO symbol_definitions
                                       (snapshot_id, symbol_name, symbol_type, line_number, language, signature)
                                       VALUES (?, ?, ?, ?, ?, ?)""",
                                    (snap_id, sym.name, sym.symbol_type, sym.line_number,
                                     sym.language, sym.signature)
                                )
                                stats["symbols_added"] += 1

                # Add missing edit
                elif fresh.operation == 'edit' and (fresh.old_string or fresh.new_string):
                    existing = db.execute_read_one(
                        CODE_HISTORY_DB,
                        "SELECT id FROM file_edits WHERE operation_id = ?",
                        (op_id,)
                    )

                    if not existing:
                        db.execute_write(
                            CODE_HISTORY_DB,
                            """INSERT INTO file_edits
                               (operation_id, old_string, new_string, line_hint)
                               VALUES (?, ?, ?, ?)""",
                            (op_id, fresh.old_string or '', fresh.new_string or '', None)
                        )
                        stats["edits_added"] += 1

            if verbose:
                log(f"[repair] {session_id[:12]}: processed {len(ops)} operations")

        except Exception as e:
            stats["errors"].append({"session": session_id[:12], "error": str(e)})

    if not stats["errors"]:
        del stats["errors"]

    return stats


def backfill_symbols(verbose: bool = False) -> Dict[str, Any]:
    """
    Backfill symbol definitions for snapshots that don't have symbols.

    This processes all snapshots where the file has a supported extension
    and extracts symbols that weren't captured during initial storage.

    Args:
        verbose: Log progress

    Returns:
        Dict with backfill statistics
    """
    init_code_history_db()
    db = get_db_manager()

    stats = {
        "status": "success",
        "snapshots_checked": 0,
        "symbols_added": 0,
        "files_processed": 0,
    }

    # Find snapshots without symbols for supported file types
    supported_ext = tuple(EXTENSION_TO_LANG.keys())
    ext_conditions = " OR ".join(f"fo.file_path LIKE '%{ext}'" for ext in supported_ext)

    orphan_snapshots = db.execute_read(
        CODE_HISTORY_DB,
        f"""SELECT fs.id, fs.content, fo.file_path
           FROM file_snapshots fs
           JOIN file_operations fo ON fs.operation_id = fo.id
           LEFT JOIN symbol_definitions sd ON sd.snapshot_id = fs.id
           WHERE sd.id IS NULL
             AND ({ext_conditions})""",
        ()
    )

    stats["snapshots_checked"] = len(orphan_snapshots)

    if not orphan_snapshots:
        stats["status"] = "no_symbols_needed"
        return stats

    for snap in orphan_snapshots:
        snap_id = snap['id']
        content = snap['content']
        file_path = snap['file_path']

        if not content:
            continue

        symbols = extract_symbols(content, file_path)
        if symbols:
            stats["files_processed"] += 1
            for sym in symbols:
                db.execute_write(
                    CODE_HISTORY_DB,
                    """INSERT INTO symbol_definitions
                       (snapshot_id, symbol_name, symbol_type, line_number, language, signature)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (snap_id, sym.name, sym.symbol_type, sym.line_number,
                     sym.language, sym.signature)
                )
                stats["symbols_added"] += 1

    if verbose:
        log(f"[backfill_symbols] Processed {stats['files_processed']} files, added {stats['symbols_added']} symbols")

    return stats


def get_code_history_stats() -> Dict[str, Any]:
    """Get statistics about code history database."""
    init_code_history_db()
    db = get_db_manager()

    stats = {}

    try:
        # Total operations
        row = db.execute_read_one(
            CODE_HISTORY_DB,
            "SELECT COUNT(*) as cnt FROM file_operations",
            ()
        )
        stats['total_operations'] = row['cnt'] if row else 0

        # Operations by type
        rows = db.execute_read(
            CODE_HISTORY_DB,
            "SELECT operation, COUNT(*) as cnt FROM file_operations GROUP BY operation",
            ()
        )
        stats['operations_by_type'] = {row['operation']: row['cnt'] for row in rows}

        # Total snapshots
        row = db.execute_read_one(
            CODE_HISTORY_DB,
            "SELECT COUNT(*) as cnt FROM file_snapshots",
            ()
        )
        stats['total_snapshots'] = row['cnt'] if row else 0

        # Total symbols
        row = db.execute_read_one(
            CODE_HISTORY_DB,
            "SELECT COUNT(*) as cnt FROM symbol_definitions",
            ()
        )
        stats['total_symbols'] = row['cnt'] if row else 0

        # Unique files tracked
        row = db.execute_read_one(
            CODE_HISTORY_DB,
            "SELECT COUNT(DISTINCT file_path) as cnt FROM file_operations",
            ()
        )
        stats['unique_files'] = row['cnt'] if row else 0

        # Sessions processed
        row = db.execute_read_one(
            CODE_HISTORY_DB,
            "SELECT COUNT(*) as cnt FROM processing_state",
            ()
        )
        stats['sessions_processed'] = row['cnt'] if row else 0

    except Exception as e:
        stats['error'] = str(e)

    return stats
