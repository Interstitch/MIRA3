"""
MIRA Error Pattern Recognition

Tracks errors and their solutions from conversation history.
Enables mira_error_lookup tool.
"""

import hashlib
import json
import re
from datetime import datetime
from typing import Dict, List, Optional

from mira.core import log, DB_INSIGHTS
from mira.core.database import get_db_manager


INSIGHTS_SCHEMA = """
-- Error patterns table
CREATE TABLE IF NOT EXISTS error_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_signature TEXT NOT NULL,
    error_type TEXT,
    error_message TEXT NOT NULL,
    normalized_message TEXT,
    solution_summary TEXT,
    solution_details TEXT,
    file_context TEXT,
    occurrence_count INTEGER DEFAULT 1,
    first_seen TEXT,
    last_seen TEXT,
    source_sessions TEXT,
    resolution_success INTEGER DEFAULT 0
);

-- Error solutions
CREATE TABLE IF NOT EXISTS error_solutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_pattern_id INTEGER,
    solution_text TEXT NOT NULL,
    solution_type TEXT,
    confidence REAL DEFAULT 0.5,
    session_id TEXT,
    timestamp TEXT,
    FOREIGN KEY (error_pattern_id) REFERENCES error_patterns(id)
);

-- Decision journal
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_hash TEXT UNIQUE,
    decision_summary TEXT NOT NULL,
    decision_details TEXT,
    reasoning TEXT,
    alternatives_considered TEXT,
    context TEXT,
    category TEXT,
    outcome TEXT,
    session_id TEXT,
    timestamp TEXT,
    confidence REAL DEFAULT 0.5
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_error_sig ON error_patterns(error_signature);
CREATE INDEX IF NOT EXISTS idx_error_type ON error_patterns(error_type);
CREATE INDEX IF NOT EXISTS idx_decision_cat ON decisions(category);

-- FTS for error search
CREATE VIRTUAL TABLE IF NOT EXISTS errors_fts USING fts5(
    error_message,
    solution_summary,
    file_context,
    content='error_patterns',
    content_rowid='id'
);

-- FTS for decision search
CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
    decision_summary,
    reasoning,
    context,
    content='decisions',
    content_rowid='id'
);
"""


def init_insights_db():
    """Initialize the insights database."""
    db = get_db_manager()
    db.init_schema(DB_INSIGHTS, INSIGHTS_SCHEMA)


def normalize_error_message(error_msg: str) -> str:
    """
    Normalize an error message for comparison.
    Removes variable parts (line numbers, file paths, etc.).
    """
    normalized = error_msg

    # Remove file paths
    normalized = re.sub(r'(?:/[^\s:,\)]+)+(?:\.[a-zA-Z0-9]+)?', '<FILE>', normalized)
    normalized = re.sub(r'[A-Z]:\\[^\s:,\)]+', '<FILE>', normalized)

    # Remove timestamps
    normalized = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?', '<TIME>', normalized)

    # Remove line/column numbers
    normalized = re.sub(r'line \d+', 'line <N>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r':\d+:\d+', ':<N>:<N>', normalized)
    normalized = re.sub(r':\d+\b', ':<N>', normalized)

    # Remove memory addresses
    normalized = re.sub(r'0x[0-9a-fA-F]+', '<ADDR>', normalized)

    # Remove UUIDs
    normalized = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<UUID>', normalized, flags=re.IGNORECASE)

    return normalized.strip()


def extract_error_type(error_msg: str) -> Optional[str]:
    """Extract the error type from an error message."""
    patterns = [
        r'^((?:[A-Z][a-z]+)+Error)(?:\s*:|\s*\()',
        r'^((?:[A-Z][a-z]+)+Exception)(?:\s*:|\s*\()',
        r'((?:[A-Z][a-z]+)+Error)\s*:',
        r'((?:[A-Z][a-z]+)+Exception)\s*:',
        r'\b(panic):\s',
        r'^(fatal error):\s',
        r'\berror\[([A-Z]\d+)\]',
    ]

    for pattern in patterns:
        match = re.search(pattern, error_msg, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def generate_error_signature(error_msg: str, error_type: Optional[str] = None) -> str:
    """Generate a unique signature for an error pattern."""
    normalized = normalize_error_message(error_msg)
    if error_type:
        sig_input = f"{error_type}:{normalized[:200]}"
    else:
        sig_input = normalized[:200]
    return hashlib.sha256(sig_input.encode()).hexdigest()[:16]


def record_error_pattern(
    error_message: str,
    solution: Optional[str] = None,
    file_context: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[int]:
    """Record an error pattern and optionally its solution."""
    init_insights_db()
    db = get_db_manager()

    error_type = extract_error_type(error_message)
    normalized = normalize_error_message(error_message)
    signature = generate_error_signature(error_message, error_type)

    # Check if exists
    row = db.execute_read_one(
        DB_INSIGHTS,
        "SELECT id, occurrence_count, source_sessions FROM error_patterns WHERE error_signature = ?",
        (signature,)
    )

    now = datetime.now().isoformat()

    if row:
        # Update existing
        sessions = json.loads(row['source_sessions'] or '[]')
        if session_id and session_id not in sessions:
            sessions.append(session_id)

        db.execute_write(
            DB_INSIGHTS,
            """UPDATE error_patterns SET
               occurrence_count = occurrence_count + 1,
               last_seen = ?,
               source_sessions = ?,
               solution_summary = COALESCE(?, solution_summary)
            WHERE id = ?""",
            (now, json.dumps(sessions), solution, row['id'])
        )
        return row['id']
    else:
        # Insert new
        sessions = [session_id] if session_id else []
        return db.execute_write(
            DB_INSIGHTS,
            """INSERT INTO error_patterns
               (error_signature, error_type, error_message, normalized_message,
                solution_summary, file_context, first_seen, last_seen, source_sessions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (signature, error_type, error_message[:2000], normalized[:1000],
             solution, file_context, now, now, json.dumps(sessions))
        )


def search_error_solutions(query: str, limit: int = 5) -> List[Dict]:
    """Search for error solutions using FTS."""
    init_insights_db()
    db = get_db_manager()

    try:
        rows = db.execute_read(
            DB_INSIGHTS,
            """SELECT e.*, rank
               FROM error_patterns e
               JOIN errors_fts fts ON e.id = fts.rowid
               WHERE errors_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit)
        )

        results = []
        for row in rows:
            result = dict(row)
            if result.get('source_sessions'):
                try:
                    result['source_sessions'] = json.loads(result['source_sessions'])
                except json.JSONDecodeError:
                    result['source_sessions'] = []
            results.append(result)
        return results
    except Exception as e:
        log(f"Error search failed: {e}")
        return []


def get_error_stats() -> Dict:
    """Get statistics about recorded errors."""
    init_insights_db()
    db = get_db_manager()

    try:
        total = db.execute_read_one(DB_INSIGHTS, "SELECT COUNT(*) as cnt FROM error_patterns", ())
        with_solution = db.execute_read_one(
            DB_INSIGHTS,
            "SELECT COUNT(*) as cnt FROM error_patterns WHERE solution_summary IS NOT NULL",
            ()
        )
        by_type = db.execute_read(
            DB_INSIGHTS,
            "SELECT error_type, COUNT(*) as cnt FROM error_patterns GROUP BY error_type ORDER BY cnt DESC LIMIT 10",
            ()
        )

        return {
            "total": total['cnt'] if total else 0,
            "with_solution": with_solution['cnt'] if with_solution else 0,
            "by_type": {row['error_type'] or 'unknown': row['cnt'] for row in by_type}
        }
    except Exception as e:
        log(f"Error stats failed: {e}")
        return {"total": 0, "with_solution": 0, "by_type": {}}


# Error detection patterns (tiered by confidence)
ERROR_PATTERNS = [
    # Tier 1: Very specific formats (0.95 confidence)
    (r'Traceback \(most recent call last\):[\s\S]*?(?:^\w+Error|^\w+Exception):\s*(.+)', 'traceback', 0.95),
    (r'^(\w+Error:\s+.+)$', 'python', 0.90),
    (r'^(\w+Exception:\s+.+)$', 'python', 0.90),
    # Tier 2: Language-specific patterns (0.85 confidence)
    (r'error\[E\d+\]:\s*(.+)', 'rust', 0.85),
    (r'error TS\d+:\s*(.+)', 'typescript', 0.85),
    (r'failed to compile[\s\S]*?error:\s*(.+)', 'compilation', 0.80),
    # Tier 3: Generic patterns (0.70 confidence)
    (r'(?:^|\n)(?:Error|ERROR):\s*(.+)', 'generic', 0.70),
    (r'(?:^|\n)(?:Failed|FAILED):\s*(.+)', 'failure', 0.70),
]


def _extract_solution_summary(assistant_content: str) -> Optional[str]:
    """Extract a solution summary from assistant response."""
    solution_patterns = [
        r'(?:the )?(?:fix|solution|answer) (?:is|was)[:\s]+(.{20,200})',
        r'(?:to fix|to solve|to resolve)[:\s]+(.{20,200})',
        r'(?:you need to|you should|try)[:\s]+(.{20,200})',
    ]

    for pattern in solution_patterns:
        match = re.search(pattern, assistant_content, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:200]

    return None


def extract_errors_from_conversation(
    conversation: dict,
    session_id: str,
    project_path: Optional[str] = None,
    storage=None
) -> int:
    """
    Extract error patterns from a conversation.

    Looks for error messages in user messages and solutions in assistant responses.
    """
    messages = conversation.get('messages', [])
    if len(messages) < 2:
        return 0

    init_insights_db()
    errors_found = 0
    seen_signatures = set()

    for i, msg in enumerate(messages):
        if msg.get('role') != 'user':
            continue

        content = msg.get('content', '')
        if isinstance(content, list):
            content = ' '.join(
                item.get('text', '') for item in content
                if isinstance(item, dict) and item.get('type') == 'text'
            )

        if len(content) < 20:
            continue

        for pattern, error_category, confidence in ERROR_PATTERNS:
            try:
                matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

                for error_match in matches:
                    if isinstance(error_match, tuple):
                        error_msg = error_match[0].strip()
                    else:
                        error_msg = error_match.strip()

                    if len(error_msg) < 10:
                        continue

                    signature = generate_error_signature(error_msg)
                    if signature in seen_signatures:
                        continue
                    seen_signatures.add(signature)

                    # Look for solution in next assistant messages
                    solution = None
                    for look_ahead in range(1, 4):
                        msg_idx = i + look_ahead
                        if msg_idx >= len(messages):
                            break
                        next_msg = messages[msg_idx]
                        if next_msg.get('role') == 'assistant':
                            next_content = next_msg.get('content', '')
                            if isinstance(next_content, list):
                                next_content = ' '.join(
                                    item.get('text', '') for item in next_content
                                    if isinstance(item, dict) and item.get('type') == 'text'
                                )
                            solution = _extract_solution_summary(next_content)
                            if solution:
                                break

                    record_error_pattern(
                        error_message=error_msg,
                        solution=solution,
                        session_id=session_id,
                    )
                    errors_found += 1

            except Exception as e:
                log(f"Error pattern extraction failed: {e}")

    return errors_found
