"""
MIRA3 Insights Module

Provides advanced analysis features:
- Error pattern recognition and solution lookup
- Decision journal with reasoning
- Conversation similarity detection
"""

import json
import re
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from .utils import log, get_mira_path, extract_query_terms


def get_insights_db_path() -> Path:
    """Get the path to the insights database."""
    return get_mira_path() / "insights.db"


def init_insights_db():
    """Initialize the insights database for errors, decisions, etc."""
    db_path = get_insights_db_path()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Error patterns table - tracks errors and their solutions
    cursor.execute("""
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
        )
    """)

    # Error solutions - links errors to specific solutions found
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS error_solutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_pattern_id INTEGER,
            solution_text TEXT NOT NULL,
            solution_type TEXT,
            confidence REAL DEFAULT 0.5,
            session_id TEXT,
            timestamp TEXT,
            FOREIGN KEY (error_pattern_id) REFERENCES error_patterns(id)
        )
    """)

    # Decision journal - tracks architectural and design decisions
    cursor.execute("""
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
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_error_sig ON error_patterns(error_signature)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_error_type ON error_patterns(error_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_decision_cat ON decisions(category)")

    # Create FTS for error search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS errors_fts USING fts5(
            error_message,
            solution_summary,
            file_context,
            content='error_patterns',
            content_rowid='id'
        )
    """)

    # Create FTS for decision search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
            decision_summary,
            reasoning,
            context,
            content='decisions',
            content_rowid='id'
        )
    """)

    conn.commit()
    conn.close()

    log("Insights database initialized")


# =============================================================================
# ERROR PATTERN RECOGNITION
# =============================================================================

def normalize_error_message(error_msg: str) -> str:
    """
    Normalize an error message for comparison.

    Removes line numbers, file paths, memory addresses, etc.
    """
    normalized = error_msg

    # Remove file paths
    normalized = re.sub(r'(?:/[^\s:]+)+\.[a-z]+', '<FILE>', normalized)
    normalized = re.sub(r'[A-Z]:\\[^\s:]+', '<FILE>', normalized)

    # Remove line numbers
    normalized = re.sub(r'line \d+', 'line <N>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r':\d+:\d+', ':<N>:<N>', normalized)
    normalized = re.sub(r':\d+', ':<N>', normalized)

    # Remove memory addresses
    normalized = re.sub(r'0x[0-9a-fA-F]+', '<ADDR>', normalized)

    # Remove UUIDs
    normalized = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<UUID>', normalized, flags=re.IGNORECASE)

    # Remove timestamps
    normalized = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '<TIME>', normalized)

    # Remove specific variable values in quotes
    normalized = re.sub(r"'[^']{20,}'", "'<VALUE>'", normalized)
    normalized = re.sub(r'"[^"]{20,}"', '"<VALUE>"', normalized)

    return normalized.strip()


def extract_error_type(error_msg: str) -> Optional[str]:
    """Extract the error type from an error message."""
    # Common error type patterns
    patterns = [
        r'^(\w+Error):',
        r'^(\w+Exception):',
        r'^(\w+Warning):',
        r'(\w+Error)\s*:',
        r'(\w+Exception)\s*:',
        r'(TypeError|ReferenceError|SyntaxError|ValueError|KeyError|AttributeError|ImportError|ModuleNotFoundError)',
        r'(Error|Exception|Panic|Fatal|Failed)',
    ]

    for pattern in patterns:
        match = re.search(pattern, error_msg)
        if match:
            return match.group(1)

    return None


def compute_error_signature(error_msg: str) -> str:
    """Compute a signature for an error for deduplication."""
    normalized = normalize_error_message(error_msg)
    error_type = extract_error_type(error_msg) or "Unknown"

    # Create signature from type + normalized message hash
    msg_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]
    return f"{error_type}:{msg_hash}"


def extract_errors_from_conversation(conversation: dict, session_id: str):
    """
    Extract error patterns and solutions from a conversation.

    Looks for:
    - Error messages in user messages
    - Solutions in subsequent assistant messages
    """
    db_path = get_insights_db_path()
    if not db_path.exists():
        init_insights_db()

    messages = conversation.get('messages', [])
    if len(messages) < 2:
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    errors_found = 0

    # Error detection patterns
    error_patterns = [
        r'((?:Error|Exception|Traceback|Failed|Panic)[^\n]*(?:\n[^\n]*){0,5})',
        r'((?:\w+Error|\w+Exception):\s*[^\n]+)',
        r'(npm ERR![^\n]+)',
        r'(error\[E\d+\]:[^\n]+)',  # Rust errors
        r'(fatal:[^\n]+)',  # Git errors
    ]

    try:
        for i, msg in enumerate(messages):
            if msg.get('role') != 'user':
                continue

            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join(
                    item.get('text', '') for item in content
                    if isinstance(item, dict) and item.get('type') == 'text'
                )

            # Look for errors in user message
            for pattern in error_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for error_match in matches:
                    error_msg = error_match.strip()
                    if len(error_msg) < 10:
                        continue

                    signature = compute_error_signature(error_msg)
                    error_type = extract_error_type(error_msg)
                    normalized = normalize_error_message(error_msg)

                    # Look for solution in next assistant message
                    solution_summary = None
                    solution_details = None
                    if i + 1 < len(messages):
                        next_msg = messages[i + 1]
                        if next_msg.get('role') == 'assistant':
                            solution_details = next_msg.get('content', '')
                            if isinstance(solution_details, list):
                                solution_details = ' '.join(
                                    item.get('text', '') for item in solution_details
                                    if isinstance(item, dict) and item.get('type') == 'text'
                                )
                            # Extract first actionable sentence as summary
                            solution_summary = _extract_solution_summary(solution_details)

                    # Check if error pattern exists
                    cursor.execute("""
                        SELECT id, occurrence_count, source_sessions FROM error_patterns
                        WHERE error_signature = ?
                    """, (signature,))

                    row = cursor.fetchone()
                    if row:
                        # Update existing
                        pattern_id, count, sources = row
                        sources_list = json.loads(sources) if sources else []
                        if session_id not in sources_list:
                            sources_list.append(session_id)

                        cursor.execute("""
                            UPDATE error_patterns
                            SET occurrence_count = ?, last_seen = ?, source_sessions = ?,
                                solution_summary = COALESCE(?, solution_summary),
                                solution_details = COALESCE(?, solution_details)
                            WHERE id = ?
                        """, (count + 1, now, json.dumps(sources_list[-10:]),
                              solution_summary, solution_details[:2000] if solution_details else None,
                              pattern_id))
                    else:
                        # Insert new
                        cursor.execute("""
                            INSERT INTO error_patterns
                            (error_signature, error_type, error_message, normalized_message,
                             solution_summary, solution_details, first_seen, last_seen, source_sessions)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (signature, error_type, error_msg[:500], normalized[:500],
                              solution_summary, solution_details[:2000] if solution_details else None,
                              now, now, json.dumps([session_id])))

                        # Update FTS
                        cursor.execute("""
                            INSERT INTO errors_fts(rowid, error_message, solution_summary, file_context)
                            VALUES (last_insert_rowid(), ?, ?, ?)
                        """, (error_msg[:500], solution_summary or '', ''))

                    errors_found += 1

        conn.commit()
    except Exception as e:
        log(f"Error extracting error patterns: {e}")
    finally:
        conn.close()

    return errors_found


def _extract_solution_summary(solution_text: str) -> Optional[str]:
    """Extract a brief summary of the solution from assistant response."""
    if not solution_text:
        return None

    # Look for solution indicators
    solution_patterns = [
        r"(?:The (?:fix|solution|issue|problem) is)[:\s]+([^.!?\n]+[.!?])",
        r"(?:To fix this|To solve this|To resolve this)[,:\s]+([^.!?\n]+[.!?])",
        r"(?:You need to|You should|Try)[:\s]+([^.!?\n]+[.!?])",
        r"(?:The error (?:occurs|happens|is caused) because)[:\s]+([^.!?\n]+[.!?])",
    ]

    for pattern in solution_patterns:
        match = re.search(pattern, solution_text, re.IGNORECASE)
        if match:
            summary = match.group(1).strip()
            if 20 < len(summary) < 200:
                return summary

    # Fallback: first substantive sentence
    sentences = re.split(r'(?<=[.!?])\s+', solution_text)
    for sent in sentences[:3]:
        sent = sent.strip()
        if 20 < len(sent) < 200 and not sent.lower().startswith(('i ', 'let me', 'sure', 'okay')):
            return sent

    return None


def search_error_solutions(query: str, limit: int = 5) -> List[Dict]:
    """
    Search for past error solutions matching the query.

    Returns list of matching errors with their solutions.
    """
    db_path = get_insights_db_path()
    if not db_path.exists():
        return []

    results = []

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Try FTS search first
        terms = extract_query_terms(query, max_terms=5)
        if terms:
            fts_query = ' OR '.join(f'"{t}"' for t in terms)

            cursor.execute("""
                SELECT e.id, e.error_type, e.error_message, e.solution_summary,
                       e.solution_details, e.occurrence_count, e.last_seen
                FROM error_patterns e
                JOIN errors_fts f ON e.id = f.rowid
                WHERE errors_fts MATCH ?
                ORDER BY e.occurrence_count DESC
                LIMIT ?
            """, (fts_query, limit))

            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'error_type': row[1],
                    'error_message': row[2],
                    'solution_summary': row[3],
                    'solution_details': row[4][:500] if row[4] else None,
                    'occurrence_count': row[5],
                    'last_seen': row[6],
                })

        # If no FTS results, try pattern matching
        if not results:
            normalized_query = normalize_error_message(query)
            cursor.execute("""
                SELECT id, error_type, error_message, solution_summary,
                       solution_details, occurrence_count, last_seen
                FROM error_patterns
                WHERE normalized_message LIKE ?
                ORDER BY occurrence_count DESC
                LIMIT ?
            """, (f'%{normalized_query[:50]}%', limit))

            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'error_type': row[1],
                    'error_message': row[2],
                    'solution_summary': row[3],
                    'solution_details': row[4][:500] if row[4] else None,
                    'occurrence_count': row[5],
                    'last_seen': row[6],
                })

        conn.close()
    except Exception as e:
        log(f"Error searching error solutions: {e}")

    return results


def get_error_stats() -> Dict:
    """Get statistics about stored error patterns."""
    db_path = get_insights_db_path()
    if not db_path.exists():
        return {'total': 0, 'by_type': {}}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM error_patterns")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT error_type, COUNT(*) as cnt
            FROM error_patterns
            WHERE error_type IS NOT NULL
            GROUP BY error_type
            ORDER BY cnt DESC
            LIMIT 10
        """)
        by_type = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()
        return {'total': total, 'by_type': by_type}
    except Exception as e:
        log(f"Error getting error stats: {e}")
        return {'total': 0, 'by_type': {}}


# =============================================================================
# DECISION JOURNAL
# =============================================================================

def extract_decisions_from_conversation(conversation: dict, session_id: str):
    """
    Extract architectural and design decisions from a conversation.

    Looks for:
    - Explicit decision statements
    - Reasoning patterns
    - Alternative considerations
    """
    db_path = get_insights_db_path()
    if not db_path.exists():
        init_insights_db()

    messages = conversation.get('messages', [])
    if not messages:
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    decisions_found = 0

    # Decision detection patterns
    decision_patterns = [
        # Explicit decisions
        (r"(?:I (?:decided|chose|went with|selected)|We should use|Let's use|The best approach is)\s+([^.!?\n]{10,150}[.!?])", 'explicit'),
        # Recommendations
        (r"(?:I recommend|I suggest|My recommendation is)\s+([^.!?\n]{10,150}[.!?])", 'recommendation'),
        # Reasoning
        (r"(?:because|since|the reason is)\s+([^.!?\n]{10,100})", 'reasoning'),
        # Trade-off discussions
        (r"(?:instead of|rather than|over)\s+(\w+[^.!?\n]{10,100})", 'alternative'),
    ]

    try:
        all_text = []
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if isinstance(content, list):
                    content = ' '.join(
                        item.get('text', '') for item in content
                        if isinstance(item, dict) and item.get('type') == 'text'
                    )
                all_text.append(content)

        combined_text = '\n'.join(all_text)

        for pattern, decision_type in decision_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:5]:  # Limit per pattern
                decision_text = match.strip()
                if len(decision_text) < 15:
                    continue

                # Skip if it's code
                if decision_text.count('(') > 2 or '{' in decision_text:
                    continue

                # Compute hash for deduplication
                decision_hash = hashlib.md5(decision_text[:100].lower().encode()).hexdigest()

                # Check if exists
                cursor.execute("SELECT id FROM decisions WHERE decision_hash = ?", (decision_hash,))
                if cursor.fetchone():
                    continue

                # Extract context (surrounding text)
                context_match = re.search(
                    rf".{{0,100}}{re.escape(decision_text[:30])}.{{0,100}}",
                    combined_text,
                    re.IGNORECASE | re.DOTALL
                )
                context = context_match.group(0) if context_match else ""

                # Determine category
                category = _categorize_decision(decision_text)

                cursor.execute("""
                    INSERT INTO decisions
                    (decision_hash, decision_summary, context, category, session_id, timestamp, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (decision_hash, decision_text[:200], context[:500], category, session_id, now, 0.5))

                # Update FTS
                cursor.execute("""
                    INSERT INTO decisions_fts(rowid, decision_summary, reasoning, context)
                    VALUES (last_insert_rowid(), ?, ?, ?)
                """, (decision_text[:200], '', context[:500]))

                decisions_found += 1

        conn.commit()
    except Exception as e:
        log(f"Error extracting decisions: {e}")
    finally:
        conn.close()

    return decisions_found


def _categorize_decision(decision_text: str) -> str:
    """Categorize a decision based on its content."""
    text_lower = decision_text.lower()

    categories = {
        'architecture': ['architecture', 'structure', 'design pattern', 'layer', 'module', 'component'],
        'technology': ['library', 'framework', 'database', 'api', 'tool', 'package'],
        'implementation': ['implement', 'approach', 'method', 'function', 'class'],
        'testing': ['test', 'testing', 'coverage', 'mock', 'stub'],
        'security': ['security', 'auth', 'permission', 'encrypt', 'token'],
        'performance': ['performance', 'optimize', 'cache', 'speed', 'memory'],
        'workflow': ['workflow', 'process', 'deploy', 'ci', 'cd', 'pipeline'],
    }

    for category, keywords in categories.items():
        if any(kw in text_lower for kw in keywords):
            return category

    return 'general'


def search_decisions(query: str, category: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """
    Search for past decisions matching the query.

    Returns list of matching decisions with context.
    """
    db_path = get_insights_db_path()
    if not db_path.exists():
        return []

    results = []

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        terms = extract_query_terms(query, max_terms=5)
        if terms:
            fts_query = ' OR '.join(f'"{t}"' for t in terms)

            if category:
                cursor.execute("""
                    SELECT d.id, d.decision_summary, d.context, d.category,
                           d.session_id, d.timestamp, d.confidence
                    FROM decisions d
                    JOIN decisions_fts f ON d.id = f.rowid
                    WHERE decisions_fts MATCH ? AND d.category = ?
                    ORDER BY d.timestamp DESC
                    LIMIT ?
                """, (fts_query, category, limit))
            else:
                cursor.execute("""
                    SELECT d.id, d.decision_summary, d.context, d.category,
                           d.session_id, d.timestamp, d.confidence
                    FROM decisions d
                    JOIN decisions_fts f ON d.id = f.rowid
                    WHERE decisions_fts MATCH ?
                    ORDER BY d.timestamp DESC
                    LIMIT ?
                """, (fts_query, limit))

            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'decision': row[1],
                    'context': row[2],
                    'category': row[3],
                    'session_id': row[4],
                    'timestamp': row[5],
                    'confidence': row[6],
                })

        conn.close()
    except Exception as e:
        log(f"Error searching decisions: {e}")

    return results


def get_decision_stats() -> Dict:
    """Get statistics about stored decisions."""
    db_path = get_insights_db_path()
    if not db_path.exists():
        return {'total': 0, 'by_category': {}}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM decisions")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT category, COUNT(*) as cnt
            FROM decisions
            GROUP BY category
            ORDER BY cnt DESC
        """)
        by_category = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()
        return {'total': total, 'by_category': by_category}
    except Exception as e:
        log(f"Error getting decision stats: {e}")
        return {'total': 0, 'by_category': {}}


# =============================================================================
# COMBINED EXTRACTION (called during ingestion)
# =============================================================================

def extract_insights_from_conversation(conversation: dict, session_id: str):
    """
    Extract all insights (errors, decisions) from a conversation.

    Called during ingestion.
    """
    errors = extract_errors_from_conversation(conversation, session_id)
    decisions = extract_decisions_from_conversation(conversation, session_id)

    return {
        'errors_found': errors or 0,
        'decisions_found': decisions or 0
    }
