"""
MIRA3 Insights Module

Provides advanced analysis features:
- Error pattern recognition and solution lookup
- Decision journal with reasoning
- Conversation similarity detection

Uses central Postgres for storage with local SQLite fallback.
"""

import json
import re
import hashlib
from datetime import datetime
from typing import Optional, List, Dict

from .utils import log, extract_query_terms
from .db_manager import get_db_manager


# Database name for insights
INSIGHTS_DB = "insights.db"

# Schema for insights database
INSIGHTS_SCHEMA = """
-- Error patterns table - tracks errors and their solutions
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

-- Error solutions - links errors to specific solutions found
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

-- Decision journal - tracks architectural and design decisions
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

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_error_sig ON error_patterns(error_signature);
CREATE INDEX IF NOT EXISTS idx_error_type ON error_patterns(error_type);
CREATE INDEX IF NOT EXISTS idx_decision_cat ON decisions(category);

-- Create FTS for error search
CREATE VIRTUAL TABLE IF NOT EXISTS errors_fts USING fts5(
    error_message,
    solution_summary,
    file_context,
    content='error_patterns',
    content_rowid='id'
);

-- Create FTS for decision search
CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
    decision_summary,
    reasoning,
    context,
    content='decisions',
    content_rowid='id'
);
"""


def init_insights_db():
    """Initialize the insights database for errors, decisions, etc."""
    db = get_db_manager()
    db.init_schema(INSIGHTS_DB, INSIGHTS_SCHEMA)
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

    msg_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]
    return f"{error_type}:{msg_hash}"


def extract_errors_from_conversation(
    conversation: dict,
    session_id: str,
    project_path: str = None,
    storage=None
) -> int:
    """
    Extract error patterns and solutions from a conversation.

    Looks for:
    - Error messages in user messages
    - Solutions in subsequent assistant messages

    Uses central Postgres if available, falls back to local SQLite.
    """
    messages = conversation.get('messages', [])
    if len(messages) < 2:
        return 0

    # Get storage if not provided
    use_central = False
    project_id = None
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            pass

    if storage and storage.using_central and project_path:
        try:
            project_id = storage.postgres.get_or_create_project(project_path)
            use_central = True
        except Exception as e:
            log(f"Failed to get project_id for errors: {e}")

    db = get_db_manager()
    now = datetime.now().isoformat()

    errors_found = 0

    # CONSERVATIVE error patterns - only match actual errors, not markdown headers
    error_patterns = [
        # Python tracebacks and errors - require specific error format
        r'(Traceback \(most recent call last\):[^\n]*(?:\n[^\n]*){1,10})',
        r'((?:\w+Error|\w+Exception):\s+[^\n]{10,200})',  # Require content after colon
        # JavaScript/Node errors
        r'(npm ERR! [A-Z][^\n]{10,150})',  # npm errors with content
        r'(TypeError:|ReferenceError:|SyntaxError:)\s+([^\n]{10,150})',
        # Rust errors
        r'(error\[E\d{4}\]:[^\n]{10,150})',  # Rust errors with code
        # Git errors
        r'(fatal: [^\n]{10,150})',
        # Generic but require specific formats
        r'(Error: [A-Z][^\n]{15,150})',  # Error with capital letter start and content
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

            for pattern in error_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for error_match in matches:
                    error_msg = error_match.strip()
                    if len(error_msg) < 10:
                        continue

                    signature = compute_error_signature(error_msg)
                    error_type = extract_error_type(error_msg)

                    solution_summary = None
                    if i + 1 < len(messages):
                        next_msg = messages[i + 1]
                        if next_msg.get('role') == 'assistant':
                            solution_details = next_msg.get('content', '')
                            if isinstance(solution_details, list):
                                solution_details = ' '.join(
                                    item.get('text', '') for item in solution_details
                                    if isinstance(item, dict) and item.get('type') == 'text'
                                )
                            solution_summary = _extract_solution_summary(solution_details)

                    # Store in central Postgres
                    if use_central and project_id:
                        try:
                            storage.postgres.upsert_error_pattern(
                                project_id=project_id,
                                signature=signature,
                                error_type=error_type,
                                error_text=error_msg[:500],
                                solution=solution_summary,
                            )
                            errors_found += 1
                            continue
                        except Exception as e:
                            log(f"Central error storage failed: {e}")

                    # Fallback to local SQLite
                    normalized = normalize_error_message(error_msg)

                    def upsert_error(cursor):
                        cursor.execute("""
                            SELECT id, occurrence_count, source_sessions FROM error_patterns
                            WHERE error_signature = ?
                        """, (signature,))

                        row = cursor.fetchone()
                        if row:
                            pattern_id, count, sources = row
                            sources_list = json.loads(sources) if sources else []
                            if session_id not in sources_list:
                                sources_list.append(session_id)

                            cursor.execute("""
                                UPDATE error_patterns
                                SET occurrence_count = ?, last_seen = ?, source_sessions = ?,
                                    solution_summary = COALESCE(?, solution_summary)
                                WHERE id = ?
                            """, (count + 1, now, json.dumps(sources_list[-10:]),
                                  solution_summary, pattern_id))
                            return 0
                        else:
                            cursor.execute("""
                                INSERT INTO error_patterns
                                (error_signature, error_type, error_message, normalized_message,
                                 solution_summary, first_seen, last_seen, source_sessions)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (signature, error_type, error_msg[:500], normalized[:500],
                                  solution_summary, now, now, json.dumps([session_id])))

                            cursor.execute("""
                                INSERT INTO errors_fts(rowid, error_message, solution_summary, file_context)
                                VALUES (last_insert_rowid(), ?, ?, ?)
                            """, (error_msg[:500], solution_summary or '', ''))
                            return 1

                    errors_found += db.execute_write_func(INSIGHTS_DB, upsert_error)

    except Exception as e:
        log(f"Error extracting error patterns: {e}")

    return errors_found


def _extract_solution_summary(solution_text: str) -> Optional[str]:
    """Extract a brief summary of the solution from assistant response.

    Only extracts if the response actually contains solution-like content.
    Returns None if no clear solution is found.
    """
    if not solution_text or len(solution_text) < 50:
        return None

    # Must contain solution-indicating words
    solution_indicators = ['fix', 'solve', 'resolv', 'chang', 'updat', 'replac', 'add', 'remov', 'correct']
    text_lower = solution_text.lower()
    if not any(ind in text_lower for ind in solution_indicators):
        return None

    # CONSERVATIVE solution patterns - require clear fix language
    solution_patterns = [
        r"(?:The (?:fix|solution) (?:is|was))[:\s]+([^.!?\n]{20,200}[.!?])",
        r"(?:I (?:fixed|resolved|corrected) (?:this|it|the error) by)[:\s]+([^.!?\n]{20,200}[.!?])",
        r"(?:To fix this)[,:\s]+([^.!?\n]{20,200}[.!?])",
        r"(?:The (?:issue|problem) was)[:\s]+([^.!?\n]{20,200}[.!?])",
        r"(?:Changed|Updated|Fixed|Replaced|Added|Removed)\s+([^.!?\n]{20,150}[.!?])",
    ]

    for pattern in solution_patterns:
        match = re.search(pattern, solution_text, re.IGNORECASE)
        if match:
            summary = match.group(1).strip()
            if 20 < len(summary) < 200:
                return summary

    # Don't fall back to arbitrary sentences - too noisy
    return None


def search_error_solutions(query: str, limit: int = 5, project_path: str = None, storage=None) -> List[Dict]:
    """
    Search for past error solutions matching the query.

    Uses central Postgres if available, falls back to local SQLite.

    Returns list of matching errors with their solutions.
    """
    # Try central Postgres first
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            pass

    if storage and storage.using_central:
        try:
            project_id = None
            if project_path:
                project_id = storage.postgres.get_or_create_project(project_path)
            results = storage.postgres.search_error_patterns(query, project_id=project_id, limit=limit)
            formatted = []
            for r in results:
                formatted.append({
                    'error_type': r.get('error_type'),
                    'error_message': r.get('error_text'),
                    'solution_summary': r.get('solution'),
                    'occurrence_count': r.get('occurrences', 1),
                    'last_seen': str(r.get('last_seen', '')),
                    'project_path': r.get('project_path', ''),
                })
            return formatted
        except Exception as e:
            log(f"Central error search failed: {e}")

    # Fallback to local SQLite
    return _search_error_solutions_local(query, limit)


def _search_error_solutions_local(query: str, limit: int = 5) -> List[Dict]:
    """Search local SQLite for error solutions."""
    db = get_db_manager()
    results = []

    try:
        terms = extract_query_terms(query, max_terms=5)
        if terms:
            fts_query = ' OR '.join(f'"{t}"' for t in terms)

            rows = db.execute_read(INSIGHTS_DB, """
                SELECT e.id, e.error_type, e.error_message, e.solution_summary,
                       e.solution_details, e.occurrence_count, e.last_seen
                FROM error_patterns e
                JOIN errors_fts f ON e.id = f.rowid
                WHERE errors_fts MATCH ?
                ORDER BY e.occurrence_count DESC
                LIMIT ?
            """, (fts_query, limit))

            for row in rows:
                results.append({
                    'id': row['id'],
                    'error_type': row['error_type'],
                    'error_message': row['error_message'],
                    'solution_summary': row['solution_summary'],
                    'solution_details': row['solution_details'][:500] if row['solution_details'] else None,
                    'occurrence_count': row['occurrence_count'],
                    'last_seen': row['last_seen'],
                })

        if not results:
            normalized_query = normalize_error_message(query)
            rows = db.execute_read(INSIGHTS_DB, """
                SELECT id, error_type, error_message, solution_summary,
                       solution_details, occurrence_count, last_seen
                FROM error_patterns
                WHERE normalized_message LIKE ?
                ORDER BY occurrence_count DESC
                LIMIT ?
            """, (f'%{normalized_query[:50]}%', limit))

            for row in rows:
                results.append({
                    'id': row['id'],
                    'error_type': row['error_type'],
                    'error_message': row['error_message'],
                    'solution_summary': row['solution_summary'],
                    'solution_details': row['solution_details'][:500] if row['solution_details'] else None,
                    'occurrence_count': row['occurrence_count'],
                    'last_seen': row['last_seen'],
                })

    except Exception as e:
        log(f"Error searching error solutions: {e}")

    return results


def get_error_stats() -> Dict:
    """Get statistics about stored error patterns."""
    db = get_db_manager()

    try:
        row = db.execute_read_one(INSIGHTS_DB, "SELECT COUNT(*) as cnt FROM error_patterns")
        total = row['cnt'] if row else 0

        rows = db.execute_read(INSIGHTS_DB, """
            SELECT error_type, COUNT(*) as cnt
            FROM error_patterns
            WHERE error_type IS NOT NULL
            GROUP BY error_type
            ORDER BY cnt DESC
            LIMIT 10
        """)
        by_type = {row['error_type']: row['cnt'] for row in rows}

        return {'total': total, 'by_type': by_type}
    except Exception as e:
        log(f"Error getting error stats: {e}")
        return {'total': 0, 'by_type': {}}


# =============================================================================
# DECISION JOURNAL
# =============================================================================

# Explicit decision recording patterns - scanned in USER messages
# These are intentional recording commands with high confidence
EXPLICIT_DECISION_PATTERNS = [
    # Tier 1: Direct recording commands (confidence 0.95)
    (r"(?:record (?:this )?decision|log (?:this )?decision|document (?:this )?decision|capture (?:this )?decision)[:\s]+(.{10,300})", 0.95),
    (r"(?:ADR|architecture decision(?: record)?|design decision|technical decision)[:\s-]+(.{10,300})", 0.95),
    (r"(?:decision|final decision|our decision|team decision)[:\s]+(.{10,300})", 0.95),
    (r"for the record[,:\s]+(.{10,300})", 0.95),
    (r"(?:MIRA,? record|for MIRA)[:\s]+(.{10,300})", 0.95),

    # Tier 2: Strong declarations (confidence 0.90)
    (r"(?:policy|rule|standard|convention|guideline)[:\s]+(.{10,300})", 0.90),
    (r"(?:going forward|from now on|henceforth)[,:\s]+(.{10,300})", 0.90),
    (r"(?:we commit to|our commitment is|we are committing to)[:\s]+(.{10,300})", 0.90),
    (r"(?:requirement|mandate|mandatory)[:\s]+(.{10,300})", 0.90),

    # Tier 3: Clear statements (confidence 0.85)
    (r"we (?:will always|will never|must always|must never)\s+(.{10,200})", 0.85),
    (r"(?:always use|never use|always do|never do)\s+(.{10,200})", 0.85),
    (r"(?:resolution|resolved|we've resolved that|conclusion|we conclude that)[:\s]+(.{10,300})", 0.85),
    (r"(?:chosen approach|selected approach|final choice)[:\s]+(.{10,300})", 0.85),
]

# Technical terms that make casual triggers more likely to be real decisions
DECISION_TECH_TERMS = [
    'api', 'database', 'db', 'code', 'function', 'class', 'file', 'config',
    'deploy', 'test', 'build', 'architecture', 'framework', 'library',
    'component', 'service', 'module', 'package', 'dependency', 'schema',
    'endpoint', 'route', 'model', 'controller', 'view', 'template',
    'query', 'cache', 'storage', 'auth', 'permission', 'security',
]


def _is_decision_false_positive(trigger_match: str, content: str) -> bool:
    """Check if a decision match is a false positive."""
    content_stripped = content.strip()

    # Questions are not decisions
    if content_stripped.endswith('?'):
        return True
    if re.search(r'\b(should we|what|how|which|can we|do we|are we)\b', content_stripped, re.I):
        return True

    # Hypotheticals are not decisions
    if re.match(r'^(if|could|might|would|may|maybe)\b', content_stripped, re.I):
        return True

    # Negations indicate no decision was made
    if re.search(r"\b(haven't|hasn't|not yet|isn't final|undecided|uncertain)\b", content_stripped, re.I):
        return True

    # Too short to be meaningful
    if len(content_stripped) < 10:
        return True

    # Skip if it looks like code
    if content_stripped.startswith('{') or content_stripped.startswith('['):
        return True
    if re.match(r'^(const|let|var|function|class|def|import)\b', content_stripped):
        return True

    return False


def _has_technical_content(content: str) -> bool:
    """Check if content contains technical terms."""
    content_lower = content.lower()
    return any(term in content_lower for term in DECISION_TECH_TERMS)


def extract_decisions_from_conversation(
    conversation: dict,
    session_id: str,
    project_path: str = None,
    postgres_session_id: int = None,
    storage=None
) -> int:
    """
    Extract architectural and design decisions from a conversation.

    Scans BOTH user and assistant messages:
    - User messages: Explicit recording commands (high confidence)
    - Assistant messages: Decision patterns and recommendations

    Uses central Postgres if available, falls back to local SQLite.
    """
    messages = conversation.get('messages', [])
    if not messages:
        return 0

    # Get storage if not provided
    use_central = False
    project_id = None
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            pass

    if storage and storage.using_central and project_path:
        try:
            project_id = storage.postgres.get_or_create_project(project_path)
            use_central = True
        except Exception as e:
            log(f"Failed to get project_id for decisions: {e}")

    db = get_db_manager()
    now = datetime.now().isoformat()

    decisions_found = 0
    seen_hashes = set()  # Dedupe within this conversation

    # Collect text from user and assistant messages separately
    user_text = []
    assistant_text = []
    for msg in messages:
        content = msg.get('content', '')
        if isinstance(content, list):
            content = ' '.join(
                item.get('text', '') for item in content
                if isinstance(item, dict) and item.get('type') == 'text'
            )
        if msg.get('role') == 'user':
            user_text.append(content)
        elif msg.get('role') == 'assistant':
            assistant_text.append(content)

    combined_user_text = '\n'.join(user_text)
    combined_assistant_text = '\n'.join(assistant_text)

    def store_decision(decision_text: str, context: str, confidence: float) -> bool:
        """Store a decision and return True if successful."""
        nonlocal decisions_found

        decision_hash = hashlib.md5(decision_text[:100].lower().encode()).hexdigest()

        # Skip duplicates within this conversation
        if decision_hash in seen_hashes:
            return False
        seen_hashes.add(decision_hash)

        category = _categorize_decision(decision_text)

        # Store in central Postgres
        if use_central and project_id:
            try:
                storage.postgres.insert_decision(
                    project_id=project_id,
                    decision=decision_text[:200],
                    category=category,
                    reasoning=context[:500] if context else None,
                    session_id=postgres_session_id,
                    confidence=confidence,
                )
                decisions_found += 1
                return True
            except Exception as e:
                log(f"Central decision storage failed: {e}")

        # Fallback to local SQLite
        def insert_decision(cursor):
            cursor.execute("SELECT id FROM decisions WHERE decision_hash = ?", (decision_hash,))
            if cursor.fetchone():
                return 0

            cursor.execute("""
                INSERT INTO decisions
                (decision_hash, decision_summary, context, category, session_id, timestamp, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (decision_hash, decision_text[:200], context[:500], category, session_id, now, confidence))

            cursor.execute("""
                INSERT INTO decisions_fts(rowid, decision_summary, reasoning, context)
                VALUES (last_insert_rowid(), ?, ?, ?)
            """, (decision_text[:200], '', context[:500]))
            return 1

        result = db.execute_write_func(INSIGHTS_DB, insert_decision)
        if result:
            decisions_found += 1
        return result > 0

    # ==========================================================================
    # PHASE 1: Scan USER messages for explicit recording patterns (high confidence)
    # ==========================================================================
    try:
        for pattern, confidence in EXPLICIT_DECISION_PATTERNS:
            matches = re.findall(pattern, combined_user_text, re.IGNORECASE | re.MULTILINE)
            for match in matches[:10]:  # Limit per pattern
                decision_text = match.strip()

                if _is_decision_false_positive(pattern, decision_text):
                    continue

                # For lower-confidence patterns, require technical content
                if confidence < 0.90 and not _has_technical_content(decision_text):
                    continue

                # Extract context around the decision
                context_match = re.search(
                    rf".{{0,100}}{re.escape(decision_text[:30])}.{{0,100}}",
                    combined_user_text,
                    re.IGNORECASE | re.DOTALL
                )
                context = context_match.group(0) if context_match else ""

                store_decision(decision_text, context, confidence)

    except Exception as e:
        log(f"Error extracting explicit decisions from user messages: {e}")

    # ==========================================================================
    # PHASE 2: Scan ASSISTANT messages for decision patterns (lower confidence)
    # ==========================================================================

    # Assistant decision patterns with confidence tiers
    # These are less explicit than user recording commands
    assistant_decision_patterns = [
        # Tier 4: Explicit decisions (0.75)
        (r"(?:I (?:decided|chose|went with)|We (?:decided|chose) to use|The (?:best|chosen) approach is)\s+([^.!?\n]{15,150}[.!?])", 0.75),
        # Tier 5: Recommendations (0.65)
        (r"(?:I recommend|I suggest|My recommendation is)\s+(using [^.!?\n]{10,100}|to use [^.!?\n]{10,100})", 0.65),
        # Tier 5: Alternative explanations (0.60)
        (r"(?:instead of|rather than)\s+(using \w+[^.!?\n]{10,60}(?:because|since|for)[^.!?\n]{10,60})", 0.60),
    ]

    try:
        for pattern, confidence in assistant_decision_patterns:
            matches = re.findall(pattern, combined_assistant_text, re.IGNORECASE)
            for match in matches[:5]:  # Limit per pattern
                decision_text = match.strip()

                if len(decision_text) < 15:
                    continue

                # Skip if too much code-like content
                if decision_text.count('(') > 2 or '{' in decision_text:
                    continue

                if _is_decision_false_positive(pattern, decision_text):
                    continue

                # Extract context around the decision
                context_match = re.search(
                    rf".{{0,100}}{re.escape(decision_text[:30])}.{{0,100}}",
                    combined_assistant_text,
                    re.IGNORECASE | re.DOTALL
                )
                context = context_match.group(0) if context_match else ""

                store_decision(decision_text, context, confidence)

    except Exception as e:
        log(f"Error extracting decisions from assistant messages: {e}")

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


def search_decisions(
    query: str,
    category: Optional[str] = None,
    limit: int = 10,
    project_path: str = None,
    storage=None
) -> List[Dict]:
    """
    Search for past decisions matching the query.

    Uses central Postgres if available, falls back to local SQLite.

    Returns list of matching decisions with context.
    """
    # Try central Postgres first
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            pass

    if storage and storage.using_central:
        try:
            project_id = None
            if project_path:
                project_id = storage.postgres.get_or_create_project(project_path)
            results = storage.postgres.search_decisions(
                query=query,
                project_id=project_id,
                category=category,
                limit=limit
            )
            formatted = []
            for r in results:
                formatted.append({
                    'decision': r.get('decision'),
                    'reasoning': r.get('reasoning'),
                    'category': r.get('category'),
                    'alternatives': r.get('alternatives', []),
                    'confidence': r.get('confidence', 0.5),
                    'created_at': str(r.get('created_at', '')),
                    'project_path': r.get('project_path', ''),
                })
            return formatted
        except Exception as e:
            log(f"Central decision search failed: {e}")

    # Fallback to local SQLite
    return _search_decisions_local(query, category, limit)


def _search_decisions_local(query: str, category: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """Search local SQLite for decisions."""
    db = get_db_manager()
    results = []

    try:
        terms = extract_query_terms(query, max_terms=5)
        if terms:
            fts_query = ' OR '.join(f'"{t}"' for t in terms)

            if category:
                rows = db.execute_read(INSIGHTS_DB, """
                    SELECT d.id, d.decision_summary, d.context, d.category,
                           d.session_id, d.timestamp, d.confidence
                    FROM decisions d
                    JOIN decisions_fts f ON d.id = f.rowid
                    WHERE decisions_fts MATCH ? AND d.category = ?
                    ORDER BY d.timestamp DESC
                    LIMIT ?
                """, (fts_query, category, limit))
            else:
                rows = db.execute_read(INSIGHTS_DB, """
                    SELECT d.id, d.decision_summary, d.context, d.category,
                           d.session_id, d.timestamp, d.confidence
                    FROM decisions d
                    JOIN decisions_fts f ON d.id = f.rowid
                    WHERE decisions_fts MATCH ?
                    ORDER BY d.timestamp DESC
                    LIMIT ?
                """, (fts_query, limit))

            for row in rows:
                results.append({
                    'id': row['id'],
                    'decision': row['decision_summary'],
                    'context': row['context'],
                    'category': row['category'],
                    'session_id': row['session_id'],
                    'timestamp': row['timestamp'],
                    'confidence': row['confidence'],
                })

    except Exception as e:
        log(f"Error searching decisions: {e}")

    return results


def get_decision_stats() -> Dict:
    """Get statistics about stored decisions."""
    db = get_db_manager()

    try:
        row = db.execute_read_one(INSIGHTS_DB, "SELECT COUNT(*) as cnt FROM decisions")
        total = row['cnt'] if row else 0

        rows = db.execute_read(INSIGHTS_DB, """
            SELECT category, COUNT(*) as cnt
            FROM decisions
            GROUP BY category
            ORDER BY cnt DESC
        """)
        by_category = {row['category']: row['cnt'] for row in rows}

        return {'total': total, 'by_category': by_category}
    except Exception as e:
        log(f"Error getting decision stats: {e}")
        return {'total': 0, 'by_category': {}}


# =============================================================================
# COMBINED EXTRACTION (called during ingestion)
# =============================================================================

def extract_insights_from_conversation(
    conversation: dict,
    session_id: str,
    project_path: str = None,
    postgres_session_id: int = None,
    storage=None
):
    """
    Extract all insights (errors, decisions) from a conversation.

    Called during ingestion.

    Args:
        conversation: Parsed conversation dict
        session_id: String session ID
        project_path: Filesystem path to project (for Postgres storage)
        postgres_session_id: Postgres session ID (for foreign keys)
        storage: Storage instance
    """
    import time
    from .utils import log

    short_id = session_id[:12]
    msg_count = len(conversation.get('messages', []))

    t0 = time.time()
    errors = extract_errors_from_conversation(
        conversation, session_id,
        project_path=project_path,
        storage=storage
    )
    t_errors = (time.time() - t0) * 1000

    t0 = time.time()
    decisions = extract_decisions_from_conversation(
        conversation, session_id,
        project_path=project_path,
        postgres_session_id=postgres_session_id,
        storage=storage
    )
    t_decisions = (time.time() - t0) * 1000

    log(f"[{short_id}] Insights detail: {msg_count} msgs | errors={errors or 0} ({t_errors:.0f}ms) decisions={decisions or 0} ({t_decisions:.0f}ms)")

    return {
        'errors_found': errors or 0,
        'decisions_found': decisions or 0
    }
