"""
MIRA Decision Journal

Tracks architectural and design decisions with reasoning.
Enables mira_decisions tool.
"""

import hashlib
import json
import re
from datetime import datetime
from typing import Dict, List, Optional

from mira.core import log, DB_INSIGHTS
from mira.core.database import get_db_manager


# Decision categories
CATEGORIES = {
    'architecture': ['architecture', 'system design', 'component', 'layer', 'service'],
    'technology': ['technology', 'library', 'framework', 'tool', 'stack', 'database'],
    'implementation': ['implementation', 'algorithm', 'pattern', 'approach', 'method'],
    'testing': ['testing', 'test', 'coverage', 'qa', 'quality'],
    'security': ['security', 'auth', 'authentication', 'encryption', 'permission'],
    'performance': ['performance', 'optimization', 'caching', 'speed', 'latency'],
    'workflow': ['workflow', 'process', 'convention', 'standard', 'rule'],
}


def categorize_decision(text: str) -> str:
    """Categorize a decision based on its content."""
    text_lower = text.lower()
    for category, keywords in CATEGORIES.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return 'general'


def generate_decision_hash(decision: str) -> str:
    """Generate a hash for deduplication."""
    normalized = decision.lower().strip()[:200]
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def record_decision(
    decision: str,
    reasoning: Optional[str] = None,
    alternatives: Optional[List[str]] = None,
    context: Optional[str] = None,
    category: Optional[str] = None,
    session_id: Optional[str] = None,
    confidence: float = 0.5,
) -> Optional[int]:
    """Record a decision in the journal."""
    from .errors import init_insights_db
    init_insights_db()
    db = get_db_manager()

    decision_hash = generate_decision_hash(decision)

    if not category:
        category = categorize_decision(decision)

    # Check if exists
    row = db.execute_read_one(
        DB_INSIGHTS,
        "SELECT id FROM decisions WHERE decision_hash = ?",
        (decision_hash,)
    )

    now = datetime.now().isoformat()
    alternatives_json = json.dumps(alternatives) if alternatives else None

    if row:
        # Update existing with higher confidence info
        db.execute_write(
            DB_INSIGHTS,
            """UPDATE decisions SET
               reasoning = COALESCE(?, reasoning),
               alternatives_considered = COALESCE(?, alternatives_considered),
               context = COALESCE(?, context),
               confidence = MAX(?, confidence),
               timestamp = ?
            WHERE id = ?""",
            (reasoning, alternatives_json, context, confidence, now, row['id'])
        )
        return row['id']
    else:
        # Insert new
        return db.execute_write(
            DB_INSIGHTS,
            """INSERT INTO decisions
               (decision_hash, decision_summary, reasoning, alternatives_considered,
                context, category, session_id, timestamp, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (decision_hash, decision[:500], reasoning, alternatives_json,
             context, category, session_id, now, confidence)
        )


def search_decisions(
    query: str = "",
    category: Optional[str] = None,
    limit: int = 10,
    min_confidence: float = 0.0,
) -> List[Dict]:
    """Search for decisions."""
    from .errors import init_insights_db
    init_insights_db()
    db = get_db_manager()

    try:
        if query:
            # FTS search
            sql = """
                SELECT d.*, rank
                FROM decisions d
                JOIN decisions_fts fts ON d.id = fts.rowid
                WHERE decisions_fts MATCH ?
                AND d.confidence >= ?
            """
            params = [query, min_confidence]

            if category:
                sql += " AND d.category = ?"
                params.append(category)

            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
        else:
            # No query - list recent by confidence
            sql = """
                SELECT * FROM decisions
                WHERE confidence >= ?
            """
            params = [min_confidence]

            if category:
                sql += " AND category = ?"
                params.append(category)

            sql += " ORDER BY confidence DESC, timestamp DESC LIMIT ?"
            params.append(limit)

        rows = db.execute_read(DB_INSIGHTS, sql, tuple(params))

        results = []
        for row in rows:
            result = dict(row)
            if result.get('alternatives_considered'):
                try:
                    result['alternatives_considered'] = json.loads(result['alternatives_considered'])
                except json.JSONDecodeError:
                    result['alternatives_considered'] = []
            results.append(result)
        return results

    except Exception as e:
        log(f"Decision search failed: {e}")
        return []


def get_decision_stats() -> Dict:
    """Get statistics about recorded decisions."""
    from .errors import init_insights_db
    init_insights_db()
    db = get_db_manager()

    try:
        total = db.execute_read_one(DB_INSIGHTS, "SELECT COUNT(*) as cnt FROM decisions", ())
        by_category = db.execute_read(
            DB_INSIGHTS,
            "SELECT category, COUNT(*) as cnt FROM decisions GROUP BY category ORDER BY cnt DESC",
            ()
        )
        high_confidence = db.execute_read_one(
            DB_INSIGHTS,
            "SELECT COUNT(*) as cnt FROM decisions WHERE confidence >= 0.8",
            ()
        )

        return {
            "total": total['cnt'] if total else 0,
            "high_confidence": high_confidence['cnt'] if high_confidence else 0,
            "by_category": {row['category'] or 'general': row['cnt'] for row in by_category}
        }
    except Exception as e:
        log(f"Decision stats failed: {e}")
        return {"total": 0, "high_confidence": 0, "by_category": {}}


# Extraction patterns for detecting decisions in conversation
EXPLICIT_PATTERNS = [
    (r"^Decision:\s*(.+?)(?:\n|$)", 0.95),
    (r"^ADR:\s*(.+?)(?:\n|$)", 0.95),
    (r"^For the record[,:]?\s*(.+?)(?:\n|$)", 0.90),
    (r"^Policy:\s*(.+?)(?:\n|$)", 0.90),
    (r"^Going forward[,:]?\s*(.+?)(?:\n|$)", 0.85),
]

IMPLICIT_PATTERNS = [
    (r"I decided to\s+(.+?)(?:\.|$)", 0.75),
    (r"We decided to\s+(.+?)(?:\.|$)", 0.75),
    (r"I recommend using\s+(.+?)(?:\.|$)", 0.65),
    (r"We should use\s+(.+?)(?:\.|$)", 0.60),
]


def extract_decisions_from_text(text: str, session_id: Optional[str] = None) -> List[Dict]:
    """Extract decisions from conversation text."""
    decisions = []

    # Try explicit patterns first
    for pattern, confidence in EXPLICIT_PATTERNS:
        for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
            decisions.append({
                "decision": match.group(1).strip(),
                "confidence": confidence,
                "source": "explicit",
                "session_id": session_id,
            })

    # Then implicit patterns
    for pattern, confidence in IMPLICIT_PATTERNS:
        for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
            decisions.append({
                "decision": match.group(1).strip(),
                "confidence": confidence,
                "source": "implicit",
                "session_id": session_id,
            })

    return decisions


def extract_decisions_from_conversation(
    conversation: dict,
    session_id: str,
    project_path: Optional[str] = None,
    postgres_session_id: Optional[int] = None,
    storage=None
) -> int:
    """
    Extract decisions from a conversation.

    Looks for explicit decision statements and implicit decision patterns.
    """
    messages = conversation.get('messages', [])
    if not messages:
        return 0

    from .errors import init_insights_db
    init_insights_db()

    decisions_found = 0
    seen_hashes = set()

    for msg in messages:
        # Only look at assistant messages for decisions
        if msg.get('role') != 'assistant':
            continue

        content = msg.get('content', '')
        if isinstance(content, list):
            content = ' '.join(
                item.get('text', '') for item in content
                if isinstance(item, dict) and item.get('type') == 'text'
            )

        if len(content) < 20:
            continue

        # Extract decisions from this message
        extracted = extract_decisions_from_text(content, session_id)

        for decision_info in extracted:
            decision_text = decision_info['decision']
            confidence = decision_info['confidence']

            # Skip very short decisions
            if len(decision_text) < 15:
                continue

            # Deduplicate
            decision_hash = generate_decision_hash(decision_text)
            if decision_hash in seen_hashes:
                continue
            seen_hashes.add(decision_hash)

            # Record the decision
            record_decision(
                decision=decision_text,
                session_id=session_id,
                confidence=confidence,
            )
            decisions_found += 1

    return decisions_found
