"""
MIRA3 Custodian Learning Module

Learns about the user (custodian) from their conversation patterns:
- Identity (name, aliases)
- Preferences (coding style, tools, frameworks)
- Rules (always/never patterns)
- Work patterns (dev loop, communication style)
- Danger zones (files/modules that caused issues)
"""

import json
import re
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Optional

from .utils import log, get_mira_path, get_custodian


# Preference categories
PREF_CODING_STYLE = "coding_style"
PREF_TOOLS = "tools"
PREF_FRAMEWORKS = "frameworks"
PREF_WORKFLOW = "workflow"
PREF_COMMUNICATION = "communication"
PREF_TESTING = "testing"


def get_custodian_db_path() -> Path:
    """Get the path to the custodian database."""
    return get_mira_path() / "custodian.db"


def init_custodian_db():
    """Initialize the custodian learning database."""
    db_path = get_custodian_db_path()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Identity table - who is the custodian
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS identity (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            source_session TEXT,
            learned_at TEXT
        )
    """)

    # Preferences table - what they prefer
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            preference TEXT NOT NULL,
            value TEXT,
            evidence TEXT,
            frequency INTEGER DEFAULT 1,
            confidence REAL DEFAULT 0.5,
            first_seen TEXT,
            last_seen TEXT,
            source_sessions TEXT
        )
    """)

    # Rules table - explicit always/never rules
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_type TEXT NOT NULL,
            rule_text TEXT NOT NULL,
            context TEXT,
            frequency INTEGER DEFAULT 1,
            first_seen TEXT,
            last_seen TEXT,
            source_sessions TEXT
        )
    """)

    # Danger zones - files/modules that caused issues
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS danger_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path_pattern TEXT NOT NULL,
            issue_description TEXT,
            issue_count INTEGER DEFAULT 1,
            last_issue TEXT,
            resolution TEXT,
            source_sessions TEXT
        )
    """)

    # Work patterns - how they work
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS work_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            pattern_description TEXT NOT NULL,
            frequency INTEGER DEFAULT 1,
            confidence REAL DEFAULT 0.5,
            first_seen TEXT,
            last_seen TEXT
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pref_category ON preferences(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rules_type ON rules(rule_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_danger_path ON danger_zones(path_pattern)")

    conn.commit()
    conn.close()

    log("Custodian database initialized")


def extract_custodian_learnings(conversation: dict, session_id: str):
    """
    Extract learnings about the custodian from a conversation.

    Called during ingestion to learn from each conversation.
    """
    messages = conversation.get('messages', [])
    if not messages:
        return

    db_path = get_custodian_db_path()
    if not db_path.exists():
        init_custodian_db()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    try:
        # Extract from user messages
        user_messages = [m for m in messages if m.get('role') == 'user']
        assistant_messages = [m for m in messages if m.get('role') == 'assistant']

        # Learn identity
        _learn_identity(cursor, user_messages, session_id, now)

        # Learn preferences from user statements
        _learn_preferences(cursor, user_messages, session_id, now)

        # Learn rules from both user and assistant
        _learn_rules(cursor, messages, session_id, now)

        # Learn danger zones from error patterns
        _learn_danger_zones(cursor, messages, session_id, now)

        # Learn work patterns
        _learn_work_patterns(cursor, messages, session_id, now)

        conn.commit()
    except Exception as e:
        log(f"Error extracting custodian learnings: {e}")
    finally:
        conn.close()


def _learn_identity(cursor, user_messages: list, session_id: str, now: str):
    """Learn identity information from user messages."""

    # Patterns to extract name
    name_patterns = [
        r"(?:my name is|i'm|i am|call me|this is)\s+([A-Z][a-z]+)",
        r"(?:^|\s)([A-Z][a-z]+)\s+here(?:\s|$|\.)",
        r"(?:regards|cheers|thanks),?\s*\n?\s*([A-Z][a-z]+)",
    ]

    for msg in user_messages:
        content = msg.get('content', '')

        for pattern in name_patterns:
            match = re.search(pattern, content)
            if match:
                name = match.group(1)
                # Avoid false positives
                if name.lower() not in {'claude', 'hello', 'please', 'thanks', 'help', 'just', 'the', 'this', 'that'}:
                    cursor.execute("""
                        INSERT OR REPLACE INTO identity (key, value, confidence, source_session, learned_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, ('name', name, 0.8, session_id, now))
                    break


def _learn_preferences(cursor, user_messages: list, session_id: str, now: str):
    """Learn preferences from user statements."""

    # Preference patterns with categories
    pref_patterns = [
        # Coding style preferences
        (PREF_CODING_STYLE, r"(?:i (?:prefer|like|want|use)|always use|my preference is)\s+([^.!?\n]{5,60})", 0.7),
        (PREF_CODING_STYLE, r"(?:don't|never|avoid)\s+(?:use|like)\s+([^.!?\n]{5,40})", 0.6),

        # Tool preferences
        (PREF_TOOLS, r"(?:i use|using|my .* is)\s+(npm|pnpm|yarn|bun|pip|poetry|cargo|go mod)", 0.8),
        (PREF_TOOLS, r"(?:run|use|prefer)\s+(vitest|jest|pytest|mocha|cargo test)", 0.8),

        # Framework preferences
        (PREF_FRAMEWORKS, r"(?:using|we use|i use|prefer)\s+(react|vue|svelte|angular|next|nuxt|express|fastapi|django|flask)", 0.7),

        # Testing preferences - capture the full phrase, not just keywords
        (PREF_TESTING, r"(?:i (?:prefer|like|want) to )(write tests? (?:before|after|first))", 0.7),
        (PREF_TESTING, r"(?:always )(run tests? before (?:commit|push|deploy))", 0.8),

        # Communication preferences
        (PREF_COMMUNICATION, r"(?:be |keep it |make it |i (?:want|prefer|like) )\s*(concise|brief|detailed|verbose|short)", 0.7),
        (PREF_COMMUNICATION, r"(no emojis?|without emojis?|don't use emojis?)", 0.9),
        (PREF_COMMUNICATION, r"(show (?:me )?code first|code before explanation)", 0.8),
    ]

    for msg in user_messages:
        content = msg.get('content', '').lower()

        for category, pattern, confidence in pref_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                pref_text = match.strip() if isinstance(match, str) else match
                if len(pref_text) < 3 or len(pref_text) > 50:
                    continue

                # Skip markdown, code, and documentation
                if '**' in pref_text or '`' in pref_text or '(' in pref_text:
                    continue

                # Check if preference exists
                cursor.execute("""
                    SELECT id, frequency, source_sessions FROM preferences
                    WHERE category = ? AND preference = ?
                """, (category, pref_text))

                row = cursor.fetchone()
                if row:
                    # Update existing
                    pref_id, freq, sources = row
                    sources_list = json.loads(sources) if sources else []
                    if session_id not in sources_list:
                        sources_list.append(session_id)
                    cursor.execute("""
                        UPDATE preferences
                        SET frequency = ?, last_seen = ?, source_sessions = ?,
                            confidence = MIN(1.0, confidence + 0.1)
                        WHERE id = ?
                    """, (freq + 1, now, json.dumps(sources_list[-10:]), pref_id))
                else:
                    # Insert new
                    cursor.execute("""
                        INSERT INTO preferences
                        (category, preference, confidence, first_seen, last_seen, source_sessions)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (category, pref_text, confidence, now, now, json.dumps([session_id])))


def _learn_rules(cursor, messages: list, session_id: str, now: str):
    """Learn explicit rules from USER messages only (custodian's own rules)."""

    # Rule patterns - require first-person language for high confidence
    rule_patterns = [
        # First-person never rules (high confidence)
        ('never', r"(?:i never|we never|i don't ever|please never|don't ever)\s+([^.!?\n]{10,80})", 0.9),
        # First-person always rules (high confidence)
        ('always', r"(?:i always|we always|always make sure|please always)\s+([^.!?\n]{10,80})", 0.9),
        # Avoid/prefer patterns
        ('avoid', r"(?:i (?:try to )?avoid|please avoid|don't use|i don't like)\s+([^.!?\n]{5,60})", 0.8),
    ]

    # Only process USER messages - these are the custodian's own rules
    user_messages = [m for m in messages if m.get('role') == 'user']

    for msg in user_messages:
        content = msg.get('content', '')

        for rule_type, pattern, base_confidence in rule_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                rule_text = match.strip()
                if len(rule_text) < 5 or len(rule_text) > 100:
                    continue

                # Skip if it's clearly not a rule (e.g., code, markdown, documentation)
                if rule_text.count('(') > 1 or rule_text.count('{') > 0:
                    continue
                if '**' in rule_text or '```' in rule_text or '##' in rule_text:
                    continue
                if '`' in rule_text or '->' in rule_text or '==' in rule_text:
                    continue
                # Skip if it looks like documentation (bullet points, numbered items)
                if rule_text.startswith('-') or (rule_text[0].isdigit() and rule_text[1] in '.):'):
                    continue
                # Skip if too many special chars (probably not natural text)
                alpha_ratio = sum(1 for c in rule_text if c.isalpha() or c.isspace()) / len(rule_text)
                if alpha_ratio < 0.8:
                    continue
                # Skip common false positives
                skip_phrases = ['the file', 'the code', 'this function', 'that method',
                               'in sampling', 'in the', 'to the', 'from the']
                if any(p in rule_text.lower() for p in skip_phrases):
                    continue

                # Check if rule exists
                cursor.execute("""
                    SELECT id, frequency, source_sessions FROM rules
                    WHERE rule_type = ? AND rule_text = ?
                """, (rule_type, rule_text[:200]))

                row = cursor.fetchone()
                if row:
                    rule_id, freq, sources = row
                    sources_list = json.loads(sources) if sources else []
                    if session_id not in sources_list:
                        sources_list.append(session_id)
                    cursor.execute("""
                        UPDATE rules SET frequency = ?, last_seen = ?, source_sessions = ?
                        WHERE id = ?
                    """, (freq + 1, now, json.dumps(sources_list[-10:]), rule_id))
                else:
                    cursor.execute("""
                        INSERT INTO rules (rule_type, rule_text, first_seen, last_seen, source_sessions)
                        VALUES (?, ?, ?, ?, ?)
                    """, (rule_type, rule_text[:200], now, now, json.dumps([session_id])))


def _learn_danger_zones(cursor, messages: list, session_id: str, now: str):
    """Learn about files/modules that caused issues."""

    # Patterns indicating problems
    problem_patterns = [
        r"(?:error|issue|bug|problem|broke|breaking|failed)\s+(?:in|with|at)\s+([^\s.!?\n]+\.[a-z]+)",
        r"([^\s.!?\n]+\.[a-z]+)\s+(?:is broken|has issues|keeps failing|caused|causing)",
        r"(?:careful with|watch out for|tricky|problematic)\s+([^\s.!?\n]+)",
    ]

    for msg in messages:
        content = msg.get('content', '')

        for pattern in problem_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                path_pattern = match.strip()
                if len(path_pattern) < 3 or len(path_pattern) > 100:
                    continue

                # Extract context around the match
                context_match = re.search(
                    rf".{{0,50}}{re.escape(path_pattern)}.{{0,50}}",
                    content,
                    re.IGNORECASE
                )
                context = context_match.group(0) if context_match else ""

                cursor.execute("""
                    SELECT id, issue_count, source_sessions FROM danger_zones
                    WHERE path_pattern = ?
                """, (path_pattern,))

                row = cursor.fetchone()
                if row:
                    zone_id, count, sources = row
                    sources_list = json.loads(sources) if sources else []
                    if session_id not in sources_list:
                        sources_list.append(session_id)
                    cursor.execute("""
                        UPDATE danger_zones
                        SET issue_count = ?, last_issue = ?,
                            issue_description = ?, source_sessions = ?
                        WHERE id = ?
                    """, (count + 1, now, context[:200], json.dumps(sources_list[-10:]), zone_id))
                else:
                    cursor.execute("""
                        INSERT INTO danger_zones
                        (path_pattern, issue_description, last_issue, source_sessions)
                        VALUES (?, ?, ?, ?)
                    """, (path_pattern, context[:200], now, json.dumps([session_id])))


def _learn_work_patterns(cursor, messages: list, session_id: str, now: str):
    """Learn work patterns from conversation flow."""

    # Detect patterns based on message flow
    user_messages = [m for m in messages if m.get('role') == 'user']

    if len(user_messages) < 3:
        return

    patterns_found = []

    # Check for iterative development pattern
    edit_mentions = sum(1 for m in messages if 'edit' in m.get('content', '').lower())
    if edit_mentions > 3:
        patterns_found.append(('workflow', 'Iterative development with frequent edits'))

    # Check for test-first pattern
    first_few = ' '.join(m.get('content', '')[:200].lower() for m in user_messages[:3])
    if 'test' in first_few:
        patterns_found.append(('workflow', 'Mentions testing early in conversation'))

    # Check for planning pattern
    if any(word in first_few for word in ['plan', 'design', 'architect', 'approach']):
        patterns_found.append(('workflow', 'Prefers planning before implementation'))

    # Check for direct action pattern
    first_msg = user_messages[0].get('content', '').lower()[:100]
    if any(word in first_msg for word in ['fix', 'change', 'update', 'add', 'remove', 'implement']):
        patterns_found.append(('workflow', 'Starts with direct action requests'))

    for pattern_type, description in patterns_found:
        cursor.execute("""
            SELECT id, frequency FROM work_patterns
            WHERE pattern_type = ? AND pattern_description = ?
        """, (pattern_type, description))

        row = cursor.fetchone()
        if row:
            cursor.execute("""
                UPDATE work_patterns SET frequency = ?, last_seen = ?
                WHERE id = ?
            """, (row[1] + 1, now, row[0]))
        else:
            cursor.execute("""
                INSERT INTO work_patterns
                (pattern_type, pattern_description, first_seen, last_seen)
                VALUES (?, ?, ?, ?)
            """, (pattern_type, description, now, now))


def get_full_custodian_profile() -> dict:
    """
    Get the complete custodian profile for providing context to Claude.

    This is the main function called by mira_init to provide rich context.
    """
    db_path = get_custodian_db_path()

    profile = {
        'name': get_custodian(),
        'identity': {},
        'preferences': {},
        'rules': {
            'always': [],
            'never': [],
            'avoid': [],
        },
        'danger_zones': [],
        'work_patterns': [],
        'summary': '',
    }

    if not db_path.exists():
        return profile

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get identity
        cursor.execute("SELECT key, value, confidence FROM identity ORDER BY confidence DESC")
        for key, value, confidence in cursor.fetchall():
            profile['identity'][key] = {'value': value, 'confidence': confidence}
            if key == 'name':
                profile['name'] = value

        # Get preferences by category
        cursor.execute("""
            SELECT category, preference, frequency, confidence
            FROM preferences
            WHERE confidence >= 0.5
            ORDER BY category, frequency DESC
        """)
        for category, pref, freq, conf in cursor.fetchall():
            if category not in profile['preferences']:
                profile['preferences'][category] = []
            profile['preferences'][category].append({
                'preference': pref,
                'frequency': freq,
                'confidence': conf
            })

        # Get rules
        cursor.execute("""
            SELECT rule_type, rule_text, frequency
            FROM rules
            WHERE frequency >= 1
            ORDER BY frequency DESC
            LIMIT 20
        """)
        for rule_type, rule_text, freq in cursor.fetchall():
            if rule_type in profile['rules']:
                profile['rules'][rule_type].append({
                    'rule': rule_text,
                    'frequency': freq
                })

        # Get danger zones
        cursor.execute("""
            SELECT path_pattern, issue_description, issue_count, last_issue
            FROM danger_zones
            WHERE issue_count >= 2
            ORDER BY issue_count DESC
            LIMIT 10
        """)
        for path, desc, count, last in cursor.fetchall():
            profile['danger_zones'].append({
                'path': path,
                'description': desc,
                'issue_count': count,
                'last_issue': last
            })

        # Get work patterns
        cursor.execute("""
            SELECT pattern_description, frequency, confidence
            FROM work_patterns
            WHERE frequency >= 2
            ORDER BY frequency DESC
            LIMIT 10
        """)
        for desc, freq, conf in cursor.fetchall():
            profile['work_patterns'].append({
                'pattern': desc,
                'frequency': freq,
                'confidence': conf
            })

        conn.close()

        # Build summary
        profile['summary'] = _build_profile_summary(profile)

    except Exception as e:
        log(f"Error loading custodian profile: {e}")

    return profile


def _build_profile_summary(profile: dict) -> str:
    """Build a human-readable summary of the custodian profile."""
    parts = []

    name = profile.get('name', 'Unknown')
    parts.append(f"Custodian: {name}")

    # Preferences summary
    prefs = profile.get('preferences', {})
    if prefs.get(PREF_TOOLS):
        tools = [p['preference'] for p in prefs[PREF_TOOLS][:3]]
        parts.append(f"Preferred tools: {', '.join(tools)}")

    if prefs.get(PREF_CODING_STYLE):
        styles = [p['preference'] for p in prefs[PREF_CODING_STYLE][:2]]
        parts.append(f"Coding style: {', '.join(styles)}")

    # Key rules
    rules = profile.get('rules', {})
    if rules.get('never'):
        never = [r['rule'][:50] for r in rules['never'][:2]]
        parts.append(f"Never: {'; '.join(never)}")

    if rules.get('always'):
        always = [r['rule'][:50] for r in rules['always'][:2]]
        parts.append(f"Always: {'; '.join(always)}")

    # Danger zones
    dangers = profile.get('danger_zones', [])
    if dangers:
        zones = [d['path'] for d in dangers[:3]]
        parts.append(f"Caution areas: {', '.join(zones)}")

    return ' | '.join(parts) if parts else "No profile data yet"


def get_danger_zones_for_files(file_paths: list) -> list:
    """
    Check if any of the given file paths match known danger zones.

    Called when about to work on files to warn Claude.
    """
    db_path = get_custodian_db_path()
    if not db_path.exists():
        return []

    warnings = []

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT path_pattern, issue_description, issue_count FROM danger_zones")
        danger_zones = cursor.fetchall()
        conn.close()

        for path in file_paths:
            path_lower = path.lower()
            for pattern, desc, count in danger_zones:
                if pattern.lower() in path_lower:
                    warnings.append({
                        'file': path,
                        'pattern': pattern,
                        'warning': desc,
                        'issue_count': count
                    })
    except Exception as e:
        log(f"Error checking danger zones: {e}")

    return warnings
