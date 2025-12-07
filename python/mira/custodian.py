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

# Development lifecycle signals - keywords that indicate each phase
LIFECYCLE_SIGNALS = {
    'plan': ['plan', 'design', 'architect', 'approach', 'outline', 'strategy', 'think through', 'let me think'],
    'test_first': ['write test', 'test first', 'tdd', 'test case', 'failing test', 'red green', 'spec first'],
    'implement': ['implement', 'build it', 'code it', 'develop', 'write the code', 'let\'s build', 'create the'],
    'test_after': ['run test', 'verify', 'check if', 'make sure', 'validate', 'confirm it works', 'test it'],
    'document': ['document', 'readme', 'add comment', 'explain', 'docstring', 'write up', 'update docs'],
    'review': ['review', 'refactor', 'clean up', 'polish', 'improve', 'optimize'],
    'commit': ['commit', 'push', 'pr', 'pull request', 'merge', 'ship it'],
}

# Human-readable phase names for output
LIFECYCLE_PHASE_NAMES = {
    'plan': 'Plan',
    'test_first': 'Write Tests',
    'implement': 'Implement',
    'test_after': 'Test',
    'document': 'Document',
    'review': 'Review',
    'commit': 'Commit',
}


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
        # More interaction preferences - but NOT generic approval words
        (PREF_COMMUNICATION, r"(?:don't |never )(ask (?:me )?questions?|prompt me)", 0.8),
        # NOTE: Removed "proceed/continue/go ahead" - these are approval words, not actual preferences
        # (PREF_COMMUNICATION, r"(?:just |please )(do it|proceed|continue|go ahead)", 0.6),
        (PREF_COMMUNICATION, r"(explain (?:as you go|while you work|your (?:thinking|reasoning)))", 0.7),
        (PREF_COMMUNICATION, r"(step by step|one step at a time)", 0.6),
        (PREF_COMMUNICATION, r"(commit (?:frequently|often|as you go)|small commits)", 0.7),
        (PREF_COMMUNICATION, r"(don't commit|no commits?|i'll commit)", 0.8),
        (PREF_COMMUNICATION, r"(run tests? (?:first|before|after))", 0.7),
        (PREF_COMMUNICATION, r"(show (?:me )?(?:the )?diff|what (?:did you )?change)", 0.6),
        # Real preferences about proceeding without questions
        (PREF_COMMUNICATION, r"(proceed without asking|don't ask.{0,10}just do)", 0.8),
        (PREF_COMMUNICATION, r"(work autonomously|be autonomous|less hand-?holding)", 0.7),
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

    # Patterns indicating problems - require actual file paths
    # File paths must have: letters/underscores, possibly slashes, and a file extension
    file_path_pattern = r'[a-zA-Z_][a-zA-Z0-9_/\-]*\.[a-zA-Z]{1,4}'

    problem_patterns = [
        # "error in src/foo.ts"
        rf"(?:error|issue|bug|problem|broke|breaking|failed)\s+(?:in|with|at)\s+`?({file_path_pattern})`?",
        # "src/foo.ts is broken"
        rf"`?({file_path_pattern})`?\s+(?:is broken|has issues|keeps failing|caused|causing)",
        # "careful with src/foo.ts"
        rf"(?:careful with|watch out for|be cautious with)\s+`?({file_path_pattern})`?",
    ]

    # Skip patterns - these indicate the text is documentation or discussion, not actual issues
    skip_patterns = [
        r'danger.?zones',  # Meta-discussion about danger zones feature
        r'problematic\s+files\)',  # Documentation text
        r'example',
        r'documentation',
    ]

    for msg in messages:
        content = msg.get('content', '')

        # Skip if content looks like documentation/meta-discussion
        if any(re.search(p, content, re.IGNORECASE) for p in skip_patterns):
            continue

        for pattern in problem_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                path_pattern = match.strip().strip('`')

                # Validate it looks like a real file path
                if len(path_pattern) < 5 or len(path_pattern) > 100:
                    continue

                # Must contain a file extension (not just end with .something)
                if not re.search(r'\.(ts|js|py|tsx|jsx|json|md|yaml|yml|go|rs|java|c|cpp|h)$', path_pattern):
                    continue

                # Skip common false positives
                if path_pattern.lower() in ['hnsw', 'files)', 'files),']:
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


def _detect_lifecycle_sequence(messages: list) -> tuple[Optional[str], float]:
    """
    Analyze conversation flow to detect the user's development lifecycle.

    Returns a tuple of (lifecycle_string, confidence) where lifecycle_string
    is like "Plan → Implement → Test" based on the order phases appear.

    Focus on the EARLY phases (what does the user do first?) as these are
    the most indicative of their preferred workflow style.
    """
    user_messages = [m for m in messages if m.get('role') == 'user']

    if len(user_messages) < 3:
        return None, 0.0

    # Track when each phase first appears (by message index)
    first_occurrence = {}
    phase_counts = Counter()

    for idx, msg in enumerate(user_messages):
        content = msg.get('content', '').lower()

        for phase, keywords in LIFECYCLE_SIGNALS.items():
            # Check if any keyword appears in this message
            if any(kw in content for kw in keywords):
                phase_counts[phase] += 1
                if phase not in first_occurrence:
                    first_occurrence[phase] = idx

    # Need at least 2 distinct phases to form a lifecycle
    if len(first_occurrence) < 2:
        return None, 0.0

    # Build sequence from ordering of first occurrences
    ordered_phases = sorted(first_occurrence.items(), key=lambda x: x[1])

    # Build the lifecycle sequence (up to 4 phases)
    # Include commit if it appears early (indicates frequent committer)
    # Skip 'review' and 'document' as they're less actionable
    lifecycle_phases = []
    for phase, idx in ordered_phases:
        if phase in ('review', 'document'):
            continue
        # Include commit only if it appears in first half of conversation
        # (indicates "commit as you go" style vs "commit at end")
        if phase == 'commit':
            midpoint = len(user_messages) // 2
            if idx > midpoint:
                continue  # Late commit - not distinctive
        lifecycle_phases.append(phase)
        if len(lifecycle_phases) >= 4:
            break

    if len(lifecycle_phases) < 2:
        return None, 0.0

    # Convert to human-readable names
    lifecycle_str = ' → '.join(LIFECYCLE_PHASE_NAMES.get(p, p.title()) for p in lifecycle_phases)

    # Calculate confidence based on:
    # - Number of core phases detected (plan, implement, test, commit)
    # - Whether plan appears first (strong signal of intentional workflow)
    core_phases = {'plan', 'implement', 'test_first', 'test_after', 'commit'}
    core_detected = sum(1 for p in lifecycle_phases if p in core_phases)

    plan_first_bonus = 0.2 if (lifecycle_phases and lifecycle_phases[0] == 'plan') else 0.0
    test_first_bonus = 0.15 if 'test_first' in lifecycle_phases else 0.0
    commit_bonus = 0.1 if 'commit' in lifecycle_phases else 0.0  # Bonus for explicit commit pattern

    base_confidence = min(core_detected / 4.0, 1.0) * 0.55
    confidence = base_confidence + plan_first_bonus + test_first_bonus + commit_bonus

    return lifecycle_str, round(min(confidence, 1.0), 2)


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

    # Detect development lifecycle sequence
    lifecycle, lifecycle_confidence = _detect_lifecycle_sequence(messages)
    if lifecycle and lifecycle_confidence >= 0.3:
        patterns_found.append(('lifecycle', lifecycle))

    for pattern_type, description in patterns_found:
        cursor.execute("""
            SELECT id, frequency, confidence FROM work_patterns
            WHERE pattern_type = ? AND pattern_description = ?
        """, (pattern_type, description))

        row = cursor.fetchone()
        if row:
            # For lifecycle patterns, update confidence based on new detection
            new_confidence = row[2]
            if pattern_type == 'lifecycle':
                # Blend existing confidence with new detection
                new_confidence = min(1.0, (row[2] + lifecycle_confidence) / 2 + 0.05)

            cursor.execute("""
                UPDATE work_patterns SET frequency = ?, last_seen = ?, confidence = ?
                WHERE id = ?
            """, (row[1] + 1, now, new_confidence, row[0]))
        else:
            initial_confidence = lifecycle_confidence if pattern_type == 'lifecycle' else 0.5
            cursor.execute("""
                INSERT INTO work_patterns
                (pattern_type, pattern_description, confidence, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
            """, (pattern_type, description, initial_confidence, now, now))


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
        'development_lifecycle': None,  # Most common lifecycle sequence
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
        # Filter out generic approval words that aren't actual preferences
        generic_approval_words = {
            'proceed', 'continue', 'yes', 'ok', 'okay', 'sure', 'thanks', 'thank',
            'good', 'great', 'nice', 'perfect', 'go', 'ahead', 'do', 'it', 'please',
            'looks', 'lgtm', 'ship', 'go ahead', 'do it', 'looks good'
        }

        cursor.execute("""
            SELECT category, preference, frequency, confidence
            FROM preferences
            WHERE confidence >= 0.5
            ORDER BY category, frequency DESC
        """)
        for category, pref, freq, conf in cursor.fetchall():
            # Skip generic approval words
            if pref.lower().strip() in generic_approval_words:
                continue

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

        # Get work patterns (non-lifecycle)
        cursor.execute("""
            SELECT pattern_description, frequency, confidence
            FROM work_patterns
            WHERE frequency >= 2 AND pattern_type = 'workflow'
            ORDER BY frequency DESC
            LIMIT 10
        """)
        for desc, freq, conf in cursor.fetchall():
            profile['work_patterns'].append({
                'pattern': desc,
                'frequency': freq,
                'confidence': conf
            })

        # Get the most common development lifecycle
        # Weight by frequency, confidence, AND recency
        # Recency bonus: patterns seen in last 7 days get 2x weight, last 30 days get 1.5x
        # This allows the lifecycle to evolve as user habits change
        cursor.execute("""
            SELECT pattern_description, frequency, confidence, last_seen,
                   (frequency * confidence *
                    CASE
                        WHEN julianday('now') - julianday(last_seen) <= 7 THEN 2.0
                        WHEN julianday('now') - julianday(last_seen) <= 30 THEN 1.5
                        ELSE 1.0
                    END
                   ) as score
            FROM work_patterns
            WHERE pattern_type = 'lifecycle' AND confidence >= 0.5
            ORDER BY score DESC
            LIMIT 1
        """)
        lifecycle_row = cursor.fetchone()
        if lifecycle_row:
            profile['development_lifecycle'] = {
                'sequence': lifecycle_row[0],
                'frequency': lifecycle_row[1],
                'confidence': lifecycle_row[2]
            }

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
