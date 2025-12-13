"""
MIRA3 Custodian Learning Module

Learns about the user (custodian) from their conversation patterns:
- Identity (name, aliases)
- Preferences (coding style, tools, frameworks)
- Rules (always/never patterns)
- Work patterns (dev loop, communication style)
- Danger zones (files/modules that caused issues)

Uses central Postgres for key preferences with local SQLite fallback.
"""

import json
import re
from datetime import datetime
from collections import Counter
from typing import Optional

from .utils import log, get_mira_path, get_custodian
from .db_manager import get_db_manager
from .rules import (
    RULE_TYPES, RULE_PATTERNS, CONDITIONAL_RULE_PATTERNS,
    RULE_REVOCATION_PATTERNS, RULE_FILTER_WORDS, RULE_FILLER_WORDS,
    normalize_rule_text, is_rule_false_positive, find_similar_rule,
    get_rules_with_decay, format_rule_for_display, extract_scope_from_content,
)
from .prerequisites import (
    PREREQ_STATEMENT_PATTERNS, PREREQ_COMMAND_PATTERNS,
    PREREQ_REASON_PATTERNS, PREREQ_CHECK_TEMPLATES, PREREQ_KEYWORDS,
)
from .migrations import CUSTODIAN_SCHEMA


# Global storage instance for custodian
_custodian_storage = None


def _get_custodian_storage():
    """Get storage instance for custodian (lazy init)."""
    global _custodian_storage
    if _custodian_storage is None:
        try:
            from .storage import get_storage
            _custodian_storage = get_storage()
        except ImportError:
            pass
    return _custodian_storage


# Database name for custodian
CUSTODIAN_DB = "custodian.db"

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





def init_custodian_db():
    """Initialize the custodian learning database."""
    db = get_db_manager()
    db.init_schema(CUSTODIAN_DB, CUSTODIAN_SCHEMA)
    log("Custodian database initialized")


def sync_from_central() -> int:
    """
    Pull critical custodian data from central PostgreSQL to local SQLite.

    This ensures local resilience when central is temporarily unavailable.
    Called during profile building to ensure local cache is populated.

    Returns:
        Number of name candidates synced
    """
    try:
        from .storage import get_storage
        storage = get_storage()
        if not storage.using_central or not storage.postgres:
            return 0

        db = get_db_manager()
        synced = 0

        # Sync name candidates from central
        try:
            candidates = storage.postgres.get_all_name_candidates()
            for candidate in candidates:
                try:
                    db.execute_write(
                        CUSTODIAN_DB,
                        """INSERT INTO name_candidates (name, confidence, pattern_type, source_session, context, extracted_at)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(name, source_session) DO UPDATE SET
                               confidence = MAX(name_candidates.confidence, excluded.confidence),
                               pattern_type = COALESCE(excluded.pattern_type, name_candidates.pattern_type),
                               context = COALESCE(excluded.context, name_candidates.context),
                               extracted_at = COALESCE(excluded.extracted_at, name_candidates.extracted_at)""",
                        (
                            candidate['name'],
                            candidate['confidence'],
                            candidate['pattern_type'],
                            candidate['source_session'],
                            candidate.get('context', ''),
                            candidate.get('extracted_at', datetime.now().isoformat())
                        )
                    )
                    synced += 1
                except Exception as e:
                    log(f"Failed to sync name candidate {candidate.get('name')}: {e}")

            if synced > 0:
                log(f"Synced {synced} name candidates from central to local")

        except Exception as e:
            log(f"Failed to fetch name candidates from central: {e}")

        return synced

    except Exception as e:
        log(f"Central-to-local sync failed: {e}")
        return 0


def extract_custodian_learnings(conversation: dict, session_id: str) -> dict:
    """
    Extract learnings about the custodian from a conversation.

    Called during ingestion to learn from each conversation.
    Returns dict with 'learned' count.
    """
    messages = conversation.get('messages', [])
    if not messages:
        return {'learned': 0}

    db = get_db_manager()
    now = datetime.now().isoformat()
    learned = 0

    try:
        # Extract from user messages
        user_messages = [m for m in messages if m.get('role') == 'user']

        # Learn identity
        learned += _learn_identity(db, user_messages, session_id, now)

        # Learn preferences from user statements
        learned += _learn_preferences(db, user_messages, session_id, now)

        # Learn rules from both user and assistant
        learned += _learn_rules(db, messages, session_id, now)

        # Learn danger zones from error patterns
        learned += _learn_danger_zones(db, messages, session_id, now)

        # Learn work patterns
        learned += _learn_work_patterns(db, messages, session_id, now)

        # Learn environment-specific prerequisites
        learned += _learn_prerequisites(db, messages, session_id, now)

    except Exception as e:
        log(f"Error extracting custodian learnings: {e}")

    return {'learned': learned}


def _learn_identity(db, user_messages: list, session_id: str, now: str) -> int:
    """Learn identity information from user messages."""
    learned = 0

    # CONSERVATIVE name patterns - only explicit first-person introductions
    # These require the user to directly state their name
    name_patterns = [
        # Highest confidence: explicit "my name is" (allow word boundary, not just sentence start)
        (r"(?:^|\s)my name is\s+([A-Z][a-z]{2,15})(?:\s|[.,!?]|$)", 0.95),
        # High confidence: "I'm [Name]" at start of message or after greeting
        (r"(?:^|[.!?]\s*|,\s*)(?:hi,?\s+)?i'?m\s+([A-Z][a-z]{2,15})(?:\s|[.,!?]|$)", 0.9),
        # Medium confidence: "call me [Name]"
        (r"(?:^|\s)(?:you can |please )?call me\s+([A-Z][a-z]{2,15})(?:\s|[.,!?]|$)", 0.85),
        # Medium confidence: "I am [Name]" (explicit statement)
        (r"(?:^|\s)i am\s+([A-Z][a-z]{2,15})(?:\s|[.,!?]|$)", 0.85),
        # Lower confidence: sign-off at END of message only (last 100 chars)
        # This is handled specially below
    ]

    # Comprehensive blocklist - tech products, tools, frameworks, common words
    name_blocklist = {
        # Greetings and conversation
        'claude', 'hello', 'please', 'thanks', 'help', 'just', 'the', 'this', 'that',
        'if', 'when', 'what', 'how', 'where', 'why', 'yes', 'no', 'okay', 'sure',
        'now', 'then', 'here', 'there', 'which', 'who', 'whom', 'hey', 'hi',

        # Common sentence starters and fillers
        'well', 'also', 'actually', 'basically', 'honestly', 'really', 'maybe',
        'perhaps', 'probably', 'certainly', 'definitely', 'absolutely', 'anyway',

        # Tech products and services (CRITICAL - prevents "Tailscale" etc.)
        'tailscale', 'docker', 'kubernetes', 'redis', 'postgres', 'postgresql',
        'mongodb', 'mysql', 'sqlite', 'elasticsearch', 'nginx', 'apache',
        'github', 'gitlab', 'bitbucket', 'vercel', 'netlify', 'heroku', 'aws',
        'azure', 'gcp', 'cloudflare', 'digitalocean', 'linode', 'vultr',
        'slack', 'discord', 'notion', 'linear', 'jira', 'asana', 'trello',
        'stripe', 'twilio', 'sendgrid', 'mailgun', 'auth0', 'okta', 'clerk',
        'supabase', 'firebase', 'planetscale', 'neon', 'turso', 'upstash',
        'openai', 'anthropic', 'cohere', 'huggingface', 'replicate', 'modal',
        'sentry', 'datadog', 'grafana', 'prometheus', 'kibana', 'splunk',
        'terraform', 'pulumi', 'ansible', 'vagrant', 'packer', 'consul',
        'chromadb', 'chroma', 'pinecone', 'weaviate', 'milvus', 'qdrant', 'faiss',

        # Programming languages and runtimes
        'python', 'javascript', 'typescript', 'golang', 'rust', 'java', 'kotlin',
        'swift', 'ruby', 'php', 'perl', 'scala', 'elixir', 'clojure', 'haskell',
        'node', 'nodejs', 'deno', 'bun', 'dotnet',

        # Frameworks and libraries
        'react', 'vue', 'angular', 'svelte', 'solid', 'qwik', 'astro', 'remix',
        'next', 'nuxt', 'gatsby', 'vite', 'webpack', 'rollup', 'esbuild', 'parcel',
        'express', 'fastapi', 'django', 'flask', 'fastify', 'koa', 'hono', 'rails',
        'spring', 'laravel', 'phoenix', 'actix', 'axum', 'rocket', 'warp', 'hyper',
        'prisma', 'drizzle', 'typeorm', 'sequelize', 'knex', 'mongoose',
        'jest', 'vitest', 'mocha', 'pytest', 'junit', 'rspec', 'cypress', 'playwright',

        # Tools and CLIs
        'npm', 'pnpm', 'yarn', 'pip', 'cargo', 'maven', 'gradle', 'brew', 'apt',
        'git', 'vim', 'neovim', 'emacs', 'vscode', 'cursor', 'zed', 'sublime',

        # Game/content words
        'planet', 'sector', 'ship', 'trade', 'port', 'warp', 'credits', 'player',
        'game', 'level', 'score', 'item', 'quest', 'mission', 'world', 'server',
        'guild', 'clan', 'team', 'alliance', 'faction', 'empire', 'kingdom',

        # Tech/code terms
        'file', 'code', 'function', 'class', 'method', 'error', 'warning', 'test',
        'api', 'sdk', 'cli', 'gui', 'url', 'uri', 'json', 'yaml', 'xml', 'html',
        'css', 'sql', 'graphql', 'rest', 'grpc', 'websocket', 'http', 'https',
        'backend', 'frontend', 'fullstack', 'devops', 'sre', 'mlops', 'dataops',
        'component', 'module', 'package', 'library', 'framework', 'runtime',
        'database', 'cache', 'queue', 'worker', 'service', 'daemon', 'process',
        'container', 'pod', 'cluster', 'node', 'instance', 'replica', 'shard',

        # Action words that appear in "Let me [action]..." or "I'm [action]..." patterns
        # These get falsely extracted as names from phrases like "Let me pause and think"
        'pause', 'think', 'thinking', 'check', 'checking', 'look', 'looking',
        'see', 'seeing', 'try', 'trying', 'start', 'starting', 'stop', 'stopping',
        'wait', 'waiting', 'continue', 'continuing', 'proceed', 'proceeding',
        'begin', 'beginning', 'finish', 'finishing', 'review', 'reviewing',
        'working', 'going', 'doing', 'done', 'done', 'ready', 'reading',
        'writing', 'running', 'testing', 'building', 'deploying', 'updating',
        'fixing', 'debugging', 'investigating', 'analyzing', 'processing',
        'wondering', 'curious', 'confused', 'stuck', 'lost', 'back',
        'sorry', 'glad', 'happy', 'excited', 'afraid', 'worried', 'concerned',
    }

    # Map patterns to their types for tracking
    pattern_types = {
        0: 'my_name_is',  # "my name is X"
        1: 'im_introduction',  # "I'm X"
        2: 'call_me',  # "call me X"
    }

    for msg in user_messages:
        content = msg.get('content', '')
        if not content or len(content) < 10:
            continue

        # Check main patterns
        for idx, (pattern, confidence) in enumerate(name_patterns):
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Validate: not in blocklist, reasonable length, looks like a name
                if (name.lower() not in name_blocklist and
                    3 <= len(name) <= 15 and
                    name[0].isupper() and
                    name[1:].islower() and
                    not any(c.isdigit() for c in name)):

                    # Get surrounding context for debugging
                    start = max(0, match.start() - 20)
                    end = min(len(content), match.end() + 20)
                    context = content[start:end]

                    pattern_type = pattern_types.get(idx, 'unknown')
                    _store_name(db, name, confidence, session_id, now, pattern_type, context)
                    learned += 1
                    break

        # Check sign-off pattern at end of message only
        if not learned:
            # Only check last 100 chars for sign-offs
            tail = content[-100:] if len(content) > 100 else content
            signoff_match = re.search(
                r"(?:regards|cheers|thanks|best),?\s*\n\s*([A-Z][a-z]{2,15})\s*$",
                tail
            )
            if signoff_match:
                name = signoff_match.group(1)
                if (name.lower() not in name_blocklist and
                    3 <= len(name) <= 15 and
                    name[0].isupper() and
                    name[1:].islower()):
                    _store_name(db, name, 0.75, session_id, now, 'signoff', tail[-50:])
                    learned += 1

    return learned


def _store_name(db, name: str, confidence: float, session_id: str, now: str, pattern_type: str = 'unknown', context: str = ''):
    """
    Store a name candidate for later scoring.

    Instead of directly storing THE name, we store all candidates and
    compute the best one when retrieving the profile.
    """
    # Store as candidate in local SQLite (name_candidates table)
    try:
        db.execute_write(
            CUSTODIAN_DB,
            """INSERT INTO name_candidates (name, confidence, pattern_type, source_session, context, extracted_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(name, source_session) DO UPDATE SET
                   confidence = MAX(name_candidates.confidence, excluded.confidence),
                   pattern_type = excluded.pattern_type,
                   context = excluded.context,
                   extracted_at = excluded.extracted_at""",
            (name, confidence, pattern_type, session_id, context[:200] if context else '', now)
        )
    except Exception as e:
        # Table might not exist yet (pre-migration), fall back to old behavior
        log(f"name_candidates insert failed (migration pending?): {e}")
        db.execute_write(
            CUSTODIAN_DB,
            """INSERT OR REPLACE INTO identity (key, value, confidence, source_session, learned_at)
               VALUES (?, ?, ?, ?, ?)""",
            ('name', name, confidence, session_id, now)
        )

    # Also store in central Postgres
    storage = _get_custodian_storage()
    if storage and storage.using_central:
        try:
            storage.postgres.upsert_name_candidate(
                name=name,
                confidence=confidence,
                pattern_type=pattern_type,
                source_session=session_id,
                context=context[:200] if context else '',
            )
        except Exception as e:
            # Fall back to old custodian storage if new method doesn't exist
            try:
                storage.postgres.upsert_custodian(
                    key='identity:name',
                    value=name,
                    category='identity',
                    confidence=confidence,
                    source_session=session_id,
                )
            except Exception as e2:
                log(f"Central identity storage failed: {e2}")


def compute_best_name() -> Optional[tuple]:
    """
    Compute the best name from all candidates using a scoring function.

    Scoring considers:
    - Total confidence across all extractions for this name
    - Number of sessions that extracted this name (frequency bonus)
    - Pattern quality: my_name_is > im_introduction > call_me > signoff
    - Recency: recent extractions weighted more

    Returns:
        Tuple of (name, score, confidence, num_sessions) or None if no candidates
    """
    import math
    db = get_db_manager()

    # Pattern quality weights (higher = more trustworthy)
    pattern_weights = {
        'my_name_is': 1.5,       # Explicit "my name is" - very reliable
        'im_introduction': 1.2,  # "I'm X" - fairly reliable
        'call_me': 1.1,          # "call me X" - reliable
        'signoff': 0.8,          # Sign-offs - less reliable
        'unknown': 0.7,          # Unknown pattern - treat cautiously
    }

    def _score_rows(rows, from_postgres=False):
        """Score name candidates from either local or central storage."""
        best_name = None
        best_score = -1

        for row in rows:
            if from_postgres:
                # PostgreSQL returns tuples
                name, total_conf, num_sessions, max_conf, patterns = row[:5]
                patterns = patterns or ['unknown']
            else:
                # SQLite returns dicts
                name = row['name']
                total_conf = row['total_conf'] or 0
                num_sessions = row['num_sessions'] or 1
                max_conf = row['max_conf'] or 0
                patterns = (row['patterns'] or 'unknown').split(',')

            # Calculate pattern quality bonus (use best pattern seen)
            if isinstance(patterns, str):
                patterns = patterns.split(',')
            pattern_bonus = max(pattern_weights.get(p.strip() if isinstance(p, str) else p, 0.7) for p in patterns)

            # Frequency bonus: more sessions = more confidence
            freq_bonus = math.log((num_sessions or 1) + 1)

            # Final score: combines confidence, frequency, and pattern quality
            score = ((total_conf or 0) * pattern_bonus) + freq_bonus

            if score > best_score:
                best_score = score
                best_name = (name, round(score, 2), max_conf or 0, num_sessions or 1)

        return best_name

    # Try local first (fast)
    try:
        rows = db.execute_read(CUSTODIAN_DB, """
            SELECT
                name,
                SUM(confidence) as total_conf,
                COUNT(DISTINCT source_session) as num_sessions,
                MAX(confidence) as max_conf,
                GROUP_CONCAT(DISTINCT pattern_type) as patterns,
                MAX(extracted_at) as last_seen
            FROM name_candidates
            GROUP BY name
        """)

        if rows:
            result = _score_rows(rows, from_postgres=False)
            if result:
                return result

    except Exception as e:
        log(f"Error reading local name candidates: {e}")

    # Fallback: Try central PostgreSQL
    try:
        from .storage import Storage
        storage = Storage()
        if storage._init_central() and storage._postgres:
            result = storage._postgres.get_best_name()
            if result:
                # Convert dict to tuple format (name, score, confidence, num_sessions)
                return (
                    result['name'],
                    round(result.get('score', 0), 2),
                    result.get('confidence', 0),  # postgres returns 'confidence'
                    result.get('sessions', 1)     # postgres returns 'sessions'
                )
    except Exception as e:
        log(f"Error reading central name candidates: {e}")

    return None


def get_all_name_candidates() -> list:
    """
    Get all name candidates with their details for debugging/display.

    Returns:
        List of dicts with name, confidence, sessions, score
    """
    db = get_db_manager()

    try:
        rows = db.execute_read(CUSTODIAN_DB, """
            SELECT
                name,
                SUM(confidence) as total_conf,
                COUNT(DISTINCT source_session) as num_sessions,
                MAX(confidence) as max_conf,
                GROUP_CONCAT(DISTINCT pattern_type) as patterns
            FROM name_candidates
            GROUP BY name
            ORDER BY SUM(confidence) DESC
            LIMIT 10
        """)

        candidates = []
        for row in rows:
            candidates.append({
                'name': row['name'],
                'total_confidence': round(row['total_conf'] or 0, 2),
                'sessions': row['num_sessions'] or 0,
                'max_confidence': round(row['max_conf'] or 0, 2),
                'patterns': (row['patterns'] or '').split(','),
            })

        return candidates

    except Exception as e:
        log(f"Error getting name candidates: {e}")
        return []


def _learn_preferences(db, user_messages: list, session_id: str, now: str) -> int:
    """Learn preferences from user statements."""
    learned = 0

    # Game/non-development content words to filter out
    game_content_words = {
        'planet', 'sector', 'ship', 'trade', 'port', 'warp', 'credits', 'player',
        'game', 'level', 'score', 'item', 'quest', 'mission', 'world', 'server',
        'attack', 'defend', 'enemy', 'spawn', 'health', 'damage', 'inventory',
        'character', 'npc', 'boss', 'dungeon', 'loot', 'xp', 'mana', 'spell',
        'population', 'resource', 'colony', 'fleet', 'station', 'galaxy',
        'allocated', 'landed', 'docked', 'warped', 'jumped',
    }

    # Preference patterns with categories - now much more specific
    pref_patterns = [
        # Coding style preferences - require development-specific context
        (PREF_CODING_STYLE, r"(?:i always prefer|i prefer to use|my coding style is)\s+([^.!?\n]{5,60})", 0.7),
        (PREF_CODING_STYLE, r"(?:i always|we always)\s+(?:use|write)\s+(type ?script|strict mode|eslint|prettier|black|ruff)", 0.8),
        (PREF_CODING_STYLE, r"(?:don't|never|avoid)\s+(?:use|write)\s+(var|any type|console\.log|print statement)", 0.7),

        # Tool preferences - explicit tool names only
        (PREF_TOOLS, r"(?:i use|we use|using|my .* is)\s+(npm|pnpm|yarn|bun|pip|poetry|cargo|go mod)", 0.8),
        (PREF_TOOLS, r"(?:run|use|prefer)\s+(vitest|jest|pytest|mocha|cargo test)", 0.8),
        (PREF_TOOLS, r"(?:i use|we use)\s+(vscode|vim|neovim|emacs|intellij|webstorm)", 0.8),

        # Framework preferences - explicit framework names only
        (PREF_FRAMEWORKS, r"(?:using|we use|i use|prefer)\s+(react|vue|svelte|angular|next|nuxt|express|fastapi|django|flask)", 0.7),

        # Testing preferences - require explicit testing context
        (PREF_TESTING, r"(?:i (?:always |usually )?(?:prefer|like|want) to )(write tests? (?:before|after|first))", 0.7),
        (PREF_TESTING, r"(?:always )(run tests? before (?:commit|push|deploy))", 0.8),
        (PREF_TESTING, r"(?:i believe in|i practice|we practice)\s+(tdd|test.driven|bdd)", 0.8),

        # Communication preferences - require explicit communication/workflow context
        (PREF_COMMUNICATION, r"(?:please )?(?:be |keep (?:it |responses? )?|make (?:it |responses? )?)(concise|brief|detailed|verbose|short)", 0.7),
        (PREF_COMMUNICATION, r"(no emojis?|without emojis?|don't use emojis?)", 0.9),
        (PREF_COMMUNICATION, r"(show (?:me )?code first|code before explanation)", 0.8),
        (PREF_COMMUNICATION, r"(?:please )?(?:don't |never )(ask (?:me )?(?:too many )?questions?|prompt me)", 0.8),
        (PREF_COMMUNICATION, r"(?:please )?(explain (?:as you go|while you work|your (?:thinking|reasoning)))", 0.7),
        (PREF_COMMUNICATION, r"(?:i prefer |please )(step by step|one step at a time)", 0.6),
        (PREF_COMMUNICATION, r"(?:please )?(commit (?:frequently|often|as you go)|small commits)", 0.7),
        (PREF_COMMUNICATION, r"(don't commit|no commits?|i'll commit|let me commit)", 0.8),
        (PREF_COMMUNICATION, r"(?:please )?(run tests? (?:first|before|after))", 0.7),
        (PREF_COMMUNICATION, r"(show (?:me )?(?:the )?diff|what (?:did you )?change)", 0.6),
        (PREF_COMMUNICATION, r"(proceed without asking|don't ask.{0,10}just do)", 0.8),
        (PREF_COMMUNICATION, r"(work autonomously|be autonomous|less hand-?holding)", 0.7),
    ]

    for msg in user_messages:
        content = msg.get('content', '').lower()

        # Skip messages that look like game content
        if any(word in content for word in ['planet', 'sector', 'warp', 'credits', 'fleet', 'colony', 'population', 'allocated']):
            continue

        for category, pattern, confidence in pref_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                pref_text = match.strip() if isinstance(match, str) else match
                if len(pref_text) < 3 or len(pref_text) > 50:
                    continue

                # Skip if contains game content words
                pref_lower = pref_text.lower()
                if any(word in pref_lower for word in game_content_words):
                    continue

                # Skip markdown, code, and documentation
                if '**' in pref_text or '`' in pref_text or '(' in pref_text:
                    continue

                def upsert_preference(cursor):
                    cursor.execute("""
                        SELECT id, frequency, source_sessions FROM preferences
                        WHERE category = ? AND preference = ?
                    """, (category, pref_text))

                    row = cursor.fetchone()
                    if row:
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
                        return 0
                    else:
                        cursor.execute("""
                            INSERT INTO preferences
                            (category, preference, confidence, first_seen, last_seen, source_sessions)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (category, pref_text, confidence, now, now, json.dumps([session_id])))
                        return 1

                learned += db.execute_write_func(CUSTODIAN_DB, upsert_preference)

                # Also store key preferences in central Postgres
                storage = _get_custodian_storage()
                if storage and storage.using_central and confidence >= 0.7:
                    try:
                        storage.postgres.upsert_custodian(
                            key=f'pref:{category}:{pref_text[:30]}',
                            value=pref_text,
                            category=category,
                            confidence=confidence,
                            source_session=session_id,
                        )
                    except Exception as e:
                        pass  # Don't log for preferences, too noisy

    return learned


def _learn_rules(db, messages: list, session_id: str, now: str) -> int:
    """
    Learn explicit rules from USER messages (custodian's own rules).

    Enhanced features:
    - Expanded pattern set (never, always, require, prefer, prohibit, avoid, style)
    - Conditional/scoped rules ("when testing, always...")
    - Semantic deduplication using normalized text
    - Improved false positive detection
    - Rule revocation detection
    """
    learned = 0
    user_messages = [m for m in messages if m.get('role') == 'user']

    for msg in user_messages:
        content = msg.get('content', '')
        content_lower = content.lower()

        # Skip messages that look like game content (quick filter)
        if any(word in content_lower for word in ['planet', 'sector', 'warp', 'credits', 'fleet', 'colony']):
            continue

        # Check for rule revocations first
        _check_rule_revocations(db, content, now)

        # Extract scope if present
        scope = extract_scope_from_content(content)

        # Process standard rule patterns
        for rule_type, pattern, base_confidence in RULE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Handle comparison patterns that return tuples
                if isinstance(match, tuple):
                    rule_text = f"{match[0]} over {match[1]}" if len(match) == 2 else match[0]
                else:
                    rule_text = match
                rule_text = rule_text.strip()

                # Use improved false positive detection
                if is_rule_false_positive(content, rule_text, rule_type):
                    continue

                # Store with normalization and deduplication
                learned += _upsert_rule_with_dedup(
                    db, rule_type, rule_text, base_confidence, scope, session_id, now
                )

        # Process conditional rule patterns
        for scope_pattern, rule_pattern, base_confidence in CONDITIONAL_RULE_PATTERNS:
            # Build combined pattern
            combined = scope_pattern + r",?\s+" + rule_pattern
            matches = re.findall(combined, content, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    cond_scope = match[0].strip()
                    rule_text = match[1].strip()

                    if is_rule_false_positive(content, rule_text, 'conditional'):
                        continue

                    # Determine rule type from the rule text
                    rule_type = 'always' if 'always' in rule_pattern else 'never' if 'never' in rule_pattern else 'require'

                    learned += _upsert_rule_with_dedup(
                        db, rule_type, rule_text, base_confidence, cond_scope, session_id, now
                    )

    return learned


def _check_rule_revocations(db, content: str, now: str) -> int:
    """
    Check for rule revocation patterns and mark matching rules as revoked.

    Returns number of rules revoked.
    """
    revoked = 0
    content_lower = content.lower()

    for pattern in RULE_REVOCATION_PATTERNS:
        matches = re.findall(pattern, content_lower)
        for match in matches:
            revocation_text = match.strip() if isinstance(match, str) else match[0].strip()
            normalized_revocation = normalize_rule_text(revocation_text)

            if not normalized_revocation or len(normalized_revocation) < 5:
                continue

            # Find rules that might match this revocation
            def revoke_matching(cursor):
                cursor.execute("""
                    SELECT id, rule_text, normalized_text FROM rules
                    WHERE revoked = 0
                """)
                rows = cursor.fetchall()

                revoked_count = 0
                revocation_words = set(normalized_revocation.split())

                for row in rows:
                    existing_norm = row[2] or normalize_rule_text(row[1])
                    existing_words = set(existing_norm.split())

                    # Check for significant word overlap
                    if not existing_words:
                        continue
                    overlap = len(revocation_words & existing_words) / len(existing_words)
                    if overlap >= 0.5:
                        cursor.execute("""
                            UPDATE rules SET revoked = 1, revoked_at = ? WHERE id = ?
                        """, (now, row[0]))
                        revoked_count += 1

                return revoked_count

            revoked += db.execute_write_func(CUSTODIAN_DB, revoke_matching)

    return revoked


def _upsert_rule_with_dedup(
    db, rule_type: str, rule_text: str, base_confidence: float,
    scope: str, session_id: str, now: str
) -> int:
    """
    Upsert a rule with semantic deduplication.

    If a similar rule exists (>70% word overlap), updates it instead of creating duplicate.
    """
    rule_text = rule_text[:200]  # Truncate if too long
    normalized = normalize_rule_text(rule_text)

    # Check for existing similar rule
    similar_id, similarity = find_similar_rule(db, rule_text, rule_type)

    def upsert(cursor):
        if similar_id:
            # Update existing similar rule
            cursor.execute("""
                SELECT frequency, source_sessions, confidence FROM rules WHERE id = ?
            """, (similar_id,))
            row = cursor.fetchone()
            if row:
                freq, sources, old_confidence = row
                sources_list = json.loads(sources) if sources else []
                if session_id not in sources_list:
                    sources_list.append(session_id)

                # Boost confidence slightly for reinforcement
                new_confidence = min(1.0, max(old_confidence, base_confidence) + 0.05)

                cursor.execute("""
                    UPDATE rules
                    SET frequency = ?, last_seen = ?, source_sessions = ?, confidence = ?
                    WHERE id = ?
                """, (freq + 1, now, json.dumps(sources_list[-10:]), new_confidence, similar_id))
                return 0  # Updated, not new
        else:
            # Insert new rule
            cursor.execute("""
                INSERT INTO rules
                (rule_type, rule_text, normalized_text, scope, confidence, first_seen, last_seen, source_sessions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (rule_type, rule_text, normalized, scope, base_confidence, now, now, json.dumps([session_id])))
            return 1  # New rule

    return db.execute_write_func(CUSTODIAN_DB, upsert)


def _learn_danger_zones(db, messages: list, session_id: str, now: str) -> int:
    """Learn about files/modules that caused issues."""
    learned = 0

    file_path_pattern = r'[a-zA-Z_][a-zA-Z0-9_/\-]*\.[a-zA-Z]{1,4}'

    problem_patterns = [
        rf"(?:error|issue|bug|problem|broke|breaking|failed)\s+(?:in|with|at)\s+`?({file_path_pattern})`?",
        rf"`?({file_path_pattern})`?\s+(?:is broken|has issues|keeps failing|caused|causing)",
        rf"(?:careful with|watch out for|be cautious with)\s+`?({file_path_pattern})`?",
    ]

    skip_patterns = [
        r'danger.?zones',
        r'problematic\s+files\)',
        r'example',
        r'documentation',
    ]

    for msg in messages:
        content = msg.get('content', '')

        if any(re.search(p, content, re.IGNORECASE) for p in skip_patterns):
            continue

        for pattern in problem_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                path_pattern = match.strip().strip('`')

                if len(path_pattern) < 5 or len(path_pattern) > 100:
                    continue

                if not re.search(r'\.(ts|js|py|tsx|jsx|json|md|yaml|yml|go|rs|java|c|cpp|h)$', path_pattern):
                    continue

                if path_pattern.lower() in ['hnsw', 'files)', 'files),']:
                    continue

                context_match = re.search(
                    rf".{{0,50}}{re.escape(path_pattern)}.{{0,50}}",
                    content,
                    re.IGNORECASE
                )
                context = context_match.group(0) if context_match else ""

                def upsert_danger_zone(cursor):
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
                        return 0
                    else:
                        cursor.execute("""
                            INSERT INTO danger_zones
                            (path_pattern, issue_description, last_issue, source_sessions)
                            VALUES (?, ?, ?, ?)
                        """, (path_pattern, context[:200], now, json.dumps([session_id])))
                        return 1

                learned += db.execute_write_func(CUSTODIAN_DB, upsert_danger_zone)

    return learned


def _detect_lifecycle_sequence(messages: list) -> tuple[Optional[str], float]:
    """
    Analyze conversation flow to detect the user's development lifecycle.

    Returns a tuple of (lifecycle_string, confidence) where lifecycle_string
    is like "Plan -> Implement -> Test" based on the order phases appear.
    """
    user_messages = [m for m in messages if m.get('role') == 'user']

    if len(user_messages) < 3:
        return None, 0.0

    first_occurrence = {}
    phase_counts = Counter()

    for idx, msg in enumerate(user_messages):
        content = msg.get('content', '').lower()

        for phase, keywords in LIFECYCLE_SIGNALS.items():
            if any(kw in content for kw in keywords):
                phase_counts[phase] += 1
                if phase not in first_occurrence:
                    first_occurrence[phase] = idx

    if len(first_occurrence) < 2:
        return None, 0.0

    ordered_phases = sorted(first_occurrence.items(), key=lambda x: x[1])

    lifecycle_phases = []
    total_msgs = len(user_messages)

    for phase, idx in ordered_phases:
        if phase in ('review', 'document'):
            continue

        if phase == 'commit':
            if idx <= 1:
                continue
            if idx > total_msgs * 0.8:
                continue

        lifecycle_phases.append(phase)
        if len(lifecycle_phases) >= 4:
            break

    if len(lifecycle_phases) < 2:
        return None, 0.0

    lifecycle_str = ' -> '.join(LIFECYCLE_PHASE_NAMES.get(p, p.title()) for p in lifecycle_phases)

    core_phases = {'plan', 'implement', 'test_first', 'test_after', 'commit'}
    core_detected = sum(1 for p in lifecycle_phases if p in core_phases)

    plan_first_bonus = 0.2 if (lifecycle_phases and lifecycle_phases[0] == 'plan') else 0.0
    test_first_bonus = 0.15 if 'test_first' in lifecycle_phases else 0.0
    commit_bonus = 0.1 if 'commit' in lifecycle_phases else 0.0

    base_confidence = min(core_detected / 4.0, 1.0) * 0.55
    confidence = base_confidence + plan_first_bonus + test_first_bonus + commit_bonus

    return lifecycle_str, round(min(confidence, 1.0), 2)


def _learn_work_patterns(db, messages: list, session_id: str, now: str) -> int:
    """Learn work patterns from conversation flow."""
    learned = 0

    user_messages = [m for m in messages if m.get('role') == 'user']

    if len(user_messages) < 3:
        return 0

    patterns_found = []

    # Check for iterative development pattern
    edit_mentions = sum(1 for m in messages if 'edit' in m.get('content', '').lower())
    if edit_mentions > 3:
        patterns_found.append(('workflow', 'Iterative development with frequent edits', 0.5))

    # Check for test-first pattern
    first_few = ' '.join(m.get('content', '')[:200].lower() for m in user_messages[:3])
    if 'test' in first_few:
        patterns_found.append(('workflow', 'Mentions testing early in conversation', 0.5))

    # Check for planning pattern
    if any(word in first_few for word in ['plan', 'design', 'architect', 'approach']):
        patterns_found.append(('workflow', 'Prefers planning before implementation', 0.5))

    # Check for direct action pattern
    first_msg = user_messages[0].get('content', '').lower()[:100]
    if any(word in first_msg for word in ['fix', 'change', 'update', 'add', 'remove', 'implement']):
        patterns_found.append(('workflow', 'Starts with direct action requests', 0.5))

    # Detect development lifecycle sequence
    lifecycle, lifecycle_confidence = _detect_lifecycle_sequence(messages)
    if lifecycle and lifecycle_confidence >= 0.3:
        patterns_found.append(('lifecycle', lifecycle, lifecycle_confidence))

    for pattern_type, description, initial_confidence in patterns_found:
        def upsert_pattern(cursor, ptype=pattern_type, desc=description, init_conf=initial_confidence):
            cursor.execute("""
                SELECT id, frequency, confidence FROM work_patterns
                WHERE pattern_type = ? AND pattern_description = ?
            """, (ptype, desc))

            row = cursor.fetchone()
            if row:
                new_confidence = row[2]
                if ptype == 'lifecycle':
                    new_confidence = min(1.0, (row[2] + init_conf) / 2 + 0.05)

                cursor.execute("""
                    UPDATE work_patterns SET frequency = ?, last_seen = ?, confidence = ?
                    WHERE id = ?
                """, (row[1] + 1, now, new_confidence, row[0]))
                return 0
            else:
                cursor.execute("""
                    INSERT INTO work_patterns
                    (pattern_type, pattern_description, confidence, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?)
                """, (ptype, desc, init_conf, now, now))
                return 1

        learned += db.execute_write_func(CUSTODIAN_DB, upsert_pattern)

        # Sync lifecycle patterns to central Postgres
        if pattern_type == 'lifecycle' and initial_confidence >= 0.5:
            storage = _get_custodian_storage()
            if storage and storage.using_central:
                try:
                    storage.postgres.upsert_lifecycle_pattern(
                        pattern=description,
                        confidence=initial_confidence,
                        source_session=session_id,
                    )
                except Exception as e:
                    log(f"Central lifecycle pattern storage failed: {e}")

    return learned


def _learn_prerequisites(db, messages: list, session_id: str, now: str) -> int:
    """
    Learn environment-specific prerequisites from conversation.

    Extracts statements like:
    - "In Codespaces, I need to start tailscaled first"
    - "On my home machine, run docker-compose up before tests"
    """
    learned = 0

    for msg in messages:
        content = msg.get('content', '')
        if isinstance(content, list):
            # Handle content blocks
            content = ' '.join(
                block.get('text', '')
                for block in content
                if isinstance(block, dict) and block.get('type') == 'text'
            )

        if not content or len(content) < 20:
            continue

        role = msg.get('role', '')
        content_lower = content.lower()

        # Quick check for prerequisite-related keywords
        if not any(kw in content_lower for kw in PREREQ_KEYWORDS):
            continue

        # Try each pattern to extract prerequisite statements
        for pattern in PREREQ_STATEMENT_PATTERNS:
            try:
                matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
            except re.error:
                continue

            for match in matches:
                groups = match.groups()
                if len(groups) < 2:
                    continue

                # Extract environment and action (order depends on pattern)
                env = None
                action = None

                # Most patterns: (env, action), some reversed
                g1, g2 = groups[0], groups[1] if len(groups) > 1 else None
                if g1 and g2:
                    # Heuristic: environments are shorter, actions are longer
                    if len(g1.strip()) < len(g2.strip()):
                        env, action = g1.strip(), g2.strip()
                    else:
                        action, env = g1.strip(), g2.strip()
                elif g1:
                    action = g1.strip()

                # Validate extraction
                if not action or len(action) < 5 or len(action) > 200:
                    continue
                if env and (len(env) < 2 or len(env) > 40):
                    continue
                if not env:
                    env = 'all'  # Applies to all environments

                # Normalize environment name
                env = env.lower().strip()

                # Extract command if present in the message
                command = None
                for cmd_pattern in PREREQ_COMMAND_PATTERNS:
                    try:
                        cmd_match = re.search(cmd_pattern, content, re.IGNORECASE | re.DOTALL)
                        if cmd_match:
                            cmd = cmd_match.group(1).strip()
                            # Validate it looks like a command
                            if cmd and len(cmd) > 3 and len(cmd) < 500:
                                command = cmd
                                break
                    except re.error:
                        continue

                # Extract reason if present
                reason = None
                for reason_pattern in PREREQ_REASON_PATTERNS:
                    try:
                        reason_match = re.search(reason_pattern, content, re.IGNORECASE)
                        if reason_match:
                            reason = reason_match.group(1).strip()
                            break
                    except re.error:
                        continue

                # Generate check command if we recognize the service
                check_command = None
                action_lower = action.lower()
                cmd_lower = (command or '').lower()

                for service, check in PREREQ_CHECK_TEMPLATES.items():
                    if service in action_lower or service in cmd_lower:
                        check_command = check
                        break

                # Calculate confidence
                confidence = 0.5
                if role == 'user':
                    confidence += 0.15  # User statements more authoritative
                if command:
                    confidence += 0.10  # Has concrete command
                if check_command:
                    confidence += 0.05  # We can verify it
                if reason:
                    confidence += 0.05  # Explained why
                confidence = min(1.0, confidence)

                # Store the prerequisite
                def upsert_prereq(cursor, e=env, a=action, c=command, cc=check_command,
                                  r=reason, conf=confidence, sid=session_id, n=now):
                    cursor.execute("""
                        SELECT id, frequency, confidence, command, check_command, reason
                        FROM prerequisites
                        WHERE environment = ? AND action = ?
                    """, (e, a))

                    row = cursor.fetchone()
                    if row:
                        # Update existing - merge data, boost confidence
                        new_conf = min(1.0, row[2] + 0.1)
                        new_cmd = c or row[3]
                        new_check = cc or row[4]
                        new_reason = r or row[5]

                        cursor.execute("""
                            UPDATE prerequisites
                            SET frequency = frequency + 1,
                                confidence = ?,
                                command = ?,
                                check_command = ?,
                                reason = ?,
                                last_confirmed = ?
                            WHERE id = ?
                        """, (new_conf, new_cmd, new_check, new_reason, n, row[0]))
                        return 0  # Not new
                    else:
                        cursor.execute("""
                            INSERT INTO prerequisites
                            (environment, action, command, check_command, reason,
                             confidence, source_session, learned_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (e, a, c, cc, r, conf, sid, n))
                        return 1  # New learning

                try:
                    result = db.execute_write_func(CUSTODIAN_DB, upsert_prereq)
                    learned += result
                    if result > 0:
                        log(f"Learned prerequisite: '{action}' for environment '{env}'")
                except Exception as e:
                    log(f"Error storing prerequisite: {e}")

    return learned


def get_full_custodian_profile() -> dict:
    """
    Get the complete custodian profile for providing context to Claude.

    This is the main function called by mira_init to provide rich context.
    Reads from both central Postgres and local SQLite.
    """
    db = get_db_manager()

    profile = {
        'name': get_custodian(),
        'identity': {},
        'preferences': {},
        'rules': {rt: [] for rt in RULE_TYPES.keys()},  # All rule types with decay
        'danger_zones': [],
        'work_patterns': [],
        'development_lifecycle': None,
        'summary': '',
    }

    # First, try to compute best name from candidates (new confidence-weighted approach)
    try:
        best_name_result = compute_best_name()
        if best_name_result:
            name, score, confidence, num_sessions = best_name_result
            profile['name'] = name
            profile['identity']['name'] = {
                'value': name,
                'confidence': confidence,
                'score': score,
                'sessions': num_sessions,
            }
            # Also include all candidates for transparency
            profile['identity']['name_candidates'] = get_all_name_candidates()
    except Exception as e:
        log(f"Error computing best name: {e}")

    # Fallback: try to get from central Postgres custodian table (old approach)
    storage = _get_custodian_storage()
    if storage and storage.using_central and not profile.get('name'):
        try:
            central_prefs = storage.postgres.get_all_custodian()
            for pref in central_prefs:
                key = pref.get('key', '')
                value = pref.get('value', '')
                category = pref.get('category', '')

                # Handle identity from central (only if not already set)
                if key == 'identity:name' and value and not profile.get('name'):
                    profile['name'] = value
                    profile['identity']['name'] = {
                        'value': value,
                        'confidence': pref.get('confidence', 0.5),
                        'source': 'central_legacy'
                    }

                # Handle preferences from central
                elif key.startswith('pref:') and category:
                    if category not in profile['preferences']:
                        profile['preferences'][category] = []
                    profile['preferences'][category].append({
                        'preference': value,
                        'frequency': pref.get('frequency', 1),
                        'confidence': pref.get('confidence', 0.5)
                    })
        except Exception as e:
            log(f"Error reading central custodian: {e}")

    try:
        # Fallback: Get identity from old SQLite table (only if not already set)
        if not profile.get('name'):
            rows = db.execute_read(CUSTODIAN_DB, "SELECT key, value, confidence FROM identity ORDER BY confidence DESC")
            for row in rows:
                profile['identity'][row['key']] = {'value': row['value'], 'confidence': row['confidence']}
                if row['key'] == 'name' and not profile.get('name'):
                    profile['name'] = row['value']

        # Get preferences by category
        generic_approval_words = {
            'proceed', 'continue', 'yes', 'ok', 'okay', 'sure', 'thanks', 'thank',
            'good', 'great', 'nice', 'perfect', 'go', 'ahead', 'do', 'it', 'please',
            'looks', 'lgtm', 'ship', 'go ahead', 'do it', 'looks good'
        }

        rows = db.execute_read(CUSTODIAN_DB, """
            SELECT category, preference, frequency, confidence
            FROM preferences
            WHERE confidence >= 0.5
            ORDER BY category, frequency DESC
        """)
        for row in rows:
            if row['preference'].lower().strip() in generic_approval_words:
                continue

            category = row['category']
            if category not in profile['preferences']:
                profile['preferences'][category] = []
            profile['preferences'][category].append({
                'preference': row['preference'],
                'frequency': row['frequency'],
                'confidence': row['confidence']
            })

        # Get rules with confidence decay (uses new enhanced function)
        profile['rules'] = get_rules_with_decay(db, max_rules=30)

        # Get danger zones
        rows = db.execute_read(CUSTODIAN_DB, """
            SELECT path_pattern, issue_description, issue_count, last_issue
            FROM danger_zones
            WHERE issue_count >= 2
            ORDER BY issue_count DESC
            LIMIT 10
        """)
        for row in rows:
            profile['danger_zones'].append({
                'path': row['path_pattern'],
                'description': row['issue_description'],
                'issue_count': row['issue_count'],
                'last_issue': row['last_issue']
            })

        # Get work patterns (non-lifecycle)
        rows = db.execute_read(CUSTODIAN_DB, """
            SELECT pattern_description, frequency, confidence
            FROM work_patterns
            WHERE frequency >= 2 AND pattern_type = 'workflow'
            ORDER BY frequency DESC
            LIMIT 10
        """)
        for row in rows:
            profile['work_patterns'].append({
                'pattern': row['pattern_description'],
                'frequency': row['frequency'],
                'confidence': row['confidence']
            })

        # Get the most common development lifecycle (try local first, then central)
        row = db.execute_read_one(CUSTODIAN_DB, """
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
        if row:
            profile['development_lifecycle'] = {
                'sequence': row['pattern_description'],
                'frequency': row['frequency'],
                'confidence': row['confidence']
            }
        else:
            # Fallback to central Postgres if local has no lifecycle data
            storage = _get_custodian_storage()
            if storage and storage.using_central:
                try:
                    patterns = storage.postgres.get_lifecycle_patterns(min_confidence=0.5)
                    if patterns:
                        # Get highest confidence pattern
                        best = max(patterns, key=lambda p: p.get('confidence', 0) * p.get('occurrences', 1))
                        profile['development_lifecycle'] = {
                            'sequence': best['pattern'],
                            'frequency': best.get('occurrences', 1),
                            'confidence': best['confidence']
                        }
                except Exception as e:
                    log(f"Central lifecycle pattern retrieval failed: {e}")

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

    prefs = profile.get('preferences', {})
    if prefs.get(PREF_TOOLS):
        tools = [p['preference'] for p in prefs[PREF_TOOLS][:3]]
        parts.append(f"Preferred tools: {', '.join(tools)}")

    if prefs.get(PREF_CODING_STYLE):
        styles = [p['preference'] for p in prefs[PREF_CODING_STYLE][:2]]
        parts.append(f"Coding style: {', '.join(styles)}")

    rules = profile.get('rules', {})
    # Show most important rules with proper truncation
    if rules.get('never'):
        never = [format_rule_for_display(r['rule'], 50) for r in rules['never'][:2]]
        parts.append(f"Never: {'; '.join(never)}")

    if rules.get('always'):
        always = [format_rule_for_display(r['rule'], 50) for r in rules['always'][:2]]
        parts.append(f"Always: {'; '.join(always)}")

    if rules.get('require'):
        require = [format_rule_for_display(r['rule'], 50) for r in rules['require'][:2]]
        parts.append(f"Required: {'; '.join(require)}")

    if rules.get('prefer'):
        prefer = [format_rule_for_display(r['rule'], 50) for r in rules['prefer'][:2]]
        parts.append(f"Prefer: {'; '.join(prefer)}")

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
    db = get_db_manager()
    warnings = []

    try:
        rows = db.execute_read(CUSTODIAN_DB, "SELECT path_pattern, issue_description, issue_count FROM danger_zones")

        for path in file_paths:
            path_lower = path.lower()
            for row in rows:
                if row['path_pattern'].lower() in path_lower:
                    warnings.append({
                        'file': path,
                        'pattern': row['path_pattern'],
                        'warning': row['issue_description'],
                        'issue_count': row['issue_count']
                    })
    except Exception as e:
        log(f"Error checking danger zones: {e}")

    return warnings


# =============================================================================
