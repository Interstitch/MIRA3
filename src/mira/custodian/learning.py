"""
MIRA Custodian - Learning Module

Core learning logic for extracting user preferences, rules, and patterns
from conversation history.
"""

import json
import re
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple

from mira.core import log
from mira.core.database import get_db_manager
from mira.core.constants import DB_CUSTODIAN
from mira.storage.migrations import CUSTODIAN_SCHEMA

from .rules import (
    RULE_TYPES, RULE_PATTERNS, CONDITIONAL_RULE_PATTERNS,
    RULE_REVOCATION_PATTERNS, RULE_FILTER_WORDS, RULE_FILLER_WORDS,
    normalize_rule_text, is_rule_false_positive, find_similar_rule,
    extract_scope_from_content,
)
from .prerequisites import (
    PREREQ_STATEMENT_PATTERNS, PREREQ_COMMAND_PATTERNS,
    PREREQ_REASON_PATTERNS, PREREQ_CHECK_TEMPLATES, PREREQ_KEYWORDS,
)

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
    db.init_schema(DB_CUSTODIAN, CUSTODIAN_SCHEMA)
    log("Custodian database initialized")


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
    name_patterns = [
        # Highest confidence: explicit "my name is"
        (r"(?:^|\s)my name is\s+([A-Z][a-z]{2,15})(?:\s|[.,!?]|$)", 0.95),
        # High confidence: "I'm [Name]" at start of message or after greeting
        (r"(?:^|[.!?]\s*|,\s*)(?:hi,?\s+)?i'?m\s+([A-Z][a-z]{2,15})(?:\s|[.,!?]|$)", 0.9),
        # Medium confidence: "call me [Name]"
        (r"(?:^|\s)(?:you can |please )?call me\s+([A-Z][a-z]{2,15})(?:\s|[.,!?]|$)", 0.85),
        # Medium confidence: "I am [Name]" (explicit statement)
        (r"(?:^|\s)i am\s+([A-Z][a-z]{2,15})(?:\s|[.,!?]|$)", 0.85),
    ]

    # Comprehensive blocklist
    name_blocklist = {
        # Greetings and conversation
        'claude', 'hello', 'please', 'thanks', 'help', 'just', 'the', 'this', 'that',
        'if', 'when', 'what', 'how', 'where', 'why', 'yes', 'no', 'okay', 'sure',
        'now', 'then', 'here', 'there', 'which', 'who', 'whom', 'hey', 'hi',

        # Common sentence starters and fillers
        'well', 'also', 'actually', 'basically', 'honestly', 'really', 'maybe',
        'perhaps', 'probably', 'certainly', 'definitely', 'absolutely', 'anyway',

        # Tech products and services
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

        # Action words
        'pause', 'think', 'thinking', 'check', 'checking', 'look', 'looking',
        'see', 'seeing', 'try', 'trying', 'start', 'starting', 'stop', 'stopping',
        'wait', 'waiting', 'continue', 'continuing', 'proceed', 'proceeding',
        'begin', 'beginning', 'finish', 'finishing', 'review', 'reviewing',
        'working', 'going', 'doing', 'done', 'ready', 'reading',
        'writing', 'running', 'testing', 'building', 'deploying', 'updating',
        'fixing', 'debugging', 'investigating', 'analyzing', 'processing',
        'wondering', 'curious', 'confused', 'stuck', 'lost', 'back',
        'sorry', 'glad', 'happy', 'excited', 'afraid', 'worried', 'concerned',
    }

    pattern_types = {
        0: 'my_name_is',
        1: 'im_introduction',
        2: 'call_me',
        3: 'i_am',
    }

    for msg in user_messages:
        content = msg.get('content', '')
        if not content or len(content) < 10:
            continue

        for idx, (pattern, confidence) in enumerate(name_patterns):
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                name = match.group(1)
                if (name.lower() not in name_blocklist and
                    3 <= len(name) <= 15 and
                    name[0].isupper() and
                    name[1:].islower() and
                    not any(c.isdigit() for c in name)):

                    start = max(0, match.start() - 20)
                    end = min(len(content), match.end() + 20)
                    context = content[start:end]

                    pattern_type = pattern_types.get(idx, 'unknown')
                    _store_name(db, name, confidence, session_id, now, pattern_type, context)
                    learned += 1
                    break

        # Check sign-off pattern at end of message only
        if not learned:
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


def _store_name(db, name: str, confidence: float, session_id: str, now: str,
                pattern_type: str = 'unknown', context: str = ''):
    """Store a name candidate for later scoring."""
    try:
        db.execute_write(
            DB_CUSTODIAN,
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
        log(f"name_candidates insert failed (migration pending?): {e}")
        db.execute_write(
            DB_CUSTODIAN,
            """INSERT OR REPLACE INTO identity (key, value, confidence, source_session, learned_at)
               VALUES (?, ?, ?, ?, ?)""",
            ('name', name, confidence, session_id, now)
        )


def _learn_preferences(db, user_messages: list, session_id: str, now: str) -> int:
    """Learn preferences from user statements."""
    learned = 0

    game_content_words = {
        'planet', 'sector', 'ship', 'trade', 'port', 'warp', 'credits', 'player',
        'game', 'level', 'score', 'item', 'quest', 'mission', 'world', 'server',
        'attack', 'defend', 'enemy', 'spawn', 'health', 'damage', 'inventory',
        'character', 'npc', 'boss', 'dungeon', 'loot', 'xp', 'mana', 'spell',
        'population', 'resource', 'colony', 'fleet', 'station', 'galaxy',
        'allocated', 'landed', 'docked', 'warped', 'jumped',
    }

    pref_patterns = [
        # Coding style
        (PREF_CODING_STYLE, r"(?:i always prefer|i prefer to use|my coding style is)\s+([^.!?\n]{5,60})", 0.7),
        (PREF_CODING_STYLE, r"(?:i always|we always)\s+(?:use|write)\s+(type ?script|strict mode|eslint|prettier|black|ruff)", 0.8),
        (PREF_CODING_STYLE, r"(?:don't|never|avoid)\s+(?:use|write)\s+(var|any type|console\.log|print statement)", 0.7),

        # Tools
        (PREF_TOOLS, r"(?:i use|we use|using|my .* is)\s+(npm|pnpm|yarn|bun|pip|poetry|cargo|go mod)", 0.8),
        (PREF_TOOLS, r"(?:run|use|prefer)\s+(vitest|jest|pytest|mocha|cargo test)", 0.8),
        (PREF_TOOLS, r"(?:i use|we use)\s+(vscode|vim|neovim|emacs|intellij|webstorm)", 0.8),

        # Frameworks
        (PREF_FRAMEWORKS, r"(?:using|we use|i use|prefer)\s+(react|vue|svelte|angular|next|nuxt|express|fastapi|django|flask)", 0.7),

        # Testing
        (PREF_TESTING, r"(?:i (?:always |usually )?(?:prefer|like|want) to )(write tests? (?:before|after|first))", 0.7),
        (PREF_TESTING, r"(?:always )(run tests? before (?:commit|push|deploy))", 0.8),
        (PREF_TESTING, r"(?:i believe in|i practice|we practice)\s+(tdd|test.driven|bdd)", 0.8),

        # Communication
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

        if any(word in content for word in ['planet', 'sector', 'warp', 'credits', 'fleet', 'colony', 'population', 'allocated']):
            continue

        for category, pattern, confidence in pref_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                pref_text = match.strip() if isinstance(match, str) else match
                if len(pref_text) < 3 or len(pref_text) > 50:
                    continue

                pref_lower = pref_text.lower()
                if any(word in pref_lower for word in game_content_words):
                    continue

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

                learned += db.execute_write_func(DB_CUSTODIAN, upsert_preference)

    return learned


def _learn_rules(db, messages: list, session_id: str, now: str) -> int:
    """Learn explicit rules from USER messages."""
    learned = 0
    user_messages = [m for m in messages if m.get('role') == 'user']

    for msg in user_messages:
        content = msg.get('content', '')
        content_lower = content.lower()

        if any(word in content_lower for word in ['planet', 'sector', 'warp', 'credits', 'fleet', 'colony']):
            continue

        _check_rule_revocations(db, content, now)

        scope = extract_scope_from_content(content)

        for rule_type, pattern, base_confidence in RULE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    rule_text = f"{match[0]} over {match[1]}" if len(match) == 2 else match[0]
                else:
                    rule_text = match
                rule_text = rule_text.strip()

                if is_rule_false_positive(content, rule_text, rule_type):
                    continue

                learned += _upsert_rule_with_dedup(
                    db, rule_type, rule_text, base_confidence, scope, session_id, now
                )

        for scope_pattern, rule_pattern, base_confidence in CONDITIONAL_RULE_PATTERNS:
            combined = scope_pattern + r",?\s+" + rule_pattern
            matches = re.findall(combined, content, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    cond_scope = match[0].strip()
                    rule_text = match[1].strip()

                    if is_rule_false_positive(content, rule_text, 'conditional'):
                        continue

                    rule_type = 'always' if 'always' in rule_pattern else 'never' if 'never' in rule_pattern else 'require'

                    learned += _upsert_rule_with_dedup(
                        db, rule_type, rule_text, base_confidence, cond_scope, session_id, now
                    )

    return learned


def _check_rule_revocations(db, content: str, now: str) -> int:
    """Check for rule revocation patterns and mark matching rules as revoked."""
    revoked = 0
    content_lower = content.lower()

    for pattern in RULE_REVOCATION_PATTERNS:
        matches = re.findall(pattern, content_lower)
        for match in matches:
            revocation_text = match.strip() if isinstance(match, str) else match[0].strip()
            normalized_revocation = normalize_rule_text(revocation_text)

            if not normalized_revocation or len(normalized_revocation) < 5:
                continue

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

                    if not existing_words:
                        continue
                    overlap = len(revocation_words & existing_words) / len(existing_words)
                    if overlap >= 0.5:
                        cursor.execute("""
                            UPDATE rules SET revoked = 1, revoked_at = ? WHERE id = ?
                        """, (now, row[0]))
                        revoked_count += 1

                return revoked_count

            revoked += db.execute_write_func(DB_CUSTODIAN, revoke_matching)

    return revoked


def _upsert_rule_with_dedup(
    db, rule_type: str, rule_text: str, base_confidence: float,
    scope: Optional[str], session_id: str, now: str
) -> int:
    """Upsert a rule with semantic deduplication."""
    rule_text = rule_text[:200]
    normalized = normalize_rule_text(rule_text)

    similar_id, similarity = find_similar_rule(db, rule_text, rule_type)

    def upsert(cursor):
        if similar_id:
            cursor.execute("""
                SELECT frequency, source_sessions, confidence FROM rules WHERE id = ?
            """, (similar_id,))
            row = cursor.fetchone()
            if row:
                freq, sources, old_confidence = row
                sources_list = json.loads(sources) if sources else []
                if session_id not in sources_list:
                    sources_list.append(session_id)

                new_confidence = min(1.0, max(old_confidence, base_confidence) + 0.05)

                cursor.execute("""
                    UPDATE rules
                    SET frequency = ?, last_seen = ?, source_sessions = ?, confidence = ?
                    WHERE id = ?
                """, (freq + 1, now, json.dumps(sources_list[-10:]), new_confidence, similar_id))
                return 0
        else:
            cursor.execute("""
                INSERT INTO rules
                (rule_type, rule_text, normalized_text, scope, confidence, first_seen, last_seen, source_sessions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (rule_type, rule_text, normalized, scope, base_confidence, now, now, json.dumps([session_id])))
            return 1

    return db.execute_write_func(DB_CUSTODIAN, upsert)


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

                learned += db.execute_write_func(DB_CUSTODIAN, upsert_danger_zone)

    return learned


def _detect_lifecycle_sequence(messages: list) -> Tuple[Optional[str], float]:
    """Analyze conversation flow to detect the user's development lifecycle."""
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

    edit_mentions = sum(1 for m in messages if 'edit' in m.get('content', '').lower())
    if edit_mentions > 3:
        patterns_found.append(('workflow', 'Iterative development with frequent edits', 0.5))

    first_few = ' '.join(m.get('content', '')[:200].lower() for m in user_messages[:3])
    if 'test' in first_few:
        patterns_found.append(('workflow', 'Mentions testing early in conversation', 0.5))

    if any(word in first_few for word in ['plan', 'design', 'architect', 'approach']):
        patterns_found.append(('workflow', 'Prefers planning before implementation', 0.5))

    first_msg = user_messages[0].get('content', '').lower()[:100]
    if any(word in first_msg for word in ['fix', 'change', 'update', 'add', 'remove', 'implement']):
        patterns_found.append(('workflow', 'Starts with direct action requests', 0.5))

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

        learned += db.execute_write_func(DB_CUSTODIAN, upsert_pattern)

    return learned


# Valid environment names for prerequisites
VALID_ENVIRONMENTS = {
    'codespaces', 'github codespaces', 'gitpod', 'wsl', 'wsl2',
    'docker', 'container', 'podman', 'kubernetes', 'k8s',
    'linux', 'macos', 'mac', 'windows', 'ubuntu', 'debian', 'arch', 'fedora',
    'ssh', 'remote', 'local', 'server', 'cloud',
    'aws', 'gcp', 'azure', 'railway', 'vercel', 'fly', 'render',
    'ci', 'github actions', 'jenkins', 'all', 'always', 'everywhere',
}

# Keywords that should appear in valid prerequisite actions
PREREQ_ACTION_KEYWORDS = {
    'start', 'run', 'launch', 'execute', 'install', 'setup', 'configure',
    'enable', 'activate', 'init', 'connect', 'login', 'authenticate',
    'source', 'export', 'set', 'load', 'mount', 'open', 'daemon',
}

# Garbage indicators - if action contains these, reject it
PREREQ_GARBAGE_INDICATORS = [
    '|', '":', '```', '**', '##', '\n\n', 'cursor', 'def ',
    'function', 'import ', 'from ', 'class ', 'return ',
    'the following', 'as follows', 'here is', 'let me',
    '.py:', '.ts:', '.js:', 'line ', 'error:', 'warning:',
]


def _is_valid_prerequisite(env: str, action: str) -> bool:
    """
    Validate that a prerequisite looks legitimate, not garbage.

    Returns True if the prerequisite should be stored, False to reject.
    """
    env_lower = env.lower().strip()
    action_lower = action.lower().strip()

    # Environment must be a known environment or very short (likely valid)
    env_valid = (
        env_lower in VALID_ENVIRONMENTS or
        any(valid in env_lower for valid in VALID_ENVIRONMENTS) or
        len(env_lower) <= 15  # Short envs like hostnames are OK
    )

    # Reject if environment is clearly garbage
    if not env_valid:
        # Check if it looks like a sentence fragment
        if ' ' in env_lower and len(env_lower) > 20:
            return False
        # Check for code indicators
        if any(ind in env_lower for ind in ['(', ')', '{', '}', '=', ':']):
            return False

    # Action must have at least one action keyword
    has_action_keyword = any(kw in action_lower for kw in PREREQ_ACTION_KEYWORDS)

    # Action must NOT contain garbage indicators
    has_garbage = any(ind in action_lower for ind in PREREQ_GARBAGE_INDICATORS)

    if has_garbage:
        return False

    # Action should be mostly alphabetic (not code/data)
    alpha_chars = sum(1 for c in action if c.isalpha() or c.isspace())
    alpha_ratio = alpha_chars / max(1, len(action))
    if alpha_ratio < 0.7:
        return False

    # Action should start with a verb or common prerequisite phrase
    action_starts_valid = any(
        action_lower.startswith(prefix) for prefix in [
            'start', 'run', 'launch', 'execute', 'install', 'setup',
            'configure', 'enable', 'make sure', 'ensure', 'first',
            'before', 'need to', 'have to', 'must', 'should',
        ]
    )

    # Accept if has action keyword OR starts with valid prefix
    return has_action_keyword or action_starts_valid


def _learn_prerequisites(db, messages: list, session_id: str, now: str) -> int:
    """Learn environment-specific prerequisites from conversation."""
    learned = 0

    for msg in messages:
        content = msg.get('content', '')
        if isinstance(content, list):
            content = ' '.join(
                block.get('text', '')
                for block in content
                if isinstance(block, dict) and block.get('type') == 'text'
            )

        if not content or len(content) < 20:
            continue

        role = msg.get('role', '')
        content_lower = content.lower()

        if not any(kw in content_lower for kw in PREREQ_KEYWORDS):
            continue

        for pattern in PREREQ_STATEMENT_PATTERNS:
            try:
                matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
            except re.error:
                continue

            for match in matches:
                groups = match.groups()
                if len(groups) < 2:
                    continue

                env = None
                action = None

                g1, g2 = groups[0], groups[1] if len(groups) > 1 else None
                if g1 and g2:
                    if len(g1.strip()) < len(g2.strip()):
                        env, action = g1.strip(), g2.strip()
                    else:
                        action, env = g1.strip(), g2.strip()
                elif g1:
                    action = g1.strip()

                if not action or len(action) < 5 or len(action) > 200:
                    continue
                if env and (len(env) < 2 or len(env) > 40):
                    continue
                if not env:
                    env = 'all'

                env = env.lower().strip()

                # Validate prerequisite quality - reject garbage
                if not _is_valid_prerequisite(env, action):
                    continue

                command = None
                for cmd_pattern in PREREQ_COMMAND_PATTERNS:
                    try:
                        cmd_match = re.search(cmd_pattern, content, re.IGNORECASE | re.DOTALL)
                        if cmd_match:
                            cmd = cmd_match.group(1).strip()
                            if cmd and len(cmd) > 3 and len(cmd) < 500:
                                command = cmd
                                break
                    except re.error:
                        continue

                reason = None
                for reason_pattern in PREREQ_REASON_PATTERNS:
                    try:
                        reason_match = re.search(reason_pattern, content, re.IGNORECASE)
                        if reason_match:
                            reason = reason_match.group(1).strip()
                            break
                    except re.error:
                        continue

                check_command = None
                action_lower = action.lower()
                cmd_lower = (command or '').lower()

                for service, check in PREREQ_CHECK_TEMPLATES.items():
                    if service in action_lower or service in cmd_lower:
                        check_command = check
                        break

                confidence = 0.5
                if role == 'user':
                    confidence += 0.15
                if command:
                    confidence += 0.10
                if check_command:
                    confidence += 0.05
                if reason:
                    confidence += 0.05
                confidence = min(1.0, confidence)

                def upsert_prereq(cursor, e=env, a=action, c=command, cc=check_command,
                                  r=reason, conf=confidence, sid=session_id, n=now):
                    cursor.execute("""
                        SELECT id, frequency, confidence, command, check_command, reason
                        FROM prerequisites
                        WHERE environment = ? AND action = ?
                    """, (e, a))

                    row = cursor.fetchone()
                    if row:
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
                        return 0
                    else:
                        cursor.execute("""
                            INSERT INTO prerequisites
                            (environment, action, command, check_command, reason,
                             confidence, source_session, learned_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (e, a, c, cc, r, conf, sid, n))
                        return 1

                try:
                    result = db.execute_write_func(DB_CUSTODIAN, upsert_prereq)
                    learned += result
                    if result > 0:
                        log(f"Learned prerequisite: '{action}' for environment '{env}'")
                except Exception as e:
                    log(f"Error storing prerequisite: {e}")

    return learned
