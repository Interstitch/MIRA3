"""
MIRA3 Rules Module

Handles rule learning patterns, normalization, deduplication, and display
for user-stated always/never/prefer/avoid rules.
"""

import re

from mira.core import log
from mira.core.database import get_db_manager
from mira.core.constants import DB_CUSTODIAN

CUSTODIAN_DB = DB_CUSTODIAN

# Supported rule types with display names
RULE_TYPES = {
    'never': 'Never',
    'always': 'Always',
    'avoid': 'Avoid',
    'require': 'Required',
    'prefer': 'Prefer',
    'prohibit': 'Don\'t',
    'style': 'Style',
}

# Expanded rule patterns - ordered by confidence (highest first)
# Format: (rule_type, pattern, base_confidence)
RULE_PATTERNS = [
    # Tier 1: Explicit first-person constraints (highest confidence)
    ('never', r"(?:i never|we never|i don't ever|please never|don't ever|never ever)\s+([^.!?\n]{10,100})", 0.95),
    ('always', r"(?:i always|we always|always make sure|please always|make sure to always)\s+([^.!?\n]{10,100})", 0.95),

    # Tier 2: Strong directives
    ('require', r"(?:make sure (?:to |you )|ensure (?:that )?(?:you )?|you must|you need to|you should always)\s+([^.!?\n]{10,100})", 0.85),
    ('prohibit', r"(?:do not|don't|never|stop|quit)\s+([^.!?\n]{10,80})", 0.80),
    ('prefer', r"(?:i (?:really )?prefer|i'd (?:rather|prefer)|please use|let's use)\s+([^.!?\n]{10,100})", 0.85),

    # Tier 3: Comparison preferences
    ('prefer', r"(?:use|prefer)\s+([a-zA-Z0-9_-]+)\s+(?:instead of|over|rather than|not)\s+([a-zA-Z0-9_-]+)", 0.80),

    # Tier 4: Style preferences
    ('style', r"(?:use|format (?:with|using)|style (?:with|using)|apply)\s+(tabs|spaces|2 spaces|4 spaces|black|prettier|eslint|ruff|gofmt|rustfmt)", 0.85),
    ('style', r"(?:i use|we use|stick to|follow)\s+(pep ?8|airbnb|google) ?(?:style)?", 0.80),

    # Tier 5: Existing avoid pattern
    ('avoid', r"(?:i (?:try to )?avoid|please avoid|i don't (?:really )?like(?: using)?)\s+([^.!?\n]{5,80})", 0.75),

    # Tier 6: Softer preferences (lower confidence)
    ('prefer', r"(?:i (?:usually|typically|normally|generally))\s+([^.!?\n]{10,80})", 0.65),
]

# Conditional rule patterns - capture scope and rule separately
# Format: (scope_pattern, rule_pattern, base_confidence)
CONDITIONAL_RULE_PATTERNS = [
    # "When X, always/never Y"
    (r"when\s+(?:working (?:on|with|in)\s+)?([^,]{5,40})", r"(?:always|never)\s+([^.!?\n]{10,80})", 0.85),
    # "In X, make sure to Y"
    (r"(?:in|on|for)\s+([a-zA-Z][a-zA-Z0-9_\s-]{2,30})", r"(?:make sure|ensure|always)\s+(?:to\s+)?([^.!?\n]{10,80})", 0.80),
    # "For X files, use Y"
    (r"for\s+([a-zA-Z]+)\s+(?:files?|code)", r"(?:use|prefer|apply)\s+([^.!?\n]{5,60})", 0.80),
    # "With X, Y"
    (r"(?:when using|with)\s+([a-zA-Z][a-zA-Z0-9_-]{2,20})", r"(?:always|never|use|prefer)\s+([^.!?\n]{10,80})", 0.75),
]

# Rule revocation patterns - detect when user is retracting a rule
RULE_REVOCATION_PATTERNS = [
    r"(?:actually,? )?(?:you can|it's ok to|it's fine to|feel free to)\s+([^.!?\n]{10,80})\s+now",
    r"(?:i |we )?(?:no longer|don't|stopped)\s+(?:need to|have to|want to)\s+([^.!?\n]{10,80})",
    r"(?:ignore|forget|disregard)\s+(?:what i said about|my rule about|the rule about)\s+([^.!?\n]{10,80})",
    r"(?:actually,? )?(?:scratch|cancel|remove)\s+(?:that|the)\s+(?:rule|constraint)\s+(?:about\s+)?([^.!?\n]{10,60})",
    r"(?:i changed my mind about|i've changed my mind about)\s+([^.!?\n]{10,80})",
]

# Words that indicate non-rule content (filter out)
RULE_FILTER_WORDS = {
    # Game content (from original)
    'planet', 'sector', 'ship', 'trade', 'port', 'warp', 'credits', 'player',
    'game', 'level', 'score', 'item', 'quest', 'mission', 'world', 'server',
    'attack', 'defend', 'enemy', 'spawn', 'health', 'damage', 'inventory',
    'character', 'npc', 'boss', 'dungeon', 'loot', 'xp', 'mana', 'spell',
    'population', 'resource', 'colony', 'fleet', 'station', 'galaxy',
    'allocated', 'landed', 'docked', 'warped', 'jumped',
    # Additional non-rule content
    'user says', 'user wants', 'customer', 'client says',
}

# Common filler words to remove during normalization
RULE_FILLER_WORDS = {'the', 'a', 'an', 'to', 'for', 'with', 'in', 'on', 'at', 'by', 'my', 'our', 'your'}


def normalize_rule_text(rule: str) -> str:
    """
    Normalize rule text for comparison and deduplication.

    Removes common filler words and normalizes whitespace to allow
    matching semantically similar rules like:
    - "never commit to main" ≈ "never commit to the main branch"
    """
    rule = rule.lower().strip()
    # Remove punctuation
    rule = re.sub(r'[^\w\s]', ' ', rule)
    # Split and filter
    words = rule.split()
    words = [w for w in words if w not in RULE_FILLER_WORDS and len(w) > 1]
    return ' '.join(words)


def is_rule_false_positive(content: str, rule_text: str, rule_type: str) -> bool:
    """
    Check if an extracted rule is likely a false positive.

    Returns True if the rule should be rejected.
    """
    content_lower = content.lower()
    rule_lower = rule_text.lower()

    # Length checks
    if len(rule_text.strip()) < 5 or len(rule_text.strip()) > 150:
        return True

    # Skip if the message is a question (asking, not stating)
    if content.strip().endswith('?'):
        return True

    # Skip if discussing someone else's rules/preferences
    third_person = ['they always', 'he always', 'she always', 'the docs say',
                    'documentation says', 'tutorial says', 'article says']
    if any(p in content_lower for p in third_person):
        return True

    # Skip if quoting/referencing external content
    external_refs = ['according to', 'the article says', 'documentation states',
                     'the guide says', 'best practice says', 'convention says']
    if any(p in content_lower for p in external_refs):
        return True

    # Skip if it's hypothetical
    hypothetical = ['would always', 'could always', 'might always', 'if i were',
                    'would never', 'could never', 'might never', 'hypothetically']
    if any(p in content_lower for p in hypothetical):
        return True

    # Skip if it's in a large code block context (discussing code, not stating rules)
    if content.count('```') >= 4:
        return True

    # Skip if contains filter words
    if any(word in rule_lower for word in RULE_FILTER_WORDS):
        return True

    # Skip if it's clearly code/syntax
    code_indicators = ['`', '->', '==', '!=', '>=', '<=', '()', '{}', '[]', '&&', '||']
    if any(ind in rule_text for ind in code_indicators):
        return True

    # Skip markdown formatting
    if '**' in rule_text or '```' in rule_text or '##' in rule_text:
        return True

    # Skip list items
    if rule_text.strip().startswith('-') or rule_text.strip().startswith('*'):
        return True
    if len(rule_text) > 1 and rule_text[0].isdigit() and rule_text[1] in '.):':
        return True

    # Require mostly alphabetic content
    alpha_ratio = sum(1 for c in rule_text if c.isalpha() or c.isspace()) / max(1, len(rule_text))
    if alpha_ratio < 0.75:
        return True

    # Skip generic phrases that aren't real rules
    skip_phrases = ['the file', 'the code', 'this function', 'that method',
                   'in sampling', 'in the', 'to the', 'from the', 'of the',
                   'such as', 'for example', 'like this']
    if any(p in rule_lower for p in skip_phrases):
        return True

    return False


def find_similar_rule(db, rule_text: str, rule_type: str, threshold: float = 0.7) -> tuple:
    """
    Find existing rule that's semantically similar using word overlap.

    Returns (rule_id, similarity_score) or (None, 0) if no match.
    """
    normalized_new = normalize_rule_text(rule_text)
    new_words = set(normalized_new.split())

    if not new_words:
        return None, 0

    rows = db.execute_read(CUSTODIAN_DB, """
        SELECT id, rule_text, normalized_text FROM rules
        WHERE rule_type = ? AND revoked = 0
    """, (rule_type,))

    best_match = None
    best_score = 0

    for row in rows:
        # Use stored normalized text if available, else compute
        if row['normalized_text']:
            existing_words = set(row['normalized_text'].split())
        else:
            existing_words = set(normalize_rule_text(row['rule_text']).split())

        if not existing_words:
            continue

        # Jaccard similarity
        intersection = len(new_words & existing_words)
        union = len(new_words | existing_words)
        similarity = intersection / union if union > 0 else 0

        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = row['id']

    return best_match, best_score


def get_rules_with_decay(db, max_rules: int = 30) -> dict:
    """
    Get rules with confidence adjusted for recency.

    Older rules that haven't been reinforced decay in confidence.
    """
    rows = db.execute_read(CUSTODIAN_DB, """
        SELECT rule_type, rule_text, frequency, confidence, first_seen, last_seen,
               julianday('now') - julianday(last_seen) as days_old
        FROM rules
        WHERE frequency >= 1 AND revoked = 0
        ORDER BY frequency DESC, confidence DESC
        LIMIT ?
    """, (max_rules,))

    rules = {rt: [] for rt in RULE_TYPES.keys()}

    for row in rows:
        days_old = row['days_old'] or 0
        base_confidence = row['confidence'] or 0.8

        # Decay: 100% for <7 days, 85% for 7-30, 70% for 30-90, 50% for 90+
        if days_old < 7:
            decay = 1.0
        elif days_old < 30:
            decay = 0.85
        elif days_old < 90:
            decay = 0.70
        else:
            decay = 0.50

        # Frequency boost: more mentions = more confident
        freq_boost = min(1.2, 1.0 + (row['frequency'] - 1) * 0.05)

        effective_confidence = min(1.0, base_confidence * decay * freq_boost)

        rule_type = row['rule_type']
        if rule_type in rules:
            rules[rule_type].append({
                'rule': row['rule_text'],
                'frequency': row['frequency'],
                'confidence': round(effective_confidence, 2),
                'days_old': int(days_old),
                'base_confidence': base_confidence,
            })

    return rules


def format_rule_for_display(rule: str, max_len: int = 60) -> str:
    """
    Format rule for display with proper truncation at word boundary.
    """
    rule = rule.strip()
    if len(rule) <= max_len:
        return rule

    # Truncate at word boundary
    truncated = rule[:max_len]
    last_space = truncated.rfind(' ')
    if last_space > max_len * 0.6:  # Keep at least 60% of content
        truncated = truncated[:last_space]

    return truncated.rstrip('.,;:') + '...'


def extract_scope_from_content(content: str) -> str:
    """
    Extract scope/context from content if present.

    Looks for patterns like "in this project", "for Python", "when testing".
    """
    scope_patterns = [
        r"(?:in|for|on)\s+(?:this\s+)?([a-zA-Z][a-zA-Z0-9_-]+(?:\s+(?:project|repo|codebase|files?))?)",
        r"(?:when|while)\s+(testing|debugging|deploying|reviewing|developing)",
        r"for\s+([a-zA-Z]+)\s+(?:code|files?)",
    ]

    content_lower = content.lower()
    for pattern in scope_patterns:
        match = re.search(pattern, content_lower)
        if match:
            scope = match.group(1).strip()
            # Filter out generic words
            if scope not in ('this', 'that', 'it', 'the', 'a', 'an'):
                return scope

    return None
