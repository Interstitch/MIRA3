"""
MIRA3 Fuzzy Matching Module

Provides typo-tolerant search via edit distance fuzzy matching.

Uses Damerau-Levenshtein distance which handles:
- Insertions: "authentcation" → "authentication"
- Deletions: "authenticaton" → "authentication"
- Substitutions: "authantication" → "authentication"
- Transpositions: "authentiacation" → "authentication"

Vocabulary is built from indexed session keywords and summaries,
ensuring corrections are terms that actually exist in the user's history.
"""

import re
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from .utils import log
from .db_manager import get_db_manager

# Configuration
MIN_TERM_LENGTH = 4  # Don't fuzzy match short terms (too many false positives)
MAX_EDIT_DISTANCE = 2  # Maximum edits for a correction
MIN_FREQUENCY = 1  # Minimum term frequency to consider

# Stopwords to exclude from vocabulary and not correct
FUZZY_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'this', 'that', 'these',
    'those', 'it', 'its', 'and', 'or', 'but', 'if', 'then', 'else',
    'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
    'from', 'with', 'for', 'about', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'under', 'again', 'further',
    'once', 'here', 'there', 'where', 'when', 'why', 'how', 'what',
    'which', 'who', 'whom', 'whose', 'this', 'that', 'these', 'those',
    'your', 'yours', 'yourself', 'yourselves', 'their', 'theirs',
}

# Database name for vocabulary
VOCAB_DB = "local_store.db"

# Cache for vocabulary (refreshed periodically)
_vocabulary_cache: Optional[Dict[str, int]] = None
_vocabulary_set: Optional[Set[str]] = None


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Damerau-Levenshtein distance between two strings.

    Handles insertions, deletions, substitutions, AND transpositions.
    Transpositions are common typos (e.g., "teh" → "the").

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (number of operations needed to transform s1 to s2)
    """
    len1, len2 = len(s1), len(s2)

    # Quick returns for edge cases
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    if s1 == s2:
        return 0

    # Early termination if length difference exceeds max distance
    if abs(len1 - len2) > MAX_EDIT_DISTANCE:
        return MAX_EDIT_DISTANCE + 1

    # Create distance matrix
    # Using dict for sparse representation (memory efficient)
    d = {}

    # Initialize
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    # Track last occurrence of each character
    last_row = {}

    for i in range(len1):
        last_col = -1

        for j in range(len2):
            i1 = last_row.get(s2[j], -1)
            j1 = last_col

            cost = 0 if s1[i] == s2[j] else 1

            if cost == 0:
                last_col = j

            # Standard Levenshtein operations
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,      # deletion
                d[(i, j - 1)] + 1,      # insertion
                d[(i - 1, j - 1)] + cost  # substitution
            )

            # Transposition
            if i1 >= 0 and j1 >= 0:
                d[(i, j)] = min(
                    d[(i, j)],
                    d[(i1 - 1, j1 - 1)] + (i - i1 - 1) + 1 + (j - j1 - 1)
                )

        last_row[s1[i]] = i

    return d[(len1 - 1, len2 - 1)]


def _load_vocabulary() -> Dict[str, int]:
    """
    Load vocabulary from database.

    Returns dict mapping term → frequency.
    """
    global _vocabulary_cache, _vocabulary_set

    if _vocabulary_cache is not None:
        return _vocabulary_cache

    db = get_db_manager()

    try:
        rows = db.execute_read(
            VOCAB_DB,
            "SELECT term, frequency FROM vocabulary WHERE frequency >= ?",
            (MIN_FREQUENCY,)
        )

        _vocabulary_cache = {row['term'].lower(): row['frequency'] for row in rows}
        _vocabulary_set = set(_vocabulary_cache.keys())

        log(f"Loaded vocabulary: {len(_vocabulary_cache)} terms")
        return _vocabulary_cache

    except Exception as e:
        # Table might not exist yet
        log(f"Could not load vocabulary: {e}")
        _vocabulary_cache = {}
        _vocabulary_set = set()
        return _vocabulary_cache


def get_vocabulary_size() -> int:
    """Get number of terms in vocabulary."""
    vocab = _load_vocabulary()
    return len(vocab)


def is_in_vocabulary(term: str) -> bool:
    """Check if a term exists in vocabulary."""
    _load_vocabulary()
    return term.lower() in _vocabulary_set


def find_closest_match(term: str, max_distance: int = MAX_EDIT_DISTANCE) -> Optional[Tuple[str, int]]:
    """
    Find the closest matching term in vocabulary.

    Args:
        term: The term to find matches for
        max_distance: Maximum edit distance to consider

    Returns:
        Tuple of (best_match, distance) or None if no match within threshold
    """
    term_lower = term.lower()

    # Don't correct short terms or stopwords
    if len(term_lower) < MIN_TERM_LENGTH:
        return None
    if term_lower in FUZZY_STOPWORDS:
        return None

    # Don't correct numbers or terms with special chars
    if re.match(r'^[\d\W]+$', term_lower):
        return None

    vocab = _load_vocabulary()

    if not vocab:
        return None

    # If term is already in vocabulary, no correction needed
    if term_lower in vocab:
        return None

    best_match = None
    best_distance = max_distance + 1
    best_frequency = 0

    for vocab_term, frequency in vocab.items():
        # Skip if length difference too large (quick filter)
        if abs(len(vocab_term) - len(term_lower)) > max_distance:
            continue

        distance = damerau_levenshtein_distance(term_lower, vocab_term)

        if distance <= max_distance:
            # Prefer lower distance, then higher frequency
            if distance < best_distance or (distance == best_distance and frequency > best_frequency):
                best_match = vocab_term
                best_distance = distance
                best_frequency = frequency

    if best_match:
        return (best_match, best_distance)

    return None


def expand_query_with_corrections(query: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Expand a search query with typo corrections.

    Tokenizes the query, checks each term against vocabulary,
    and replaces misspelled terms with corrections.

    Args:
        query: Original search query

    Returns:
        Tuple of (corrected_query, list of corrections)
        Each correction is {"original": "...", "corrected": "...", "distance": N}
    """
    if not query or not query.strip():
        return query, []

    # Tokenize query (simple word splitting)
    tokens = re.findall(r'\b\w+\b', query.lower())

    if not tokens:
        return query, []

    corrections = []
    corrected_tokens = []

    for token in tokens:
        # Try to find a correction
        match = find_closest_match(token)

        if match:
            corrected_term, distance = match
            corrections.append({
                "original": token,
                "corrected": corrected_term,
                "distance": distance
            })
            corrected_tokens.append(corrected_term)
        else:
            corrected_tokens.append(token)

    corrected_query = ' '.join(corrected_tokens)

    return corrected_query, corrections


def add_terms_to_vocabulary(terms: List[str], source: str = "unknown"):
    """
    Add terms to the vocabulary.

    Args:
        terms: List of terms to add
        source: Source of terms (e.g., "keyword", "summary", "fact")
    """
    if not terms:
        return

    db = get_db_manager()

    # Filter and normalize terms
    valid_terms = []
    for term in terms:
        if not term:
            continue
        term_lower = term.lower().strip()
        # Skip short terms, stopwords, and non-alpha terms
        if len(term_lower) < MIN_TERM_LENGTH:
            continue
        if term_lower in FUZZY_STOPWORDS:
            continue
        if not re.match(r'^[a-z]', term_lower):
            continue
        valid_terms.append(term_lower)

    if not valid_terms:
        return

    # Batch upsert
    for term in valid_terms:
        try:
            db.execute_write(
                VOCAB_DB,
                """INSERT INTO vocabulary (term, frequency, source)
                   VALUES (?, 1, ?)
                   ON CONFLICT(term) DO UPDATE SET
                       frequency = frequency + 1,
                       source = CASE WHEN source NOT LIKE '%' || excluded.source || '%'
                                     THEN source || ',' || excluded.source
                                     ELSE source END""",
                (term, source)
            )
        except Exception as e:
            # Table might not exist yet - will be created by migration
            log(f"Could not add term to vocabulary: {e}")
            break

    # Invalidate cache
    global _vocabulary_cache, _vocabulary_set
    _vocabulary_cache = None
    _vocabulary_set = None


def extract_terms_from_text(text: str) -> List[str]:
    """
    Extract significant terms from text for vocabulary.

    Filters out stopwords, short terms, and non-alphabetic terms.

    Args:
        text: Text to extract terms from

    Returns:
        List of significant terms
    """
    if not text:
        return []

    # Tokenize
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', text.lower())

    # Filter
    terms = []
    for token in tokens:
        if len(token) < MIN_TERM_LENGTH:
            continue
        if token in FUZZY_STOPWORDS:
            continue
        terms.append(token)

    return list(set(terms))  # Dedupe


# Schema for vocabulary table (added via migration)
VOCABULARY_SCHEMA = """
CREATE TABLE IF NOT EXISTS vocabulary (
    term TEXT PRIMARY KEY,
    frequency INTEGER DEFAULT 1,
    source TEXT DEFAULT 'unknown',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_vocabulary_frequency ON vocabulary(frequency DESC);
"""
