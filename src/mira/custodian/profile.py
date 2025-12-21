"""
MIRA Custodian - Profile Module

Handles building and retrieving the custodian profile for providing
context to Claude sessions.
"""

import json
import math
from typing import Dict, List, Optional, Tuple

from mira.core import log, get_custodian
from mira.core.database import get_db_manager
from mira.core.constants import DB_CUSTODIAN

from .rules import RULE_TYPES, get_rules_with_decay, format_rule_for_display
from .learning import (
    PREF_CODING_STYLE, PREF_TOOLS, PREF_FRAMEWORKS,
    PREF_WORKFLOW, PREF_COMMUNICATION, PREF_TESTING,
)

# Global storage instance for custodian
_custodian_storage = None


def _get_custodian_storage():
    """Get storage instance for custodian (lazy init)."""
    global _custodian_storage
    if _custodian_storage is None:
        try:
            from mira.storage import get_storage
            _custodian_storage = get_storage()
        except ImportError:
            pass
    return _custodian_storage


def compute_best_name() -> Optional[Tuple[str, float, float, int]]:
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
    db = get_db_manager()

    # Pattern quality weights (higher = more trustworthy)
    pattern_weights = {
        'my_name_is': 1.5,
        'im_introduction': 1.2,
        'call_me': 1.1,
        'signoff': 0.8,
        'unknown': 0.7,
    }

    def _score_rows(rows, from_postgres=False):
        """Score name candidates from either local or central storage."""
        best_name = None
        best_score = -1

        for row in rows:
            if from_postgres:
                name, total_conf, num_sessions, max_conf, patterns = row[:5]
                patterns = patterns or ['unknown']
            else:
                name = row['name']
                total_conf = row['total_conf'] or 0
                num_sessions = row['num_sessions'] or 1
                max_conf = row['max_conf'] or 0
                patterns = (row['patterns'] or 'unknown').split(',')

            if isinstance(patterns, str):
                patterns = patterns.split(',')
            pattern_bonus = max(pattern_weights.get(p.strip() if isinstance(p, str) else p, 0.7) for p in patterns)

            freq_bonus = math.log((num_sessions or 1) + 1)

            score = ((total_conf or 0) * pattern_bonus) + freq_bonus

            if score > best_score:
                best_score = score
                best_name = (name, round(score, 2), max_conf or 0, num_sessions or 1)

        return best_name

    # Try local first (fast)
    try:
        rows = db.execute_read(DB_CUSTODIAN, """
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
        from mira.storage import Storage
        storage = Storage()
        if storage._init_central() and storage._postgres:
            result = storage._postgres.get_best_name()
            if result:
                return (
                    result['name'],
                    round(result.get('score', 0), 2),
                    result.get('confidence', 0),
                    result.get('sessions', 1)
                )
    except Exception as e:
        log(f"Error reading central name candidates: {e}")

    return None


def get_all_name_candidates() -> List[Dict]:
    """Get all name candidates with their details for debugging/display."""
    db = get_db_manager()

    try:
        rows = db.execute_read(DB_CUSTODIAN, """
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


def sync_from_central() -> int:
    """
    Pull critical custodian data from central PostgreSQL to local SQLite.

    Returns:
        Number of name candidates synced
    """
    try:
        from mira.storage import get_storage
        storage = get_storage()
        if not storage.using_central or not storage.postgres:
            return 0

        db = get_db_manager()
        synced = 0

        try:
            candidates = storage.postgres.get_all_name_candidates()
            for candidate in candidates:
                try:
                    from datetime import datetime
                    db.execute_write(
                        DB_CUSTODIAN,
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
        'rules': {rt: [] for rt in RULE_TYPES.keys()},
        'danger_zones': [],
        'work_patterns': [],
        'development_lifecycle': None,
        'summary': '',
    }

    # First, try to compute best name from candidates
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
            profile['identity']['name_candidates'] = get_all_name_candidates()
    except Exception as e:
        log(f"Error computing best name: {e}")

    # Fallback: try to get from central Postgres custodian table
    storage = _get_custodian_storage()
    if storage and storage.using_central and not profile.get('name'):
        try:
            central_prefs = storage.postgres.get_all_custodian()
            for pref in central_prefs:
                key = pref.get('key', '')
                value = pref.get('value', '')
                category = pref.get('category', '')

                if key == 'identity:name' and value and not profile.get('name'):
                    profile['name'] = value
                    profile['identity']['name'] = {
                        'value': value,
                        'confidence': pref.get('confidence', 0.5),
                        'source': 'central_legacy'
                    }

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
        # Fallback: Get identity from old SQLite table
        if not profile.get('name'):
            rows = db.execute_read(DB_CUSTODIAN, "SELECT key, value, confidence FROM identity ORDER BY confidence DESC")
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

        rows = db.execute_read(DB_CUSTODIAN, """
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

        # Get rules with confidence decay
        profile['rules'] = get_rules_with_decay(db, max_rules=30)

        # Get danger zones
        rows = db.execute_read(DB_CUSTODIAN, """
            SELECT path_pattern, issue_description, issue_count, last_issue
            FROM danger_zones
            WHERE issue_count >= 2
            ORDER BY issue_count DESC
            LIMIT 10
        """)
        for row in rows:
            # Clean up description - filter garbage context fragments
            desc = row['issue_description'] or ''
            # Skip descriptions that are code fragments or incomplete
            if any(c in desc for c in ['`', '```', '- ', '  - ', '":', 'line ']):
                desc = f"Had {row['issue_count']} issues in past sessions"
            profile['danger_zones'].append({
                'path': row['path_pattern'],
                'description': desc,
                'issue_count': row['issue_count'],
                'last_issue': row['last_issue']
            })

        # Get work patterns (non-lifecycle)
        rows = db.execute_read(DB_CUSTODIAN, """
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

        # Get the most common development lifecycle
        row = db.execute_read_one(DB_CUSTODIAN, """
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
            # Fallback to central Postgres
            storage = _get_custodian_storage()
            if storage and storage.using_central:
                try:
                    patterns = storage.postgres.get_lifecycle_patterns(min_confidence=0.5)
                    if patterns:
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
    """Check if any of the given file paths match known danger zones."""
    db = get_db_manager()
    warnings = []

    try:
        rows = db.execute_read(DB_CUSTODIAN, "SELECT path_pattern, issue_description, issue_count FROM danger_zones")

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


def get_custodian_stats() -> dict:
    """Get statistics about custodian data (for mira_status)."""
    db = get_db_manager()

    stats = {
        'name': None,
        'preferences': 0,
        'rules': 0,
        'danger_zones': 0,
        'work_patterns': 0,
        'name_candidates': 0,
    }

    try:
        # Get name
        best_name = compute_best_name()
        if best_name:
            stats['name'] = best_name[0]

        # Count preferences
        row = db.execute_read_one(DB_CUSTODIAN, "SELECT COUNT(*) as cnt FROM preferences")
        stats['preferences'] = row['cnt'] if row else 0

        # Count rules
        row = db.execute_read_one(DB_CUSTODIAN, "SELECT COUNT(*) as cnt FROM rules")
        stats['rules'] = row['cnt'] if row else 0

        # Count danger zones
        row = db.execute_read_one(DB_CUSTODIAN, "SELECT COUNT(*) as cnt FROM danger_zones")
        stats['danger_zones'] = row['cnt'] if row else 0

        # Count work patterns
        row = db.execute_read_one(DB_CUSTODIAN, "SELECT COUNT(*) as cnt FROM work_patterns")
        stats['work_patterns'] = row['cnt'] if row else 0

        # Count name candidates
        row = db.execute_read_one(DB_CUSTODIAN, "SELECT COUNT(*) as cnt FROM name_candidates")
        stats['name_candidates'] = row['cnt'] if row else 0

    except Exception as e:
        log(f"Error getting custodian stats: {e}")

    return stats
