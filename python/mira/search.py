"""
MIRA3 Search Module

Handles semantic search, archive enrichment, and fulltext fallback.

Three-Tier Search Architecture:
- Tier 1: Remote semantic (Qdrant + embedding-service) - cross-machine, best quality
- Tier 2: Local semantic (sqlite-vec + fastembed) - offline, same quality
- Tier 3: FTS5 keyword (SQLite) - always available, fast

Time Decay Scoring:
- Exponential decay with 90-day half-life: recent results weighted higher
- Formula: decayed_score = relevance × e^(-λ × age_days) where λ = ln(2)/90
- Floor of 0.1 ensures old but highly relevant results still appear
- Applied after all search tiers, before final ranking

Lazy Loading:
- Local semantic model (~100MB) only downloads when:
  1. Remote storage is unavailable AND
  2. User actually performs a search
- First offline search returns FTS5 results, triggers background download
- Subsequent searches use local semantic

Response Format:
- compact=True (default): Optimized for Claude context (~79% smaller)
- compact=False: Full verbose format for debugging
"""

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .utils import log, get_mira_path, extract_text_content, extract_query_terms
from .artifacts import search_artifacts_for_query


# =============================================================================
# TIME DECAY SCORING
# =============================================================================

# Default parameters for exponential time decay
DEFAULT_HALF_LIFE_DAYS = 90  # Score halves every 90 days
MIN_DECAY_FACTOR = 0.1  # Floor to prevent old results from disappearing entirely


def _calculate_time_decay(timestamp_str: str, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """
    Calculate exponential time decay factor for a result.

    Uses formula: decay = e^(-λ × age_days) where λ = ln(2) / half_life

    This implements a "forgetting curve" where recent results are weighted more
    heavily, but old results with high relevance can still surface.

    Args:
        timestamp_str: ISO format timestamp string (e.g., "2025-01-15T10:30:00")
        half_life_days: Days until score is halved (default 90)

    Returns:
        Decay factor between MIN_DECAY_FACTOR and 1.0
    """
    if not timestamp_str:
        return 0.5  # Neutral for missing timestamps

    try:
        # Parse timestamp (handle various formats)
        ts_str = str(timestamp_str)
        if "T" in ts_str:
            # ISO format: 2025-01-15T10:30:00Z or 2025-01-15T10:30:00+00:00
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00").split("+")[0])
        else:
            ts = datetime.fromisoformat(ts_str)

        # Calculate age in days
        age_days = (datetime.now() - ts).total_seconds() / 86400

        if age_days < 0:
            return 1.0  # Future timestamps get no decay

        # Exponential decay: e^(-λ × age) where λ = ln(2) / half_life
        decay_rate = math.log(2) / half_life_days
        decay = math.exp(-decay_rate * age_days)

        # Apply floor to prevent old results from disappearing
        return max(decay, MIN_DECAY_FACTOR)

    except (ValueError, TypeError, AttributeError):
        return 0.5  # Neutral for unparseable timestamps


def _apply_time_decay(
    results: List[Dict[str, Any]],
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
) -> List[Dict[str, Any]]:
    """
    Apply exponential time decay to search results and re-sort.

    Modifies results in place, adding 'decayed_score' and 'decay_factor' fields.
    The original 'relevance' is preserved. Results are re-sorted by decayed_score.

    Args:
        results: List of search result dicts with 'relevance' and 'timestamp'
        half_life_days: Days until score is halved (default 90)

    Returns:
        Results sorted by decayed score (descending)
    """
    for result in results:
        original_score = result.get('relevance', 0.5)
        # Try multiple timestamp fields
        timestamp = result.get('timestamp') or result.get('started_at', '')

        decay_factor = _calculate_time_decay(timestamp, half_life_days)
        decayed_score = original_score * decay_factor

        result['decayed_score'] = round(decayed_score, 4)
        result['decay_factor'] = round(decay_factor, 3)

    # Re-sort by decayed score
    results.sort(key=lambda x: x.get('decayed_score', 0), reverse=True)

    return results


# =============================================================================
# COMPACT RESPONSE OPTIMIZATION
# =============================================================================

# Stopwords to filter from keywords/topics
SEARCH_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'this', 'that', 'these',
    'those', 'it', 'its', 'and', 'or', 'but', 'if', 'then', 'else',
    'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
    'here', 'there', 'please', 'check', 'look', 'see', 'run', 'use',
    'can', 'get', 'got', 'make', 'made', 'want', 'need', 'try', 'let',
    'like', 'know', 'think', 'going', 'want', 'file', 'files', 'code',
    'work', 'working', 'works', 'error', 'errors', 'using', 'used',
}


def _select_best_excerpt(excerpts: List[Dict], query_terms: List[str]) -> str:
    """
    Select the single most relevant excerpt (full length preserved).

    Scoring:
    - Prefer assistant responses (more informative)
    - Count query term matches
    - Prefer longer excerpts (more context)
    """
    if not excerpts:
        return ""

    query_terms_set = set(t.lower() for t in query_terms)
    scored = []

    for exc in excerpts:
        score = 0
        # Prefer assistant responses (more informative)
        if exc.get('role') == 'assistant':
            score += 2
        # Count query term matches
        matched = set(t.lower() for t in exc.get('matched_terms', []))
        score += len(matched & query_terms_set)
        # Prefer longer excerpts (more context) - up to 2 points
        excerpt_len = len(exc.get('excerpt', ''))
        score += min(excerpt_len / 150, 2)

        scored.append((score, exc))

    best = max(scored, key=lambda x: x[0])[1]
    return best.get('excerpt', '')


def _filter_keywords_to_topics(keywords: List[str], query_terms: List[str], limit: int = 5) -> List[str]:
    """
    Filter keywords to most relevant topics.

    - Remove stopwords
    - Prioritize query term matches
    - Limit to top N
    """
    if not keywords:
        return []

    query_terms_lower = set(t.lower() for t in query_terms)

    # Filter out stopwords and very short words
    filtered = [kw for kw in keywords
                if kw.lower() not in SEARCH_STOPWORDS and len(kw) > 2]

    # Prioritize query term matches
    query_matches = [kw for kw in filtered if kw.lower() in query_terms_lower]
    others = [kw for kw in filtered if kw.lower() not in query_terms_lower]

    return (query_matches + others)[:limit]


def _consolidate_summary(summary: str, task_description: str = "", max_length: int = 100) -> str:
    """
    Merge summary and task_description into concise form.

    If summary has "Task: X | Outcome: Y" format, extract outcome.
    Otherwise prefer shorter of the two.
    """
    if not summary and not task_description:
        return ""

    # If summary has structured format, extract the outcome
    if summary and ' | Outcome: ' in summary:
        outcome = summary.split(' | Outcome: ')[1]
        if len(outcome) <= max_length:
            return outcome
        return outcome[:max_length - 3] + "..."

    # Prefer task_description if it's shorter and non-empty
    if task_description and (not summary or len(task_description) < len(summary)):
        text = task_description
    else:
        text = summary or ""

    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def _compact_result(result: Dict[str, Any], query_terms: List[str]) -> Dict[str, Any]:
    """
    Transform a verbose search result into compact format.

    Removes: slug, git_branch, search_source, has_archive_matches, task_description
    Transforms: session_id (8 chars), keywords->topics (5 max), excerpts->excerpt (single)
    Consolidates: summary + task_description
    """
    # Shorten session_id to first 8 chars (still unique enough)
    session_id = result.get('session_id', '')
    short_id = session_id[:8] if session_id else ''

    # Consolidate summary
    summary = _consolidate_summary(
        result.get('summary', ''),
        result.get('task_description', '')
    )

    # Extract date from timestamp
    timestamp = result.get('timestamp', '')
    date = ''
    if timestamp:
        # Handle ISO format: 2025-12-07T21:22:48.586Z
        date = timestamp[:10] if len(timestamp) >= 10 else timestamp

    # Filter keywords to topics
    keywords = result.get('keywords', [])
    if isinstance(keywords, str):
        keywords = keywords.split(',') if keywords else []
    topics = _filter_keywords_to_topics(keywords, query_terms)

    # Select best excerpt (full length)
    excerpts = result.get('excerpts', [])
    # Handle case where excerpts is a list of strings instead of dicts
    if excerpts and isinstance(excerpts[0], str):
        # Convert string excerpts to dict format
        excerpts = [{'excerpt': e, 'role': 'unknown', 'matched_terms': []} for e in excerpts]
    excerpt = _select_best_excerpt(excerpts, query_terms) if excerpts else ''

    # Message count
    msg_count = result.get('message_count', 0)
    if isinstance(msg_count, str):
        try:
            msg_count = int(msg_count)
        except ValueError:
            msg_count = 0

    compact = {
        'id': short_id,
        'summary': summary,
        'date': date,
        'topics': topics,
        'excerpt': excerpt,
        'messages': msg_count,
    }

    # Include score if high relevance (use decayed_score if available for ranking transparency)
    decayed_score = result.get('decayed_score')
    relevance = result.get('relevance', 0)

    # Use decayed score for display since that's what we're ranking by
    display_score = decayed_score if decayed_score is not None else relevance
    if display_score and display_score > 0.5:
        compact['score'] = round(display_score, 2)
        # Show age indicator if decay significantly affected score
        decay_factor = result.get('decay_factor', 1.0)
        if decay_factor < 0.7:
            # Result is older - show the original relevance for context
            compact['raw_score'] = round(relevance, 2) if relevance else None

    return compact


def _compact_results(results: List[Dict], query: str) -> List[Dict]:
    """Transform all results to compact format."""
    query_terms = extract_query_terms(query)
    return [_compact_result(r, query_terms) for r in results]


def handle_search(params: dict, collection, storage=None) -> dict:
    """
    Search conversations with tiered/layered search strategy.

    Search tiers (in order, falls through if no results):
    1. Central hybrid (vector + metadata FTS) - semantic understanding
    2. Archive FTS - raw content search in conversation archives
    3. Artifact search - code blocks, lists, tables, etc.
    4. Local FTS fallback - if central unavailable

    Always attempts to return results by trying each tier.

    Args:
        params: Search parameters
            - query: Search query string
            - limit: Max results (default 10)
            - project_path: Optional project filter
            - compact: Return compact format (default True, ~79% smaller)
            - days: Filter to sessions from last N days (hard cutoff)
            - recency_bias: Apply time decay to boost recent results (default True).
                           Set to False for historical searches where old content matters.
        collection: Deprecated - kept for API compatibility, ignored
        storage: Storage instance

    Returns:
        Dict with results, total, and query (compact) or artifacts/search_type (verbose)
    """
    from datetime import datetime, timedelta

    query = params.get("query", "")
    limit = params.get("limit", 10)
    project_path = params.get("project_path")  # Optional: filter to specific project
    compact = params.get("compact", True)  # Default to compact for token efficiency
    days = params.get("days")  # Optional: filter to last N days
    recency_bias = params.get("recency_bias", True)  # Apply time decay by default
    mira_path = get_mira_path()

    # Calculate cutoff time if days specified
    cutoff_time = None
    if days is not None and days > 0:
        cutoff_time = datetime.now() - timedelta(days=days)

    if not query:
        return {"results": [], "total": 0}

    # Get storage instance if not provided
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            log("ERROR: Storage not available")
            return {"results": [], "total": 0, "error": "Storage not available"}

    results = []
    search_type = "none"
    artifact_results = []
    searched_global = False  # Track if we've expanded beyond project

    # Always search artifacts (fast, runs in parallel conceptually)
    try:
        artifact_results = search_artifacts_for_query(query, limit=5) or []
    except Exception as e:
        log(f"Artifact search failed: {e}")

    # PROJECT-FIRST SEARCH STRATEGY:
    # 1. First search within project_path if provided
    # 2. If no results, expand search to all projects

    # TIER 1: Central hybrid search (vector + metadata FTS) - project first
    if storage.using_central:
        try:
            # First: search within project only
            central_results = _search_central_parallel(
                storage, query, project_path, limit
            )
            if central_results:
                enriched = enrich_results_from_archives(central_results, query, mira_path, storage)
                results = enriched
                search_type = "central_hybrid"

            # If no results and we had a project filter, try global
            if not results and project_path:
                log(f"No results in project, expanding search globally")
                central_results = _search_central_parallel(
                    storage, query, None, limit  # No project filter
                )
                if central_results:
                    enriched = enrich_results_from_archives(central_results, query, mira_path, storage)
                    results = enriched
                    search_type = "central_hybrid_global"
                    searched_global = True
        except Exception as e:
            log(f"Central hybrid search failed: {e}")

    # TIER 2: Archive FTS (raw content search) - ALWAYS run for exact matches
    # Run regardless of vector results to find exact keyword matches in archive content
    archive_fts_results = []
    if storage.using_central:
        try:
            # First: search within project only
            archive_results = storage.search_archives_fts(query, project_path=project_path, limit=limit)

            # If no results and we had a project filter, try global
            if not archive_results and project_path and not searched_global:
                log(f"No archive results in project, expanding search globally")
                archive_results = storage.search_archives_fts(query, project_path=None, limit=limit)
                searched_global = True

            if archive_results:
                # Format archive results
                for r in archive_results:
                    # Extract excerpts from content
                    content = r.get("content", "")
                    excerpts = _extract_excerpts(content, query, max_excerpts=3)
                    # Convert rank to float (Postgres returns Decimal)
                    rank = r.get("rank", 0.5)
                    if hasattr(rank, '__float__'):
                        rank = float(rank)
                    archive_fts_results.append({
                        "session_id": r.get("session_id", ""),
                        "summary": r.get("summary", ""),
                        "project_path": r.get("project_path", ""),
                        "excerpts": excerpts,
                        "relevance": rank,
                        "search_source": "archive_fts" + ("_global" if searched_global else "")
                    })
        except Exception as e:
            log(f"Archive FTS search failed: {e}")

    # Merge vector results with archive FTS results
    # Archive FTS results (exact matches) are prioritized and merged with vector results
    if archive_fts_results:
        # Get session IDs already in results from vector search
        existing_session_ids = {r.get("session_id") for r in results}

        # Add archive FTS results that aren't already in vector results
        # These go first since they're exact matches
        new_results = []
        for ar in archive_fts_results:
            session_id = ar.get("session_id")
            if session_id not in existing_session_ids:
                new_results.append(ar)
            else:
                # Update existing result with excerpts from archive FTS
                for r in results:
                    if r.get("session_id") == session_id:
                        r["excerpts"] = ar.get("excerpts", [])
                        r["has_archive_matches"] = True
                        break

        # Prepend new archive FTS results (exact matches first)
        results = new_results + results
        if new_results:
            search_type = "combined" if search_type != "none" else "archive_fts"

    # TIER 2: Local semantic search (if remote unavailable)
    # Only triggered when remote is down - this is where lazy loading kicks in
    local_semantic_notice = None
    if not results and not storage.using_central:
        try:
            from .local_semantic import is_local_semantic_available, get_local_semantic, trigger_local_semantic_download

            if is_local_semantic_available():
                # Local semantic ready - use it
                ls = get_local_semantic()
                local_results = ls.search(query, project_path=project_path, limit=limit)
                if local_results:
                    for r in local_results:
                        results.append({
                            "session_id": r.get("session_id", ""),
                            "summary": "",  # Will be enriched from archives
                            "keywords": [],
                            "relevance": r.get("score", 0.5),
                            "timestamp": "",
                            "project_path": "",
                            "search_source": "local_semantic"
                        })
                    enriched = enrich_results_from_archives(results, query, mira_path, storage)
                    results = enriched
                    search_type = "local_semantic"
            else:
                # Local semantic not ready - trigger download, will fall through to FTS5
                local_semantic_notice = trigger_local_semantic_download()

        except Exception as e:
            log(f"Local semantic search failed: {e}")
            # Fall through to FTS5

    # TIER 3: Local FTS fallback (if central unavailable or no results)
    if not results:
        try:
            fts_results = storage.search_sessions_fts(query, project_path, limit)
            if fts_results:
                for r in fts_results:
                    results.append({
                        "session_id": r.get("session_id", ""),
                        "summary": r.get("summary", ""),
                        "keywords": r.get("keywords", []) if isinstance(r.get("keywords"), list) else [],
                        "relevance": r.get("rank", 0.5),
                        "timestamp": r.get("started_at", ""),
                        "project_path": r.get("project_path", ""),
                        "search_source": "local_fts"
                    })
                enriched = enrich_results_from_archives(results, query, mira_path, storage)
                results = enriched
                search_type = "local_fts"
        except Exception as e:
            log(f"Local FTS search failed: {e}")

    # TIER 4: Fulltext archive search (last resort)
    if not results:
        try:
            fallback_results = fulltext_search_archives(query, limit, mira_path, storage)
            if fallback_results:
                results = fallback_results
                search_type = "fulltext_fallback"
        except Exception as e:
            log(f"Fulltext fallback failed: {e}")

    # Apply time filter if days specified
    if cutoff_time and results:
        filtered_results = []
        for r in results:
            # Try to parse timestamp from various fields
            ts_str = r.get("timestamp") or r.get("started_at") or ""
            if ts_str:
                try:
                    # Handle various timestamp formats
                    ts_str = str(ts_str)
                    if "T" in ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00").split("+")[0])
                    else:
                        ts = datetime.fromisoformat(ts_str)
                    if ts >= cutoff_time:
                        filtered_results.append(r)
                except (ValueError, TypeError):
                    # If we can't parse, include the result
                    filtered_results.append(r)
            else:
                # No timestamp, include the result
                filtered_results.append(r)
        results = filtered_results

    # Apply exponential time decay to scores and re-sort (unless disabled)
    # Recent results are boosted relative to older ones (90-day half-life)
    if results and recency_bias:
        results = _apply_time_decay(results)

    # Apply compact transformation if requested (default)
    if compact:
        response = {
            "results": _compact_results(results, query),
            "total": len(results),
            "query": query,  # Include query for context
        }
        if days:
            response["filtered_to_days"] = days
        # Include notice about local semantic download if triggered
        if local_semantic_notice and local_semantic_notice.get("notice"):
            response["notice"] = local_semantic_notice["notice"]
        return response
    else:
        # Verbose format for debugging
        response = {
            "results": results,
            "total": len(results),
            "search_type": search_type,
            "artifacts": artifact_results
        }
        if days:
            response["filtered_to_days"] = days
        # Include notice about local semantic download if triggered
        if local_semantic_notice and local_semantic_notice.get("notice"):
            response["notice"] = local_semantic_notice["notice"]
        return response


def _extract_excerpts(content: str, query: str, max_excerpts: int = 3) -> List[str]:
    """Extract relevant excerpts from content around query matches."""
    if not content or not query:
        return []

    excerpts = []
    query_lower = query.lower()
    content_lower = content.lower()

    # Find all occurrences
    start = 0
    while len(excerpts) < max_excerpts:
        idx = content_lower.find(query_lower, start)
        if idx == -1:
            break

        # Extract context around match
        excerpt_start = max(0, idx - 100)
        excerpt_end = min(len(content), idx + len(query) + 100)

        # Try to break at word boundaries
        if excerpt_start > 0:
            space_idx = content.rfind(' ', excerpt_start, idx)
            if space_idx > excerpt_start:
                excerpt_start = space_idx + 1

        excerpt = content[excerpt_start:excerpt_end].strip()
        if excerpt_start > 0:
            excerpt = "..." + excerpt
        if excerpt_end < len(content):
            excerpt = excerpt + "..."

        excerpts.append(excerpt)
        start = idx + len(query)

    return excerpts


def _search_central_parallel(
    storage,
    query: str,
    project_path: Optional[str],
    limit: int
) -> List[Dict[str, Any]]:
    """
    Search central storage using parallel vector + FTS queries.

    Merges and deduplicates results from both search types.
    Vector search uses remote embedding service.
    """
    from .embedding_client import get_embedding_client

    vector_results = []
    fts_results = []

    # Get embedding client for semantic search
    embed_client = get_embedding_client()

    # Run vector and FTS searches in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Vector search via remote embedding service
        if embed_client:
            # Get project_id if filtering by project
            project_id = None
            if project_path:
                project_id = storage.postgres.get_project_id(project_path)

            vector_future = executor.submit(
                embed_client.search,
                query,
                project_id,
                project_path,
                limit
            )
        else:
            vector_future = None

        fts_future = executor.submit(
            storage.search_sessions_fts,
            query,
            project_path,
            limit
        )

        if vector_future:
            try:
                result = vector_future.result(timeout=60)
                # Transform embedding service results to match expected format
                for r in result.get("results", []):
                    vector_results.append({
                        "session_id": r.get("session_id"),
                        "score": r.get("score", 0),
                        "metadata": r.get("metadata", {}),
                    })
            except Exception as e:
                log(f"Vector search failed: {e}")

        try:
            fts_results = fts_future.result(timeout=30)
        except Exception as e:
            log(f"FTS search failed: {e}")

    # Merge and deduplicate results
    return _merge_search_results(vector_results, fts_results, limit)


def _merge_search_results(
    vector_results: List[Dict[str, Any]],
    fts_results: List[Dict[str, Any]],
    limit: int
) -> List[Dict[str, Any]]:
    """
    Merge vector and FTS search results, deduplicating by session_id.

    Vector results are prioritized (semantic match), FTS results fill in gaps.
    """
    seen_sessions = set()
    merged = []

    # Add vector results first (higher priority)
    for r in vector_results:
        session_id = r.get("session_id", "")
        if session_id and session_id not in seen_sessions:
            seen_sessions.add(session_id)
            merged.append({
                "session_id": session_id,
                "summary": r.get("content", r.get("summary", ""))[:500],
                "keywords": [],
                "relevance": r.get("score", 0.5),
                "timestamp": "",
                "project_path": r.get("project_path", ""),
                "search_source": "vector"
            })

    # Add FTS results that weren't in vector results
    for r in fts_results:
        session_id = r.get("session_id", "")
        if session_id and session_id not in seen_sessions:
            seen_sessions.add(session_id)
            merged.append({
                "session_id": session_id,
                "summary": r.get("summary", ""),
                "keywords": r.get("keywords", []) if isinstance(r.get("keywords"), list) else [],
                "relevance": r.get("rank", 0.3),
                "timestamp": r.get("started_at", ""),
                "project_path": r.get("project_path", ""),
                "search_source": "fts"
            })

    # Sort by relevance and limit
    merged.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return merged[:limit]


def enrich_results_from_archives(results: list, query: str, mira_path: Path, storage=None) -> list:
    """
    Enrich search results with relevant excerpts from full conversation archives.

    This solves the embedding truncation problem: while we can only index ~900 chars
    for semantic search, we can search the FULL conversation archive for specific
    content once we know which conversations are relevant.

    Tries remote archives first, falls back to local if unavailable.
    """
    archives_path = mira_path / "archives"

    # Extract meaningful search terms from query
    query_terms = extract_query_terms(query)
    if not query_terms:
        return results

    # Try to get storage if not provided
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            pass

    enriched = []
    for result in results:
        session_id = result.get("session_id", "")
        excerpts = []

        # Try remote archive first
        if storage and storage.using_central:
            try:
                archive_content = storage.get_archive(session_id)
                if archive_content:
                    excerpts = search_archive_content_for_excerpts(archive_content, query_terms, max_excerpts=3)
            except Exception as e:
                log(f"Remote archive search failed for {session_id}: {e}")

        # Fall back to local archive if no remote excerpts
        if not excerpts:
            archive_file = archives_path / f"{session_id}.jsonl"
            if archive_file.exists():
                try:
                    excerpts = search_archive_for_excerpts(archive_file, query_terms, max_excerpts=3)
                except Exception as e:
                    log(f"Error searching local archive {session_id}: {e}")

        # Add excerpts to result
        result_copy = result.copy()
        result_copy["excerpts"] = excerpts
        result_copy["has_archive_matches"] = len(excerpts) > 0
        enriched.append(result_copy)

    return enriched


def search_archive_content_for_excerpts(content: str, query_terms: list, max_excerpts: int = 3) -> list:
    """
    Search archive content (string) for excerpts matching query terms.
    Used for remote archives stored in Postgres.
    """
    excerpts = []
    seen_excerpts = set()

    for line in content.split('\n'):
        if len(excerpts) >= max_excerpts:
            break

        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
            msg_type = obj.get('type', '')

            # Only search user and assistant messages
            if msg_type not in ('user', 'assistant'):
                continue

            message = obj.get('message', {})
            msg_content = extract_text_content(message)

            if not msg_content:
                continue

            content_lower = msg_content.lower()

            # Check if any query terms appear
            matching_terms = [t for t in query_terms if t in content_lower]

            if matching_terms:
                excerpt = extract_excerpt_around_terms(msg_content, matching_terms)

                # Avoid duplicates
                excerpt_key = excerpt[:100].lower()
                if excerpt_key not in seen_excerpts:
                    seen_excerpts.add(excerpt_key)
                    excerpts.append({
                        "role": msg_type,
                        "excerpt": excerpt,
                        "matched_terms": matching_terms,
                        "timestamp": obj.get("timestamp", "")
                    })

        except json.JSONDecodeError:
            continue

    return excerpts


def search_archive_for_excerpts(archive_path: Path, query_terms: list, max_excerpts: int = 3) -> list:
    """
    Search a conversation archive for excerpts matching query terms.
    """
    excerpts = []
    seen_excerpts = set()

    try:
        with open(archive_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(excerpts) >= max_excerpts:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    msg_type = obj.get('type', '')

                    # Only search user and assistant messages
                    if msg_type not in ('user', 'assistant'):
                        continue

                    message = obj.get('message', {})
                    content = extract_text_content(message)

                    if not content:
                        continue

                    content_lower = content.lower()

                    # Check if any query terms appear
                    matching_terms = [t for t in query_terms if t in content_lower]

                    if matching_terms:
                        excerpt = extract_excerpt_around_terms(content, matching_terms)

                        # Avoid duplicates
                        excerpt_key = excerpt[:100].lower()
                        if excerpt_key not in seen_excerpts:
                            seen_excerpts.add(excerpt_key)
                            excerpts.append({
                                "role": msg_type,
                                "excerpt": excerpt,
                                "matched_terms": matching_terms,
                                "timestamp": obj.get("timestamp", "")
                            })

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        log(f"Error reading archive: {e}")

    return excerpts


def extract_excerpt_around_terms(content: str, terms: list, context_chars: int = 200) -> str:
    """
    Extract an excerpt from content centered around the first matching term.
    """
    content_lower = content.lower()

    # Find the first occurrence of any term
    first_pos = len(content)
    matched_term = terms[0]
    for term in terms:
        pos = content_lower.find(term)
        if pos != -1 and pos < first_pos:
            first_pos = pos
            matched_term = term

    if first_pos == len(content):
        return content[:context_chars * 2] + "..." if len(content) > context_chars * 2 else content

    # Calculate excerpt boundaries
    start = max(0, first_pos - context_chars)
    end = min(len(content), first_pos + len(matched_term) + context_chars)

    # Adjust to word boundaries
    if start > 0:
        while start > 0 and content[start - 1].isalnum():
            start -= 1
        space_pos = content.find(' ', start)
        if space_pos != -1 and space_pos < first_pos:
            start = space_pos + 1

    if end < len(content):
        space_pos = content.rfind(' ', first_pos, end)
        if space_pos != -1:
            end = space_pos

    excerpt = content[start:end].strip()

    # Add ellipsis if truncated
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(content):
        excerpt = excerpt + "..."

    return excerpt


def fulltext_search_archives(query: str, limit: int, mira_path: Path, storage=None) -> list:
    """
    Full-text search across all archived conversations.

    Used as fallback when semantic search returns no results.
    Tries remote Postgres FTS first, falls back to local archives.
    """
    # Extract search terms
    query_terms = extract_query_terms(query)
    if not query_terms:
        return []

    # Try to get storage if not provided
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            pass

    # Try remote FTS search first
    if storage and storage.using_central:
        try:
            remote_results = storage.search_archives_fts(query, limit=limit)
            if remote_results:
                results = []
                for r in remote_results:
                    # Parse archive content for excerpts
                    excerpts = search_archive_content_for_excerpts(
                        r.get("content", ""), query_terms, max_excerpts=5
                    )
                    results.append({
                        "session_id": r.get("session_id", ""),
                        "summary": r.get("summary", ""),
                        "project_path": r.get("project_path", ""),
                        "excerpts": excerpts,
                        "relevance": r.get("rank", 0.5),
                        "has_archive_matches": len(excerpts) > 0,
                        "search_source": "remote_fts"
                    })
                return results
        except Exception as e:
            log(f"Remote archive FTS failed: {e}")

    # Fall back to local archive search
    archives_path = mira_path / "archives"
    metadata_path = mira_path / "metadata"

    if not archives_path.exists():
        return []

    results = []

    for archive_file in archives_path.glob("*.jsonl"):
        session_id = archive_file.stem

        # Load metadata
        meta_file = metadata_path / f"{session_id}.json"
        metadata = {}
        if meta_file.exists():
            try:
                metadata = json.loads(meta_file.read_text())
            except:
                pass

        # Search archive
        excerpts = search_archive_for_excerpts(archive_file, query_terms, max_excerpts=5)

        if excerpts:
            results.append({
                "session_id": session_id,
                "slug": metadata.get("slug", ""),
                "summary": metadata.get("summary", ""),
                "task_description": metadata.get("task_description", ""),
                "project_path": metadata.get("project_path", ""),
                "git_branch": metadata.get("git_branch", ""),
                "keywords": metadata.get("keywords", []),
                "excerpts": excerpts,
                "relevance": 0.5,
                "has_archive_matches": True,
                "timestamp": metadata.get("extracted_at", ""),
                "message_count": str(metadata.get("message_count", 0)),
                "search_source": "local_fts"
            })

        if len(results) >= limit:
            break

    # Sort by number of excerpt matches
    results.sort(key=lambda x: len(x.get("excerpts", [])), reverse=True)

    return results[:limit]
