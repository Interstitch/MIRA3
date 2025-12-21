"""
MIRA Search - Core Search Module

Handles semantic search, archive enrichment, and fulltext fallback.

Three-Tier Search Architecture:
- Tier 1: Remote semantic (Qdrant + embedding-service) - cross-machine, best quality
- Tier 2: Local semantic (sqlite-vec + fastembed) - offline, same quality
- Tier 3: FTS5 keyword (SQLite) - always available, fast

Time Decay Scoring:
- Exponential decay with 90-day half-life: recent results weighted higher
- Formula: decayed_score = relevance × e^(-λ × age_days) where λ = ln(2)/90
- Floor of 0.1 ensures old but highly relevant results still appear

Response Format:
- compact=True (default): Optimized for Claude context (~79% smaller)
- compact=False: Full verbose format for debugging
"""

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from mira.core import log, MIRA_PATH
from mira.core.utils import extract_text_content, extract_query_terms

# Time decay constants
DEFAULT_HALF_LIFE_DAYS = 90
MIN_DECAY_FACTOR = 0.1

# Stopwords for keyword filtering
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


def _calculate_time_decay(timestamp_str: str, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """
    Calculate exponential time decay factor for a result.

    Uses formula: decay = e^(-λ × age_days) where λ = ln(2) / half_life
    """
    if not timestamp_str:
        return 0.5  # Neutral for missing timestamps

    try:
        ts_str = str(timestamp_str)
        if "T" in ts_str:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00").split("+")[0])
        else:
            ts = datetime.fromisoformat(ts_str)

        age_days = (datetime.now() - ts).total_seconds() / 86400

        if age_days < 0:
            return 1.0

        decay_rate = math.log(2) / half_life_days
        decay = math.exp(-decay_rate * age_days)

        return max(decay, MIN_DECAY_FACTOR)

    except (ValueError, TypeError, AttributeError):
        return 0.5


def _apply_time_decay(
    results: List[Dict[str, Any]],
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
) -> List[Dict[str, Any]]:
    """Apply exponential time decay to search results and re-sort."""
    for result in results:
        original_score = result.get('relevance', 0.5)
        timestamp = result.get('timestamp') or result.get('started_at', '')

        decay_factor = _calculate_time_decay(timestamp, half_life_days)
        decayed_score = original_score * decay_factor

        result['decayed_score'] = round(decayed_score, 4)
        result['decay_factor'] = round(decay_factor, 3)

    results.sort(key=lambda x: x.get('decayed_score', 0), reverse=True)
    return results


def _select_best_excerpt(excerpts: List[Dict], query_terms: List[str]) -> str:
    """Select the single most relevant excerpt."""
    if not excerpts:
        return ""

    query_terms_set = set(t.lower() for t in query_terms)
    scored = []

    for exc in excerpts:
        score = 0
        if exc.get('role') == 'assistant':
            score += 2
        matched = set(t.lower() for t in exc.get('matched_terms', []))
        score += len(matched & query_terms_set)
        excerpt_len = len(exc.get('excerpt', ''))
        score += min(excerpt_len / 150, 2)
        scored.append((score, exc))

    best = max(scored, key=lambda x: x[0])[1]
    return best.get('excerpt', '')


def _filter_keywords_to_topics(keywords: List[str], query_terms: List[str], limit: int = 5) -> List[str]:
    """Filter keywords to most relevant topics."""
    if not keywords:
        return []

    query_terms_lower = set(t.lower() for t in query_terms)
    filtered = [kw for kw in keywords
                if kw.lower() not in SEARCH_STOPWORDS and len(kw) > 2]

    query_matches = [kw for kw in filtered if kw.lower() in query_terms_lower]
    others = [kw for kw in filtered if kw.lower() not in query_terms_lower]

    return (query_matches + others)[:limit]


def _consolidate_summary(summary: str, task_description: str = "", max_length: int = 100) -> str:
    """Merge summary and task_description into concise form."""
    if not summary and not task_description:
        return ""

    if summary and ' | Outcome: ' in summary:
        outcome = summary.split(' | Outcome: ')[1]
        if len(outcome) <= max_length:
            return outcome
        return outcome[:max_length - 3] + "..."

    if task_description and (not summary or len(task_description) < len(summary)):
        text = task_description
    else:
        text = summary or ""

    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def _compact_result(result: Dict[str, Any], query_terms: List[str]) -> Dict[str, Any]:
    """Transform a verbose search result into compact format."""
    session_id = result.get('session_id', '')
    short_id = session_id[:8] if session_id else ''

    summary = _consolidate_summary(
        result.get('summary', ''),
        result.get('task_description', '')
    )

    timestamp = result.get('timestamp') or ''
    date = timestamp[:10] if len(timestamp) >= 10 else timestamp

    keywords = result.get('keywords', [])
    if isinstance(keywords, str):
        keywords = keywords.split(',') if keywords else []
    topics = _filter_keywords_to_topics(keywords, query_terms)

    excerpts = result.get('excerpts', [])
    if excerpts and isinstance(excerpts[0], str):
        excerpts = [{'excerpt': e, 'role': 'unknown', 'matched_terms': []} for e in excerpts]
    excerpt = _select_best_excerpt(excerpts, query_terms) if excerpts else ''

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

    decayed_score = result.get('decayed_score')
    relevance = result.get('relevance', 0)
    display_score = decayed_score if decayed_score is not None else relevance

    if display_score and display_score > 0.5:
        compact['score'] = round(display_score, 2)
        decay_factor = result.get('decay_factor', 1.0)
        if decay_factor < 0.7:
            compact['raw_score'] = round(relevance, 2) if relevance else None

    return compact


def _compact_results(results: List[Dict], query: str) -> List[Dict]:
    """Transform all results to compact format."""
    query_terms = extract_query_terms(query)
    return [_compact_result(r, query_terms) for r in results]


def _extract_excerpts(content: str, query: str, max_excerpts: int = 3) -> List[str]:
    """Extract relevant excerpts from content around query matches."""
    if not content or not query:
        return []

    excerpts = []
    query_lower = query.lower()
    content_lower = content.lower()

    start = 0
    while len(excerpts) < max_excerpts:
        idx = content_lower.find(query_lower, start)
        if idx == -1:
            break

        excerpt_start = max(0, idx - 100)
        excerpt_end = min(len(content), idx + len(query) + 100)

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


def extract_excerpt_around_terms(content: str, terms: list, context_chars: int = 200) -> str:
    """Extract an excerpt from content centered around the first matching term."""
    content_lower = content.lower()

    first_pos = len(content)
    matched_term = terms[0]
    for term in terms:
        pos = content_lower.find(term)
        if pos != -1 and pos < first_pos:
            first_pos = pos
            matched_term = term

    if first_pos == len(content):
        return content[:context_chars * 2] + "..." if len(content) > context_chars * 2 else content

    start = max(0, first_pos - context_chars)
    end = min(len(content), first_pos + len(matched_term) + context_chars)

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

    if start > 0:
        excerpt = "..." + excerpt
    if end < len(content):
        excerpt = excerpt + "..."

    return excerpt


def search_archive_content_for_excerpts(content: str, query_terms: list, max_excerpts: int = 3) -> list:
    """Search archive content (string) for excerpts matching query terms."""
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

            if msg_type not in ('user', 'assistant'):
                continue

            message = obj.get('message', {})
            msg_content = extract_text_content(message)

            if not msg_content:
                continue

            content_lower = msg_content.lower()
            matching_terms = [t for t in query_terms if t in content_lower]

            if matching_terms:
                excerpt = extract_excerpt_around_terms(msg_content, matching_terms)

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
    """Search a conversation archive for excerpts matching query terms."""
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

                    if msg_type not in ('user', 'assistant'):
                        continue

                    message = obj.get('message', {})
                    content = extract_text_content(message)

                    if not content:
                        continue

                    content_lower = content.lower()
                    matching_terms = [t for t in query_terms if t in content_lower]

                    if matching_terms:
                        excerpt = extract_excerpt_around_terms(content, matching_terms)

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


def enrich_results_from_archives(results: list, query: str, storage=None) -> list:
    """
    Enrich search results with relevant excerpts from full conversation archives.

    Tries remote archives first, falls back to local if unavailable.
    """
    archives_path = MIRA_PATH / "archives"
    query_terms = extract_query_terms(query)

    if not query_terms:
        return results

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

        result_copy = result.copy()
        result_copy["excerpts"] = excerpts
        result_copy["has_archive_matches"] = len(excerpts) > 0
        enriched.append(result_copy)

    return enriched


def fulltext_search_archives(query: str, limit: int, storage=None) -> list:
    """
    Full-text search across all archived conversations.

    Used as fallback when semantic search returns no results.
    """
    query_terms = extract_query_terms(query)
    if not query_terms:
        return []

    # Try remote FTS search first
    if storage and storage.using_central:
        try:
            remote_results = storage.search_archives_fts(query, limit=limit)
            if remote_results:
                results = []
                for r in remote_results:
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
    archives_path = MIRA_PATH / "archives"
    metadata_path = MIRA_PATH / "metadata"

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
                metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
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

    results.sort(key=lambda x: len(x.get("excerpts", [])), reverse=True)
    return results[:limit]


def _search_central_parallel(
    storage,
    query: str,
    project_path: Optional[str],
    limit: int
) -> List[Dict[str, Any]]:
    """Search central storage using parallel vector + FTS queries."""
    from mira.external.embedding_client import get_embedding_client

    vector_results = []
    fts_results = []

    embed_client = get_embedding_client()

    with ThreadPoolExecutor(max_workers=2) as executor:
        if embed_client:
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

    return _merge_search_results(vector_results, fts_results, limit)


def _merge_search_results(
    vector_results: List[Dict[str, Any]],
    fts_results: List[Dict[str, Any]],
    limit: int
) -> List[Dict[str, Any]]:
    """Merge vector and FTS search results, deduplicating by session_id."""
    seen_sessions = set()
    merged = []

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

    merged.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return merged[:limit]


def handle_search(params: dict, storage=None) -> dict:
    """
    Search conversations with tiered/layered search strategy.

    Search tiers (in order, falls through if no results):
    1. Central hybrid (vector + metadata FTS) - semantic understanding
    2. Archive FTS - raw content search in conversation archives
    3. Local semantic - if remote unavailable
    4. Local FTS fallback - always available

    Args:
        params: Search parameters
            - query: Search query string
            - limit: Max results (default 10)
            - project_path: Optional project filter
            - compact: Return compact format (default True)
            - days: Filter to sessions from last N days
            - recency_bias: Apply time decay (default True)
        storage: Storage instance

    Returns:
        Dict with results, total, and query (compact) or search_type (verbose)
    """
    from mira.search.fuzzy import expand_query_with_corrections, get_vocabulary_size, is_in_vocabulary
    from mira.search.local_semantic import is_local_semantic_available, get_local_semantic, trigger_local_semantic_download
    from mira.extraction.artifacts import search_artifacts_for_query

    query = params.get("query", "")
    limit = params.get("limit", 10)
    project_path = params.get("project_path")
    compact = params.get("compact", True)
    days = params.get("days")
    recency_bias = params.get("recency_bias", True)

    cutoff_time = None
    if days is not None and days > 0:
        cutoff_time = datetime.now() - timedelta(days=days)

    if not query:
        return {
            "results": [],
            "total": 0,
            "message": "No query provided. Specify a search query to search conversation history."
        }

    # Apply fuzzy matching for typo correction
    original_query = query
    corrections = []
    unknown_terms = []
    try:
        vocab_size = get_vocabulary_size()
        if vocab_size > 0:
            tokens = re.findall(r'\b\w{3,}\b', query.lower())
            unknown_terms = [t for t in tokens if not is_in_vocabulary(t)]

            corrected_query, corrections = expand_query_with_corrections(query)
            if corrections:
                query = corrected_query
                log(f"Fuzzy corrected: '{original_query}' → '{query}' ({len(corrections)} corrections)")
    except Exception as e:
        log(f"Fuzzy matching failed: {e}")

    # Get storage instance if not provided
    if storage is None:
        try:
            from mira.storage import get_storage
            storage = get_storage()
        except ImportError:
            log("ERROR: Storage not available")
            return {"results": [], "total": 0, "error": "Storage not available"}

    results = []
    search_type = "none"
    artifact_results = []
    searched_global = False

    # Always search artifacts
    try:
        artifact_results = search_artifacts_for_query(query, limit=5) or []
    except Exception as e:
        log(f"Artifact search failed: {e}")

    # TIER 1: Central hybrid search (vector + metadata FTS)
    if storage.using_central:
        try:
            central_results = _search_central_parallel(storage, query, project_path, limit)
            if central_results:
                enriched = enrich_results_from_archives(central_results, query, storage)
                results = enriched
                search_type = "central_hybrid"

            if not results and project_path:
                log(f"No results in project, expanding search globally")
                central_results = _search_central_parallel(storage, query, None, limit)
                if central_results:
                    enriched = enrich_results_from_archives(central_results, query, storage)
                    results = enriched
                    search_type = "central_hybrid_global"
                    searched_global = True
        except Exception as e:
            log(f"Central hybrid search failed: {e}")

    # TIER 2: Archive FTS (raw content search)
    archive_fts_results = []
    if storage.using_central:
        try:
            archive_results = storage.search_archives_fts(query, project_path=project_path, limit=limit)

            if not archive_results and project_path and not searched_global:
                log(f"No archive results in project, expanding search globally")
                archive_results = storage.search_archives_fts(query, project_path=None, limit=limit)
                searched_global = True

            if archive_results:
                for r in archive_results:
                    content = r.get("content", "")
                    excerpts = _extract_excerpts(content, query, max_excerpts=3)
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

    # Merge archive FTS results
    if archive_fts_results:
        existing_session_ids = {r.get("session_id") for r in results}

        new_results = []
        for ar in archive_fts_results:
            session_id = ar.get("session_id")
            if session_id not in existing_session_ids:
                new_results.append(ar)
            else:
                for r in results:
                    if r.get("session_id") == session_id:
                        r["excerpts"] = ar.get("excerpts", [])
                        r["has_archive_matches"] = True
                        break

        results = new_results + results
        if new_results:
            search_type = "combined" if search_type != "none" else "archive_fts"

    # TIER 3: Local semantic search (if remote unavailable)
    local_semantic_notice = None
    if not results and not storage.using_central:
        try:
            if is_local_semantic_available():
                ls = get_local_semantic()
                local_results = ls.search(query, project_path=project_path, limit=limit)
                if local_results:
                    for r in local_results:
                        results.append({
                            "session_id": r.get("session_id", ""),
                            "summary": "",
                            "keywords": [],
                            "relevance": r.get("score", 0.5),
                            "timestamp": "",
                            "project_path": "",
                            "search_source": "local_semantic"
                        })
                    enriched = enrich_results_from_archives(results, query, storage)
                    results = enriched
                    search_type = "local_semantic"
            else:
                local_semantic_notice = trigger_local_semantic_download()

        except Exception as e:
            log(f"Local semantic search failed: {e}")

    # TIER 4: Local FTS fallback
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
                enriched = enrich_results_from_archives(results, query, storage)
                results = enriched
                search_type = "local_fts"
        except Exception as e:
            log(f"Local FTS search failed: {e}")

    # TIER 5: Fulltext archive search (last resort)
    if not results:
        try:
            fallback_results = fulltext_search_archives(query, limit, storage)
            if fallback_results:
                results = fallback_results
                search_type = "fulltext_fallback"
        except Exception as e:
            log(f"Fulltext fallback failed: {e}")

    # Apply time filter if days specified
    if cutoff_time and results:
        filtered_results = []
        for r in results:
            ts_str = r.get("timestamp") or r.get("started_at") or ""
            if ts_str:
                try:
                    ts_str = str(ts_str)
                    if "T" in ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00").split("+")[0])
                    else:
                        ts = datetime.fromisoformat(ts_str)
                    if ts >= cutoff_time:
                        filtered_results.append(r)
                except (ValueError, TypeError):
                    filtered_results.append(r)
            else:
                filtered_results.append(r)
        results = filtered_results

    # Apply time decay
    if results and recency_bias:
        results = _apply_time_decay(results)

    # Build response
    if compact:
        response = {
            "results": _compact_results(results, query),
            "total": len(results),
            "query": query,
        }

        if not results:
            try:
                from mira.storage.local_store import get_session_count
                session_count = get_session_count()
                if session_count == 0:
                    response["message"] = "No conversations indexed yet. MIRA is still processing your conversation history. Try again in a few moments."
                    response["hint"] = "Run mira_status to check ingestion progress."
                else:
                    response["message"] = f"No results found for '{query}'. Searched {session_count} indexed conversations."
                    response["suggestions"] = [
                        "Try different keywords or broader terms",
                        "Use mira_recent to see what topics are available",
                        "Check spelling - typo correction is automatic"
                    ]
            except Exception:
                response["message"] = f"No results found for '{query}'."

        if days:
            response["filtered_to_days"] = days
        if corrections:
            response["corrections"] = corrections
            response["original_query"] = original_query
        if unknown_terms:
            response["unknown_terms"] = unknown_terms
        if local_semantic_notice and local_semantic_notice.get("notice"):
            response["notice"] = local_semantic_notice["notice"]
        return response
    else:
        response = {
            "results": results,
            "total": len(results),
            "search_type": search_type,
            "artifacts": artifact_results
        }

        if not results:
            try:
                from mira.storage.local_store import get_session_count
                session_count = get_session_count()
                if session_count == 0:
                    response["message"] = "No conversations indexed yet. MIRA is still processing."
                else:
                    response["message"] = f"No results found for '{query}'. Searched {session_count} conversations."
            except Exception:
                response["message"] = f"No results found for '{query}'."

        if days:
            response["filtered_to_days"] = days
        if corrections:
            response["corrections"] = corrections
            response["original_query"] = original_query
        if unknown_terms:
            response["unknown_terms"] = unknown_terms
        if local_semantic_notice and local_semantic_notice.get("notice"):
            response["notice"] = local_semantic_notice["notice"]
        return response
