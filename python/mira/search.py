"""
MIRA3 Search Module

Handles semantic search, archive enrichment, and fulltext fallback.

Primary: Central Qdrant (vector) + Postgres (FTS) for hybrid search
Fallback: Local SQLite with FTS5 (keyword search only)

In local mode, only FTS search is available (no semantic/vector search).
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional

from .utils import log, get_mira_path, extract_text_content, extract_query_terms
from .artifacts import search_artifacts_for_query


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
        params: Search parameters (query, limit, project_path)
        collection: Deprecated - kept for API compatibility, ignored
        storage: Storage instance

    Returns:
        Dict with results, total, artifacts, and search_type
    """
    query = params.get("query", "")
    limit = params.get("limit", 10)
    project_path = params.get("project_path")  # Optional: filter to specific project
    mira_path = get_mira_path()

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

    # TIER 2: Archive FTS (raw content search) - project first, then global
    if not results and storage.using_central:
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
                    results.append({
                        "session_id": r.get("session_id", ""),
                        "summary": r.get("summary", ""),
                        "project_path": r.get("project_path", ""),
                        "excerpts": excerpts,
                        "relevance": rank,
                        "search_source": "archive_fts" + ("_global" if searched_global else "")
                    })
                search_type = "archive_fts" + ("_global" if searched_global else "")
        except Exception as e:
            log(f"Archive FTS search failed: {e}")

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

    return {
        "results": results,
        "total": len(results),
        "search_type": search_type,
        "artifacts": artifact_results
    }


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
    """
    from .embedding import get_embedding_function

    # Get query embedding
    embed_fn = get_embedding_function()
    query_vector = embed_fn([query])[0]

    vector_results = []
    fts_results = []

    # Run vector and FTS searches in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(
            storage.vector_search,
            query_vector,
            project_path,
            limit
        )
        fts_future = executor.submit(
            storage.search_sessions_fts,
            query,
            project_path,
            limit
        )

        try:
            vector_results = vector_future.result(timeout=30)
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
