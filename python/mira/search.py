"""
MIRA3 Search Module

Handles semantic search, archive enrichment, and fulltext fallback.
"""

import json
import re
from pathlib import Path

from .utils import log, get_mira_path, extract_text_content, extract_query_terms
from .artifacts import search_artifacts_for_query


def handle_search(params: dict, collection) -> dict:
    """Search conversations using ChromaDB semantic search."""
    query = params.get("query", "")
    limit = params.get("limit", 10)
    mira_path = get_mira_path()

    if not query:
        return {"results": [], "total": 0}

    # Check if collection is empty
    if collection.count() == 0:
        log(f"Collection is empty, using fulltext fallback for '{query}'")
        fallback_results = fulltext_search_archives(query, limit, mira_path)
        artifact_results = search_artifacts_for_query(query, limit=5)
        return {
            "results": fallback_results,
            "total": len(fallback_results),
            "search_type": "fulltext_fallback",
            "artifacts": artifact_results if artifact_results else []
        }

    # Query ChromaDB
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(limit, 100)
        )
    except Exception as e:
        log(f"ChromaDB query error: {e}, falling back to fulltext")
        fallback_results = fulltext_search_archives(query, limit, mira_path)
        artifact_results = search_artifacts_for_query(query, limit=5)
        return {
            "results": fallback_results,
            "total": len(fallback_results),
            "search_type": "fulltext_fallback",
            "artifacts": artifact_results if artifact_results else []
        }

    formatted_results = []
    if results and results.get("ids") and len(results["ids"]) > 0 and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
            distance = results["distances"][0][i] if results.get("distances") else 0

            formatted_results.append({
                "session_id": doc_id,
                "summary": metadata.get("summary", ""),
                "keywords": metadata.get("keywords", "").split(",") if metadata.get("keywords") else [],
                "relevance": 1.0 - (distance / 2.0),  # Convert distance to relevance
                "timestamp": metadata.get("timestamp", "")
            })

    # If semantic search found results, enrich them with archive excerpts
    if formatted_results:
        enriched_results = enrich_results_from_archives(formatted_results, query, mira_path)

        # Also search artifacts for matching structured content
        artifact_results = search_artifacts_for_query(query, limit=5)

        return {
            "results": enriched_results,
            "total": len(enriched_results),
            "artifacts": artifact_results if artifact_results else []
        }

    # Fallback: no semantic matches, search archives directly
    log(f"No semantic matches for '{query}', falling back to fulltext archive search")
    fallback_results = fulltext_search_archives(query, limit, mira_path)

    artifact_results = search_artifacts_for_query(query, limit=5)

    return {
        "results": fallback_results,
        "total": len(fallback_results),
        "search_type": "fulltext_fallback" if fallback_results else "no_results",
        "artifacts": artifact_results if artifact_results else []
    }


def enrich_results_from_archives(results: list, query: str, mira_path: Path) -> list:
    """
    Enrich search results with relevant excerpts from full conversation archives.

    This solves the embedding truncation problem: while we can only index ~900 chars
    for semantic search, we can search the FULL conversation archive for specific
    content once we know which conversations are relevant.
    """
    archives_path = mira_path / "archives"
    if not archives_path.exists():
        return results

    # Extract meaningful search terms from query
    query_terms = extract_query_terms(query)
    if not query_terms:
        return results

    enriched = []
    for result in results:
        session_id = result.get("session_id", "")
        archive_file = archives_path / f"{session_id}.jsonl"

        excerpts = []
        if archive_file.exists():
            try:
                excerpts = search_archive_for_excerpts(archive_file, query_terms, max_excerpts=3)
            except Exception as e:
                log(f"Error searching archive {session_id}: {e}")

        # Add excerpts to result
        result_copy = result.copy()
        result_copy["excerpts"] = excerpts
        result_copy["has_archive_matches"] = len(excerpts) > 0
        enriched.append(result_copy)

    return enriched


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


def fulltext_search_archives(query: str, limit: int, mira_path: Path) -> list:
    """
    Full-text search across all archived conversations.

    Used as fallback when semantic search returns no results.
    """
    archives_path = mira_path / "archives"
    metadata_path = mira_path / "metadata"

    if not archives_path.exists():
        return []

    # Extract search terms
    query_terms = extract_query_terms(query)
    if not query_terms:
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
                "message_count": str(metadata.get("message_count", 0))
            })

        if len(results) >= limit:
            break

    # Sort by number of excerpt matches
    results.sort(key=lambda x: len(x.get("excerpts", [])), reverse=True)

    return results[:limit]
