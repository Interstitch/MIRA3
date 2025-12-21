"""
MIRA Search Package

Provides multi-tier search capabilities:
- Tier 1: Remote semantic (Qdrant + embedding-service)
- Tier 2: Local semantic (sqlite-vec + fastembed)
- Tier 3: FTS5 keyword (SQLite)

Also includes fuzzy matching for typo correction.
"""

from .core import (
    handle_search,
    enrich_results_from_archives,
    fulltext_search_archives,
    search_archive_for_excerpts,
    search_archive_content_for_excerpts,
    extract_excerpt_around_terms,
)

from .fuzzy import (
    damerau_levenshtein_distance,
    find_closest_match,
    expand_query_with_corrections,
    add_terms_to_vocabulary,
    extract_terms_from_text,
    get_vocabulary_size,
    is_in_vocabulary,
    VOCABULARY_SCHEMA,
)

from .local_semantic import (
    LocalSemanticSearch,
    get_local_semantic,
    is_local_semantic_available,
    trigger_local_semantic_download,
    queue_session_for_indexing,
    get_pending_indexing_count,
    LocalSemanticIndexer,
    start_local_indexer,
    stop_local_indexer,
    LOCAL_VECTORS_SCHEMA,
)

__all__ = [
    # Core search
    "handle_search",
    "enrich_results_from_archives",
    "fulltext_search_archives",
    "search_archive_for_excerpts",
    "search_archive_content_for_excerpts",
    "extract_excerpt_around_terms",
    # Fuzzy matching
    "damerau_levenshtein_distance",
    "find_closest_match",
    "expand_query_with_corrections",
    "add_terms_to_vocabulary",
    "extract_terms_from_text",
    "get_vocabulary_size",
    "is_in_vocabulary",
    "VOCABULARY_SCHEMA",
    # Local semantic
    "LocalSemanticSearch",
    "get_local_semantic",
    "is_local_semantic_available",
    "trigger_local_semantic_download",
    "queue_session_for_indexing",
    "get_pending_indexing_count",
    "LocalSemanticIndexer",
    "start_local_indexer",
    "stop_local_indexer",
    "LOCAL_VECTORS_SCHEMA",
]
