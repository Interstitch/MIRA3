"""
MIRA Decisions Tool

Search architectural and design decisions with reasoning.
"""

from mira.core import log


def handle_decisions(params: dict, storage=None) -> dict:
    """
    Search for past architectural/design decisions.

    Uses project-first search strategy:
    1. Search within current project first
    2. If no results, expand to search all projects globally

    Args:
        params: Request parameters (query, category, limit, project_path, min_confidence)
        storage: Storage instance for central Qdrant + Postgres

    Params:
        query: Search query
        category: Optional category filter (architecture, technology, etc.)
        limit: Maximum results (default: 10)
        project_path: Optional project path to search first
        min_confidence: Minimum confidence threshold (0.0-1.0, default: 0.0)
                       Use 0.8+ for explicit decisions only, 0.6+ to include implicit

    Returns matching decisions with context.
    """
    query = params.get("query", "")
    category = params.get("category")
    limit = params.get("limit", 10)
    project_path = params.get("project_path")
    min_confidence = params.get("min_confidence", 0.0)

    # Apply fuzzy matching for typo correction
    original_query = query
    corrections = []
    if query:  # Only if query provided
        try:
            from mira.search.fuzzy import expand_query_with_corrections, get_vocabulary_size
            if get_vocabulary_size() > 0:
                corrected_query, corrections = expand_query_with_corrections(query)
                if corrections:
                    query = corrected_query
                    log(f"Decisions fuzzy corrected: '{original_query}' → '{query}'")
        except Exception as e:
            log(f"Decisions fuzzy matching failed: {e}")

    # Try central storage with project-first strategy
    if storage and storage.using_central:
        try:
            # Get project_id if project_path provided
            project_id = None
            if project_path:
                project_id = storage.get_project_id(project_path)

            # First: search within project only
            results = []
            searched_global = False
            if project_id:
                results = storage.search_decisions(
                    query=query or "",
                    project_id=project_id,
                    category=category,
                    limit=limit
                )

            # If no results and we had a project filter, search globally
            if not results:
                results = storage.search_decisions(
                    query=query or "",
                    project_id=None,
                    category=category,
                    limit=limit
                )
                searched_global = True if project_id else False

            if results:
                response = {
                    "decisions": results,
                    "total": len(results),
                    "query": query,
                    "category": category,
                    "source": "central" + ("_global" if searched_global else "")
                }
                if corrections:
                    response["corrections"] = corrections
                    response["original_query"] = original_query
                return response
        except Exception as e:
            log(f"Central decisions search failed: {e}")

    # Fall back to local search
    try:
        from mira.extraction.decisions import search_decisions
        if not query:
            results = search_decisions("", category=category, limit=limit, min_confidence=min_confidence)
        else:
            results = search_decisions(query, category=category, limit=limit, min_confidence=min_confidence)
    except ImportError:
        results = []

    response = {
        "decisions": results,
        "total": len(results),
        "query": query,
        "category": category,
        "min_confidence": min_confidence,
        "source": "local"
    }
    if corrections:
        response["corrections"] = corrections
        response["original_query"] = original_query

    # Add helpful message for empty results
    if not results:
        response["message"] = f"No past decisions found matching '{query}'."
        response["suggestions"] = [
            "Record decisions explicitly: 'Decision: use PostgreSQL for the database'",
            "Try broader keywords or remove category filter",
            "Lower min_confidence to include implicit decisions"
        ]

    return response
