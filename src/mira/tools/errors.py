"""
MIRA Error Lookup Tool

Search past error patterns and their solutions.
"""

from mira.core import log


def handle_error_lookup(params: dict, storage=None) -> dict:
    """
    Search for past error solutions.

    Uses project-first search strategy:
    1. Search within current project first
    2. If no results, expand to search all projects globally

    Args:
        params: Request parameters (query, limit, project_path)
        storage: Storage instance for central Qdrant + Postgres

    Params:
        query: Error message or description to search for
        limit: Maximum results (default: 5)
        project_path: Optional project path to search first

    Returns matching errors with their solutions.
    """
    query = params.get("query", "")
    limit = params.get("limit", 5)
    project_path = params.get("project_path")

    if not query:
        return {"results": [], "total": 0}

    # Apply fuzzy matching for typo correction
    original_query = query
    corrections = []
    try:
        from mira.search.fuzzy import expand_query_with_corrections, get_vocabulary_size
        if get_vocabulary_size() > 0:
            corrected_query, corrections = expand_query_with_corrections(query)
            if corrections:
                query = corrected_query
                log(f"Error lookup fuzzy corrected: '{original_query}' → '{query}'")
    except Exception as e:
        log(f"Error lookup fuzzy matching failed: {e}")

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
                results = storage.search_error_patterns(query, project_id=project_id, limit=limit)

            # If no results and we had a project filter, search globally
            if not results:
                results = storage.search_error_patterns(query, project_id=None, limit=limit)
                searched_global = True if project_id else False

            if results:
                response = {
                    "solutions": results,
                    "total": len(results),
                    "query": query,
                    "source": "central" + ("_global" if searched_global else "")
                }
                if corrections:
                    response["corrections"] = corrections
                    response["original_query"] = original_query
                return response
        except Exception as e:
            log(f"Central error lookup failed: {e}")

    # Fall back to local search
    try:
        from mira.extraction.errors import search_error_solutions
        results = search_error_solutions(query, limit=limit)
    except ImportError:
        results = []

    response = {
        "solutions": results,
        "total": len(results),
        "query": query,
        "source": "local"
    }
    if corrections:
        response["corrections"] = corrections
        response["original_query"] = original_query

    # Add helpful message for empty results
    if not results:
        response["message"] = f"No past errors found matching '{query}'."
        response["suggestions"] = [
            "Try simpler keywords (e.g., 'TypeError' instead of full message)",
            "Error patterns are learned from past conversations",
            "Use mira_search for broader conversation search"
        ]

    return response
