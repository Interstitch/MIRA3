"""
MIRA Code History Tool

Track file changes across conversation sessions.
"""

from datetime import datetime, timedelta, timezone


def _normalize_file_path_pattern(file_path: str) -> str:
    """Convert user file path input to SQL LIKE pattern.

    - Converts * to % for glob patterns
    - Adds % prefix for simple filenames (not absolute paths)
    - Passes through absolute paths and existing % patterns
    """
    if not file_path:
        return file_path
    if '*' in file_path:
        return file_path.replace('*', '%')
    if not file_path.startswith('/') and '%' not in file_path:
        return '%' + file_path
    return file_path


def handle_code_history(params: dict, storage=None) -> dict:
    """
    Search code history by file path or symbol name.

    Provides three modes:
    - timeline: List of sessions that touched a file/symbol
    - snapshot: Reconstruct file content at a specific date
    - changes: List of edits made to a file

    Args:
        params: Request parameters
        storage: Storage instance (unused for local code history)

    Params:
        path: File path or pattern (supports % wildcards)
        symbol: Function/class name to search
        mode: "timeline" | "snapshot" | "changes" (default: timeline)
        date: Target date for snapshot mode (ISO format)
        limit: Maximum results (default: 20)

    Returns:
        Mode-specific response with file history data.
    """
    try:
        from mira.extraction.code_history import (
            get_file_timeline,
            get_symbol_history,
            reconstruct_file_at_date,
            get_edits_between,
            get_file_snapshot_at_date,
        )
    except ImportError:
        return {"error": "Code history module not available"}

    file_path = params.get("path", "")
    symbol = params.get("symbol", "")
    mode = params.get("mode", "timeline")
    target_date = params.get("date", "")
    limit = params.get("limit", 20)

    # Require at least one search criterion
    if not file_path and not symbol:
        return {
            "error": "Must provide 'path' or 'symbol' parameter",
            "usage": {
                "path": "File path or pattern (e.g., 'handlers.py', 'src/%.py')",
                "symbol": "Function/class name (e.g., 'handle_search')",
                "mode": "timeline | snapshot | changes",
                "date": "ISO date for snapshot mode (e.g., '2025-12-01')",
                "limit": "Max results (default 20)",
            }
        }

    # MODE: timeline - list of changes over time
    if mode == "timeline":
        if symbol and not file_path:
            # Symbol-only search
            results = get_symbol_history(symbol, limit=limit)
            return {
                "mode": "timeline",
                "symbol": symbol,
                "appearances": results,
                "total": len(results),
            }
        else:
            # File-based timeline (optionally filtered by symbol)
            search_path = _normalize_file_path_pattern(file_path)

            results = get_file_timeline(
                file_path=search_path,
                symbol=symbol if symbol else None,
                limit=limit
            )
            response = {
                "mode": "timeline",
                "file_path": file_path,
                "timeline": results,
                "total": len(results),
            }
            if symbol:
                response["filtered_by_symbol"] = symbol
            return response

    # MODE: snapshot - reconstruct file at a date
    elif mode == "snapshot":
        if not file_path:
            return {"error": "snapshot mode requires 'path' parameter"}
        if not target_date:
            return {"error": "snapshot mode requires 'date' parameter (ISO format)"}

        # Normalize file path for pattern matching
        search_path = _normalize_file_path_pattern(file_path)

        # Add end-of-day time if only date provided
        if len(target_date) == 10:  # YYYY-MM-DD format
            target_date = target_date + "T23:59:59.999Z"

        result = reconstruct_file_at_date(search_path, target_date)

        response = {
            "mode": "snapshot",
            "file_path": result.file_path,
            "target_date": result.target_date,
            "confidence": result.confidence,
        }

        if result.content:
            response["content"] = result.content
            response["line_count"] = result.content.count('\n') + 1
            response["source_snapshot_date"] = result.source_snapshot_date
            response["edits_applied"] = result.edits_applied
            if result.edits_failed > 0:
                response["edits_failed"] = result.edits_failed
            if result.gaps:
                response["gaps"] = result.gaps
        else:
            response["error"] = "Could not reconstruct file"
            response["gaps"] = result.gaps

        return response

    # MODE: changes - list of edits
    elif mode == "changes":
        if not file_path:
            return {"error": "changes mode requires 'path' parameter"}

        # Normalize file path for SQL LIKE pattern
        search_path = _normalize_file_path_pattern(file_path)

        if target_date:
            end_date = target_date
        else:
            end_date = datetime.now(timezone.utc).isoformat()

        # Get earliest snapshot as start date
        earliest_snapshot = get_file_snapshot_at_date(search_path, "2000-01-01")
        if earliest_snapshot:
            start_date = "2000-01-01"  # Get all history
        else:
            start_date = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()

        edits = get_edits_between(search_path, start_date, end_date)

        # Format edits for display
        changes = []
        for edit in edits[:limit]:
            change = {
                "date": edit.get("timestamp", ""),
                "session_id": edit.get("session_id", ""),
                "type": "edit",
            }
            # Truncate long strings for readability
            old_str = edit.get("old_string", "")
            new_str = edit.get("new_string", "")

            if len(old_str) > 200:
                change["before"] = old_str[:200] + "..."
                change["before_truncated"] = True
            else:
                change["before"] = old_str

            if len(new_str) > 200:
                change["after"] = new_str[:200] + "..."
                change["after_truncated"] = True
            else:
                change["after"] = new_str

            changes.append(change)

        return {
            "mode": "changes",
            "file_path": file_path,
            "changes": changes,
            "total": len(changes),
        }

    else:
        return {
            "error": f"Unknown mode: {mode}",
            "valid_modes": ["timeline", "snapshot", "changes"]
        }
