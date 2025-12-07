"""
MIRA3 RPC Request Handlers Module

Handles JSON-RPC requests from the Node.js MCP server.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter

from .utils import log, get_mira_path, get_custodian
from .search import handle_search
from .artifacts import get_artifact_stats
from .custodian import get_full_custodian_profile, get_danger_zones_for_files
from .insights import search_error_solutions, search_decisions, get_error_stats, get_decision_stats


def handle_recent(params: dict) -> dict:
    """Get recent conversation sessions."""
    limit = params.get("limit", 10)

    mira_path = get_mira_path()
    metadata_path = mira_path / "metadata"

    sessions = []
    if metadata_path.exists():
        for meta_file in sorted(metadata_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            try:
                meta = json.loads(meta_file.read_text())
                sessions.append({
                    "session_id": meta_file.stem,
                    "summary": meta.get("summary", ""),
                    "project_path": meta.get("project_path", ""),
                    "timestamp": meta.get("extracted_at", "")
                })
            except (json.JSONDecodeError, IOError, OSError):
                pass

    # Group by project
    projects = {}
    for session in sessions:
        project = session.get("project_path", "unknown")
        if project not in projects:
            projects[project] = []
        projects[project].append(session)

    return {
        "projects": [{"path": k, "sessions": v} for k, v in projects.items()],
        "total": len(sessions)
    }


def handle_init(params: dict, collection) -> dict:
    """
    Get comprehensive initialization context for the current session.

    Returns:
    - Storage statistics
    - Custodian information
    - Current work context
    - System health information
    """
    project_path = params.get("project_path", "")
    mira_path = get_mira_path()

    # Get recent sessions
    recent = handle_recent({"limit": 5})

    # Get collection stats
    count = collection.count()

    # Get artifact stats
    artifact_stats = get_artifact_stats()

    # Calculate storage sizes
    storage_stats = calculate_storage_stats(mira_path)

    # Get rich custodian profile (learned from conversations)
    custodian_profile = get_full_custodian_profile()

    # Merge with legacy profile stats
    legacy_profile = get_custodian_profile()
    custodian_profile['tech_stack'] = legacy_profile.get('tech_stack', [])
    custodian_profile['active_projects'] = legacy_profile.get('active_projects', [])
    custodian_profile['common_tools'] = legacy_profile.get('common_tools', [])
    custodian_profile['total_sessions'] = legacy_profile.get('total_sessions', 0)
    custodian_profile['total_messages'] = legacy_profile.get('total_messages', 0)

    # Get current work context
    work_context = get_current_work_context()

    return {
        "message": "MIRA3 initialized",
        "indexed_conversations": count,
        "recent_sessions": recent.get("projects", []),
        "custodian": custodian_profile,
        "storage": storage_stats,
        "artifacts": artifact_stats,
        "current_work": work_context
    }


def calculate_storage_stats(mira_path: Path) -> dict:
    """Calculate storage usage for .mira directory."""
    def get_dir_size(path: Path) -> int:
        total = 0
        if path.exists():
            for item in path.rglob('*'):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except (OSError, PermissionError):
                        pass
        return total

    def format_size(bytes_size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f} TB"

    components = {
        'venv': get_dir_size(mira_path / '.venv'),
        'chroma': get_dir_size(mira_path / 'chroma'),
        'archives': get_dir_size(mira_path / 'archives'),
        'metadata': get_dir_size(mira_path / 'metadata'),
        'models': get_dir_size(mira_path / 'models'),
    }

    artifacts_db = mira_path / 'artifacts.db'
    if artifacts_db.exists():
        components['artifacts_db'] = artifacts_db.stat().st_size
    else:
        components['artifacts_db'] = 0

    total_mira = sum(components.values())

    # Calculate codebase size
    codebase_path = mira_path.parent
    codebase_size = 0
    excluded_dirs = {'.mira', '.git', 'node_modules', '.venv', '__pycache__', 'dist', '.next', '.cache'}

    if codebase_path.exists():
        for item in codebase_path.rglob('*'):
            if item.is_file():
                skip = False
                for parent in item.parents:
                    if parent.name in excluded_dirs:
                        skip = True
                        break
                if not skip:
                    try:
                        codebase_size += item.stat().st_size
                    except (OSError, PermissionError):
                        pass

    # Calculate data-only storage (excluding venv which is just dependencies)
    data_storage = total_mira - components['venv']
    ratio = (data_storage / codebase_size * 100) if codebase_size > 0 else 0

    # Create a more meaningful note
    if codebase_size > 0:
        if ratio < 10:
            note = f"MIRA data is minimal ({ratio:.1f}% of codebase)"
        elif ratio < 50:
            note = f"MIRA data is {ratio:.1f}% of codebase size"
        else:
            note = f"MIRA data is larger than codebase ({format_size(data_storage)})"
    else:
        note = "Could not calculate codebase size"

    return {
        'total_mira': format_size(total_mira),
        'total_mira_bytes': total_mira,
        'data_storage': format_size(data_storage),
        'data_storage_bytes': data_storage,
        'codebase': format_size(codebase_size),
        'codebase_bytes': codebase_size,
        'ratio_percent': round(ratio, 1),
        'components': {
            'venv': format_size(components['venv']),
            'chroma': format_size(components['chroma']),
            'archives': format_size(components['archives']),
            'metadata': format_size(components['metadata']),
            'models': format_size(components['models']),
            'artifacts_db': format_size(components['artifacts_db']),
        },
        'note': note
    }


def get_custodian_profile() -> dict:
    """Build a profile of the custodian based on conversation history."""
    mira_path = get_mira_path()
    metadata_path = mira_path / "metadata"

    profile = {
        'name': get_custodian(),
        'tech_stack': [],
        'active_projects': [],
        'total_sessions': 0,
        'total_messages': 0,
    }

    if not metadata_path.exists():
        return profile

    tech_counter = Counter()
    project_counter = Counter()
    total_messages = 0
    tools_used = Counter()

    for meta_file in metadata_path.glob("*.json"):
        try:
            meta = json.loads(meta_file.read_text())

            project = meta.get('project_path', 'unknown')
            if project:
                project_counter[project] += 1

            keywords = meta.get('keywords', [])
            for kw in keywords[:10]:
                tech_counter[kw.lower()] += 1

            session_tools = meta.get('tools_used', {})
            for tool, count in session_tools.items():
                tools_used[tool] += count

            total_messages += meta.get('message_count', 0)
            profile['total_sessions'] += 1

        except Exception:
            pass

    profile['total_messages'] = total_messages

    # Filter tech stack - remove generic/common words
    tech_filter = {
        # Common words
        'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'been',
        'code', 'file', 'files', 'data', 'line', 'text', 'string', 'list',
        # Action words
        'fix', 'add', 'remove', 'update', 'change', 'make', 'get', 'set',
        'create', 'delete', 'run', 'check', 'test', 'use', 'using',
        # Context words
        'search', 'function', 'method', 'class', 'module', 'command',
        'error', 'issue', 'problem', 'solution', 'result', 'output',
        # Descriptors
        'new', 'old', 'first', 'last', 'current', 'previous', 'next',
        'extraction', 'inaccuracies', 'implement', 'implementation',
        # More generic words from actual output
        'please', 'gets', 'deep', 'dive', 'model', 'include', 'index',
        'idf', 'should', 'would', 'need', 'want', 'will', 'can', 'may',
        'information', 'ensure', 'analyze', 'just', 'also', 'specific',
        'look', 'want', 'work', 'about', 'into', 'here', 'there',
        'indexing', 'technologies', 'perform', 'before', 'after', 'server',
        'being', 'since', 'conversation', 'conversations', 'messages',
        'summary', 'keywords', 'embedding', 'embeddings', 'vector', 'vectors',
        'phase', 'stdin', 'inside', 'best', 'source', 'more', 'over', 'under',
        'only', 'same', 'each', 'other', 'well', 'done', 'good', 'like',
        'write', 'read', 'call', 'find', 'show', 'type', 'name', 'path',
    }
    # Only include terms that look like technology names (usually lowercase, no common verbs)
    profile['tech_stack'] = [
        tech for tech, count in tech_counter.most_common(50)
        if tech not in tech_filter and len(tech) > 3 and not tech.endswith('ing')
    ][:10]

    profile['active_projects'] = [
        {'path': proj, 'sessions': count}
        for proj, count in project_counter.most_common(5)
    ]

    profile['common_tools'] = [
        {'tool': tool, 'uses': count}
        for tool, count in tools_used.most_common(5)
    ]

    return profile


def get_current_work_context() -> dict:
    """Get context about current/recent work."""
    mira_path = get_mira_path()
    metadata_path = mira_path / "metadata"

    context = {
        'recent_tasks': [],
        'active_topics': [],
        'recent_decisions': [],
    }

    if not metadata_path.exists():
        return context

    recent_files = sorted(
        metadata_path.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:5]

    for meta_file in recent_files:
        try:
            meta = json.loads(meta_file.read_text())

            task = meta.get('task_description', '')
            # Skip command messages and empty tasks
            if task and task not in context['recent_tasks']:
                if not task.startswith('<command-') and not task.startswith('/'):
                    context['recent_tasks'].append(task)

            todos = meta.get('todo_topics', [])
            for todo in todos[:3]:
                if todo not in context['active_topics']:
                    context['active_topics'].append(todo)

            facts = meta.get('key_facts', [])
            for fact in facts[:2]:
                if fact not in context['recent_decisions']:
                    context['recent_decisions'].append(fact)

        except Exception:
            pass

    context['recent_tasks'] = context['recent_tasks'][:5]
    context['active_topics'] = context['active_topics'][:10]
    context['recent_decisions'] = context['recent_decisions'][:5]

    return context


def handle_status(collection) -> dict:
    """Get system status and statistics."""
    mira_path = get_mira_path()
    claude_path = Path.home() / ".claude" / "projects"

    # Count source files
    total_files = 0
    if claude_path.exists():
        total_files = sum(1 for _ in claude_path.rglob("*.jsonl"))

    # Count archived files
    archives_path = mira_path / "archives"
    archived = sum(1 for _ in archives_path.glob("*.jsonl")) if archives_path.exists() else 0

    # Count indexed
    indexed = collection.count()

    # Get insights stats
    error_stats = get_error_stats()
    decision_stats = get_decision_stats()

    return {
        "total_files": total_files,
        "archived": archived,
        "indexed": indexed,
        "pending": total_files - archived,
        "storage_path": str(mira_path),
        "last_sync": datetime.now().isoformat(),
        "errors": error_stats,
        "decisions": decision_stats
    }


def handle_error_lookup(params: dict) -> dict:
    """
    Search for past error solutions.

    Params:
        query: Error message or description to search for
        limit: Maximum results (default: 5)

    Returns matching errors with their solutions.
    """
    query = params.get("query", "")
    limit = params.get("limit", 5)

    if not query:
        return {"results": [], "total": 0}

    results = search_error_solutions(query, limit=limit)

    return {
        "results": results,
        "total": len(results),
        "query": query
    }


def handle_decisions(params: dict) -> dict:
    """
    Search for past architectural/design decisions.

    Params:
        query: Search query
        category: Optional category filter (architecture, technology, etc.)
        limit: Maximum results (default: 10)

    Returns matching decisions with context.
    """
    query = params.get("query", "")
    category = params.get("category")
    limit = params.get("limit", 10)

    if not query:
        # Return recent decisions if no query
        results = search_decisions("", category=category, limit=limit)
    else:
        results = search_decisions(query, category=category, limit=limit)

    return {
        "results": results,
        "total": len(results),
        "query": query,
        "category": category
    }


def handle_rpc_request(request: dict, collection) -> dict:
    """Handle a JSON-RPC request and return response."""
    method = request.get("method", "")
    params = request.get("params", {})
    request_id = request.get("id")

    result = None
    error = None

    try:
        if method == "search":
            result = handle_search(params, collection)
        elif method == "recent":
            result = handle_recent(params)
        elif method == "init":
            result = handle_init(params, collection)
        elif method == "status":
            result = handle_status(collection)
        elif method == "error_lookup":
            result = handle_error_lookup(params)
        elif method == "decisions":
            result = handle_decisions(params)
        elif method == "shutdown":
            result = {"status": "shutting_down"}
        else:
            error = {"code": -32601, "message": f"Method not found: {method}"}
    except Exception as e:
        # Log full error for debugging, return sanitized message
        log(f"RPC handler error for {method}: {e}")
        error = {"code": -32603, "message": "Internal error processing request"}

    response = {"jsonrpc": "2.0", "id": request_id}
    if error:
        response["error"] = error
    else:
        response["result"] = result

    return response
