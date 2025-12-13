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
from .artifacts import get_artifact_stats, get_journey_stats
from .custodian import get_full_custodian_profile, get_danger_zones_for_files, get_custodian_stats
from .rules import format_rule_for_display, RULE_TYPES
from .insights import search_error_solutions, search_decisions, get_error_stats, get_decision_stats
from .concepts import get_codebase_knowledge, ConceptStore, get_concepts_stats
from .ingestion import get_active_ingestions
from .guidance import (
    build_claude_guidance,
    filter_codebase_knowledge,
    get_actionable_alerts,
    get_simplified_storage_stats,
    build_enriched_custodian_summary,
    build_interaction_tips,
)
from .work_context import get_current_work_context


def _format_project_path(encoded_path: str) -> str:
    """Convert encoded project path to readable format."""
    if not encoded_path:
        return "unknown"
    # Convert -workspaces-MIRA3 to /workspaces/MIRA3
    readable = encoded_path.replace('-', '/')
    # Handle leading slash
    if not readable.startswith('/'):
        readable = '/' + readable
    return readable


def handle_recent(params: dict, storage=None) -> dict:
    """Get recent conversation sessions.

    Tries central Postgres first, falls back to local SQLite, then local metadata files.

    Args:
        params: dict with 'limit' (int) and optional 'days' (int) to filter by time
        storage: Storage instance
    """
    from datetime import datetime, timedelta

    limit = params.get("limit", 10)
    days = params.get("days")  # Optional: filter to last N days

    # Calculate cutoff time if days specified
    cutoff_time = None
    if days is not None and days > 0:
        cutoff_time = datetime.now() - timedelta(days=days)

    # Try storage (central or local SQLite via storage abstraction)
    if storage:
        try:
            sessions = storage.get_recent_sessions(limit=limit, since=cutoff_time)
            if sessions:
                # Group by project
                projects = {}
                for session in sessions:
                    project = session.get("project_path", "unknown")
                    if project not in projects:
                        projects[project] = []
                    projects[project].append({
                        "session_id": session.get("session_id", ""),
                        "summary": session.get("summary", ""),
                        "project_path": project,
                        "timestamp": str(session.get("started_at", "")),
                        "accomplishments": session.get("accomplishments", []),
                    })

                source = "central" if storage.using_central else "local_sqlite"
                result = {
                    "projects": [{"path": k, "sessions": v} for k, v in projects.items()],
                    "total": len(sessions),
                    "source": source
                }
                if days:
                    result["filtered_to_days"] = days
                return result
        except Exception as e:
            log(f"Storage recent query failed: {e}")

    # Fallback to local metadata files
    mira_path = get_mira_path()
    metadata_path = mira_path / "metadata"

    sessions = []
    if metadata_path.exists():
        for meta_file in sorted(metadata_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            # Check time filter first (before limit)
            if cutoff_time:
                file_mtime = datetime.fromtimestamp(meta_file.stat().st_mtime)
                if file_mtime < cutoff_time:
                    continue  # Skip files older than cutoff

            if len(sessions) >= limit:
                break

            try:
                meta = json.loads(meta_file.read_text())
                raw_path = meta.get("project_path", "")
                sessions.append({
                    "session_id": meta_file.stem,
                    "summary": meta.get("summary", ""),
                    "project_path": _format_project_path(raw_path),
                    "timestamp": meta.get("extracted_at", ""),
                    "accomplishments": meta.get("accomplishments", []),
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

    result = {
        "projects": [{"path": k, "sessions": v} for k, v in projects.items()],
        "total": len(sessions),
        "source": "local"
    }
    if days:
        result["filtered_to_days"] = days
    return result


def handle_init(params: dict, collection, storage=None) -> dict:
    """
    Get comprehensive initialization context for the current session.

    Returns TIERED output:
    - TIER 1 (alerts): Actionable items requiring attention
    - TIER 2 (core): Essential context for immediate work
    - TIER 3 (details): Deeper context when needed

    The output is organized to prioritize actionable information first.

    Args:
        params: Request parameters (project_path)
        collection: Deprecated - kept for API compatibility, ignored
        storage: Storage instance for central Qdrant + Postgres
    """
    project_path = params.get("project_path", "")
    mira_path = get_mira_path()

    # Get indexed count from storage (central or local)
    count = 0
    if storage and storage.using_central and storage.postgres:
        try:
            # Count sessions in postgres (central mode)
            with storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM sessions")
                    count = cur.fetchone()[0]
        except Exception:
            pass
    else:
        # Count sessions in local SQLite
        try:
            from .local_store import get_session_count
            count = get_session_count()
        except Exception:
            pass

    # Get artifact stats (global and project-scoped)
    artifact_stats = get_artifact_stats(project_path)

    # Get insights stats for guidance (decision count for triggers)
    decision_stats = get_decision_stats()
    decision_count = decision_stats.get('total', 0)

    # Calculate storage sizes (simplified)
    storage_stats = get_simplified_storage_stats(mira_path)

    # Get rich custodian profile (learned from conversations)
    custodian_profile = get_full_custodian_profile()

    # Merge with legacy profile stats (for total_sessions used in summary)
    legacy_profile = get_custodian_profile()
    custodian_profile['total_sessions'] = legacy_profile.get('total_sessions', 0)
    custodian_profile['total_messages'] = legacy_profile.get('total_messages', 0)

    # Build a richer summary now that we have all the data
    custodian_profile['summary'] = build_enriched_custodian_summary(custodian_profile)

    # Add interaction tips based on learned preferences
    custodian_profile['interaction_tips'] = build_interaction_tips(custodian_profile)

    # Get current work context (filtered by project)
    work_context = get_current_work_context(project_path)

    # Get codebase knowledge learned from conversations
    # NOTE: This only returns genuinely learned content - no CLAUDE.md echoes
    codebase_knowledge = get_codebase_knowledge(project_path)

    # === BUILD TIERED OUTPUT ===

    # TIER 1: Actionable alerts (always check these)
    alerts = get_actionable_alerts(mira_path, project_path, custodian_profile)

    # TIER 2: Core context (essential for immediate work)
    # Custodian - behavioral info only (tech_stack is in CLAUDE.md, not actionable here)
    custodian_data = {
        'name': custodian_profile.get('name', 'Unknown'),
        'summary': custodian_profile.get('summary', ''),
        'interaction_tips': custodian_profile.get('interaction_tips', [])[:5],
        'total_sessions': custodian_profile.get('total_sessions', 0),
    }

    # Add development lifecycle if detected (this is key behavioral context)
    dev_lifecycle = custodian_profile.get('development_lifecycle')
    if dev_lifecycle:
        confidence_pct = int(dev_lifecycle.get('confidence', 0) * 100)
        custodian_data['development_lifecycle'] = f"{dev_lifecycle.get('sequence')} ({confidence_pct}% confidence)"

    # Only include danger_zones if non-empty
    danger_zones = custodian_profile.get('danger_zones', [])
    if danger_zones:
        custodian_data['danger_zones'] = danger_zones

    # NOTE: Removed journey_data (hot_files with operation counts) - not actionable.
    # The data is used internally for danger_zone detection but raw counts don't
    # help Claude make decisions. Files under active development are flagged via alerts.

    # TIER 3: Detailed context (for deep-dive when needed)
    # Project info - only non-CLAUDE.md data (description/commands already in CLAUDE.md)
    # Only include if we have meaningful learned knowledge

    # Filter codebase_knowledge to remove empty arrays and CLAUDE.md duplicates
    filtered_knowledge = filter_codebase_knowledge(codebase_knowledge)

    # Only include details if there's meaningful learned knowledge
    has_meaningful_knowledge = any([
        filtered_knowledge.get('integrations'),
        filtered_knowledge.get('patterns'),
        filtered_knowledge.get('facts'),
        filtered_knowledge.get('rules'),
    ])

    details = {}
    if has_meaningful_knowledge:
        details['codebase_knowledge'] = filtered_knowledge

    # Artifact stats - add to guidance if significant (tells Claude there's searchable history)
    # artifact_stats has 'global' and 'project' keys when project_path is provided
    if 'global' in artifact_stats:
        global_artifact_total = artifact_stats['global'].get('total', 0)
        global_error_count = artifact_stats['global'].get('by_type', {}).get('error', 0)
        project_artifact_total = artifact_stats['project'].get('total', 0)
        project_error_count = artifact_stats['project'].get('by_type', {}).get('error', 0)
    else:
        # Backwards compatible: flat stats (no project_path provided)
        global_artifact_total = artifact_stats.get('total', 0)
        global_error_count = artifact_stats.get('by_type', {}).get('error', 0)
        project_artifact_total = 0
        project_error_count = 0

    # Check if storage is concerning (>500MB data or >1GB total)
    data_bytes = storage_stats.get('data_bytes', 0)
    models_bytes = storage_stats.get('models_bytes', 0)
    total_bytes = data_bytes + models_bytes

    if data_bytes > 500 * 1024 * 1024:  # >500MB data
        alerts.append({
            'type': 'storage_warning',
            'priority': 'medium',
            'message': f"MIRA data storage is large: {storage_stats['data']}",
            'suggestion': 'Consider pruning old archives with mira_status',
        })
    elif total_bytes > 1024 * 1024 * 1024:  # >1GB total
        alerts.append({
            'type': 'storage_info',
            'priority': 'low',
            'message': f"MIRA total storage: {storage_stats['data']} data + {storage_stats['models']} models",
        })

    # Build guidance for Claude on how to use this context
    guidance = build_claude_guidance(
        custodian_data, alerts, work_context,
        global_artifact_total=global_artifact_total, global_error_count=global_error_count,
        project_artifact_total=project_artifact_total, project_error_count=project_error_count,
        decision_count=decision_count
    )

    # Get storage mode information
    storage_mode = None
    if storage:
        storage_mode = storage.get_storage_mode()
    else:
        storage_mode = {
            "mode": "local",
            "description": "Using local SQLite storage (keyword search only, single-machine)",
            "limitations": [
                "Keyword search only (no semantic/vector search)",
                "History stays on this machine only",
                "Codebase concepts not available",
            ],
            "setup": {
                "summary": "To enable semantic search and cross-machine sync, set up central storage.",
                "note": "Central storage is optional. MIRA works in local mode with keyword search."
            }
        }

    # Build response - only include fields that provide value
    response = {
        # GUIDANCE: How Claude should use this context (most important)
        "guidance": guidance,

        # TIER 1: Alerts - check these first
        "alerts": alerts,

        # TIER 2: Core context (no duplication - single source of truth)
        "core": {
            "custodian": custodian_data,
            "current_work": work_context,
        },

        # Storage mode - always show so user knows sync status
        "storage": storage_mode,
    }

    # Add alert if using local storage (for new user awareness)
    if storage_mode.get("mode") == "local":
        limitations = storage_mode.get("limitations", [])
        alerts.insert(0, {
            "type": "storage_mode",
            "priority": "info",
            "message": "Running in local mode (keyword search only, single-machine).",
            "limitations": limitations,
            "note": "Set up central storage for semantic search and cross-machine sync.",
        })

    # Show indexing progress if not fully caught up
    claude_path = Path.home() / ".claude" / "projects"
    total_files = sum(1 for _ in claude_path.rglob("*.jsonl")) if claude_path.exists() else 0

    if count < total_files:
        pending = total_files - count
        response["indexing"] = {
            "indexed": count,
            "total": total_files,
            "pending": pending,
            "storage_mode": storage_mode.get("mode", "local"),
        }
        if pending > 0:
            response["guidance"]["actions"].append(
                f"Indexing in progress: {count}/{total_files} sessions ({pending} pending)"
            )
    elif count < 5:
        response["indexing"] = {
            "indexed": count,
            "storage_mode": storage_mode.get("mode", "local"),
        }
        response["guidance"]["actions"].append(
            f"Limited history: only {count} indexed sessions. Context may be sparse."
        )

    # Check for active ingestion jobs and inform Claude
    active_jobs = get_active_ingestions()
    if active_jobs:
        # Add ingestion status to response
        response["active_ingestion"] = {
            "count": len(active_jobs),
            "jobs": [
                {
                    "session_id": job["session_id"][:12] + "...",
                    "project": job.get("project_path", "").split("-")[-1] if job.get("project_path") else "unknown",
                    "elapsed_sec": round(job.get("elapsed_ms", 0) / 1000, 1),
                    "worker": job.get("worker", "unknown"),
                }
                for job in active_jobs[:5]  # Limit to 5 jobs in output
            ],
        }

        # Add alert for ongoing ingestion
        if len(active_jobs) == 1:
            job = active_jobs[0]
            elapsed = round(job.get("elapsed_ms", 0) / 1000, 1)
            alerts.insert(0, {
                "type": "ingestion_active",
                "priority": "info",
                "message": f"MIRA is currently ingesting 1 conversation ({elapsed}s elapsed)",
                "suggestion": "Search results may be incomplete until ingestion finishes",
            })
        else:
            alerts.insert(0, {
                "type": "ingestion_active",
                "priority": "info",
                "message": f"MIRA is currently ingesting {len(active_jobs)} conversations",
                "suggestion": "Search results may be incomplete until ingestion finishes",
            })

        # Add guidance for Claude to inform user
        response["guidance"]["actions"].insert(0,
            f"[ACTIVE INGESTION] {len(active_jobs)} conversation(s) being processed. "
            "Inform user that search results may be incomplete if they search for very recent work."
        )

    # Only include details if there's meaningful learned knowledge
    if details:
        response["details"] = details

    # Add estimated token count to help users understand context usage
    # Using ~4 chars per token as rough approximation (GPT/Claude typical)
    response_json = json.dumps(response, separators=(',', ':'))
    char_count = len(response_json)
    estimated_tokens = char_count // 4  # Conservative estimate
    response["token_estimate"] = {
        "chars": char_count,
        "tokens": estimated_tokens,
        "note": "Estimated tokens injected into context (~4 chars/token)"
    }

    return response


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

    # Sync name candidates from central to local for resilience
    try:
        from .custodian import sync_from_central
        sync_from_central()
    except Exception as e:
        log.debug(f"Central sync skipped: {e}")

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

            # Increment session count FIRST (before any code that might fail)
            profile['total_sessions'] += 1
            total_messages += meta.get('message_count', 0)

            # tools_used can be a list OR dict depending on metadata version
            session_tools = meta.get('tools_used', {})
            if isinstance(session_tools, dict):
                for tool, count in session_tools.items():
                    tools_used[tool] += count
            elif isinstance(session_tools, list):
                # Legacy format: list of tool names without counts
                for tool in session_tools:
                    if tool:  # Skip null/None entries
                        tools_used[tool] += 1

        except Exception:
            pass

    profile['total_messages'] = total_messages

    # Use allowlist approach - only recognize actual technology names
    # This is more reliable than trying to blocklist all possible noise words
    known_technologies = {
        # Databases
        'chromadb', 'postgresql', 'postgres', 'mysql', 'mongodb', 'redis',
        'sqlite', 'elasticsearch', 'dynamodb', 'cassandra', 'neo4j',
        # Languages
        'python', 'typescript', 'javascript', 'rust', 'golang', 'java',
        'ruby', 'swift', 'kotlin', 'scala', 'haskell', 'elixir',
        # Frameworks - Backend
        'fastapi', 'django', 'flask', 'express', 'nestjs', 'rails',
        'spring', 'actix', 'axum', 'gin', 'echo', 'fiber',
        # Frameworks - Frontend
        'react', 'vue', 'angular', 'svelte', 'nextjs', 'nuxt', 'remix',
        'solidjs', 'preact', 'astro', 'gatsby',
        # ML/AI
        'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'sklearn',
        'transformers', 'langchain', 'openai', 'anthropic', 'huggingface',
        'sentence-transformers', 'minilm', 'pinecone', 'weaviate',
        # Tools
        'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins',
        'github', 'gitlab', 'circleci', 'webpack', 'vite', 'esbuild',
        'pnpm', 'yarn', 'npm', 'poetry', 'pipenv', 'conda',
        # Protocols/Standards
        'graphql', 'grpc', 'rest', 'websocket', 'mqtt', 'kafka',
        'rabbitmq', 'zeromq', 'nats', 'mcp',
        # Testing
        'pytest', 'jest', 'vitest', 'mocha', 'cypress', 'playwright',
        # Other common tech
        'nginx', 'apache', 'caddy', 'traefik', 'haproxy',
        'aws', 'gcp', 'azure', 'vercel', 'netlify', 'heroku',
        'linux', 'macos', 'windows', 'ubuntu', 'debian',
        'git', 'vim', 'neovim', 'emacs', 'vscode',
        'watchdog', 'celery', 'dramatiq', 'rq',
    }

    # Map file extensions to technology names (keywords extract extensions)
    extension_to_tech = {
        'py': 'Python',
        'ts': 'TypeScript',
        'tsx': 'TypeScript',
        'js': 'JavaScript',
        'jsx': 'JavaScript',
        'rs': 'Rust',
        'go': 'Go',
        'java': 'Java',
        'rb': 'Ruby',
        'swift': 'Swift',
        'kt': 'Kotlin',
        'scala': 'Scala',
        'ex': 'Elixir',
        'exs': 'Elixir',
        'hs': 'Haskell',
        'vue': 'Vue',
        'svelte': 'Svelte',
    }

    # Build tech stack from both direct matches and extension mappings
    tech_stack = []
    seen = set()

    for tech, count in tech_counter.most_common(50):
        tech_lower = tech.lower()

        # Direct match against known technologies
        if tech_lower in known_technologies:
            if tech_lower not in seen:
                tech_stack.append(tech_lower)
                seen.add(tech_lower)

        # Map file extensions to technology names
        elif tech_lower in extension_to_tech:
            mapped = extension_to_tech[tech_lower]
            if mapped.lower() not in seen:
                tech_stack.append(mapped)
                seen.add(mapped.lower())

        if len(tech_stack) >= 10:
            break

    profile['tech_stack'] = tech_stack

    profile['active_projects'] = [
        {'path': _format_project_path(proj), 'sessions': count}
        for proj, count in project_counter.most_common(5)
    ]

    profile['common_tools'] = [
        {'tool': tool, 'uses': count}
        for tool, count in tools_used.most_common(5)
    ]

    return profile


# Work context functions moved to work_context.py


def handle_status(params: dict, collection, storage=None) -> dict:
    """
    Get system status and statistics.

    Returns both project-scoped and global statistics when project_path is provided.

    Args:
        params: Request parameters (project_path for scoped stats)
        collection: Deprecated - kept for API compatibility, ignored
        storage: Storage instance for central Qdrant + Postgres
    """
    project_path = params.get("project_path") if params else None
    mira_path = get_mira_path()
    claude_path = Path.home() / ".claude" / "projects"

    # Get project_id for scoped queries
    project_id = None
    if project_path and storage and storage.using_central:
        try:
            project_id = storage.get_project_id(project_path)
        except Exception:
            pass

    # Count and categorize source files - global and project-scoped
    # All files with 3+ user/assistant messages are worth indexing
    file_categories = {
        'sessions': 0,         # Main session files with messages
        'agents': 0,           # Subagent task files with messages
        'no_messages': 0,      # Files with 0 user/assistant messages (snapshots only)
        'minimal': 0,          # 1-2 messages (likely abandoned sessions)
    }
    project_files = 0
    project_conversations = 0
    indexable_session_ids = []  # Track session IDs of indexable files
    project_session_ids = []    # Track session IDs for this project

    if claude_path.exists():
        encoded_project = project_path.replace('/', '-').lstrip('-') if project_path else None

        for f in claude_path.rglob("*.jsonl"):
            is_project_file = encoded_project and encoded_project in str(f.parent)
            is_agent_file = f.name.startswith('agent-')
            session_id = f.stem

            # Count messages (user/assistant only, not file-history-snapshot etc.)
            try:
                msg_count = 0
                with f.open() as fp:
                    for line in fp:
                        try:
                            data = json.loads(line)
                            if data.get('type') in ('user', 'assistant'):
                                msg_count += 1
                                if msg_count > 2:  # Early exit once we know it's a real conversation
                                    break
                        except (json.JSONDecodeError, KeyError):
                            pass

                if msg_count == 0:
                    file_categories['no_messages'] += 1
                elif msg_count <= 2:
                    file_categories['minimal'] += 1
                else:
                    # Worth indexing - categorize by file type
                    if is_agent_file:
                        file_categories['agents'] += 1
                    else:
                        file_categories['sessions'] += 1
                    indexable_session_ids.append(session_id)
                    if is_project_file:
                        project_conversations += 1
                        project_session_ids.append(session_id)

                if is_project_file:
                    project_files += 1
            except (IOError, OSError):
                file_categories['no_messages'] += 1
                if is_project_file:
                    project_files += 1

    total_files = sum(file_categories.values())
    indexable_files = file_categories['sessions'] + file_categories['agents']

    # Count archived files
    archives_path = mira_path / "archives"
    archived = sum(1 for _ in archives_path.glob("*.jsonl")) if archives_path.exists() else 0

    # Count indexed from storage - check which current files are indexed
    indexed_global = 0
    indexed_project = 0
    indexed_of_indexable = 0      # How many of current indexable files are in DB
    indexed_of_project = 0        # How many of current project files are in DB
    session_count_source = "unknown"

    # Helper to query local SQLite for session counts
    def _query_local_session_counts():
        nonlocal indexed_global, indexed_of_indexable, indexed_of_project, session_count_source
        try:
            import sqlite3
            local_db = mira_path / "local_store.db"
            if local_db.exists():
                conn = sqlite3.connect(str(local_db))
                cur = conn.cursor()
                try:
                    # Total sessions in local DB
                    cur.execute("SELECT COUNT(*) FROM sessions")
                    indexed_global = cur.fetchone()[0]

                    # Check which of the current indexable files are indexed
                    if indexable_session_ids:
                        placeholders = ','.join(['?'] * len(indexable_session_ids))
                        cur.execute(
                            f"SELECT COUNT(*) FROM sessions WHERE session_id IN ({placeholders})",
                            indexable_session_ids
                        )
                        indexed_of_indexable = cur.fetchone()[0]

                    # Check which of the current project files are indexed
                    if project_session_ids:
                        placeholders = ','.join(['?'] * len(project_session_ids))
                        cur.execute(
                            f"SELECT COUNT(*) FROM sessions WHERE session_id IN ({placeholders})",
                            project_session_ids
                        )
                        indexed_of_project = cur.fetchone()[0]

                    session_count_source = "local"
                finally:
                    conn.close()
        except Exception:
            pass

    if storage and storage.using_central and storage.postgres:
        # Query central Postgres
        try:
            with storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    # Total sessions in DB (historical)
                    cur.execute("SELECT COUNT(*) FROM sessions")
                    indexed_global = cur.fetchone()[0]

                    # Project-scoped count (historical)
                    if project_id:
                        cur.execute("SELECT COUNT(*) FROM sessions WHERE project_id = %s", (project_id,))
                        indexed_project = cur.fetchone()[0]

                    # Check which of the current indexable files are indexed
                    if indexable_session_ids:
                        placeholders = ','.join(['%s'] * len(indexable_session_ids))
                        cur.execute(
                            f"SELECT COUNT(*) FROM sessions WHERE session_id IN ({placeholders})",
                            indexable_session_ids
                        )
                        indexed_of_indexable = cur.fetchone()[0]

                    # Check which of the current project files are indexed
                    if project_session_ids:
                        placeholders = ','.join(['%s'] * len(project_session_ids))
                        cur.execute(
                            f"SELECT COUNT(*) FROM sessions WHERE session_id IN ({placeholders})",
                            project_session_ids
                        )
                        indexed_of_project = cur.fetchone()[0]
                    session_count_source = "central"
        except Exception as e:
            # Fall back to local SQLite if central queries fail
            log(f"Central session count query failed, falling back to local: {e}")
            _query_local_session_counts()
            if session_count_source == "local":
                session_count_source = "local_fallback"
    else:
        # Query local SQLite
        _query_local_session_counts()

    # Get insights stats (local SQLite - always global for now)
    error_stats = get_error_stats()
    decision_stats = get_decision_stats()
    concepts_stats = get_concepts_stats()
    custodian_stats = get_custodian_stats()

    # Get health check info
    health = {}
    if storage:
        health = storage.health_check()

    # Get sync queue stats
    sync_queue_stats = {}
    try:
        from .sync_queue import get_sync_queue
        queue = get_sync_queue()
        sync_queue_stats = queue.get_stats()
    except Exception:
        pass

    # Get active ingestion jobs
    active_ingestions = []
    try:
        from .ingestion import get_active_ingestions
        active_ingestions = get_active_ingestions()
    except Exception:
        pass

    # Count active ingestions for this project (if project_path specified)
    project_active_count = 0
    if project_path:
        encoded_project = project_path.replace('/', '-').lstrip('-')
        for job in active_ingestions:
            job_project = job.get('project_path', '')
            if encoded_project in job_project or job_project.lstrip('-') == encoded_project:
                project_active_count += 1

    # Get artifact stats - global and project-scoped
    artifact_stats = {'global': {}, 'project': {}}

    if storage and storage.using_central and storage.postgres:
        try:
            with storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    # GLOBAL artifact stats
                    cur.execute("SELECT COUNT(*) FROM artifacts")
                    artifact_stats['global']['total'] = cur.fetchone()[0]

                    cur.execute("""
                        SELECT artifact_type, COUNT(*)
                        FROM artifacts
                        GROUP BY artifact_type
                        ORDER BY COUNT(*) DESC
                    """)
                    artifact_stats['global']['by_type'] = {
                        row[0]: row[1] for row in cur.fetchall()
                    }

                    # PROJECT-scoped artifact stats
                    if project_id:
                        cur.execute("""
                            SELECT COUNT(*) FROM artifacts a
                            JOIN sessions s ON a.session_id = s.id
                            WHERE s.project_id = %s
                        """, (project_id,))
                        artifact_stats['project']['total'] = cur.fetchone()[0]

                        cur.execute("""
                            SELECT a.artifact_type, COUNT(*)
                            FROM artifacts a
                            JOIN sessions s ON a.session_id = s.id
                            WHERE s.project_id = %s
                            GROUP BY a.artifact_type
                            ORDER BY COUNT(*) DESC
                        """, (project_id,))
                        artifact_stats['project']['by_type'] = {
                            row[0]: row[1] for row in cur.fetchall()
                        }

                    artifact_stats['storage'] = 'central'
        except Exception as e:
            # Fall back to local stats if central query fails
            log(f"Central artifact count query failed, falling back to local: {e}")
            local_stats = get_artifact_stats()
            artifact_stats['global'] = local_stats
            artifact_stats['storage'] = 'local_fallback'
    else:
        # No central storage - use local stats
        local_stats = get_artifact_stats()
        artifact_stats['global'] = local_stats
        artifact_stats['storage'] = 'local_only'

    # Get file operations stats - global and project-scoped
    file_ops_stats = {'global': {}, 'project': {}}
    if storage and storage.using_central and storage.postgres:
        try:
            with storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    # GLOBAL file operations stats
                    cur.execute("SELECT COUNT(*) FROM file_operations")
                    file_ops_stats['global']['total_operations'] = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(DISTINCT file_path) FROM file_operations")
                    file_ops_stats['global']['unique_files'] = cur.fetchone()[0]

                    # PROJECT-scoped file operations stats
                    if project_id:
                        cur.execute("""
                            SELECT COUNT(*) FROM file_operations fo
                            JOIN sessions s ON fo.session_id = s.id
                            WHERE s.project_id = %s
                        """, (project_id,))
                        file_ops_stats['project']['total_operations'] = cur.fetchone()[0]

                        cur.execute("""
                            SELECT COUNT(DISTINCT fo.file_path) FROM file_operations fo
                            JOIN sessions s ON fo.session_id = s.id
                            WHERE s.project_id = %s
                        """, (project_id,))
                        file_ops_stats['project']['unique_files'] = cur.fetchone()[0]

                    file_ops_stats['storage'] = 'central'
        except Exception:
            file_ops_stats['global'] = {'total_operations': 0, 'unique_files': 0}
            file_ops_stats['storage'] = 'pending_migration'
    else:
        # Get from local storage
        try:
            from .artifacts import get_journey_stats
            journey = get_journey_stats()
            file_ops_stats['global'] = {
                'total_operations': journey.get('files_created', 0) + journey.get('total_edits', 0),
                'unique_files': journey.get('unique_files', 0),
            }
            file_ops_stats['storage'] = 'local_only'
        except Exception:
            file_ops_stats['global'] = {'total_operations': 0, 'unique_files': 0}
            file_ops_stats['storage'] = 'error'

    # Get decision stats from central storage - global and project-scoped
    decision_stats_central = {'global': {}, 'project': {}}
    if storage and storage.using_central and storage.postgres:
        try:
            with storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    # GLOBAL decision stats
                    cur.execute("SELECT COUNT(*) FROM decisions")
                    decision_stats_central['global']['total'] = cur.fetchone()[0]

                    cur.execute("""
                        SELECT category, COUNT(*)
                        FROM decisions
                        GROUP BY category
                        ORDER BY COUNT(*) DESC
                    """)
                    decision_stats_central['global']['by_category'] = {
                        row[0]: row[1] for row in cur.fetchall()
                    }

                    # PROJECT-scoped decision stats
                    if project_id:
                        cur.execute("""
                            SELECT COUNT(*) FROM decisions d
                            JOIN sessions s ON d.session_id = s.id
                            WHERE s.project_id = %s
                        """, (project_id,))
                        decision_stats_central['project']['total'] = cur.fetchone()[0]

                        cur.execute("""
                            SELECT d.category, COUNT(*)
                            FROM decisions d
                            JOIN sessions s ON d.session_id = s.id
                            WHERE s.project_id = %s
                            GROUP BY d.category
                            ORDER BY COUNT(*) DESC
                        """, (project_id,))
                        decision_stats_central['project']['by_category'] = {
                            row[0]: row[1] for row in cur.fetchall()
                        }

                    decision_stats_central['storage'] = 'central'
        except Exception:
            decision_stats_central['global'] = decision_stats
            decision_stats_central['storage'] = 'local_fallback'
    else:
        decision_stats_central['global'] = decision_stats
        decision_stats_central['storage'] = 'local_only'

    # Build response with clear project vs global distinction
    result = {
        "storage_path": str(mira_path),
        "last_sync": datetime.now().isoformat(),
        "storage_health": health,
        "sync_queue": sync_queue_stats,
        "active_ingestions": active_ingestions,
        # Global stats
        "global": {
            "files": {
                "total": total_files,
                "indexable": indexable_files,              # Worth indexing (sessions + agents with 3+ msgs)
                "sessions": file_categories['sessions'],   # Main conversations (3+ messages)
                "agents": file_categories['agents'],       # Subagent tasks (3+ messages)
                "skipped": {
                    "no_messages": file_categories['no_messages'],  # Empty/snapshots only
                    "minimal": file_categories['minimal'],          # 1-2 msgs, likely abandoned
                },
            },
            "ingestion": {
                "indexed": min(indexed_of_indexable, indexable_files),  # Current files that are indexed
                "pending": max(0, indexable_files - indexed_of_indexable - len(active_ingestions)),  # Not yet started
                "in_progress": len(active_ingestions),     # Currently being processed
                "complete": indexed_of_indexable >= indexable_files and len(active_ingestions) == 0,
                "percent": min(100, round(100 * indexed_of_indexable / indexable_files)) if indexable_files > 0 else 100,
                "total_in_db": indexed_global,             # Historical: all sessions ever indexed
                "_note": "Local ingestion only (files → SQLite). See central_sync for remote sync status.",
            },
            "central_sync": {
                "pending": sync_queue_stats.get('total_pending', 0),
                "failed": sync_queue_stats.get('total_failed', 0),
                "status": "idle" if sync_queue_stats.get('total_pending', 0) == 0 and sync_queue_stats.get('total_failed', 0) == 0
                         else "syncing" if sync_queue_stats.get('total_pending', 0) > 0
                         else "has_failures",
                "_note": "Remote sync (SQLite → Postgres/Qdrant). Failed items retry on restart.",
            },
            "archived": archived,
            "artifacts": artifact_stats['global'],
            "decisions": decision_stats_central['global'],
            "file_operations": file_ops_stats['global'],
            "errors": error_stats,  # Local SQLite, always global
            "concepts": concepts_stats,  # Local SQLite, codebase concepts
            "custodian": custodian_stats,  # Local SQLite, user profile data
        },
    }

    # Add project-scoped stats if project_path was provided
    if project_path:
        result["project"] = {
            "path": project_path,
            "project_id": project_id,
            "files": {
                "total": project_files,
                "conversations": project_conversations,
            },
            "ingestion": {
                "indexed": min(indexed_of_project, project_conversations),
                "pending": max(0, project_conversations - indexed_of_project - project_active_count),
                "in_progress": project_active_count,
                "complete": indexed_of_project >= project_conversations and project_active_count == 0,
                "percent": min(100, round(100 * indexed_of_project / project_conversations)) if project_conversations > 0 else 100,
                "total_in_db": indexed_project,
            },
            "artifacts": artifact_stats.get('project', {}),
            "decisions": decision_stats_central.get('project', {}),
            "file_operations": file_ops_stats.get('project', {}),
        }

    # Add storage info
    result["storage_mode"] = {
        "sessions": session_count_source,
        "artifacts": artifact_stats.get('storage', 'unknown'),
        "decisions": decision_stats_central.get('storage', 'unknown'),
        "file_operations": file_ops_stats.get('storage', 'unknown'),
    }

    # Add local semantic search status
    try:
        from .local_semantic import get_local_semantic
        ls = get_local_semantic()
        result["local_semantic"] = ls.get_status()
    except Exception as e:
        result["local_semantic"] = {"available": False, "error": str(e)}

    return result


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
        from .fuzzy import expand_query_with_corrections, get_vocabulary_size
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
                    "solutions": results,  # Key matches TypeScript expectation
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
    results = search_error_solutions(query, limit=limit)

    response = {
        "solutions": results,  # Key matches TypeScript expectation
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
            from .fuzzy import expand_query_with_corrections, get_vocabulary_size
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
                    "decisions": results,  # Key matches TypeScript expectation
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
    if not query:
        results = search_decisions("", category=category, limit=limit, min_confidence=min_confidence)
    else:
        results = search_decisions(query, category=category, limit=limit, min_confidence=min_confidence)

    response = {
        "decisions": results,  # Key matches TypeScript expectation
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
    from .code_history import (
        get_file_timeline,
        get_symbol_history,
        reconstruct_file_at_date,
        get_edits_between,
        get_file_snapshot_at_date,
        get_code_history_stats,
    )

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

        # Add end-of-day time if only date provided (to include all snapshots on that day)
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

        # Get all snapshots and edits for the file
        # Default to last 30 days if no date specified
        from datetime import datetime, timedelta, timezone

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


def handle_rpc_request(request: dict, collection, storage=None) -> dict:
    """
    Handle a JSON-RPC request and return response.

    Args:
        request: JSON-RPC request dict
        collection: Deprecated - kept for API compatibility, ignored
        storage: Storage instance for central Qdrant + Postgres
    """
    method = request.get("method", "")
    params = request.get("params", {})
    request_id = request.get("id")

    result = None
    error = None

    try:
        if method == "search":
            result = handle_search(params, collection, storage)
        elif method == "recent":
            result = handle_recent(params, storage)
        elif method == "init":
            result = handle_init(params, collection, storage)
        elif method == "status":
            result = handle_status(params, collection, storage)
        elif method == "error_lookup":
            result = handle_error_lookup(params, storage)
        elif method == "decisions":
            result = handle_decisions(params, storage)
        elif method == "code_history":
            result = handle_code_history(params, storage)
        elif method == "shutdown":
            result = {"status": "shutting_down"}
        else:
            error = {"code": -32601, "message": f"Method not found: {method}"}
    except Exception as e:
        log(f"RPC handler error for {method}: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        error = {"code": -32603, "message": "Internal error processing request"}

    response = {"jsonrpc": "2.0", "id": request_id}
    if error:
        response["error"] = error
    else:
        response["result"] = result

    return response
