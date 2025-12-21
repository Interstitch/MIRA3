"""
MIRA Status Tool

Get system status, health, and statistics.
"""

import json
from datetime import datetime
from pathlib import Path

from mira.core import get_mira_path, get_global_mira_path, get_project_mira_path, log


def _get_dir_size_mb(path: Path) -> float:
    """Get directory size in MB. Returns 0 if path doesn't exist."""
    if not path.exists():
        return 0.0
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except (OSError, PermissionError):
        pass
    return round(total / (1024 * 1024), 2)


def handle_status(params: dict, storage=None) -> dict:
    """
    Get system status and statistics.

    Returns both project-scoped and global statistics when project_path is provided.

    Args:
        params: Request parameters (project_path for scoped stats)
        storage: Storage instance for central Qdrant + Postgres
    """
    log("[STATUS] Starting handle_status")

    project_path = params.get("project_path") if params else None
    global_mira_path = get_global_mira_path()
    project_mira_path = get_project_mira_path()
    mira_path = project_mira_path  # Backwards compat for rest of function
    claude_path = Path.home() / ".claude" / "projects"

    # Calculate storage sizes
    global_venv_path = global_mira_path / ".venv"
    storage_info = {
        "global_path": str(global_mira_path),
        "global_size_mb": _get_dir_size_mb(global_mira_path),
        "venv_path": str(global_venv_path),
        "venv_size_mb": _get_dir_size_mb(global_venv_path),
        "project_path": str(project_mira_path),
        "project_size_mb": _get_dir_size_mb(project_mira_path),
    }

    # Get project_id for scoped queries
    project_id = None
    if project_path and storage and storage.using_central:
        try:
            project_id = storage.get_project_id(project_path)
        except Exception:
            pass

    # Count and categorize source files
    file_categories = {
        'sessions': 0,
        'agents': 0,
        'no_messages': 0,
        'minimal': 0,
    }
    project_files = 0
    project_conversations = 0
    indexable_session_ids = []
    project_session_ids = []

    if claude_path.exists():
        encoded_project = project_path.replace('/', '-').lstrip('-') if project_path else None

        for f in claude_path.rglob("*.jsonl"):
            is_project_file = encoded_project and encoded_project in str(f.parent)
            is_agent_file = f.name.startswith('agent-')
            session_id = f.stem

            try:
                msg_count = 0
                with f.open(encoding="utf-8") as fp:
                    for line in fp:
                        try:
                            data = json.loads(line)
                            if data.get('type') in ('user', 'assistant'):
                                msg_count += 1
                                if msg_count > 2:
                                    break
                        except (json.JSONDecodeError, KeyError):
                            pass

                if msg_count == 0:
                    file_categories['no_messages'] += 1
                elif msg_count <= 2:
                    file_categories['minimal'] += 1
                else:
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

    # Count indexed from storage
    indexed_global = 0
    indexed_project = 0
    indexed_of_indexable = 0
    indexed_of_project = 0
    session_count_source = "unknown"

    # Query local SQLite for session counts
    def _query_local_session_counts():
        nonlocal indexed_global, indexed_of_indexable, indexed_of_project, session_count_source
        try:
            import sqlite3
            local_db = mira_path / "local_store.db"
            if local_db.exists():
                conn = sqlite3.connect(str(local_db))
                cur = conn.cursor()
                try:
                    cur.execute("SELECT COUNT(*) FROM sessions")
                    indexed_global = cur.fetchone()[0]

                    if indexable_session_ids:
                        placeholders = ','.join(['?'] * len(indexable_session_ids))
                        cur.execute(
                            f"SELECT COUNT(*) FROM sessions WHERE session_id IN ({placeholders})",
                            indexable_session_ids
                        )
                        indexed_of_indexable = cur.fetchone()[0]

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

    if storage and storage.using_central and hasattr(storage, 'postgres') and storage.postgres:
        try:
            with storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM sessions")
                    indexed_global = cur.fetchone()[0]

                    if project_id:
                        cur.execute("SELECT COUNT(*) FROM sessions WHERE project_id = %s", (project_id,))
                        indexed_project = cur.fetchone()[0]

                    if indexable_session_ids:
                        placeholders = ','.join(['%s'] * len(indexable_session_ids))
                        cur.execute(
                            f"SELECT COUNT(*) FROM sessions WHERE session_id IN ({placeholders})",
                            indexable_session_ids
                        )
                        indexed_of_indexable = cur.fetchone()[0]

                    if project_session_ids:
                        placeholders = ','.join(['%s'] * len(project_session_ids))
                        cur.execute(
                            f"SELECT COUNT(*) FROM sessions WHERE session_id IN ({placeholders})",
                            project_session_ids
                        )
                        indexed_of_project = cur.fetchone()[0]
                    session_count_source = "central"
        except Exception as e:
            log(f"[STATUS] Central session count query failed, falling back to local: {e}")
            _query_local_session_counts()
            if session_count_source == "local":
                session_count_source = "local_fallback"
    else:
        _query_local_session_counts()

    # Get insights stats
    error_stats = {}
    decision_stats = {}
    concepts_stats = {}
    custodian_stats = {}

    try:
        from mira.extraction.errors import get_error_stats
        error_stats = get_error_stats()
    except ImportError:
        pass

    try:
        from mira.extraction.decisions import get_decision_stats
        decision_stats = get_decision_stats()
    except ImportError:
        pass

    try:
        from mira.extraction.concepts import get_concepts_stats
        concepts_stats = get_concepts_stats()
    except ImportError:
        pass

    try:
        from mira.custodian import get_custodian_stats
        custodian_stats = get_custodian_stats()
    except ImportError:
        pass

    # Get health check info
    health = {}
    if storage:
        try:
            health = storage.health_check()
        except Exception:
            pass

    # Get sync queue stats
    sync_queue_stats = {}
    try:
        from mira.storage.sync.queue import get_sync_queue
        queue = get_sync_queue()
        sync_queue_stats = queue.get_stats()
    except Exception:
        pass

    # Get active ingestion jobs
    active_ingestions = []
    try:
        from mira.ingestion import get_active_ingestions
        active_ingestions = get_active_ingestions()
    except Exception:
        pass

    # Get artifact stats
    artifact_stats = {'global': {}, 'local': {}}
    try:
        from mira.extraction.artifacts import get_artifact_stats
        local_stats = get_artifact_stats()
        artifact_stats['local'] = local_stats
        artifact_stats['global'] = local_stats
    except ImportError:
        pass

    # Build response
    result = {
        "storage_path": str(mira_path),  # Backwards compat (project path)
        "storage": storage_info,  # New: detailed storage paths and sizes
        "last_sync": datetime.now().isoformat(),
        "storage_health": health,
        "sync_queue": sync_queue_stats,
        "active_ingestions": active_ingestions,
        "global": {
            "files": {
                "total": total_files,
                "indexable": indexable_files,
                "sessions": file_categories['sessions'],
                "agents": file_categories['agents'],
                "skipped": {
                    "no_messages": file_categories['no_messages'],
                    "minimal": file_categories['minimal'],
                },
            },
            "ingestion": {
                "indexed": min(indexed_of_indexable, indexable_files),
                "pending": max(0, indexable_files - indexed_of_indexable - len(active_ingestions)),
                "in_progress": len(active_ingestions),
                "complete": indexed_of_indexable >= indexable_files and len(active_ingestions) == 0,
                "percent": min(100, round(100 * indexed_of_indexable / indexable_files)) if indexable_files > 0 else 100,
                "total_in_db": indexed_global,
            },
            "central_sync": {
                "pending": sync_queue_stats.get('total_pending', 0),
                "failed": sync_queue_stats.get('total_failed', 0),
                "status": "idle" if sync_queue_stats.get('total_pending', 0) == 0 and sync_queue_stats.get('total_failed', 0) == 0
                         else "syncing" if sync_queue_stats.get('total_pending', 0) > 0
                         else "has_failures",
            },
            "archived": archived,
            "artifacts": artifact_stats['global'],
            "decisions": decision_stats,
            "errors": error_stats,
            "concepts": concepts_stats,
            "custodian": custodian_stats,
        },
    }

    # Add project-scoped stats if project_path was provided
    if project_path:
        project_active_count = 0
        if project_path:
            encoded_project = project_path.replace('/', '-').lstrip('-')
            for job in active_ingestions:
                job_project = job.get('project_path', '')
                if encoded_project in job_project:
                    project_active_count += 1

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
        }

    # Add storage mode info
    result["storage_mode"] = {
        "sessions": session_count_source,
        "artifacts": artifact_stats.get('storage', 'local'),
    }

    # Add local semantic search status
    try:
        from mira.search.local_semantic import get_local_semantic, get_pending_indexing_count
        ls = get_local_semantic()
        status = ls.get_status()
        status["pending_indexing"] = get_pending_indexing_count()
        result["local_semantic"] = status
    except Exception as e:
        result["local_semantic"] = {"available": False, "error": str(e)}

    log("[STATUS] handle_status complete")
    return result
