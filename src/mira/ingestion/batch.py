"""
MIRA Ingestion Batch Module

Handles parallel batch ingestion of multiple conversations.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Optional

from mira.core import log, MIRA_PATH

from .core import (
    ingest_conversation,
    discover_conversations,
    _mark_ingestion_active,
    _mark_ingestion_done,
)


def run_full_ingestion(
    collection,
    mira_path: Path = None,
    max_workers: int = 4,
    storage=None
) -> Dict[str, Any]:
    """
    Run full ingestion of all discovered conversations.

    Uses thread pool for parallel processing of conversations.
    Indexes to central storage if available, falls back to local SQLite.

    Args:
        collection: Deprecated - kept for API compatibility, ignored
        mira_path: Path to .mira directory
        max_workers: Number of parallel ingestion threads (default: 4)
        storage: Storage instance

    Returns stats dict with counts.
    """
    if mira_path is None:
        mira_path = MIRA_PATH

    # Get storage instance if not provided
    if storage is None:
        try:
            from mira.storage import get_storage
            storage = get_storage()
        except ImportError:
            log("ERROR: Storage not available")
            return {'discovered': 0, 'ingested': 0, 'skipped': 0, 'failed': 0}

    storage_mode = "central" if storage.using_central else "local"
    pool_size = storage.postgres.pool_size if storage.postgres else "N/A"
    log(f"======================================================================")
    log(f" PARALLEL INGESTION: {max_workers} workers, pool_size={pool_size}, mode={storage_mode}")
    log(f"======================================================================")

    t_discover_start = time.time()
    all_conversations = discover_conversations()
    t_discover = (time.time() - t_discover_start) * 1000

    # Filter out agent sub-conversations
    conversations = [c for c in all_conversations if not c.get('is_agent', False)]
    agent_count = len(all_conversations) - len(conversations)

    # Group by project for visibility
    by_project: Dict[str, int] = {}
    for c in conversations:
        proj = c.get('project_path', 'unknown')
        by_project[proj] = by_project.get(proj, 0) + 1

    log(f"Discovered {len(all_conversations)} files, {agent_count} agent files filtered ({t_discover:.0f}ms)")
    for proj, count in sorted(by_project.items(), key=lambda x: -x[1])[:5]:
        log(f"  - {proj}: {count} files")

    stats = {
        'discovered': len(all_conversations),
        'ingested': 0,
        'skipped': 0,
        'skipped_agent_files': agent_count,
        'skipped_in_central': 0,
        'skipped_no_messages': 0,
        'skipped_unchanged': 0,
        'failed': 0
    }
    stats_lock = threading.Lock()

    processed_count = [0]
    active_workers = [0]
    max_concurrent = [0]

    t_ingestion_start = time.time()

    def ingest_one(file_info):
        """Ingest a single conversation and return result."""
        worker_id = threading.current_thread().name
        session_id = file_info['session_id']
        with stats_lock:
            active_workers[0] += 1
            if active_workers[0] > max_concurrent[0]:
                max_concurrent[0] = active_workers[0]

        _mark_ingestion_active(
            session_id,
            file_info.get('file_path', ''),
            file_info.get('project_path', ''),
            worker_id
        )
        try:
            result = ingest_conversation(file_info, None, mira_path, storage)
            with stats_lock:
                processed_count[0] += 1
                cnt = processed_count[0]
            if result:
                log(f"[{cnt}/{len(conversations)}] [{worker_id}] Ingested: {session_id[:12]}...")
                return ('ingested', session_id)
            else:
                meta_file = Path(mira_path) / "metadata" / f"{session_id}.json"
                if meta_file.exists():
                    if storage.using_central and storage.session_exists_in_central(session_id):
                        return ('skipped_in_central', session_id)
                    else:
                        return ('skipped_unchanged', session_id)
                else:
                    return ('skipped_no_messages', session_id)
        except Exception as e:
            with stats_lock:
                processed_count[0] += 1
                cnt = processed_count[0]
            log(f"[{cnt}/{len(conversations)}] [{worker_id}] Failed {session_id[:12]}: {e}")
            return ('failed', session_id)
        finally:
            _mark_ingestion_done(session_id)
            with stats_lock:
                active_workers[0] -= 1

    # Use thread pool for parallel ingestion
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Ingest") as executor:
        futures = {executor.submit(ingest_one, fi): fi for fi in conversations}

        for future in as_completed(futures):
            result_type, session_id = future.result()
            with stats_lock:
                if result_type in stats:
                    stats[result_type] += 1
                if result_type.startswith('skipped'):
                    stats['skipped'] += 1

    t_total = (time.time() - t_ingestion_start) * 1000
    rate = stats['ingested'] * 1000 / max(1, t_total) * 60

    log(f"======================================================================")
    log(f" INGESTION COMPLETE")
    log(f"   Ingested: {stats['ingested']}, Failed: {stats['failed']}")
    log(f"   Skipped: {stats['skipped'] + stats['skipped_agent_files']} total")
    log(f"     - Agent files (filtered): {stats['skipped_agent_files']}")
    log(f"     - Already in central: {stats['skipped_in_central']}")
    log(f"     - No messages: {stats['skipped_no_messages']}")
    log(f"     - Unchanged (local only): {stats['skipped_unchanged']}")
    log(f"   Time: {t_total:.0f}ms, Rate: {rate:.1f} sessions/min")
    log(f"   Peak concurrency: {max_concurrent[0]} workers")
    log(f"======================================================================")
    return stats
