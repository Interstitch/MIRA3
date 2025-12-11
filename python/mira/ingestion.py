"""
MIRA3 Conversation Ingestion Module

Handles the full pipeline of parsing, extracting, archiving, and indexing conversations.

Primary: Central Qdrant + Postgres storage (full semantic search, cross-machine sync)
Fallback: Local SQLite with FTS (keyword search only, single machine)

Archives are stored in the available storage backend.
"""

import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .utils import log, get_mira_path
from .parsing import parse_conversation
from .metadata import extract_metadata, build_document_content
from .artifacts import extract_file_operations_from_messages, extract_artifacts_from_messages
from .custodian import extract_custodian_learnings
from .insights import extract_insights_from_conversation
from .concepts import extract_concepts_from_conversation
from .audit import audit_log

import threading

# Module-level tracking of active ingestions for status reporting
_active_ingestions = {}  # session_id -> {file_path, project_path, started_at, worker}
_active_lock = threading.Lock()


def get_active_ingestions() -> list:
    """Return list of currently active ingestion jobs."""
    with _active_lock:
        return [
            {
                'session_id': sid,
                'file_path': info['file_path'],
                'project_path': info['project_path'],
                'started_at': info['started_at'],
                'worker': info['worker'],
                'elapsed_ms': int((time.time() - info['started_at']) * 1000)
            }
            for sid, info in _active_ingestions.items()
        ]


def _mark_ingestion_active(session_id: str, file_path: str, project_path: str, worker: str):
    """Mark a session as currently being ingested."""
    with _active_lock:
        _active_ingestions[session_id] = {
            'file_path': file_path,
            'project_path': project_path,
            'started_at': time.time(),
            'worker': worker
        }


def _mark_ingestion_done(session_id: str):
    """Mark a session as done ingesting."""
    with _active_lock:
        _active_ingestions.pop(session_id, None)


def ingest_conversation(file_info: dict, collection, mira_path: Path = None, storage=None) -> bool:
    """
    Ingest a single conversation: parse, extract, archive, index.

    Supports incremental ingestion - only processes new messages since last run.

    Args:
        file_info: Dict with session_id, file_path, project_path, last_modified
        collection: Deprecated - kept for API compatibility, ignored
        mira_path: Path to .mira directory (optional, uses default if not provided)
        storage: Storage instance for central Qdrant + Postgres

    Returns True if successfully ingested, False if skipped or failed.
    """
    if mira_path is None:
        mira_path = get_mira_path()

    # Get storage instance if not provided
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            log("ERROR: Storage not available")
            return False

    session_id = file_info['session_id']
    file_path = Path(file_info['file_path'])

    archives_path = mira_path / "archives"
    metadata_path = mira_path / "metadata"

    # Ensure directories exist
    archives_path.mkdir(parents=True, exist_ok=True)
    metadata_path.mkdir(parents=True, exist_ok=True)

    # Check if already ingested and track incremental state
    meta_file = metadata_path / f"{session_id}.json"
    existing_meta = None
    last_indexed_message_count = 0
    is_incremental = False

    if meta_file.exists():
        try:
            existing_meta = json.loads(meta_file.read_text())
            last_indexed_message_count = existing_meta.get('last_indexed_message_count', 0)

            if existing_meta.get('last_modified') == file_info.get('last_modified'):
                # File hasn't changed - but check if we need to sync to central
                if storage.using_central:
                    if not storage.session_exists_in_central(session_id):
                        log(f"[{session_id[:12]}] Local session not in central, will sync")
                        # Continue with full processing for central sync
                    else:
                        return False  # Already in central, skip
                else:
                    return False  # Local only mode, already processed
            else:
                # File changed - check if we can do incremental
                is_incremental = last_indexed_message_count > 0
        except (json.JSONDecodeError, IOError, OSError):
            pass

    short_id = session_id[:12]
    t_start = time.time()
    log(f"[{short_id}] Starting ingestion{'(incremental)' if is_incremental else ''}...")

    # Parse conversation
    t0 = time.time()
    conversation = parse_conversation(file_path)
    messages = conversation.get('messages', [])
    msg_count = len(messages)
    if not msg_count:
        log(f"[{short_id}] Skipped: no messages")
        return False
    t_parse = (time.time() - t0) * 1000

    # Determine new messages to process
    new_message_start = 0
    if is_incremental and last_indexed_message_count < msg_count:
        new_message_start = last_indexed_message_count
        new_msg_count = msg_count - last_indexed_message_count
        log(f"[{short_id}] Parsed {msg_count} messages, {new_msg_count} NEW (from idx {new_message_start}) ({t_parse:.0f}ms)")
    else:
        log(f"[{short_id}] Parsed {msg_count} messages ({t_parse:.0f}ms)")

    # Extract metadata (always from full conversation for accurate summary)
    t0 = time.time()
    metadata = extract_metadata(conversation, file_info)
    kw_count = len(metadata.get('keywords', []))
    facts_count = len(metadata.get('key_facts', []))
    t_meta = (time.time() - t0) * 1000
    log(f"[{short_id}] Metadata: {kw_count} keywords, {facts_count} facts ({t_meta:.0f}ms)")

    # Read file content for remote archiving
    try:
        file_content = file_path.read_text(encoding='utf-8')
        content_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
    except Exception as e:
        log(f"[{short_id}] Failed to read file: {e}")
        return False

    # Add incremental tracking to metadata
    metadata['last_indexed_message_count'] = msg_count

    # Save metadata
    meta_file.write_text(json.dumps(metadata, indent=2))

    # Read raw messages for extraction
    raw_messages = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_messages.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # For incremental, only process new messages
    raw_messages_to_process = raw_messages[new_message_start:] if is_incremental else raw_messages

    # NOTE: File operations extraction moved after session upsert to get postgres_session_id

    # Build document content for vector search
    doc_content = build_document_content(conversation, metadata)
    log(f"[{short_id}] Built doc ({len(doc_content)} chars)")

    # Determine project path for central storage
    project_path_encoded = file_info.get('project_path', '')
    project_path_normalized = ''
    if project_path_encoded:
        # Convert from "-workspaces-MIRA3" to "/workspaces/MIRA3"
        project_path_normalized = '/' + project_path_encoded.replace('-', '/')

    # Get git remote for cross-machine project identification
    from .utils import get_git_remote_for_claude_path
    git_remote = get_git_remote_for_claude_path(project_path_encoded)
    if git_remote:
        log(f"[{short_id}] Git remote: {git_remote}")

    # Index to storage (central preferred, local fallback)
    try:
        # Upsert session (uses central if available, falls back to local)
        # This returns the session ID needed for archive and artifact foreign keys
        t0 = time.time()
        db_session_id = storage.upsert_session(
            project_path=project_path_normalized,
            session_id=session_id,
            git_remote=git_remote,
            summary=metadata.get('summary', ''),
            keywords=metadata.get('keywords', []),
            facts=metadata.get('key_facts', []),
            task_description=metadata.get('task_description', ''),
            git_branch=metadata.get('git_branch'),
            models_used=metadata.get('models_used', []),
            tools_used=metadata.get('tools_used', []),
            files_touched=metadata.get('files_touched', []),
            message_count=metadata.get('message_count', 0),
            started_at=metadata.get('started_at'),
            ended_at=metadata.get('last_modified'),
        )
        t_session = (time.time() - t0) * 1000

        if db_session_id is None:
            log(f"[{short_id}] ERROR: Failed to create session")
            return False

        storage_mode = "central" if storage.using_central else "local"
        log(f"[{short_id}] Session upserted ({storage_mode}, id={db_session_id}) ({t_session:.0f}ms)")

        # Archive conversation (full JSONL content)
        try:
            t0 = time.time()
            archive_id = storage.upsert_archive(
                postgres_session_id=db_session_id,
                content=file_content,
                content_hash=content_hash,
            )
            t_archive = (time.time() - t0) * 1000
            log(f"[{short_id}] Archived (archive_id={archive_id}) ({t_archive:.0f}ms)")
        except Exception as e:
            log(f"[{short_id}] Archive failed: {e}")

        # Extract artifacts with session ID for proper foreign keys
        # Use incremental message slice if available
        try:
            t0 = time.time()
            artifact_count = extract_artifacts_from_messages(
                raw_messages_to_process, session_id,
                postgres_session_id=db_session_id,
                storage=storage,
                message_start_index=new_message_start,  # For correct message_index in artifacts
            )
            t_artifacts = (time.time() - t0) * 1000
            if artifact_count > 0:
                incr_note = f" (incremental from msg {new_message_start})" if is_incremental else ""
                log(f"[{short_id}] Artifacts: {artifact_count} ({t_artifacts:.0f}ms, {artifact_count*1000/max(1,t_artifacts):.0f}/sec){incr_note}")
        except Exception as e:
            log(f"[{short_id}] Artifacts failed: {e}")

        # Extract file operations (Write/Edit tool uses) for file history
        try:
            t0 = time.time()
            file_ops_count = extract_file_operations_from_messages(
                raw_messages_to_process,
                session_id,
                postgres_session_id=db_session_id,
                storage=storage,
            )
            t_file_ops = (time.time() - t0) * 1000
            if file_ops_count > 0:
                incr_note = f" (incremental from msg {new_message_start})" if is_incremental else ""
                log(f"[{short_id}] File ops: {file_ops_count} ({t_file_ops:.0f}ms){incr_note}")
        except Exception as e:
            log(f"[{short_id}] File ops failed: {e}")

        # Create incremental conversation for custodian/insights/concepts
        # Only process new messages for these extractions
        if is_incremental and new_message_start > 0:
            conversation_to_process = {
                **conversation,
                'messages': messages[new_message_start:]
            }
            incr_msg_count = len(messages) - new_message_start
        else:
            conversation_to_process = conversation
            incr_msg_count = len(messages)

        # Learn about the custodian from this conversation
        try:
            t0 = time.time()
            custodian_result = extract_custodian_learnings(conversation_to_process, session_id)
            learned = custodian_result.get('learned', 0) if isinstance(custodian_result, dict) else 0
            t_custodian = (time.time() - t0) * 1000
            if learned > 0:
                incr_note = f" (from {incr_msg_count} new msgs)" if is_incremental else ""
                log(f"[{short_id}] Custodian: {learned} learnings ({t_custodian:.0f}ms){incr_note}")
        except Exception as e:
            log(f"[{short_id}] Custodian failed: {e}")

        # Extract insights (errors, decisions) from this conversation
        try:
            t0 = time.time()
            insights = extract_insights_from_conversation(
                conversation_to_process, session_id,
                project_path=project_path_normalized,
                postgres_session_id=db_session_id,
                storage=storage
            )
            err_count = insights.get('errors_found', 0)
            dec_count = insights.get('decisions_found', 0)
            t_insights = (time.time() - t0) * 1000
            if err_count > 0 or dec_count > 0:
                incr_note = f" (from {incr_msg_count} new msgs)" if is_incremental else ""
                log(f"[{short_id}] Insights: {err_count} errors, {dec_count} decisions ({t_insights:.0f}ms){incr_note}")
        except Exception as e:
            log(f"[{short_id}] Insights failed: {e}")

        # Extract codebase concepts (central only - requires vector storage)
        if storage.using_central:
            try:
                t0 = time.time()
                concepts = extract_concepts_from_conversation(
                    conversation_to_process, session_id,
                    project_path=project_path_normalized,
                    storage=storage
                )
                concept_count = concepts.get('concepts_found', 0)
                t_concepts = (time.time() - t0) * 1000
                if concept_count > 0:
                    incr_note = f" (from {incr_msg_count} new msgs)" if is_incremental else ""
                    log(f"[{short_id}] Concepts: {concept_count} ({t_concepts:.0f}ms){incr_note}")
            except Exception as e:
                log(f"[{short_id}] Concepts failed: {e}")

        # Vector indexing (central only - local uses FTS instead)
        if storage.using_central:
            try:
                t0 = time.time()
                from .embedding import get_embedding_function
                embed_fn = get_embedding_function()
                doc_vector = embed_fn([doc_content])[0]

                storage.vector_upsert(
                    vector=doc_vector,
                    content=doc_content[:2000],  # Truncate for storage efficiency
                    session_id=session_id,
                    project_path=project_path_normalized,
                    chunk_type="session",
                )
                t_vector = (time.time() - t0) * 1000
                log(f"[{short_id}] Indexed to central storage ({t_vector:.0f}ms)")
            except Exception as e:
                log(f"[{short_id}] Vector indexing failed: {e}")

        t_total = (time.time() - t_start) * 1000
        log(f"[{short_id}] Ingestion complete ({storage_mode} mode) - TOTAL: {t_total:.0f}ms")

        # Audit log successful ingestion
        audit_log(
            action="ingest",
            resource_type="session",
            resource_id=session_id,
            parameters={"project_path": project_path_normalized, "message_count": msg_count},
            result_summary={"storage_mode": storage_mode, "db_session_id": db_session_id},
            status="success",
        )
        return True
    except Exception as e:
        log(f"[{short_id}] Ingestion failed: {e}")

        # Audit log failed ingestion
        audit_log(
            action="ingest",
            resource_type="session",
            resource_id=session_id,
            parameters={"project_path": project_path_normalized},
            status="failure",
            error_message=str(e),
        )
        return False


def sync_active_session(
    file_path: str,
    session_id: str,
    project_path: str,
    mira_path: Path = None,
    storage=None
) -> bool:
    """
    Sync the active session to remote storage.

    This is a lightweight sync that updates the archive content and re-indexes
    vectors without full re-extraction of metadata. Used by the active session
    watcher for near real-time sync.

    Args:
        file_path: Full path to the session JSONL file
        session_id: Session UUID
        project_path: Project directory name
        mira_path: Path to .mira directory
        storage: Storage instance

    Returns:
        True if synced successfully, False if skipped or failed
    """
    if mira_path is None:
        mira_path = get_mira_path()

    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            log("[active-sync] Storage not available")
            return False

    # Only sync to central storage
    if not storage.using_central:
        return False

    short_id = f"{session_id[:12]}"
    file_path = Path(file_path)

    try:
        # Parse conversation
        parsed = parse_conversation(file_path)
        messages = parsed.get('messages', [])

        if not messages:
            return False

        # Get file stats for archive
        file_stat = file_path.stat()
        file_size = file_stat.st_size

        # Read raw content for archive
        archive_content = file_path.read_text()
        line_count = archive_content.count('\n')

        # Check if session exists in central
        if not storage.session_exists_in_central(session_id):
            # Session doesn't exist yet - do full ingestion instead
            log(f"[{short_id}] Session not in central, doing full ingest")
            file_info = {
                'session_id': session_id,
                'file_path': str(file_path),
                'project_path': project_path,
                'last_modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            }
            return ingest_conversation(file_info, None, mira_path, storage)

        # Update archive content in central storage
        try:
            storage.update_archive(
                session_id=session_id,
                content=archive_content,
                size_bytes=file_size,
                line_count=line_count
            )
            log(f"[{short_id}] Archive updated ({file_size} bytes)")
        except Exception as e:
            log(f"[{short_id}] Archive update failed: {e}")
            # Continue anyway - archive update is not critical

        # Re-extract and update metadata/keywords for better search
        file_info = {'project_path': project_path}
        raw_meta = extract_metadata(parsed, file_info)

        # Update session metadata in central
        try:
            storage.update_session_metadata(
                session_id=session_id,
                summary=raw_meta.get('summary', ''),
                keywords=raw_meta.get('keywords', []),
            )
        except Exception as e:
            log(f"[{short_id}] Metadata update failed: {e}")

        # Audit log
        audit_log(
            action="active_sync",
            resource_type="session",
            resource_id=session_id,
            parameters={"project_path": project_path},
            result_summary={"size_bytes": file_size, "line_count": line_count},
            status="success",
        )

        return True

    except Exception as e:
        log(f"[{short_id}] Active sync failed: {e}")
        audit_log(
            action="active_sync",
            resource_type="session",
            resource_id=session_id,
            parameters={"project_path": project_path},
            status="failure",
            error_message=str(e),
        )
        return False


def discover_conversations(claude_path: Path = None) -> list:
    """
    Discover all conversation files from Claude Code projects.

    Returns list of file_info dicts with:
    - session_id: Unique identifier
    - file_path: Full path to JSONL file
    - project_path: Project directory
    - last_modified: ISO timestamp
    """
    if claude_path is None:
        claude_path = Path.home() / ".claude" / "projects"

    if not claude_path.exists():
        return []

    conversations = []

    for jsonl_file in claude_path.rglob("*.jsonl"):
        # Skip agent files (subagent task logs)
        if jsonl_file.name.startswith("agent-"):
            continue

        session_id = jsonl_file.stem

        # Extract project path from directory structure
        # e.g., ~/.claude/projects/-workspaces-MIRA3/session.jsonl
        project_dir = jsonl_file.parent.name

        # Get last modified time
        try:
            mtime = jsonl_file.stat().st_mtime
            last_modified = datetime.fromtimestamp(mtime).isoformat()
        except (OSError, ValueError):
            last_modified = ""

        conversations.append({
            'session_id': session_id,
            'file_path': str(jsonl_file),
            'project_path': project_dir,
            'last_modified': last_modified
        })

    return conversations


def run_full_ingestion(collection, mira_path: Path = None, max_workers: int = 4, storage=None) -> dict:
    """
    Run full ingestion of all discovered conversations.

    Uses thread pool for parallel processing of conversations.
    Indexes to central storage if available, falls back to local SQLite.

    Args:
        collection: Deprecated - kept for API compatibility, ignored
        mira_path: Path to .mira directory
        max_workers: Number of parallel ingestion threads (default: 4, pool_size=12)
        storage: Storage instance

    Returns stats dict with counts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    if mira_path is None:
        mira_path = get_mira_path()

    # Get storage instance if not provided
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            log("ERROR: Storage not available")
            return {'discovered': 0, 'ingested': 0, 'skipped': 0, 'failed': 0}

    storage_mode = "central" if storage.using_central else "local"
    pool_size = storage.postgres.pool_size if storage.postgres else "N/A"
    log(f"╔══════════════════════════════════════════════════════════════╗")
    log(f"║ PARALLEL INGESTION: {max_workers} workers, pool_size={pool_size}, mode={storage_mode}")
    log(f"╚══════════════════════════════════════════════════════════════╝")

    t_discover_start = time.time()
    conversations = discover_conversations()
    t_discover = (time.time() - t_discover_start) * 1000
    log(f"Discovered {len(conversations)} conversation files ({t_discover:.0f}ms)")

    stats = {
        'discovered': len(conversations),
        'ingested': 0,
        'skipped': 0,
        'failed': 0
    }
    stats_lock = threading.Lock()

    processed_count = [0]  # Use list to allow mutation in nested function
    active_workers = [0]   # Track concurrent workers
    max_concurrent = [0]   # Peak concurrency observed

    t_ingestion_start = time.time()

    def ingest_one(file_info):
        """Ingest a single conversation and return result."""
        worker_id = threading.current_thread().name
        session_id = file_info['session_id']
        with stats_lock:
            active_workers[0] += 1
            if active_workers[0] > max_concurrent[0]:
                max_concurrent[0] = active_workers[0]

        # Track this ingestion as active
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
            return ('ingested' if result else 'skipped', session_id)
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
                stats[result_type] += 1

    t_total = (time.time() - t_ingestion_start) * 1000
    rate = stats['ingested'] * 1000 / max(1, t_total) * 60  # sessions per minute

    log(f"╔══════════════════════════════════════════════════════════════╗")
    log(f"║ INGESTION COMPLETE")
    log(f"║   New: {stats['ingested']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
    log(f"║   Time: {t_total:.0f}ms, Rate: {rate:.1f} sessions/min")
    log(f"║   Peak concurrency: {max_concurrent[0]} workers")
    log(f"╚══════════════════════════════════════════════════════════════╝")
    return stats
