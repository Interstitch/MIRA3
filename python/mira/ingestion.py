"""
MIRA3 Conversation Ingestion Module

Handles the full pipeline of parsing, extracting, archiving, and indexing conversations.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from .utils import log, get_mira_path
from .parsing import parse_conversation
from .metadata import extract_metadata, build_document_content
from .artifacts import extract_file_operations_from_messages, extract_artifacts_from_messages
from .custodian import extract_custodian_learnings
from .insights import extract_insights_from_conversation
from .concepts import extract_concepts_from_conversation


def ingest_conversation(file_info: dict, collection, mira_path: Path = None) -> bool:
    """
    Ingest a single conversation: parse, extract, archive, index.

    Args:
        file_info: Dict with session_id, file_path, project_path, last_modified
        collection: ChromaDB collection
        mira_path: Path to .mira directory (optional, uses default if not provided)

    Returns True if successfully ingested, False if skipped or failed.
    """
    if mira_path is None:
        mira_path = get_mira_path()

    session_id = file_info['session_id']
    file_path = Path(file_info['file_path'])

    archives_path = mira_path / "archives"
    metadata_path = mira_path / "metadata"

    # Ensure directories exist
    archives_path.mkdir(parents=True, exist_ok=True)
    metadata_path.mkdir(parents=True, exist_ok=True)

    # Check if already ingested (by checking metadata file)
    meta_file = metadata_path / f"{session_id}.json"
    if meta_file.exists():
        # Check if source file was modified
        try:
            existing_meta = json.loads(meta_file.read_text())
            if existing_meta.get('last_modified') == file_info.get('last_modified'):
                return False  # Already up to date
        except (json.JSONDecodeError, IOError, OSError):
            pass

    short_id = session_id[:12]
    log(f"[{short_id}] Starting ingestion...")

    # Parse conversation
    conversation = parse_conversation(file_path)
    msg_count = len(conversation.get('messages', []))
    if not msg_count:
        log(f"[{short_id}] Skipped: no messages")
        return False
    log(f"[{short_id}] Parsed {msg_count} messages")

    # Extract metadata
    metadata = extract_metadata(conversation, file_info)
    kw_count = len(metadata.get('keywords', []))
    facts_count = len(metadata.get('key_facts', []))
    log(f"[{short_id}] Metadata: {kw_count} keywords, {facts_count} facts")

    # Archive the conversation (copy to .mira/archives/)
    archive_file = archives_path / f"{session_id}.jsonl"
    try:
        shutil.copy2(file_path, archive_file)
        log(f"[{short_id}] Archived")
    except Exception as e:
        log(f"[{short_id}] Archive failed: {e}")

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

    # Extract and store file operations for reconstruction capability
    try:
        ops_count = extract_file_operations_from_messages(raw_messages, session_id)
        if ops_count > 0:
            log(f"[{short_id}] File ops: {ops_count}")
    except Exception as e:
        log(f"[{short_id}] File ops failed: {e}")

    # Extract and store artifacts (code blocks, lists, tables, etc.)
    try:
        artifact_count = extract_artifacts_from_messages(raw_messages, session_id)
        if artifact_count > 0:
            log(f"[{short_id}] Artifacts: {artifact_count}")
    except Exception as e:
        log(f"[{short_id}] Artifacts failed: {e}")

    # Learn about the custodian from this conversation
    try:
        custodian_result = extract_custodian_learnings(conversation, session_id)
        learned = custodian_result.get('learned', 0) if isinstance(custodian_result, dict) else 0
        if learned > 0:
            log(f"[{short_id}] Custodian: {learned} learnings")
    except Exception as e:
        log(f"[{short_id}] Custodian failed: {e}")

    # Extract insights (errors, decisions) from this conversation
    try:
        insights = extract_insights_from_conversation(conversation, session_id)
        err_count = insights.get('errors_found', 0)
        dec_count = insights.get('decisions_found', 0)
        if err_count > 0 or dec_count > 0:
            log(f"[{short_id}] Insights: {err_count} errors, {dec_count} decisions")
    except Exception as e:
        log(f"[{short_id}] Insights failed: {e}")

    # Extract codebase concepts from this conversation
    try:
        project_path = file_info.get('project_path', '')
        if project_path:
            project_path = '/' + project_path.replace('-', '/')

        concepts = extract_concepts_from_conversation(conversation, session_id, project_path)
        concept_count = concepts.get('concepts_found', 0)
        if concept_count > 0:
            log(f"[{short_id}] Concepts: {concept_count}")
    except Exception as e:
        log(f"[{short_id}] Concepts failed: {e}")

    # Build document content for ChromaDB
    doc_content = build_document_content(conversation, metadata)
    log(f"[{short_id}] Built doc ({len(doc_content)} chars)")

    # Index to ChromaDB
    try:
        collection.upsert(
            ids=[session_id],
            documents=[doc_content],
            metadatas=[{
                'summary': metadata.get('summary', '')[:500],
                'keywords': ','.join(metadata.get('keywords', [])),
                'project_path': metadata.get('project_path', ''),
                'timestamp': metadata.get('last_modified', ''),
                'message_count': str(metadata.get('message_count', 0))
            }]
        )
        log(f"[{short_id}] Indexed to ChromaDB ✓")
    except Exception as e:
        log(f"[{short_id}] ChromaDB failed: {e}")
        return False

    return True


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


def run_full_ingestion(collection, mira_path: Path = None, max_workers: int = 4) -> dict:
    """
    Run full ingestion of all discovered conversations.

    Uses thread pool for parallel processing of conversations.
    SQLite writes are serialized through db_manager's write queue (WAL mode enabled).
    ChromaDB handles concurrent writes safely.

    Args:
        collection: ChromaDB collection
        mira_path: Path to .mira directory
        max_workers: Number of parallel ingestion threads (default: 4)

    Returns stats dict with counts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    if mira_path is None:
        mira_path = get_mira_path()

    conversations = discover_conversations()
    log(f"Discovered {len(conversations)} conversation files")

    stats = {
        'discovered': len(conversations),
        'ingested': 0,
        'skipped': 0,
        'failed': 0
    }
    stats_lock = threading.Lock()

    processed_count = [0]  # Use list to allow mutation in nested function

    def ingest_one(file_info):
        """Ingest a single conversation and return result."""
        try:
            result = ingest_conversation(file_info, collection, mira_path)
            processed_count[0] += 1
            if result:
                log(f"[{processed_count[0]}/{len(conversations)}] Ingested: {file_info['session_id'][:12]}...")
            return ('ingested' if result else 'skipped', file_info['session_id'])
        except Exception as e:
            processed_count[0] += 1
            log(f"[{processed_count[0]}/{len(conversations)}] Failed {file_info['session_id'][:12]}: {e}")
            return ('failed', file_info['session_id'])

    # Use thread pool for parallel ingestion
    # Limit workers to avoid overwhelming ChromaDB/SQLite
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(ingest_one, fi): fi for fi in conversations}

        for future in as_completed(futures):
            result_type, session_id = future.result()
            with stats_lock:
                stats[result_type] += 1

    log(f"Ingestion complete: {stats['ingested']} new, {stats['skipped']} skipped, {stats['failed']} failed")
    return stats
