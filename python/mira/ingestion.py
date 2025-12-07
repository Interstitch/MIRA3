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

    log(f"Ingesting: {session_id}")

    # Parse conversation
    conversation = parse_conversation(file_path)
    if not conversation.get('messages'):
        log(f"  Skipping {session_id}: no messages")
        return False

    # Extract metadata
    metadata = extract_metadata(conversation, file_info)

    # Archive the conversation (copy to .mira/archives/)
    archive_file = archives_path / f"{session_id}.jsonl"
    try:
        shutil.copy2(file_path, archive_file)
    except Exception as e:
        log(f"  Failed to archive {session_id}: {e}")

    # Save metadata
    meta_file.write_text(json.dumps(metadata, indent=2))

    # Extract and store file operations for reconstruction capability
    try:
        # Read raw messages for file operation extraction
        raw_messages = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw_messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        ops_count = extract_file_operations_from_messages(raw_messages, session_id)
        if ops_count > 0:
            log(f"  Stored {ops_count} file operations for {session_id}")
    except Exception as e:
        log(f"  Failed to extract file operations: {e}")

    # Extract and store artifacts (code blocks, lists, tables, etc.)
    try:
        artifact_count = extract_artifacts_from_messages(raw_messages, session_id)
        if artifact_count > 0:
            log(f"  Stored {artifact_count} artifacts for {session_id}")
    except Exception as e:
        log(f"  Failed to extract artifacts: {e}")

    # Learn about the custodian from this conversation
    try:
        extract_custodian_learnings(conversation, session_id)
    except Exception as e:
        log(f"  Failed to extract custodian learnings: {e}")

    # Extract insights (errors, decisions) from this conversation
    try:
        insights = extract_insights_from_conversation(conversation, session_id)
        if insights['errors_found'] > 0 or insights['decisions_found'] > 0:
            log(f"  Extracted {insights['errors_found']} errors, {insights['decisions_found']} decisions")
    except Exception as e:
        log(f"  Failed to extract insights: {e}")

    # Build document content for ChromaDB
    doc_content = build_document_content(conversation, metadata)

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
        log(f"  Indexed {session_id} ({metadata.get('message_count', 0)} messages)")
    except Exception as e:
        log(f"  Failed to index {session_id}: {e}")
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


def run_full_ingestion(collection, mira_path: Path = None) -> dict:
    """
    Run full ingestion of all discovered conversations.

    Returns stats dict with counts.
    """
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

    for file_info in conversations:
        try:
            if ingest_conversation(file_info, collection, mira_path):
                stats['ingested'] += 1
            else:
                stats['skipped'] += 1
        except Exception as e:
            log(f"Failed to ingest {file_info['session_id']}: {e}")
            stats['failed'] += 1

    log(f"Ingestion complete: {stats['ingested']} new, {stats['skipped']} skipped, {stats['failed']} failed")
    return stats
