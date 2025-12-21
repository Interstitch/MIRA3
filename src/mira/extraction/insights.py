"""
MIRA Insights Extraction Coordinator

Coordinates extraction of all insights (errors, decisions) from conversations.
This module ties together the errors and decisions extractors.
"""

import time
from typing import Dict, Any, Optional

from mira.core import log


def extract_insights_from_conversation(
    conversation: dict,
    session_id: str,
    project_path: Optional[str] = None,
    postgres_session_id: Optional[int] = None,
    storage=None
) -> Dict[str, Any]:
    """
    Extract all insights (errors, decisions) from a conversation.

    Called during ingestion.

    Args:
        conversation: Parsed conversation dict
        session_id: String session ID
        project_path: Filesystem path to project (for Postgres storage)
        postgres_session_id: Postgres session ID (for foreign keys)
        storage: Storage instance

    Returns:
        Dict with 'errors_found' and 'decisions_found' counts
    """
    from .errors import extract_errors_from_conversation
    from .decisions import extract_decisions_from_conversation

    short_id = session_id[:12]
    msg_count = len(conversation.get('messages', []))

    t0 = time.time()
    errors = extract_errors_from_conversation(
        conversation, session_id,
        project_path=project_path,
        storage=storage
    )
    t_errors = (time.time() - t0) * 1000

    t0 = time.time()
    decisions = extract_decisions_from_conversation(
        conversation, session_id,
        project_path=project_path,
        postgres_session_id=postgres_session_id,
        storage=storage
    )
    t_decisions = (time.time() - t0) * 1000

    log(f"[{short_id}] Insights detail: {msg_count} msgs | errors={errors or 0} ({t_errors:.0f}ms) decisions={decisions or 0} ({t_decisions:.0f}ms)")

    return {
        'errors_found': errors or 0,
        'decisions_found': decisions or 0
    }
