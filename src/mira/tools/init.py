"""
MIRA Init Tool

Initialize session context with user profile, recent work, and usage guidance.
"""

import json
import os
import platform
from datetime import datetime
from pathlib import Path

from mira.core import get_mira_path, log
from mira.tools.guidance import (
    build_claude_guidance,
    filter_codebase_knowledge,
    get_actionable_alerts,
    get_simplified_storage_stats,
    build_enriched_custodian_summary,
    build_interaction_tips,
)


def _format_project_path(encoded_path: str) -> str:
    """Convert encoded project path to readable format."""
    if not encoded_path:
        return "unknown"
    readable = encoded_path.replace('-', '/')
    if not readable.startswith('/'):
        readable = '/' + readable
    return readable


def _get_environment_summary() -> dict:
    """
    Get a clear summary of the current environment.

    Returns structured info about where we're running.
    """
    env = {
        "os": platform.system().lower(),
        "type": "local",
    }

    # Detect cloud/container environments
    if os.environ.get('CODESPACES'):
        env["type"] = "codespaces"
        env["description"] = "GitHub Codespaces (cloud container)"
    elif os.environ.get('GITPOD_WORKSPACE_ID'):
        env["type"] = "gitpod"
        env["description"] = "Gitpod workspace"
    elif os.environ.get('WSL_DISTRO_NAME'):
        env["type"] = "wsl"
        env["description"] = f"WSL ({os.environ.get('WSL_DISTRO_NAME')})"
    elif os.path.exists('/.dockerenv'):
        env["type"] = "docker"
        env["description"] = "Docker container"
    elif os.environ.get('SSH_CONNECTION'):
        env["type"] = "ssh"
        env["description"] = "Remote SSH session"
    else:
        env["description"] = f"Local {platform.system()} machine"

    return env


def _get_last_session_summary(mira_path: Path, project_path: str = "") -> dict:
    """
    Get summary of the most recent session for continuity.

    Returns dict with last session's summary, timestamp, and open tasks.
    Aggressively filters meta-questions and status checks.
    """
    metadata_path = mira_path / "metadata"
    if not metadata_path.exists():
        return {}

    # Patterns indicating meta-questions or status checks (not real work)
    skip_patterns = [
        'what were we', 'what did we', 'where were we',
        'show me', 'can you show', 'status', 'check ',
        'hi claude', 'hello', 'hey ',
    ]

    # Find the most recent metadata file (optionally filtered by project)
    recent_files = sorted(
        metadata_path.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    for meta_file in recent_files[:20]:  # Check more files to find real work
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))

            # Filter by project if specified
            if project_path:
                file_project = meta.get('project_path', '')
                normalized = _format_project_path(file_project)
                if project_path not in normalized and normalized not in project_path:
                    continue

            # Extract session info
            summary = meta.get('summary', '')
            task = meta.get('task_description', '')
            timestamp = meta.get('extracted_at', '')

            # Skip empty sessions
            if not summary or len(summary) < 20:
                continue

            summary_lower = summary.lower()
            task_lower = (task or '').lower()

            # Skip meta-questions and status checks
            if any(pattern in summary_lower for pattern in skip_patterns):
                continue
            if any(pattern in task_lower for pattern in skip_patterns):
                continue

            # Skip if task is a question
            if task and task.strip().endswith('?'):
                continue

            # Clean the summary - remove "Task: " prefix if present
            clean_summary = summary
            if clean_summary.startswith('Task:'):
                # Extract just the outcome part after "|"
                if '|' in clean_summary:
                    parts = clean_summary.split('|')
                    clean_summary = parts[-1].strip()
                else:
                    clean_summary = clean_summary[5:].strip()

            # Skip garbage summaries (code fragments, too short, etc.)
            if len(clean_summary) < 30:
                continue
            if clean_summary.startswith('`') or clean_summary.startswith('"'):
                continue
            if not clean_summary[0].isalpha():
                continue

            result = {
                "summary": clean_summary[:200] if clean_summary else None,
                "timestamp": timestamp,
            }

            # Add task description if meaningful (not a question, not too short)
            if task and len(task) > 20 and not task.strip().endswith('?') and not task.startswith('/'):
                # Skip if it's a meta-question
                if not any(pattern in task_lower for pattern in skip_patterns):
                    # Skip command messages
                    if '<command-' in task or '</command' in task:
                        pass  # Don't add
                    else:
                        result["last_task"] = task[:150]

            # Check for incomplete TODOs
            todos = meta.get('todo_topics', [])
            incomplete = [t for t in todos if isinstance(t, str) and 'pending' in t.lower()]
            if incomplete:
                result["open_tasks"] = incomplete[:3]

            return result

        except (json.JSONDecodeError, IOError):
            pass

    return {}


def _get_structured_decisions(limit: int = 5) -> list:
    """
    Get recent decisions in structured format with reasoning.

    Returns list of decision dicts, not fragments.
    Filters out garbage extractions aggressively.
    """
    decisions = []

    # Garbage indicators - skip decisions containing these
    # Garbage indicators - focus on formatting markers and meta-text, NOT action verbs
    # (Action verbs like "added", "fixed", "updated" are legitimate in decisions)
    garbage_patterns = [
        # Formatting markers (indicate incomplete extraction)
        '**', '```', '|', '#{',
        # Process narration (assistant meta-text, not decisions)
        'let me', 'now let', 'i will', 'i\'ll', 'i am going',
        # Filler phrases (incomplete extraction)
        'here is', 'the following', 'as follows',
    ]

    try:
        from mira.core.database import get_db_manager
        from mira.core.constants import DB_INSIGHTS

        db = get_db_manager()
        rows = db.execute_read(DB_INSIGHTS, """
            SELECT decision_summary, reasoning, category, confidence, timestamp
            FROM decisions
            WHERE confidence >= 0.75
            ORDER BY confidence DESC, timestamp DESC
            LIMIT ?
        """, (limit * 3,))  # Fetch extra to filter

        for row in rows:
            if len(decisions) >= limit:
                break

            summary = row['decision_summary']
            if not summary or len(summary) < 15:
                continue

            summary_lower = summary.lower()

            # Skip garbage patterns
            if any(pattern in summary_lower for pattern in garbage_patterns):
                continue

            # Must start with capital letter (proper sentence)
            if not summary[0].isupper():
                continue

            # Skip instruction-style text (not decisions, but directives)
            instruction_starts = ['yes ', 'no ', 'do ', 'please ', 'try ', 'make ']
            if any(summary_lower.startswith(s) for s in instruction_starts):
                continue

            # Skip problem descriptions (not decisions, but issues)
            problem_indicators = [' issues ', ' issue ', ' problem ', ' bug ', ' error ']
            if any(ind in summary_lower for ind in problem_indicators):
                continue

            # Skip list item fragments (starts with dash after potential prefix)
            if ' - ' in summary[:20]:
                continue

            # Skip unbalanced brackets (indicates incomplete extraction)
            if summary.count('[') != summary.count(']'):
                continue
            if summary.count('(') != summary.count(')'):
                continue

            # Skip if ends with incomplete structural markers
            if summary.rstrip().endswith(('[', '(', '-', ':', ',')):
                continue

            # Must be mostly alphabetic
            alpha_ratio = sum(1 for c in summary if c.isalpha() or c.isspace()) / len(summary)
            if alpha_ratio < 0.8:
                continue

            dec = {
                "decision": summary[:120],
                "category": row['category'] or 'general',
            }
            if row['reasoning'] and len(row['reasoning']) > 15:
                reasoning = row['reasoning']
                # Skip garbage reasoning too
                if not any(p in reasoning.lower() for p in garbage_patterns[:5]):
                    dec["reasoning"] = reasoning[:150]

            decisions.append(dec)

    except Exception:
        pass

    return decisions


def _clean_recent_tasks(tasks: list) -> list:
    """
    Clean recent tasks - remove meta-questions and format properly.
    """
    cleaned = []

    # Patterns to filter out
    skip_patterns = [
        'what were we working on',
        'what did we',
        'show me',
        'can you',
        'please',
        '?',  # Questions aren't tasks
    ]

    for task in tasks:
        if not task or len(task) < 10:
            continue

        task_lower = task.lower()

        # Skip meta-questions and simple queries
        if any(pattern in task_lower for pattern in skip_patterns):
            continue

        # Skip slash commands
        if task.startswith('/'):
            continue

        # Truncate cleanly
        if len(task) > 100:
            task = task[:97] + "..."

        cleaned.append(task)

    return cleaned[:5]


def handle_init(params: dict, storage=None) -> dict:
    """
    Get comprehensive initialization context for the current session.

    Returns TIERED output:
    - TIER 1 (alerts): Actionable items requiring attention
    - TIER 2 (core): Essential context for immediate work
    - TIER 3 (details): Deeper context when needed

    Args:
        params: Request parameters (project_path)
        storage: Storage instance for central Qdrant + Postgres
    """
    project_path = params.get("project_path", "")
    mira_path = get_mira_path()

    # === PROJECT DESCRIPTION ===
    # This is what MIRA is - critical for fresh Claude understanding
    project_description = (
        "MIRA (Memory Information Retriever and Archiver) gives Claude Code "
        "persistent memory across sessions - the ability to recall past conversations, "
        "solutions, decisions, and patterns that would otherwise be lost."
    )

    # === ENVIRONMENT DETECTION ===
    environment = _get_environment_summary()

    # === SESSION COUNTS ===
    count = 0
    if storage and storage.using_central and hasattr(storage, 'postgres') and storage.postgres:
        try:
            with storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM sessions")
                    count = cur.fetchone()[0]
        except Exception:
            pass
    else:
        try:
            from mira.storage.local_store import get_session_count
            count = get_session_count()
        except Exception:
            pass

    # === ARTIFACT STATS ===
    artifact_stats = {'global': {'total': 0, 'by_type': {}}}
    try:
        from mira.extraction.artifacts import get_artifact_stats
        artifact_stats = get_artifact_stats(project_path)
    except ImportError:
        pass

    # === DECISION STATS ===
    decision_count = 0
    try:
        from mira.extraction.decisions import get_decision_stats
        decision_stats = get_decision_stats()
        decision_count = decision_stats.get('total', 0)
    except ImportError:
        pass

    # === STORAGE STATS ===
    storage_stats = get_simplified_storage_stats(mira_path)

    # === CUSTODIAN PROFILE ===
    custodian_profile = {'name': 'Unknown', 'summary': '', 'interaction_tips': []}
    try:
        from mira.custodian import get_full_custodian_profile
        custodian_profile = get_full_custodian_profile()
    except ImportError:
        pass

    if not custodian_profile.get('total_sessions'):
        custodian_profile['total_sessions'] = count

    # Build enriched summary
    custodian_profile['summary'] = build_enriched_custodian_summary(custodian_profile)

    # Build interaction tips - ensure full sentences
    raw_tips = build_interaction_tips(custodian_profile)
    # Filter out truncated tips (those ending with ...)
    custodian_profile['interaction_tips'] = [
        tip for tip in raw_tips
        if not tip.endswith('...') or len(tip) > 50
    ][:7]

    # === WORK CONTEXT ===
    work_context = {}
    try:
        from mira.custodian.work_context import get_current_work_context
        work_context = get_current_work_context(project_path)
    except ImportError:
        pass

    # Clean recent tasks - remove meta-questions
    if work_context.get('recent_tasks'):
        work_context['recent_tasks'] = _clean_recent_tasks(work_context['recent_tasks'])

    # === LAST SESSION SUMMARY ===
    last_session = _get_last_session_summary(mira_path, project_path)

    # === STRUCTURED DECISIONS ===
    structured_decisions = _get_structured_decisions(5)

    # === CODEBASE KNOWLEDGE ===
    codebase_knowledge = {}
    try:
        from mira.extraction.concepts import get_codebase_knowledge
        codebase_knowledge = get_codebase_knowledge(project_path)
    except ImportError:
        pass

    # === BUILD TIERED OUTPUT ===

    # TIER 1: Actionable alerts
    alerts = get_actionable_alerts(mira_path, project_path, custodian_profile)

    # TIER 2: Core context
    custodian_data = {
        'name': custodian_profile.get('name', 'Unknown'),
        'summary': custodian_profile.get('summary', ''),
        'interaction_tips': custodian_profile.get('interaction_tips', [])[:5],
        'total_sessions': custodian_profile.get('total_sessions', 0),
    }

    # Add development lifecycle
    dev_lifecycle = custodian_profile.get('development_lifecycle')
    if dev_lifecycle:
        confidence_pct = int(dev_lifecycle.get('confidence', 0) * 100)
        custodian_data['development_lifecycle'] = f"{dev_lifecycle.get('sequence')} ({confidence_pct}% confidence)"

    # Add danger zones
    danger_zones = custodian_profile.get('danger_zones', [])
    if danger_zones:
        custodian_data['danger_zones'] = danger_zones

    # TIER 3: Details
    filtered_knowledge = filter_codebase_knowledge(codebase_knowledge)
    has_meaningful_knowledge = any([
        filtered_knowledge.get('integrations'),
        filtered_knowledge.get('patterns'),
        filtered_knowledge.get('facts'),
        filtered_knowledge.get('rules'),
    ])

    details = {}
    if has_meaningful_knowledge:
        details['codebase_knowledge'] = filtered_knowledge

    # Artifact counts for guidance
    if 'global' in artifact_stats:
        global_artifact_total = artifact_stats['global'].get('total', 0)
        global_error_count = artifact_stats['global'].get('by_type', {}).get('error', 0)
        project_artifact_total = artifact_stats.get('project', {}).get('total', 0)
        project_error_count = artifact_stats.get('project', {}).get('by_type', {}).get('error', 0)
    else:
        global_artifact_total = artifact_stats.get('total', 0)
        global_error_count = artifact_stats.get('by_type', {}).get('error', 0)
        project_artifact_total = 0
        project_error_count = 0

    # Storage warnings
    data_bytes = storage_stats.get('data_bytes', 0)
    models_bytes = storage_stats.get('models_bytes', 0)
    total_bytes = data_bytes + models_bytes

    if data_bytes > 500 * 1024 * 1024:
        alerts.append({
            'type': 'storage_warning',
            'priority': 'medium',
            'message': f"MIRA data storage is large: {storage_stats['data']}",
            'suggestion': 'Consider pruning old archives',
        })

    # Build guidance
    guidance = build_claude_guidance(
        custodian_data, alerts, work_context,
        global_artifact_total=global_artifact_total, global_error_count=global_error_count,
        project_artifact_total=project_artifact_total, project_error_count=project_error_count,
        decision_count=decision_count
    )

    # === PERSONA CONTEXT ===
    # Reference to CLAUDE.md personality if it exists
    persona_context = None
    claude_md_path = Path(project_path) / "CLAUDE.md" if project_path else None
    if claude_md_path and claude_md_path.exists():
        persona_context = {
            "note": "Check CLAUDE.md for persona/personality instructions",
            "path": str(claude_md_path),
        }

    # Storage mode
    storage_mode = None
    if storage:
        try:
            storage_mode = storage.get_storage_mode()
        except Exception:
            storage_mode = {"mode": "local", "description": "Local storage only"}
    else:
        storage_mode = {
            "mode": "local",
            "description": "Using local SQLite storage (keyword search only, single-machine)",
            "limitations": [
                "Keyword search only (no semantic/vector search)",
                "History stays on this machine only",
            ],
        }

    # === BUILD RESPONSE ===
    response = {
        # PROJECT: What this is
        "project": {
            "description": project_description,
            "path": project_path or "unknown",
        },

        # ENVIRONMENT: Where we're running
        "environment": environment,

        # GUIDANCE: How Claude should use this context
        "guidance": guidance,

        # ALERTS: Check these first
        "alerts": alerts,

        # CORE: Essential context
        "core": {
            "custodian": custodian_data,
            "current_work": work_context,
        },

        # STORAGE: Sync status
        "storage": storage_mode,
    }

    # Add persona context if CLAUDE.md exists
    if persona_context:
        response["persona"] = persona_context

    # Add last session for continuity
    if last_session:
        response["last_session"] = last_session

    # Add structured decisions if available
    if structured_decisions:
        response["recent_decisions"] = structured_decisions

    # Local storage alert
    if storage_mode.get("mode") == "local":
        alerts.insert(0, {
            "type": "storage_mode",
            "priority": "info",
            "message": "Running in local mode (keyword search only, single-machine).",
        })

    # Indexing progress
    claude_path = Path.home() / ".claude" / "projects"
    total_files = sum(1 for _ in claude_path.rglob("*.jsonl")) if claude_path.exists() else 0

    if count < total_files:
        pending = total_files - count
        response["indexing"] = {
            "indexed": count,
            "total": total_files,
            "pending": pending,
        }
        if pending > 0:
            response["guidance"]["actions"].append(
                f"Indexing in progress: {count}/{total_files} sessions ({pending} pending)"
            )

    # Active ingestion
    try:
        from mira.ingestion import get_active_ingestions
        active_jobs = get_active_ingestions()
        if active_jobs:
            response["active_ingestion"] = {"count": len(active_jobs)}
            alerts.insert(0, {
                "type": "ingestion_active",
                "priority": "info",
                "message": f"MIRA is ingesting {len(active_jobs)} conversation(s)",
            })
    except ImportError:
        pass

    # Details section
    if details:
        response["details"] = details

    # Token estimate
    response_json = json.dumps(response, separators=(',', ':'))
    response["token_estimate"] = {
        "chars": len(response_json),
        "tokens": len(response_json) // 4,
    }

    return response
