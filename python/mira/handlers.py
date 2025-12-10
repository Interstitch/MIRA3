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
from .custodian import get_full_custodian_profile, get_danger_zones_for_files
from .insights import search_error_solutions, search_decisions, get_error_stats, get_decision_stats
from .concepts import get_codebase_knowledge, ConceptStore


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
    """
    limit = params.get("limit", 10)

    # Try storage (central or local SQLite via storage abstraction)
    if storage:
        try:
            sessions = storage.get_recent_sessions(limit=limit)
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
                        "timestamp": str(session.get("started_at", ""))
                    })

                source = "central" if storage.using_central else "local_sqlite"
                return {
                    "projects": [{"path": k, "sessions": v} for k, v in projects.items()],
                    "total": len(sessions),
                    "source": source
                }
        except Exception as e:
            log(f"Storage recent query failed: {e}")

    # Fallback to local metadata files
    mira_path = get_mira_path()
    metadata_path = mira_path / "metadata"

    sessions = []
    if metadata_path.exists():
        for meta_file in sorted(metadata_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            try:
                meta = json.loads(meta_file.read_text())
                raw_path = meta.get("project_path", "")
                sessions.append({
                    "session_id": meta_file.stem,
                    "summary": meta.get("summary", ""),
                    "project_path": _format_project_path(raw_path),
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
        "total": len(sessions),
        "source": "local"
    }


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

    # Get indexed count from central storage
    count = 0
    if storage and storage.using_central and storage.postgres:
        try:
            # Count sessions in postgres
            with storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM sessions")
                    count = cur.fetchone()[0]
        except Exception:
            pass

    # Get artifact stats
    artifact_stats = get_artifact_stats()

    # Calculate storage sizes (simplified)
    storage_stats = _get_simplified_storage_stats(mira_path)

    # Get rich custodian profile (learned from conversations)
    custodian_profile = get_full_custodian_profile()

    # Merge with legacy profile stats (for total_sessions used in summary)
    legacy_profile = get_custodian_profile()
    custodian_profile['total_sessions'] = legacy_profile.get('total_sessions', 0)
    custodian_profile['total_messages'] = legacy_profile.get('total_messages', 0)

    # Build a richer summary now that we have all the data
    custodian_profile['summary'] = _build_enriched_custodian_summary(custodian_profile)

    # Add interaction tips based on learned preferences
    custodian_profile['interaction_tips'] = _build_interaction_tips(custodian_profile)

    # Get current work context
    work_context = get_current_work_context()

    # Get codebase knowledge learned from conversations
    # NOTE: This only returns genuinely learned content - no CLAUDE.md echoes
    codebase_knowledge = get_codebase_knowledge(project_path)

    # === BUILD TIERED OUTPUT ===

    # TIER 1: Actionable alerts (always check these)
    alerts = _get_actionable_alerts(mira_path, project_path, custodian_profile)

    # TIER 2: Core context (essential for immediate work)
    # Custodian - behavioral info only (tech_stack is in CLAUDE.md, not actionable here)
    custodian_data = {
        'name': custodian_profile.get('name', 'Unknown'),
        'summary': custodian_profile.get('summary', ''),
        'interaction_tips': custodian_profile.get('interaction_tips', [])[:5],
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
    filtered_knowledge = _filter_codebase_knowledge(codebase_knowledge)

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
    artifact_total = artifact_stats.get('total', 0)
    error_count = artifact_stats.get('by_type', {}).get('error', 0)

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
    guidance = _build_claude_guidance(
        custodian_data, alerts, work_context,
        artifact_total=artifact_total, error_count=error_count
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

    # Only include details if there's meaningful learned knowledge
    if details:
        response["details"] = details

    return response


def _build_claude_guidance(
    custodian: dict, alerts: list, work_context: dict,
    artifact_total: int = 0, error_count: int = 0
) -> dict:
    """
    Build actionable guidance for Claude on how to use the MIRA context.

    This tells a future Claude session exactly what to DO with the information,
    not just what the information IS.
    """
    guidance = {
        "how_to_use_this": "This context is from past sessions. Use it to personalize your approach.",
        "actions": []
    }

    # Artifact guidance - tell Claude there's searchable history
    if artifact_total > 100:
        if error_count > 20:
            guidance["actions"].append(
                f"Searchable history: {artifact_total} artifacts including {error_count} resolved errors. "
                "Use mira_search or mira_error_lookup for past solutions."
            )
        else:
            guidance["actions"].append(
                f"Searchable history: {artifact_total} artifacts available. "
                "Use mira_search for past code, decisions, or patterns."
            )

    # User identity guidance
    name = custodian.get('name')
    if name and name != 'Unknown':
        guidance["actions"].append(f"Address user as {name} naturally (don't announce you know their name)")

    # Development lifecycle guidance - ENFORCE the user's established workflow
    # This is key: Claude should actively push back if user skips steps
    lifecycle = custodian.get('development_lifecycle')
    if lifecycle:
        # Parse the lifecycle to give specific enforcement guidance
        lifecycle_lower = lifecycle.lower()
        has_commit = 'commit' in lifecycle_lower
        has_test = 'test' in lifecycle_lower
        has_plan = 'plan' in lifecycle_lower

        # Add the workflow enforcement action
        guidance["actions"].append(f"User's established workflow: {lifecycle}. ENFORCE this sequence.")

        # Add specific enforcement prompts for each phase
        if has_plan:
            guidance["actions"].append(
                "If user jumps straight to implementation, PAUSE and ask: "
                "'Should we outline the approach first?'"
            )
        if has_test:
            guidance["actions"].append(
                "Before marking work complete, prompt: 'Should we write/run tests for this?'"
            )
        if has_commit:
            guidance["actions"].append(
                "After completing a logical unit of work, prompt: 'Ready to commit these changes?'"
            )

    # Interaction tips - convert to actions
    tips = custodian.get('interaction_tips', [])
    for tip in tips:
        tip_lower = tip.lower()
        if 'iterative' in tip_lower:
            guidance["actions"].append("Make incremental changes rather than large rewrites")
        elif 'planning' in tip_lower:
            guidance["actions"].append("Outline your approach before writing code")
        elif 'concise' in tip_lower:
            guidance["actions"].append("Keep responses brief - avoid over-explaining")
        elif 'detailed' in tip_lower:
            guidance["actions"].append("Provide thorough explanations with your code")

    # Alert-based guidance
    high_priority_alerts = [a for a in alerts if a.get('priority') == 'high']
    if high_priority_alerts:
        for alert in high_priority_alerts[:2]:
            if alert.get('type') == 'git_uncommitted':
                modified = alert.get('modified', [])
                if modified:
                    guidance["actions"].append(
                        f"User has uncommitted changes in: {', '.join(modified[:3])}. "
                        "Acknowledge this context if relevant to their request."
                    )
            elif alert.get('type') == 'danger_zone':
                guidance["actions"].append(
                    f"CAUTION: {alert.get('message')}. Proceed carefully and confirm changes."
                )

    # Current work context guidance
    active_topics = work_context.get('active_topics', [])
    if active_topics:
        guidance["actions"].append(
            f"Recent work context: '{active_topics[0][:60]}...'. "
            "Reference this if the user's request seems related."
        )

    # Danger zones guidance
    danger_zones = custodian.get('danger_zones', [])
    if danger_zones:
        paths = [dz.get('path', '') for dz in danger_zones[:2]]
        guidance["actions"].append(
            f"Files that caused past issues: {', '.join(paths)}. "
            "Be extra careful when modifying these."
        )

    # Deduplicate and limit actions
    seen = set()
    unique_actions = []
    for action in guidance["actions"]:
        key = action[:50].lower()
        if key not in seen:
            seen.add(key)
            unique_actions.append(action)
    guidance["actions"] = unique_actions[:8]  # Max 8 actions

    return guidance


def _filter_codebase_knowledge(knowledge: dict) -> dict:
    """
    Filter codebase_knowledge to ONLY learned content.

    Removes:
    1. Empty arrays (no value)
    2. CLAUDE.md-sourced entries (already in context)
    3. Redundant/low-value entries
    4. architecture_summary (derived from CLAUDE.md)

    Keeps only genuinely learned knowledge from conversation analysis.
    """
    filtered = {}

    # NOTE: Removed architecture_summary - it's derived from CLAUDE.md parsing,
    # which Claude already has in context. Only include genuinely learned content.

    # Integrations - learned communication patterns between components
    integrations = knowledge.get('integrations', [])
    if integrations:
        filtered['integrations'] = integrations

    # Patterns - learned design patterns from conversations
    patterns = knowledge.get('patterns', [])
    if patterns:
        filtered['patterns'] = patterns

    # Facts - user-provided facts about the codebase
    facts = knowledge.get('facts', [])
    if facts:
        filtered['facts'] = facts

    # Rules - user-provided conventions and requirements
    rules = knowledge.get('rules', [])
    if rules:
        filtered['rules'] = rules

    # Skip: architecture_summary, components, technologies, key_modules, hot_files
    # These either duplicate CLAUDE.md or aren't actionable

    return filtered


def _get_actionable_alerts(mira_path: Path, project_path: str, custodian_profile: dict) -> list:
    """
    Generate actionable alerts that require attention.

    Alerts are prioritized issues or context that Claude should act on.
    """
    import subprocess
    alerts = []

    # Check for uncommitted git changes
    project_root = mira_path.parent
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            # Git porcelain format: XY filename (X=index, Y=worktree)
            # Extract filename by skipping first 3 chars (XY + space)
            # But handle edge case where there might be extra/fewer spaces
            def extract_path(line):
                # Skip the 2-char status prefix, then strip any leading space
                return line[2:].lstrip() if len(line) > 2 else line

            modified = [extract_path(l) for l in lines if l[1:2] == 'M' or l[0:1] == 'M']
            added = [extract_path(l) for l in lines if l.startswith('A ') or l.startswith('??')]
            deleted = [extract_path(l) for l in lines if l[1:2] == 'D' or l[0:1] == 'D']

            if modified or added or deleted:
                alert = {
                    'type': 'git_uncommitted',
                    'priority': 'high',
                    'message': f"Uncommitted changes: {len(modified)} modified, {len(added)} new, {len(deleted)} deleted",
                }
                if modified:
                    alert['modified'] = modified[:10]
                if added:
                    alert['new'] = added[:10]
                if deleted:
                    alert['deleted'] = deleted[:5]
                alerts.append(alert)
    except Exception:
        pass

    # Check for danger zones in recently touched files
    danger_zones = custodian_profile.get('danger_zones', [])
    if danger_zones:
        recent_files = []
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD~5..HEAD'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                recent_files = result.stdout.strip().split('\n')
        except Exception:
            pass

        for dz in danger_zones:
            dz_path = dz.get('path', '')
            for rf in recent_files:
                if dz_path in rf:
                    alerts.append({
                        'type': 'danger_zone',
                        'priority': 'medium',
                        'message': f"Recent changes to danger zone: {dz_path}",
                        'reason': dz.get('reason', 'Has caused issues before'),
                    })
                    break

    # Check for any "never" rules that might apply
    rules = custodian_profile.get('rules', {})
    never_rules = rules.get('never', [])
    if never_rules:
        alerts.append({
            'type': 'reminder',
            'priority': 'low',
            'message': f"User rule: never {never_rules[0].get('rule', '')}",
        })

    return alerts


def _get_simplified_storage_stats(mira_path: Path) -> dict:
    """Get simplified storage stats - just the essentials."""
    def format_size(bytes_size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f} TB"

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

    # Only calculate essential sizes (chroma no longer used)
    data_size = (
        get_dir_size(mira_path / 'archives') +
        get_dir_size(mira_path / 'metadata')
    )

    # Add databases
    for db in ['artifacts.db', 'custodian.db', 'insights.db', 'concepts.db']:
        db_path = mira_path / db
        if db_path.exists():
            try:
                data_size += db_path.stat().st_size
            except (OSError, PermissionError):
                pass

    models_size = get_dir_size(mira_path / 'models')

    # Return stats with raw bytes for threshold checks
    return {
        'data': format_size(data_size),
        'data_bytes': data_size,
        'models': format_size(models_size),
        'models_bytes': models_size,
    }


def _build_enriched_custodian_summary(profile: dict) -> str:
    """
    Build a natural language summary of the custodian.

    Creates a concise, readable paragraph instead of pipe-separated fields.
    Emphasizes team context (sole developer vs team member) as this affects
    how Claude should interact (no coordination needed vs collaborative context).
    """
    name = profile.get('name', 'Unknown')
    total_sessions = profile.get('total_sessions', 0)
    total_messages = profile.get('total_messages', 0)

    if total_sessions == 0:
        return f"New user: {name}. No conversation history yet."

    # Start with basic info
    sentences = []

    # Team context - single user means sole developer (no coordination needed)
    # This is determined by custodian detection - one name = sole developer
    # Future: could track multiple custodians per project for team context
    if total_sessions >= 5:
        sentences.append(f"{name} is the sole developer on this project ({total_sessions} sessions).")
    else:
        sentences.append(f"{name} is working on this project ({total_sessions} sessions).")

    # Key preferences (communication style)
    # Filter out generic approval words that aren't actual preferences
    generic_approval_words = {
        'proceed', 'continue', 'yes', 'ok', 'okay', 'sure', 'thanks', 'thank',
        'good', 'great', 'nice', 'perfect', 'go', 'ahead', 'do', 'it', 'please',
        'looks', 'lgtm', 'ship'
    }
    preferences = profile.get('preferences', {})
    comm_prefs = preferences.get('communication', [])
    if comm_prefs:
        # Filter to actual preferences, not approval words
        real_prefs = [
            p['preference'] for p in comm_prefs
            if p.get('preference') and p['preference'].lower() not in generic_approval_words
        ]
        if real_prefs:
            sentences.append(f"Prefers {real_prefs[0].lower()}.")

    # Important rules (most critical)
    rules = profile.get('rules', {})
    never_rules = rules.get('never', [])
    always_rules = rules.get('always', [])

    if never_rules:
        rule = never_rules[0].get('rule', '')
        if rule:
            sentences.append(f"Important: never {rule[:50]}.")

    if always_rules:
        rule = always_rules[0].get('rule', '')
        if rule:
            sentences.append(f"Always {rule[:50]}.")

    # Danger zones
    danger_zones = profile.get('danger_zones', [])
    if danger_zones:
        paths = [dz.get('path', '').split('/')[-1] for dz in danger_zones[:2] if dz.get('path')]
        if paths:
            sentences.append(f"Caution with: {', '.join(paths)}.")

    return ' '.join(sentences)


def _build_interaction_tips(profile: dict) -> list:
    """
    Build a list of interaction tips for Claude based on learned custodian preferences.

    These tips help a future Claude session understand how to interact with this
    specific custodian based on their observed communication patterns and rules.

    NOTE: Development lifecycle is NOT included here - it's shown separately in
    custodian_data['development_lifecycle'] and enforced via guidance.actions.
    """
    tips = []

    # NOTE: Skipping development lifecycle here - it's already in custodian_data['development_lifecycle']
    # and more importantly, it's enforced via guidance.actions with specific prompts

    # Communication preferences
    preferences = profile.get('preferences', {})
    comm_prefs = preferences.get('communication', [])

    for pref in comm_prefs:
        pref_text = pref.get('preference', '').lower()

        # Map preferences to actionable tips
        if 'concise' in pref_text or 'brief' in pref_text:
            tips.append("Prefers concise responses - avoid verbose explanations")
        elif 'detailed' in pref_text or 'verbose' in pref_text:
            tips.append("Prefers detailed explanations - be thorough")
        elif 'no emoji' in pref_text:
            tips.append("Do not use emojis in responses")
        elif 'code first' in pref_text:
            tips.append("Show code before explanations")
        elif "don't ask" in pref_text or 'prompt me' in pref_text:
            tips.append("Proceed without asking questions when task is clear")
        elif 'step by step' in pref_text:
            tips.append("Break down complex tasks step by step")
        elif "don't commit" in pref_text or "i'll commit" in pref_text:
            tips.append("Don't commit changes - user prefers to commit manually")
        elif 'commit' in pref_text and ('often' in pref_text or 'frequently' in pref_text):
            tips.append("Make frequent, small commits as you work")
        elif 'explain' in pref_text:
            tips.append("Explain your reasoning as you work")

    # Rules
    rules = profile.get('rules', {})

    for never_rule in rules.get('never', [])[:3]:
        rule_text = never_rule.get('rule', '')
        if rule_text:
            tips.append(f"Never: {rule_text}")

    for always_rule in rules.get('always', [])[:3]:
        rule_text = always_rule.get('rule', '')
        if rule_text:
            tips.append(f"Always: {rule_text}")

    # Work patterns
    work_patterns = profile.get('work_patterns', [])
    for pattern in work_patterns[:2]:
        pattern_desc = pattern.get('pattern', '')
        if pattern_desc:
            tips.append(f"Work pattern: {pattern_desc}")

    # Danger zones
    danger_zones = profile.get('danger_zones', [])
    if danger_zones:
        tips.append(f"Be careful with: {', '.join(dz.get('path', '') for dz in danger_zones[:3])}")

    return tips[:10]  # Limit to 10 most relevant tips


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


def _normalize_task(task: str) -> str:
    """Normalize a task string for deduplication."""
    # Strip common prefixes that add no value
    normalized = task.strip()
    for prefix in ['Task: ', 'task: ', 'TODO: ', 'todo: ']:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    # Normalize whitespace and truncate for comparison
    normalized = ' '.join(normalized.split())[:100].lower()
    return normalized


def _extract_topic_keywords(text: str) -> set:
    """Extract significant keywords from a topic for similarity matching."""
    # Common words to ignore
    stopwords = {
        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'is',
        'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'this', 'that', 'these', 'those', 'it', 'its', 'with', 'from', 'by', 'as',
        'task', 'analyze', 'analysis', 'regarding', 'about', 'context', 'project',
        'efficiency', 'init', 'mira', 'ignore', 'any', 'finding', 'findings',
    }

    words = set()
    for word in text.lower().split():
        # Clean punctuation
        word = word.strip('.,!?:;()[]{}"\'-')
        if len(word) >= 3 and word not in stopwords:
            words.add(word)
    return words


def _is_duplicate_task(new_task: str, existing_tasks: list) -> bool:
    """Check if a task is a duplicate of an existing one."""
    new_norm = _normalize_task(new_task)
    new_keywords = _extract_topic_keywords(new_task)

    for existing in existing_tasks:
        existing_norm = _normalize_task(existing)

        # Check for prefix similarity (one contains the start of the other)
        if new_norm.startswith(existing_norm[:40]) or existing_norm.startswith(new_norm[:40]):
            return True

        # Check for keyword overlap (>60% shared keywords = duplicate)
        if new_keywords:
            existing_keywords = _extract_topic_keywords(existing)
            if existing_keywords:
                overlap = len(new_keywords & existing_keywords)
                smaller_set = min(len(new_keywords), len(existing_keywords))
                if smaller_set > 0 and overlap / smaller_set > 0.6:
                    return True

    return False


def _string_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity ratio between two strings.

    Uses multiple methods for robustness.
    """
    if not s1 or not s2:
        return 0.0

    # Normalize
    s1 = ' '.join(s1.lower().split())
    s2 = ' '.join(s2.lower().split())

    # Quick check for near-identical
    if s1 == s2:
        return 1.0

    # Method 1: Word overlap (Jaccard on words)
    words1 = set(s1.split())
    words2 = set(s2.split())
    if words1 and words2:
        word_intersection = len(words1 & words2)
        word_union = len(words1 | words2)
        word_sim = word_intersection / word_union if word_union > 0 else 0.0
    else:
        word_sim = 0.0

    # Method 2: Character bigrams (catches typos)
    def get_bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1))

    b1 = get_bigrams(s1)
    b2 = get_bigrams(s2)

    if b1 and b2:
        bigram_intersection = len(b1 & b2)
        bigram_union = len(b1 | b2)
        bigram_sim = bigram_intersection / bigram_union if bigram_union > 0 else 0.0
    else:
        bigram_sim = 0.0

    # Return the higher of the two (catches both word-level and char-level similarity)
    return max(word_sim, bigram_sim)


def _dedupe_task_list(tasks: list) -> list:
    """
    Aggressively deduplicate a list of tasks.

    Keeps the shortest/cleanest version when duplicates are found.
    """
    if not tasks:
        return []

    # Group similar tasks together
    groups = []
    for task in tasks:
        task_norm = _normalize_task(task)

        found_group = False
        for group in groups:
            # Compare against first item in group (the representative)
            rep = group[0]
            rep_norm = _normalize_task(rep)

            # Check prefix similarity
            if task_norm.startswith(rep_norm[:35]) or rep_norm.startswith(task_norm[:35]):
                group.append(task)
                found_group = True
                break

            # Check string similarity (catches typos and minor rewording)
            # Use 0.5 threshold - tasks about the same topic share ~50%+ words
            if _string_similarity(task_norm, rep_norm) > 0.5:
                group.append(task)
                found_group = True
                break

        if not found_group:
            groups.append([task])

    # Pick the best representative from each group (prefer shorter, cleaner)
    result = []
    for group in groups:
        # Sort by length, prefer shorter
        group.sort(key=len)
        result.append(group[0])

    return result


def _is_completed_topic(topic: str) -> bool:
    """
    Check if a topic appears to be completed based on various signals.

    Signals that a topic is complete:
    1. Contains past-tense completion words (completed, done, finished, implemented)
    2. Refers to known completed work in MIRA3
    3. Contains TODO status markers indicating completion
    """
    topic_lower = topic.lower()

    # Explicit completion markers in the topic text itself
    completion_markers = [
        'completed', ' done', 'finished', 'implemented', 'fixed',
        'resolved', 'merged', 'shipped', 'deployed', 'working',
        'added', 'created', 'built', 'verified', 'tested', 'passed',
        'setup complete', 'successfully', 'success', 'now works',
    ]

    for marker in completion_markers:
        if marker in topic_lower:
            return True

    # Skip TODOs that show their status as completed
    if 'status: completed' in topic_lower or '"status": "completed"' in topic_lower:
        return True

    # Skip topics from TODO snapshots that include status info
    if '"status":' in topic_lower:
        # If we see status info, check if it's not pending
        if '"pending"' not in topic_lower and '"in_progress"' not in topic_lower:
            return True

    # Known completed topics for this project (stale if they persist)
    known_completed = [
        'remove faiss',  # FAISS was removed
        'faiss keyword',  # Was removed
        'faiss index',  # Was removed
        'faiss reference',  # Checking FAISS removal - done
        'add embedding model',  # Using all-MiniLM-L6-v2
        'switch to chromadb',  # Already using ChromaDB
        'chromadb indexing',  # Already done
        'tf-idf index',  # Replaced by ChromaDB
        'full mcp server integration',  # Done
        'conversation parsing',  # Already exists
        'ingestion pipeline',  # Already exists
        'cosine distance',  # Already configured
        'add conversation',  # Generic "add" tasks are usually done
        'custodian learning',  # Already implemented
        'insights extraction',  # Already implemented
        'codebase concepts',  # Already implemented
        'tech_stack noise',  # Fixed
        'architecture field',  # Fixed
        # Recently completed improvements
        'add get_codebase_knowledge',  # Done
        'add technology extraction',  # Done - technologies from CLAUDE.md
        'technology extraction from claude',  # Done
        'improve key facts',  # Done
        'improve keyword extraction',  # Done
        'improve summary generation',  # Done
        'analyze chromadb',  # Done - reviewed usage
        'analyze all-minilm',  # Done - reviewed embedding model
        'custodian interaction',  # Done - added interaction tips
        'journey stats',  # Done
        'milestones',  # Done
        'key files table',  # Done - parsing CLAUDE.md table
        'architecture details',  # Done - extracting component details
        'preferences filtering',  # Done
    ]

    for completed in known_completed:
        if completed in topic_lower:
            return True

    return False


def _filter_active_topics(topics: list) -> list:
    """Filter out completed or stale topics."""
    active = []
    seen_normalized = set()

    for topic in topics:
        # Skip empty or very short topics
        if not topic or len(topic.strip()) < 10:
            continue

        # Check if completed
        if _is_completed_topic(topic):
            continue

        # Deduplicate similar topics
        normalized = _normalize_task(topic)[:60]
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)

        active.append(topic)

    return active


def _is_valid_decision(fact: str) -> bool:
    """
    Validate that a fact/decision is meaningful and not garbage.

    Filters out:
    - Concatenated/truncated sentences
    - Tool output fragments
    - Generic filler text
    """
    if not fact or len(fact.strip()) < 15:
        return False

    fact_lower = fact.lower()

    # Garbage patterns (concatenated text, tool fragments, debug output)
    garbage_patterns = [
        'continues with',  # Tool continuation
        'tool invocation',  # Tool output
        'no manual setup',  # Generic setup text
        '<tool_use>',  # Tool markup
        '</tool_use>',
        'function_calls',  # Internal markup
        '```',  # Code block markers
        'let me check',  # Process narration
        'i\'ll now',
        'please wait',
        'looking at the',
        'reviewing the',
        # Debug/status output patterns
        'now contain',  # "The key_facts now contain..."
        'now has',  # Status output
        'now shows',
        'now includes',
        '- empty arrays',  # List items in debug output
        '- "',  # Quoted list items
        'which is better than',  # Comparison/reasoning
    ]

    for pattern in garbage_patterns:
        if pattern in fact_lower:
            return False

    # Must start with a capital letter (proper sentence)
    if not fact[0].isupper():
        return False

    # Must end with proper punctuation
    if not fact.rstrip()[-1] in '.!?':
        return False

    # Check for sentence coherence - must have at least a subject/verb structure
    # Simple heuristic: must have at least 3 words
    words = fact.split()
    if len(words) < 4:
        return False

    # Must be primarily alphabetic text
    alpha_ratio = sum(1 for c in fact if c.isalpha() or c.isspace()) / len(fact)
    if alpha_ratio < 0.75:
        return False

    return True


def _filter_recent_decisions(facts: list) -> list:
    """Filter and deduplicate recent decisions/facts."""
    valid = []
    seen_normalized = set()

    for fact in facts:
        if not _is_valid_decision(fact):
            continue

        # Deduplicate
        normalized = _normalize_task(fact)[:60]
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)

        valid.append(fact)

    return valid


def get_current_work_context() -> dict:
    """
    Get context about current/recent work.

    Returns only non-empty fields to minimize token waste.
    """
    mira_path = get_mira_path()
    metadata_path = mira_path / "metadata"

    recent_tasks = []
    active_topics = []

    if not metadata_path.exists():
        return {}

    # Scan more files to find diverse tasks (not just recent repeats of same work)
    recent_files = sorted(
        metadata_path.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:15]

    for meta_file in recent_files:
        try:
            meta = json.loads(meta_file.read_text())

            task = meta.get('task_description', '')
            # Skip command messages and empty tasks
            if task and not task.startswith('<command-') and not task.startswith('/'):
                # Check for duplicates (similar task descriptions)
                if not _is_duplicate_task(task, recent_tasks):
                    recent_tasks.append(task)

            # Use session SUMMARY as active topic, not granular todo items
            # Summaries capture the high-level work theme ("MIRA Init Optimization")
            # while todo_topics are implementation details ("Fix full_path")
            summary = meta.get('summary', '')
            if summary and len(summary) >= 10:
                if not _is_duplicate_task(summary, active_topics):
                    active_topics.append(summary)

        except Exception:
            pass

    # Final aggressive deduplication - keep only truly distinct tasks
    recent_tasks = _dedupe_task_list(recent_tasks)

    # Build context with only non-empty fields
    context = {}

    recent_tasks = recent_tasks[:5]
    if recent_tasks:
        context['recent_tasks'] = recent_tasks

    # Filter out completed/stale topics AND cross-dedupe against recent_tasks
    active_topics = _filter_active_topics(active_topics)

    # Remove topics that duplicate recent_tasks (summaries often echo task descriptions)
    if recent_tasks:
        unique_topics = []
        for topic in active_topics:
            if not _is_duplicate_task(topic, recent_tasks):
                unique_topics.append(topic)
        active_topics = unique_topics

    active_topics = active_topics[:3]  # Reduced from 5 - topics should be distinct themes
    if active_topics:
        context['active_topics'] = active_topics

    return context


def handle_status(collection, storage=None) -> dict:
    """
    Get system status and statistics.

    Args:
        collection: Deprecated - kept for API compatibility, ignored
        storage: Storage instance for central Qdrant + Postgres
    """
    mira_path = get_mira_path()
    claude_path = Path.home() / ".claude" / "projects"

    # Count source files
    total_files = 0
    if claude_path.exists():
        total_files = sum(1 for _ in claude_path.rglob("*.jsonl"))

    # Count archived files
    archives_path = mira_path / "archives"
    archived = sum(1 for _ in archives_path.glob("*.jsonl")) if archives_path.exists() else 0

    # Count indexed from central storage
    indexed = 0
    if storage and storage.using_central and storage.postgres:
        try:
            with storage.postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM sessions")
                    indexed = cur.fetchone()[0]
        except Exception:
            pass

    # Get insights stats
    error_stats = get_error_stats()
    decision_stats = get_decision_stats()

    # Get health check info
    health = {}
    if storage:
        health = storage.health_check()

    return {
        "total_files": total_files,
        "archived": archived,
        "indexed": indexed,
        "pending": total_files - archived,
        "storage_path": str(mira_path),
        "last_sync": datetime.now().isoformat(),
        "errors": error_stats,
        "decisions": decision_stats,
        "storage_health": health
    }


def handle_error_lookup(params: dict, storage=None) -> dict:
    """
    Search for past error solutions.

    Args:
        params: Request parameters (query, limit)
        storage: Storage instance for central Qdrant + Postgres

    Params:
        query: Error message or description to search for
        limit: Maximum results (default: 5)

    Returns matching errors with their solutions.
    """
    query = params.get("query", "")
    limit = params.get("limit", 5)

    if not query:
        return {"results": [], "total": 0}

    # Try central storage first
    if storage and storage.using_central:
        try:
            results = storage.search_error_patterns(query, limit=limit)
            if results:
                return {
                    "results": results,
                    "total": len(results),
                    "query": query,
                    "source": "central"
                }
        except Exception:
            pass

    # Fall back to local search
    results = search_error_solutions(query, limit=limit)

    return {
        "results": results,
        "total": len(results),
        "query": query,
        "source": "local"
    }


def handle_decisions(params: dict, storage=None) -> dict:
    """
    Search for past architectural/design decisions.

    Args:
        params: Request parameters (query, category, limit)
        storage: Storage instance for central Qdrant + Postgres

    Params:
        query: Search query
        category: Optional category filter (architecture, technology, etc.)
        limit: Maximum results (default: 10)

    Returns matching decisions with context.
    """
    query = params.get("query", "")
    category = params.get("category")
    limit = params.get("limit", 10)

    # Try central storage first
    if storage and storage.using_central:
        try:
            results = storage.search_decisions(
                query=query or "",
                category=category,
                limit=limit
            )
            if results:
                return {
                    "results": results,
                    "total": len(results),
                    "query": query,
                    "category": category,
                    "source": "central"
                }
        except Exception:
            pass

    # Fall back to local search
    if not query:
        results = search_decisions("", category=category, limit=limit)
    else:
        results = search_decisions(query, category=category, limit=limit)

    return {
        "results": results,
        "total": len(results),
        "query": query,
        "category": category,
        "source": "local"
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
            # search.py's handle_search uses storage for central, ignores collection
            result = handle_search(params, collection, storage)
        elif method == "recent":
            result = handle_recent(params, storage)
        elif method == "init":
            result = handle_init(params, collection, storage)
        elif method == "status":
            result = handle_status(collection, storage)
        elif method == "error_lookup":
            result = handle_error_lookup(params, storage)
        elif method == "decisions":
            result = handle_decisions(params, storage)
        elif method == "shutdown":
            result = {"status": "shutting_down"}
        else:
            error = {"code": -32601, "message": f"Method not found: {method}"}
    except Exception as e:
        # Log full error for debugging, return sanitized message
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
