"""
MIRA3 Claude Guidance Module

Builds actionable guidance for Claude sessions based on custodian profile,
alerts, work context, and MIRA statistics.
"""

import subprocess
from pathlib import Path

from mira.core import log
from mira.custodian.rules import format_rule_for_display, RULE_TYPES
from mira.custodian.prerequisites import check_prerequisites_and_alert


def build_claude_guidance(
    custodian: dict, alerts: list, work_context: dict,
    global_artifact_total: int = 0, global_error_count: int = 0,
    project_artifact_total: int = 0, project_error_count: int = 0,
    decision_count: int = 0
) -> dict:
    """
    Build actionable guidance for Claude on how to use the MIRA context.

    This tells a future Claude session exactly what to DO with the information,
    not just what the information IS.

    Includes:
    - mira_usage_triggers: When Claude should pause and consult MIRA
    - tool_quick_reference: Syntax and examples for each MIRA tool
    - actions: Specific behavioral guidance for this session
    """
    guidance = {
        "how_to_use_this": "This context is from past sessions. Use it to personalize your approach and CONSULT MIRA before reinventing solutions.",
        "mira_usage_triggers": [],
        "tool_quick_reference": {},
        "actions": []
    }

    # === BUILD USAGE TRIGGERS ===
    # These tell Claude WHEN to pause and consult MIRA
    triggers = []

    # Critical triggers - always include
    triggers.append({
        "situation": "Encountering an error, exception, or unexpected failure",
        "action": "BEFORE attempting to debug, call mira_error_lookup(query='<error message>')",
        "reason": f"Past solutions exist for {global_error_count} resolved errors - avoid duplicate debugging work" if global_error_count > 0 else "Past solutions may exist - check before debugging from scratch",
        "priority": "critical"
    })

    triggers.append({
        "situation": "User asks about unfamiliar system, process, or code area",
        "action": "PAUSE and call mira_search(query='<topic>') before exploring codebase",
        "reason": f"{global_artifact_total} artifacts from past sessions may document this - check memory first" if global_artifact_total > 0 else "Past sessions may document this - check memory before investigation",
        "priority": "critical"
    })

    triggers.append({
        "situation": "About to say 'I don't know' or 'I'm not familiar with' ANY name, term, or concept",
        "action": "STOP - first call mira_search(query='<unknown term>') before admitting ignorance",
        "reason": "The user may have discussed this in past sessions - NEVER say 'I don't know' without searching MIRA first",
        "priority": "critical"
    })

    triggers.append({
        "situation": "Making architectural, technology, or design decisions",
        "action": "Call mira_decisions(query='<decision topic>') to check precedents",
        "reason": f"{decision_count} past decisions with reasoning are logged - maintain project consistency" if decision_count > 0 else "Past decisions may be logged - check for precedents",
        "priority": "critical"
    })

    triggers.append({
        "situation": "User references past work ('we discussed this', 'we talked about', 'remember when', 'like last time', 'as we did before')",
        "action": "Call mira_search(query='<referenced topic>') immediately",
        "reason": "User expects continuity across sessions - search MIRA before asking them to repeat context",
        "priority": "critical"
    })

    # Danger zone trigger - dynamic based on custodian data
    danger_zones = custodian.get('danger_zones', [])
    if danger_zones:
        paths = [dz.get('path', '') for dz in danger_zones if dz.get('path')]
        if paths:
            total_issues = sum(dz.get('issue_count', 0) for dz in danger_zones)
            triggers.append({
                "situation": f"About to modify: {', '.join(paths[:4])}",
                "action": "Call mira_search(query='<filename>') to understand past issues with these files",
                "reason": f"These danger_zone files have {total_issues} combined recorded issues - learn from history before editing" if total_issues > 0 else "These files have caused issues before - learn from history before editing",
                "priority": "critical"
            })

    # Recommended triggers
    triggers.append({
        "situation": "Implementing a feature similar to existing functionality",
        "action": "Call mira_search(query='<feature type>') to find established patterns",
        "reason": "Maintain consistency with patterns already established in this codebase",
        "priority": "recommended"
    })

    triggers.append({
        "situation": "User seems frustrated or mentions something not working as expected",
        "action": "Call mira_error_lookup or mira_search for the problematic area",
        "reason": "This may be a recurring issue with known context and workarounds",
        "priority": "recommended"
    })

    # Optional trigger
    triggers.append({
        "situation": "Starting implementation of a multi-step or complex task",
        "action": "Call mira_search(query='<task description>') for prior attempts or related work",
        "reason": "Avoid repeating failed approaches or reinventing existing solutions",
        "priority": "optional"
    })

    guidance["mira_usage_triggers"] = triggers

    # === BUILD TOOL QUICK REFERENCE ===
    guidance["tool_quick_reference"] = {
        "mira_search": {
            "purpose": "Semantic search across all conversation history",
            "when": "Looking for past discussions, implementations, decisions, or any historical context",
            "syntax": "mira_search(query='<search terms>', limit=10, project_path='<optional>', days=<optional>, recency_bias=True)",
            "parameters": {
                "days": "Filter to last N days (hard cutoff)",
                "recency_bias": "Time decay boosts recent results (default True). Recent content ranks higher than old."
            },
            "recency_bias_guidance": {
                "default_true": "Most searches - recent context is usually more relevant",
                "set_false_when": [
                    "User asks about 'original', 'first', or 'initial' implementations",
                    "User asks 'why did we decide X' or 'when did we start doing Y'",
                    "User wants comprehensive results regardless of age",
                    "Searching for historical decisions or early architecture"
                ]
            },
            "examples": [
                "mira_search(query='authentication implementation')",
                "mira_search(query='recent bugs', days=7)",
                "mira_search(query='original architecture decision', recency_bias=False)",
                "mira_search(query='when did we first add caching', recency_bias=False)"
            ]
        },
        "mira_error_lookup": {
            "purpose": "Find past solutions to similar errors - searches error-specific index",
            "when": "Encountering ANY error, exception, stack trace, or unexpected failure",
            "syntax": "mira_error_lookup(query='<error message or description>', limit=5)",
            "examples": [
                "mira_error_lookup(query='TypeError: Cannot read property of undefined')",
                "mira_error_lookup(query='connection refused postgres')",
                "mira_error_lookup(query='CORS policy blocked')"
            ]
        },
        "mira_decisions": {
            "purpose": "Search architectural and design decisions with their reasoning and context",
            "when": "Making technology choices, architectural decisions, or wondering 'why was it done this way?'",
            "syntax": "mira_decisions(query='<decision topic>', category='<optional>', limit=10)",
            "categories": ["architecture", "technology", "implementation", "testing", "security", "performance", "workflow"],
            "examples": [
                "mira_decisions(query='state management')",
                "mira_decisions(query='database schema', category='architecture')",
                "mira_decisions(query='testing strategy')"
            ]
        },
        "mira_recent": {
            "purpose": "View summaries of recent conversation sessions",
            "when": "Starting a new session, need to understand recent work context, or user asks 'what were we working on?'",
            "syntax": "mira_recent(limit=10)"
        },
        "mira_status": {
            "purpose": "Check MIRA system health, ingestion progress, storage stats, and sync status",
            "when": "Debugging MIRA itself, checking if data is available, or verifying sync status",
            "syntax": "mira_status(project_path='<optional>')"
        }
    }

    # === BUILD ACTIONS (existing logic) ===

    # Artifact guidance - tell Claude there's searchable history
    # Show both project-specific and global counts for clarity
    if global_artifact_total > 100:
        # Build the message showing both scopes
        if project_artifact_total > 0:
            # Have project-specific data
            msg_parts = [f"Searchable history: {global_artifact_total} artifacts"]
            if global_error_count > 0:
                msg_parts.append(f"including {global_error_count} resolved errors")
            msg_parts[0] = msg_parts[0] + " (global)"

            # Add project-specific counts
            project_msg = f"{project_artifact_total} for this project"
            if project_error_count > 0:
                project_msg += f" ({project_error_count} errors)"
            msg_parts.append(project_msg)

            guidance["actions"].append(
                f"{', '.join(msg_parts)}. Use mira_search or mira_error_lookup for past solutions."
            )
        else:
            # No project data, just show global
            if global_error_count > 20:
                guidance["actions"].append(
                    f"Searchable history: {global_artifact_total} artifacts (global) including {global_error_count} resolved errors. "
                    "Use mira_search or mira_error_lookup for past solutions."
                )
            else:
                guidance["actions"].append(
                    f"Searchable history: {global_artifact_total} artifacts (global). "
                    "Use mira_search for past code, decisions, or patterns."
                )

    # User identity guidance - include session count to convey shared history
    name = custodian.get('name')
    total_sessions = custodian.get('total_sessions', 0)
    if name and name != 'Unknown':
        if total_sessions >= 50:
            guidance["actions"].append(
                f"Address user as {name} naturally (don't announce you know their name). "
                f"You have {total_sessions} sessions of shared history - reference past work when relevant."
            )
        elif total_sessions >= 10:
            guidance["actions"].append(
                f"Address user as {name} naturally (don't announce you know their name). "
                f"You have {total_sessions} sessions of shared context."
            )
        else:
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


def filter_codebase_knowledge(knowledge: dict) -> dict:
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


def get_actionable_alerts(mira_path: Path, project_path: str, custodian_profile: dict) -> list:
    """
    Generate actionable alerts that require attention.

    Alerts are prioritized issues or context that Claude should act on.
    """
    alerts = []

    # Check for uncommitted git changes
    project_root = mira_path.parent
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2  # Quick - don't block startup
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
                timeout=2  # Quick - don't block startup
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

    # Check for environment-specific prerequisites
    try:
        prereq_alerts = check_prerequisites_and_alert()
        # Insert at beginning since these are high priority
        alerts = prereq_alerts + alerts
    except Exception as e:
        log(f"Error checking prerequisites: {e}")

    return alerts


def get_simplified_storage_stats(mira_path: Path) -> dict:
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


def build_enriched_custodian_summary(profile: dict) -> str:
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

    # Important rules (most critical - show highest confidence)
    rules = profile.get('rules', {})
    never_rules = rules.get('never', [])
    always_rules = rules.get('always', [])
    require_rules = rules.get('require', [])

    if never_rules:
        rule = never_rules[0].get('rule', '')
        if rule:
            sentences.append(f"Important: never {format_rule_for_display(rule, 45)}.")

    if always_rules:
        rule = always_rules[0].get('rule', '')
        if rule:
            sentences.append(f"Always {format_rule_for_display(rule, 45)}.")

    if require_rules and not always_rules:  # Only if no always rules
        rule = require_rules[0].get('rule', '')
        if rule:
            sentences.append(f"Required: {format_rule_for_display(rule, 45)}.")

    # Danger zones
    danger_zones = profile.get('danger_zones', [])
    if danger_zones:
        paths = [dz.get('path', '').split('/')[-1] for dz in danger_zones[:2] if dz.get('path')]
        if paths:
            sentences.append(f"Caution with: {', '.join(paths)}.")

    return ' '.join(sentences)


def build_interaction_tips(profile: dict) -> list:
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

    # Rules - handle all rule types with proper formatting
    rules = profile.get('rules', {})

    # Priority order for display: never, always, require, prefer, avoid, prohibit, style
    rule_display_order = ['never', 'always', 'require', 'prefer', 'avoid', 'prohibit', 'style']
    rules_added = 0
    max_rules = 6  # Limit total rules in tips

    for rule_type in rule_display_order:
        if rules_added >= max_rules:
            break
        type_rules = rules.get(rule_type, [])
        display_name = RULE_TYPES.get(rule_type, rule_type.capitalize())

        for rule in type_rules[:2]:  # Max 2 per type
            if rules_added >= max_rules:
                break
            rule_text = rule.get('rule', '')
            if rule_text and len(rule_text) >= 10:
                # Use longer limit and skip if truncation is awkward
                formatted = format_rule_for_display(rule_text, 80)
                # Skip if truncation cuts off mid-thought
                if formatted.endswith('...'):
                    # Bad truncation indicators:
                    # - Too short after truncation
                    # - Ends with articles, prepositions, conjunctions
                    # - Ends with incomplete words (no space before ...)
                    bad_endings = [' a...', ' an...', ' the...', ' with...', ' to...',
                                   ' for...', ' and...', ' or...', ' in...', ' on...',
                                   ' by...', ' of...', ' that...', ' this...', ' is...']
                    if len(formatted) < 45 or any(formatted.endswith(e) for e in bad_endings):
                        continue
                tips.append(f"{display_name}: {formatted}")
                rules_added += 1

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
