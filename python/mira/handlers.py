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


def handle_recent(params: dict) -> dict:
    """Get recent conversation sessions."""
    limit = params.get("limit", 10)

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
        "total": len(sessions)
    }


def handle_init(params: dict, collection) -> dict:
    """
    Get comprehensive initialization context for the current session.

    Returns TIERED output:
    - TIER 1 (alerts): Actionable items requiring attention
    - TIER 2 (core): Essential context for immediate work
    - TIER 3 (details): Deeper context when needed

    The output is organized to prioritize actionable information first.
    """
    project_path = params.get("project_path", "")
    mira_path = get_mira_path()

    # Get project overview from CLAUDE.md
    project_overview = _get_project_overview(mira_path)

    # Get recent sessions
    recent = handle_recent({"limit": 5})

    # Get collection stats
    count = collection.count()

    # Get artifact stats
    artifact_stats = get_artifact_stats()

    # Calculate storage sizes (simplified)
    storage_stats = _get_simplified_storage_stats(mira_path)

    # Get rich custodian profile (learned from conversations)
    custodian_profile = get_full_custodian_profile()

    # Merge with legacy profile stats
    legacy_profile = get_custodian_profile()
    custodian_profile['tech_stack'] = legacy_profile.get('tech_stack', [])
    custodian_profile['active_projects'] = legacy_profile.get('active_projects', [])
    custodian_profile['common_tools'] = legacy_profile.get('common_tools', [])
    custodian_profile['total_sessions'] = legacy_profile.get('total_sessions', 0)
    custodian_profile['total_messages'] = legacy_profile.get('total_messages', 0)

    # Build a richer summary now that we have all the data
    custodian_profile['summary'] = _build_enriched_custodian_summary(custodian_profile)

    # Add interaction tips based on learned preferences
    custodian_profile['interaction_tips'] = _build_interaction_tips(custodian_profile)

    # Get current work context
    work_context = get_current_work_context()

    # Get journey statistics (files created, edited, etc.)
    journey_stats = get_journey_stats()

    # Extract milestones from recent session summaries
    milestones = _extract_milestones(recent.get("projects", []))
    journey_stats['milestones'] = milestones

    # Get codebase knowledge learned from conversations
    codebase_knowledge = get_codebase_knowledge(project_path)

    # Merge technologies from CLAUDE.md into codebase_knowledge
    claude_md_techs = set(project_overview.get('key_technologies', []))
    existing_tech_names = {t.get('name', '').lower() for t in codebase_knowledge.get('technologies', [])}

    for tech in claude_md_techs:
        if tech.lower() not in existing_tech_names:
            codebase_knowledge['technologies'].append({
                'name': tech,
                'role': 'Documented in CLAUDE.md',
                'confidence': 0.95,
                'source': 'CLAUDE.md'
            })

    # Merge key_files from CLAUDE.md into codebase_knowledge.key_modules
    claude_md_files = project_overview.get('key_files', {})
    existing_module_names = {m.get('file', '').lower() for m in codebase_knowledge.get('key_modules', [])}

    for filename, purpose in claude_md_files.items():
        short_name = filename.split('/')[-1] if '/' in filename else filename
        if short_name.lower() not in existing_module_names:
            codebase_knowledge['key_modules'].append({
                'file': short_name,
                'full_path': filename,
                'purpose': purpose,
                'confidence': 0.95,
                'source': 'CLAUDE.md'
            })
        else:
            for mod in codebase_knowledge['key_modules']:
                if mod.get('file', '').lower() == short_name.lower():
                    if 'Frequently discussed' in mod.get('purpose', ''):
                        mod['purpose'] = purpose
                        mod['source'] = 'CLAUDE.md'
                    break

    # Add architecture_details as components if not already present
    arch_details = project_overview.get('architecture_details', [])
    if arch_details and not codebase_knowledge.get('components'):
        codebase_knowledge['components'] = []
        for detail in arch_details:
            codebase_knowledge['components'].append({
                'name': detail.get('name', ''),
                'purpose': detail.get('description', ''),
                'details': detail.get('details', []),
                'confidence': 0.95,
                'source': 'CLAUDE.md'
            })

    if arch_details and not codebase_knowledge.get('architecture_summary'):
        component_names = [d.get('name', '') for d in arch_details[:3]]
        codebase_knowledge['architecture_summary'] = f"Key components: {', '.join(component_names)}"

    # Get concept stats
    concept_store = ConceptStore(project_path)
    concept_stats = concept_store.get_stats()

    # === BUILD TIERED OUTPUT ===

    # TIER 1: Actionable alerts (always check these)
    alerts = _get_actionable_alerts(mira_path, project_path, custodian_profile)

    # TIER 2: Core context (essential for immediate work)
    # Simplified custodian - just the essentials
    custodian_core = {
        'name': custodian_profile.get('name', 'Unknown'),
        'summary': custodian_profile.get('summary', ''),
        'interaction_tips': custodian_profile.get('interaction_tips', [])[:5],
    }

    # Simplified journey - just the highlights
    journey_core = {
        'total_edits': journey_stats.get('total_edits', 0),
        'unique_files': journey_stats.get('unique_files', 0),
        'hot_files': [f['file'] for f in journey_stats.get('most_active_files', [])[:5]],
        'recent_files': journey_stats.get('recent_files', [])[:5],
    }

    # Recent sessions - just summaries
    recent_sessions_core = []
    for proj in recent.get("projects", []):
        for sess in proj.get('sessions', [])[:3]:
            recent_sessions_core.append({
                'summary': sess.get('summary', '')[:100],
                'timestamp': sess.get('timestamp', '')[:10],  # Just date
            })

    # TIER 3: Detailed context (for deep-dive when needed)
    details = {
        'project': project_overview,
        'codebase_knowledge': codebase_knowledge,
        'custodian_full': custodian_profile,
        'journey_full': journey_stats,
        'artifacts': _get_simplified_artifact_stats(artifact_stats),
        'concept_stats': concept_stats,
    }

    return {
        # Top-level status
        "message": "MIRA3 initialized",
        "indexed_conversations": count,

        # TIER 1: Alerts - check these first
        "alerts": alerts,

        # TIER 2: Core context
        "core": {
            "custodian": custodian_core,
            "recent_sessions": recent_sessions_core[:5],
            "current_work": work_context,
            "journey": journey_core,
            "storage": storage_stats,
        },

        # TIER 3: Details - available when needed
        "details": details,
    }


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
            modified = [l[3:] for l in lines if l.startswith(' M') or l.startswith('M ')]
            added = [l[3:] for l in lines if l.startswith('A ') or l.startswith('??')]
            deleted = [l[3:] for l in lines if l.startswith(' D') or l.startswith('D ')]

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

    # Only calculate essential sizes
    data_size = (
        get_dir_size(mira_path / 'chroma') +
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

    return {
        'data': format_size(data_size),
        'models': format_size(models_size),
        'note': 'venv excluded from stats',
    }


def _get_simplified_artifact_stats(artifact_stats: dict) -> dict:
    """Simplify artifact stats to just counts."""
    return {
        'total': artifact_stats.get('total', 0),
        'code_blocks': artifact_stats.get('by_type', {}).get('code_block', 0),
        'errors': artifact_stats.get('by_type', {}).get('error', 0),
        'file_operations': artifact_stats.get('file_operations', 0),
    }


def _get_project_overview(mira_path: Path) -> dict:
    """
    Extract comprehensive project overview from CLAUDE.md if it exists.

    Parses the CLAUDE.md file to extract:
    - Project name and description
    - Full architecture description with components
    - Key Files table (file → purpose mappings)
    - Build/run commands
    - Key technologies
    """
    project_root = mira_path.parent
    claude_md = project_root / "CLAUDE.md"

    overview = {
        "name": project_root.name,
        "description": None,
        "architecture": None,
        "architecture_details": [],  # List of component descriptions
        "key_files": {},  # file → purpose mapping from table
        "commands": {},
        "key_technologies": [],
    }

    if not claude_md.exists():
        # Try README.md as fallback
        readme = project_root / "README.md"
        if readme.exists():
            try:
                content = readme.read_text()[:2000]
                # Extract first paragraph as description
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('```'):
                        overview["description"] = line[:200]
                        break
            except Exception:
                pass
        return overview

    try:
        content = claude_md.read_text()
        import re

        # Extract project overview section
        overview_match = re.search(
            r'##\s*Project\s*Overview\s*\n+(.*?)(?=\n##|\Z)',
            content, re.IGNORECASE | re.DOTALL
        )
        if overview_match:
            overview_text = overview_match.group(1).strip()
            # Get first paragraph
            para = overview_text.split('\n\n')[0].replace('\n', ' ').strip()
            overview["description"] = para[:300]

        # Look for architecture section and extract FULL description
        arch_match = re.search(
            r'##\s*Architecture\s*\n+(.*?)(?=\n##\s+[A-Z]|\Z)',
            content, re.IGNORECASE | re.DOTALL
        )
        if arch_match:
            arch_text = arch_match.group(1).strip()

            # Get the architecture type (first header or line)
            for line in arch_text.split('\n'):
                line = line.strip()
                if not line or line.startswith('|') or line.startswith('```'):
                    continue
                # Strip markdown header prefixes
                while line.startswith('#'):
                    line = line[1:].strip()
                if line:
                    overview["architecture"] = line[:200]
                    break

            # Extract numbered component descriptions (1. **Name** - description)
            component_pattern = r'\d+\.\s*\*\*([^*]+)\*\*[^-]*-\s*([^:]+)(?::\n((?:\s+-[^\n]+\n?)*))?'
            for match in re.finditer(component_pattern, arch_text, re.MULTILINE):
                name = match.group(1).strip()
                desc = match.group(2).strip()
                bullets = match.group(3) or ""

                # Parse bullet points
                details = []
                for bullet in bullets.split('\n'):
                    bullet = bullet.strip()
                    if bullet.startswith('-'):
                        details.append(bullet[1:].strip())

                overview["architecture_details"].append({
                    "name": name,
                    "description": desc,
                    "details": details[:4]  # Max 4 details
                })

            # Extract Key Files table
            # Pattern: | `file.py` | Purpose description |
            table_match = re.search(
                r'###?\s*Key\s*Files\s*\n+\|[^\n]+\|\s*\n\|[-|\s]+\|\s*\n((?:\|[^\n]+\|\s*\n?)+)',
                arch_text, re.IGNORECASE
            )
            if table_match:
                table_rows = table_match.group(1)
                for row in table_rows.split('\n'):
                    row = row.strip()
                    if not row or row.startswith('|---'):
                        continue
                    # Parse: | `file.py` | purpose |
                    cells = [c.strip() for c in row.split('|')[1:-1]]
                    if len(cells) >= 2:
                        file_cell = cells[0].strip('`').strip()
                        purpose = cells[1].strip()
                        if file_cell and purpose:
                            overview["key_files"][file_cell] = purpose

        # Extract build commands
        cmd_match = re.search(
            r'```bash\s*\n(.*?)```',
            content, re.DOTALL
        )
        if cmd_match:
            cmds = cmd_match.group(1).strip().split('\n')
            for cmd in cmds[:5]:
                cmd = cmd.strip()
                if cmd and not cmd.startswith('#'):
                    # Parse "npm run build  # comment" format
                    parts = cmd.split('#')
                    cmd_text = parts[0].strip()
                    comment = parts[1].strip() if len(parts) > 1 else ""
                    if cmd_text:
                        overview["commands"][cmd_text] = comment

        # Extract technologies from content
        tech_patterns = [
            r'ChromaDB', r'all-MiniLM-L6-v2', r'sentence-transformers',
            r'TypeScript', r'Python', r'Node\.js', r'MCP', r'watchdog',
            r'SQLite', r'JSON-RPC'
        ]
        found_tech = set()
        for pattern in tech_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                # Normalize the name
                tech_name = pattern.replace(r'\.', '.').replace('\\', '')
                found_tech.add(tech_name)
        overview["key_technologies"] = sorted(found_tech)

    except Exception as e:
        log(f"Error parsing CLAUDE.md: {e}")

    return overview


def _extract_milestones(projects: list) -> list:
    """
    Extract milestones from session summaries.

    Looks for accomplishment patterns in summaries to identify
    major completed work items.
    """
    milestones = []
    seen = set()

    # Patterns that indicate accomplishments
    accomplishment_patterns = [
        r'(?:implemented|added|created|built|completed|finished|fixed)\s+(.+?)(?:\.|$)',
        r'(.+?)\s+(?:complete|implemented|working|finished)',
        r'(?:Task|Outcome):\s*(.+?)(?:\||$)',
    ]

    import re

    for project in projects:
        for session in project.get('sessions', []):
            summary = session.get('summary', '')
            if not summary:
                continue

            # Try to extract accomplishments
            for pattern in accomplishment_patterns:
                for match in re.finditer(pattern, summary, re.IGNORECASE):
                    milestone = match.group(1).strip()

                    # Clean up the milestone text
                    milestone = milestone.strip('.,!').strip()

                    # Skip if too short or too long
                    if len(milestone) < 10 or len(milestone) > 100:
                        continue

                    # Skip generic or incomplete phrases
                    skip_phrases = ['task', 'the', 'a', 'an', 'now', 'let me', 'i will']
                    if milestone.lower() in skip_phrases:
                        continue
                    if milestone.lower().startswith(tuple(skip_phrases)):
                        continue

                    # Deduplicate
                    key = milestone.lower()[:50]
                    if key in seen:
                        continue
                    seen.add(key)

                    milestones.append({
                        'description': milestone,
                        'session': session.get('session_id', '')[:8]
                    })

    return milestones[:10]  # Return top 10 milestones


def _build_enriched_custodian_summary(profile: dict) -> str:
    """
    Build a natural language summary of the custodian.

    Creates a concise, readable paragraph instead of pipe-separated fields.
    """
    name = profile.get('name', 'Unknown')
    total_sessions = profile.get('total_sessions', 0)
    total_messages = profile.get('total_messages', 0)

    if total_sessions == 0:
        return f"New user: {name}. No conversation history yet."

    # Start with basic info
    sentences = []

    # Activity level description
    if total_sessions >= 50:
        activity = "very active"
    elif total_sessions >= 20:
        activity = "active"
    elif total_sessions >= 5:
        activity = "regular"
    else:
        activity = "occasional"

    # Use "a" or "an" based on whether activity starts with vowel
    article = "an" if activity[0] in 'aeiou' else "a"
    sentences.append(f"{name} is {article} {activity} user with {total_sessions} sessions and {total_messages} messages.")

    # Tech focus
    tech_stack = profile.get('tech_stack', [])
    if tech_stack:
        if len(tech_stack) >= 3:
            sentences.append(f"Works primarily with {', '.join(tech_stack[:3])}.")
        elif len(tech_stack) >= 1:
            sentences.append(f"Works with {', '.join(tech_stack)}.")

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
    """
    tips = []

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
        'chroma': get_dir_size(mira_path / 'chroma'),
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
            'chroma': format_size(components['chroma']),
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

    # Filter tech stack - remove generic/common words
    tech_filter = {
        # Common words
        'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'been',
        'code', 'file', 'files', 'data', 'line', 'text', 'string', 'list',
        # Action words
        'fix', 'add', 'remove', 'update', 'change', 'make', 'get', 'set',
        'create', 'delete', 'run', 'check', 'test', 'use', 'using', 'uses',
        # Context words
        'search', 'function', 'method', 'class', 'module', 'command',
        'error', 'issue', 'problem', 'solution', 'result', 'output',
        # Descriptors
        'new', 'old', 'first', 'last', 'current', 'previous', 'next',
        'extraction', 'inaccuracies', 'implement', 'implementation',
        # More generic words from actual output
        'please', 'gets', 'deep', 'dive', 'model', 'include', 'index',
        'idf', 'should', 'would', 'need', 'want', 'will', 'can', 'may',
        'information', 'ensure', 'analyze', 'just', 'also', 'specific',
        'look', 'want', 'work', 'about', 'into', 'here', 'there',
        'indexing', 'technologies', 'perform', 'before', 'after', 'server',
        'being', 'since', 'conversation', 'conversations', 'messages',
        'summary', 'keywords', 'embedding', 'embeddings', 'vector', 'vectors',
        'phase', 'stdin', 'inside', 'best', 'source', 'more', 'over', 'under',
        'only', 'same', 'each', 'other', 'well', 'done', 'good', 'like',
        'write', 'read', 'call', 'find', 'show', 'type', 'name', 'path',
        # AI/assistant related (not actual tech)
        'claude', 'assistant', 'user', 'message', 'response', 'tool',
        # Deprecated/removed tech should be removed from keywords at source
        'faiss',
        # More generic words found in actual output
        'where', 'pythonpath', 'these', 'those', 'which', 'what', 'when',
        'optional', 'structure', 'historian', 'such', 'very', 'some',
        'pattern', 'patterns', 'content', 'based', 'using', 'since',
        'both', 'also', 'even', 'still', 'then', 'than', 'have', 'been',
        'were', 'does', 'done', 'make', 'made', 'takes', 'took', 'given',
        # Additional garbage words found in recent output
        'understand', 'again', 'readme', 'confidence', 'codebase', 'concepts',
        'integration', 'component', 'session', 'sessions', 'project', 'projects',
        'init', 'status', 'recent', 'knowledge', 'custodian', 'learn', 'learned',
        'extract', 'handler', 'handlers', 'backend', 'frontend', 'layer',
        # More noise from latest output
        'todo', 'explicit', 'ultrathink', 'failed', 'imports', 'indexed',
        'datetime', 'ingestion', 'architecture', 'technology', 'think',
        'analyze', 'analysis', 'perspective', 'fresh', 'remaining', 'issues',
        'milestones', 'journey', 'stats', 'improve', 'improvement', 'improvements',
        'filter', 'filtering', 'generic', 'approval', 'words', 'preferences',
        'active', 'topics', 'staleness', 'completed', 'items',
    }
    # Only include terms that look like technology names
    # Filter out: common words, verbs ending in 'ing', function names (contain _), too short
    def is_valid_tech(tech: str) -> bool:
        if tech in tech_filter:
            return False
        if len(tech) < 4:
            return False
        if tech.endswith('ing'):
            return False
        # Filter out function/variable names (contain underscore)
        if '_' in tech:
            return False
        # Filter out camelCase words (likely variable/function names)
        if any(c.isupper() for c in tech[1:]):
            return False
        return True

    profile['tech_stack'] = [
        tech for tech, count in tech_counter.most_common(50)
        if is_valid_tech(tech)
    ][:10]

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
    # Normalize whitespace and truncate for comparison
    normalized = ' '.join(task.split())[:100].lower()
    return normalized


def _is_duplicate_task(new_task: str, existing_tasks: list) -> bool:
    """Check if a task is a duplicate of an existing one."""
    new_norm = _normalize_task(new_task)
    for existing in existing_tasks:
        existing_norm = _normalize_task(existing)
        # Check for high similarity (one is prefix of the other, or very similar)
        if new_norm.startswith(existing_norm[:50]) or existing_norm.startswith(new_norm[:50]):
            return True
    return False


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
    """Get context about current/recent work."""
    mira_path = get_mira_path()
    metadata_path = mira_path / "metadata"

    context = {
        'recent_tasks': [],
        'active_topics': [],
        'recent_decisions': [],
    }

    if not metadata_path.exists():
        return context

    recent_files = sorted(
        metadata_path.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:5]

    for meta_file in recent_files:
        try:
            meta = json.loads(meta_file.read_text())

            task = meta.get('task_description', '')
            # Skip command messages and empty tasks
            if task and not task.startswith('<command-') and not task.startswith('/'):
                # Check for duplicates (similar task descriptions)
                if not _is_duplicate_task(task, context['recent_tasks']):
                    context['recent_tasks'].append(task)

            todos = meta.get('todo_topics', [])
            for todo in todos[:3]:
                if todo not in context['active_topics']:
                    context['active_topics'].append(todo)

            facts = meta.get('key_facts', [])
            for fact in facts[:2]:
                if fact not in context['recent_decisions']:
                    context['recent_decisions'].append(fact)

        except Exception:
            pass

    context['recent_tasks'] = context['recent_tasks'][:5]
    # Filter out completed/stale topics
    context['active_topics'] = _filter_active_topics(context['active_topics'])[:10]
    # Filter out garbage/invalid decisions
    context['recent_decisions'] = _filter_recent_decisions(context['recent_decisions'])[:5]

    return context


def handle_status(collection) -> dict:
    """Get system status and statistics."""
    mira_path = get_mira_path()
    claude_path = Path.home() / ".claude" / "projects"

    # Count source files
    total_files = 0
    if claude_path.exists():
        total_files = sum(1 for _ in claude_path.rglob("*.jsonl"))

    # Count archived files
    archives_path = mira_path / "archives"
    archived = sum(1 for _ in archives_path.glob("*.jsonl")) if archives_path.exists() else 0

    # Count indexed
    indexed = collection.count()

    # Get insights stats
    error_stats = get_error_stats()
    decision_stats = get_decision_stats()

    return {
        "total_files": total_files,
        "archived": archived,
        "indexed": indexed,
        "pending": total_files - archived,
        "storage_path": str(mira_path),
        "last_sync": datetime.now().isoformat(),
        "errors": error_stats,
        "decisions": decision_stats
    }


def handle_error_lookup(params: dict) -> dict:
    """
    Search for past error solutions.

    Params:
        query: Error message or description to search for
        limit: Maximum results (default: 5)

    Returns matching errors with their solutions.
    """
    query = params.get("query", "")
    limit = params.get("limit", 5)

    if not query:
        return {"results": [], "total": 0}

    results = search_error_solutions(query, limit=limit)

    return {
        "results": results,
        "total": len(results),
        "query": query
    }


def handle_decisions(params: dict) -> dict:
    """
    Search for past architectural/design decisions.

    Params:
        query: Search query
        category: Optional category filter (architecture, technology, etc.)
        limit: Maximum results (default: 10)

    Returns matching decisions with context.
    """
    query = params.get("query", "")
    category = params.get("category")
    limit = params.get("limit", 10)

    if not query:
        # Return recent decisions if no query
        results = search_decisions("", category=category, limit=limit)
    else:
        results = search_decisions(query, category=category, limit=limit)

    return {
        "results": results,
        "total": len(results),
        "query": query,
        "category": category
    }


def handle_rpc_request(request: dict, collection) -> dict:
    """Handle a JSON-RPC request and return response."""
    method = request.get("method", "")
    params = request.get("params", {})
    request_id = request.get("id")

    result = None
    error = None

    try:
        if method == "search":
            result = handle_search(params, collection)
        elif method == "recent":
            result = handle_recent(params)
        elif method == "init":
            result = handle_init(params, collection)
        elif method == "status":
            result = handle_status(collection)
        elif method == "error_lookup":
            result = handle_error_lookup(params)
        elif method == "decisions":
            result = handle_decisions(params)
        elif method == "shutdown":
            result = {"status": "shutting_down"}
        else:
            error = {"code": -32601, "message": f"Method not found: {method}"}
    except Exception as e:
        # Log full error for debugging, return sanitized message
        log(f"RPC handler error for {method}: {e}")
        error = {"code": -32603, "message": "Internal error processing request"}

    response = {"jsonrpc": "2.0", "id": request_id}
    if error:
        response["error"] = error
    else:
        response["result"] = result

    return response
