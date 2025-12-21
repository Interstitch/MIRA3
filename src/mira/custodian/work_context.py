"""
MIRA3 Work Context Module

Handles detection and deduplication of recent work context including
tasks, topics, and decisions from conversation metadata.
"""

import json
from pathlib import Path

from mira.core import get_mira_path


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


def normalize_task(task: str) -> str:
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


def extract_topic_keywords(text: str) -> set:
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


def is_duplicate_task(new_task: str, existing_tasks: list) -> bool:
    """Check if a task is a duplicate of an existing one."""
    new_norm = normalize_task(new_task)
    new_keywords = extract_topic_keywords(new_task)

    for existing in existing_tasks:
        existing_norm = normalize_task(existing)

        # Check for prefix similarity (one contains the start of the other)
        if new_norm.startswith(existing_norm[:40]) or existing_norm.startswith(new_norm[:40]):
            return True

        # Check for keyword overlap (>60% shared keywords = duplicate)
        if new_keywords:
            existing_keywords = extract_topic_keywords(existing)
            if existing_keywords:
                overlap = len(new_keywords & existing_keywords)
                smaller_set = min(len(new_keywords), len(existing_keywords))
                if smaller_set > 0 and overlap / smaller_set > 0.6:
                    return True

    return False


def string_similarity(s1: str, s2: str) -> float:
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


def dedupe_task_list(tasks: list) -> list:
    """
    Aggressively deduplicate a list of tasks.

    Keeps the shortest/cleanest version when duplicates are found.
    """
    if not tasks:
        return []

    # Group similar tasks together
    groups = []
    for task in tasks:
        task_norm = normalize_task(task)

        found_group = False
        for group in groups:
            # Compare against first item in group (the representative)
            rep = group[0]
            rep_norm = normalize_task(rep)

            # Check prefix similarity
            if task_norm.startswith(rep_norm[:35]) or rep_norm.startswith(task_norm[:35]):
                group.append(task)
                found_group = True
                break

            # Check string similarity (catches typos and minor rewording)
            # Use 0.5 threshold - tasks about the same topic share ~50%+ words
            if string_similarity(task_norm, rep_norm) > 0.5:
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


def is_completed_topic(topic: str) -> bool:
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


def filter_active_topics(topics: list) -> list:
    """Filter out completed or stale topics."""
    active = []
    seen_normalized = set()

    for topic in topics:
        # Skip empty or very short topics
        if not topic or len(topic.strip()) < 10:
            continue

        # Check if completed
        if is_completed_topic(topic):
            continue

        # Deduplicate similar topics
        normalized = normalize_task(topic)[:60]
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)

        active.append(topic)

    return active


def is_valid_decision(fact: str) -> bool:
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
        'let me add',  # Process narration
        'let me fix',  # Process narration
        'let me update',  # Process narration
        'now let me',  # Process narration
        'i\'ll now',
        'i will now',
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
        # Generic status/test output (not decisions)
        'tests pass',  # Test output
        'deprecation warning',  # Test/build output
        'build succeeded',  # Build output
        'build failed',  # Build output
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


def filter_recent_decisions(facts: list) -> list:
    """Filter and deduplicate recent decisions/facts."""
    valid = []
    seen_normalized = set()

    for fact in facts:
        if not is_valid_decision(fact):
            continue

        # Deduplicate
        normalized = normalize_task(fact)[:60]
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)

        valid.append(fact)

    return valid


def get_current_work_context(project_path: str = "") -> dict:
    """
    Get context about current/recent work for a specific project.

    Args:
        project_path: Filter to sessions from this project path only.
                      If empty, returns work from all projects.

    Returns only non-empty fields to minimize token waste.
    """
    mira_path = get_mira_path()
    metadata_path = mira_path / "metadata"

    recent_tasks = []
    active_topics = []
    recent_decisions = []

    if not metadata_path.exists():
        return {}

    # Scan more files to find diverse tasks (not just recent repeats of same work)
    # Scan extra files when filtering by project since many may be skipped
    scan_limit = 50 if project_path else 15
    recent_files = sorted(
        metadata_path.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:scan_limit]

    for meta_file in recent_files:
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))

            # Filter by project if specified
            if project_path:
                file_project = meta.get('project_path', '')
                # Convert encoded path format (-workspaces-MIRA3 -> /workspaces/MIRA3)
                normalized = _format_project_path(file_project)
                # Check if this session belongs to the requested project
                if project_path not in normalized and normalized not in project_path:
                    continue  # Skip sessions from other projects

            task = meta.get('task_description', '')
            # Skip command messages and empty tasks
            if task and not task.startswith('<command-') and not task.startswith('/'):
                # Check for duplicates (similar task descriptions)
                if not is_duplicate_task(task, recent_tasks):
                    recent_tasks.append(task)

            # Use session SUMMARY as active topic, not granular todo items
            # Summaries capture the high-level work theme ("MIRA Init Optimization")
            # while todo_topics are implementation details ("Fix full_path")
            summary = meta.get('summary', '')
            if summary and len(summary) >= 10:
                if not is_duplicate_task(summary, active_topics):
                    active_topics.append(summary)

            # Collect key_facts as potential decisions
            key_facts = meta.get('key_facts', [])
            recent_decisions.extend(key_facts)

        except Exception:
            pass

    # Final aggressive deduplication - keep only truly distinct tasks
    recent_tasks = dedupe_task_list(recent_tasks)

    # Build context with only non-empty fields
    context = {}

    recent_tasks = recent_tasks[:5]
    if recent_tasks:
        context['recent_tasks'] = recent_tasks

    # Filter out completed/stale topics AND cross-dedupe against recent_tasks
    active_topics = filter_active_topics(active_topics)

    # Remove topics that duplicate recent_tasks (summaries often echo task descriptions)
    if recent_tasks:
        unique_topics = []
        for topic in active_topics:
            if not is_duplicate_task(topic, recent_tasks):
                unique_topics.append(topic)
        active_topics = unique_topics

    active_topics = active_topics[:3]  # Reduced from 5 - topics should be distinct themes
    if active_topics:
        context['active_topics'] = active_topics

    # Filter and deduplicate decisions from key_facts
    recent_decisions = filter_recent_decisions(recent_decisions)
    recent_decisions = recent_decisions[:5]
    if recent_decisions:
        context['recent_decisions'] = recent_decisions

    return context
