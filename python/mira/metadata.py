"""
MIRA3 Metadata Extraction Module

Extracts summary, keywords, key facts, and other metadata from conversations.
"""

import re
from pathlib import Path
from datetime import datetime
from collections import Counter

from .utils import parse_timestamp
from .constants import TIME_GAP_THRESHOLD, CHARS_PER_TOKEN


# Comprehensive stopwords list
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'where', 'why',
    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'would', 'could', 'ought', 'im',
    'youre', 'hes', 'shes', 'were', 'theyre', 'ive', 'youve',
    'weve', 'theyve', 'id', 'youd', 'hed', 'shed', 'wed', 'theyd',
    'ill', 'youll', 'hell', 'shell', 'well', 'theyll', 'isnt', 'arent',
    'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'doesnt', 'dont',
    'didnt', 'wont', 'wouldnt', 'shouldnt', 'couldnt', 'cant', 'cannot',
    'mustnt', 'lets', 'thats', 'whos', 'whats', 'heres', 'theres',
    'whens', 'wheres', 'whys', 'hows', 'also', 'like', 'get', 'got',
    'make', 'made', 'use', 'used', 'using', 'want', 'need', 'try',
    'look', 'see', 'know', 'think', 'take', 'come', 'go', 'way', 'new',
    'one', 'two', 'first', 'last', 'long', 'little', 'own', 'other',
    'yes', 'let', 'sure', 'ok', 'okay', 'really', 'actually', 'probably',
    'maybe', 'still', 'already', 'always', 'never', 'ever', 'yet',
    'must', 'may', 'might', 'shall', 'since', 'until', 'while',
    'of', 'as', 'because', 'although', 'though', 'however', 'therefore',
}


def build_summary(messages: list, first_msg: str, existing_summary: str) -> str:
    """Build a comprehensive summary of the conversation."""
    # If we have an existing summary from Claude Code, use it
    if existing_summary and len(existing_summary) > 20:
        return existing_summary

    # Otherwise, build one from the conversation
    parts = []

    # Start with the first user request (cleaned)
    if first_msg:
        # Clean and truncate first message
        clean_first = first_msg.strip()
        if len(clean_first) > 100:
            # Find a good break point
            break_point = clean_first[:100].rfind(' ')
            if break_point > 50:
                clean_first = clean_first[:break_point] + '...'
            else:
                clean_first = clean_first[:100] + '...'
        parts.append(f"Task: {clean_first}")

    # Look for key outcomes in assistant messages
    outcomes = []
    for msg in reversed(messages[-10:]):  # Check last 10 messages
        if msg.get('role') != 'assistant':
            continue
        content = msg.get('content', '')
        # Look for completion indicators
        if any(phrase in content.lower() for phrase in [
            'complete', 'finished', 'done', 'implemented', 'fixed', 'created',
            'successfully', 'working', 'ready', 'deployed'
        ]):
            # Extract a short excerpt
            sentences = content.split('.')
            for sent in sentences[:3]:
                sent = sent.strip()
                if len(sent) > 20 and len(sent) < 200:
                    if any(word in sent.lower() for word in ['complete', 'done', 'implemented', 'fixed', 'created', 'working']):
                        outcomes.append(sent)
                        break
            if outcomes:
                break

    if outcomes:
        parts.append(f"Outcome: {outcomes[0]}")

    if parts:
        return ' | '.join(parts)

    # Fallback to first message
    return first_msg[:200] if first_msg else "No summary available"


def extract_keywords(messages: list) -> list:
    """Extract keywords from conversation messages."""
    keywords = set()

    # Patterns for extracting meaningful terms
    file_pattern = re.compile(r'[\w/\\]+\.(py|ts|js|tsx|jsx|json|md|yaml|yml|toml|sh|sql|go|rs|java|c|cpp|h|css|html|vue|svelte)')
    error_pattern = re.compile(r'(Error|Exception|Failed|error|exception|failed):\s*(\w+)')
    package_pattern = re.compile(r'(?:import|from|require)\s+[\'"]?([a-zA-Z@][a-zA-Z0-9_\-/]*)')
    function_pattern = re.compile(r'(?:function|def|async|const|let|var)\s+(\w+)')

    all_text = ' '.join(m.get('content', '') for m in messages)

    # Extract file extensions as tech indicators
    for match in file_pattern.findall(all_text):
        keywords.add(match.lower())

    # Extract error types
    for match in error_pattern.findall(all_text):
        keywords.add(match[1].lower())

    # Extract package names
    for match in package_pattern.findall(all_text):
        keywords.add(match.split('/')[0].lower())

    # Extract function names (first 20)
    for match in function_pattern.findall(all_text)[:20]:
        if len(match) > 2:
            keywords.add(match.lower())

    # Extract significant words (TF-based)
    word_counter = Counter()
    word_pattern = re.compile(r'\b[a-zA-Z][a-zA-Z0-9_]{2,}\b')
    for msg in messages:
        content = msg.get('content', '')
        words = word_pattern.findall(content.lower())
        for word in words:
            if word not in STOPWORDS and len(word) > 2:
                word_counter[word] += 1

    # Add most frequent meaningful words
    for word, count in word_counter.most_common(30):
        if count >= 2:  # Appears at least twice
            keywords.add(word)

    return list(keywords)[:50]  # Limit to 50 keywords


def extract_accomplishments(messages: list) -> list:
    """
    Extract accomplishments from conversation - things that were completed/achieved.

    Looks for:
    - Commit messages
    - ✅ emoji patterns
    - "Fixed/Implemented/Added" statements
    - Test results
    - Deployment/publish success
    - Version bumps
    """
    accomplishments = []
    seen = set()

    # Patterns for accomplishment extraction
    patterns = [
        # Commit messages
        (re.compile(r'git commit.*?["\']([^"\']{10,100})["\']', re.IGNORECASE), 'commit'),
        (re.compile(r'committed[:\s]+["\']?([^"\'.\n]{10,80})', re.IGNORECASE), 'commit'),

        # Emoji checkmarks with context
        (re.compile(r'✅\s*([^\n]{5,80})'), 'done'),
        (re.compile(r'✓\s*([^\n]{5,80})'), 'done'),

        # Fixed/resolved patterns
        (re.compile(r'(?:fixed|resolved|solved)[:\s]+([^\n.]{10,80})', re.IGNORECASE), 'fix'),
        (re.compile(r'(?:the )?(?:bug|issue|error|problem) (?:is |was |has been )?(?:fixed|resolved|solved)', re.IGNORECASE), 'fix'),

        # Implementation patterns
        (re.compile(r'(?:implemented|added|created|built)[:\s]+([^\n.]{10,80})', re.IGNORECASE), 'implement'),
        (re.compile(r'(?:successfully |now )?(?:implemented|added|created) (?:the |a )?([^\n.]{10,60})', re.IGNORECASE), 'implement'),

        # Completion patterns
        (re.compile(r'(?:completed|finished|done with)[:\s]+([^\n.]{10,80})', re.IGNORECASE), 'complete'),

        # Test results
        (re.compile(r'(\d+)\s*(?:tests?|specs?)\s*(?:passed|passing)', re.IGNORECASE), 'test'),
        (re.compile(r'all\s*(?:tests?|specs?)\s*(?:pass(?:ed|ing)?|green)', re.IGNORECASE), 'test'),

        # Version/publish patterns
        (re.compile(r'(?:published|released|deployed)\s+(?:v(?:ersion)?\s*)?(\d+\.\d+(?:\.\d+)?)', re.IGNORECASE), 'release'),
        (re.compile(r'v(\d+\.\d+\.\d+)\s*(?:published|released|deployed)', re.IGNORECASE), 'release'),
        (re.compile(r'bumped?\s*(?:to\s*)?v?(\d+\.\d+\.\d+)', re.IGNORECASE), 'release'),
    ]

    for msg in messages:
        if msg.get('role') != 'assistant':
            continue
        content = msg.get('content', '')

        for pattern, acc_type in patterns:
            matches = pattern.findall(content)
            for match in matches:
                # Clean and normalize
                if isinstance(match, tuple):
                    match = match[0] if match else ''
                text = match.strip() if isinstance(match, str) else str(match)

                # Skip if too short, too long, or already seen
                if len(text) < 5 or len(text) > 100:
                    continue

                # Normalize for dedup
                normalized = text.lower()[:50]
                if normalized in seen:
                    continue
                seen.add(normalized)

                # Format based on type
                if acc_type == 'test' and text.isdigit():
                    accomplishments.append(f"{text} tests passed")
                elif acc_type == 'release':
                    accomplishments.append(f"Released v{text}")
                elif acc_type == 'commit':
                    accomplishments.append(text)
                else:
                    # Capitalize first letter
                    accomplishments.append(text[0].upper() + text[1:] if text else text)

                if len(accomplishments) >= 10:
                    break

        if len(accomplishments) >= 10:
            break

    return accomplishments[:10]


def extract_key_facts(messages: list) -> list:
    """
    Extract key facts, rules, and important decisions from assistant messages.

    Uses priority scoring to surface the most important facts.
    Filters out debug observations and troubleshooting statements.
    """
    # Priority levels
    PRIORITY_CRITICAL = 5  # Warnings, errors, breaking changes
    PRIORITY_HIGH = 4      # Never/always rules
    PRIORITY_MEDIUM = 3    # Technical constraints, requirements
    PRIORITY_NORMAL = 2    # Decisions, recommendations

    scored_facts = []

    # Pattern definitions with associated priorities
    patterns_with_priority = [
        # Critical: Warnings and breaking issues (but not debug observations)
        (PRIORITY_CRITICAL, re.compile(
            r'([A-Z][^.!?]*(?:warning|breaking change|security|vulnerability|deprecated)[^.!?]*[.!?])',
            re.IGNORECASE
        )),
        # High: Strong directives (never/always) - but require actionable context
        (PRIORITY_HIGH, re.compile(
            r'([A-Z][^.!?]*(?:you should never|we should never|must not|must always|absolutely must)[^.!?]*[.!?])',
            re.IGNORECASE
        )),
        # Medium: Requirements and constraints
        (PRIORITY_MEDIUM, re.compile(
            r'([A-Z][^.!?]*(?:required|essential|critical requirement|necessary for)[^.!?]*[.!?])',
            re.IGNORECASE
        )),
        # Normal: Explicit decisions
        (PRIORITY_NORMAL, re.compile(
            r'(?:I (?:decided|chose|recommend)|The (?:solution|fix|approach) (?:is|was)|We (?:decided|chose) to)[:\s]*([^.!?]+[.!?])',
            re.IGNORECASE
        )),
    ]

    seen_facts = set()

    # Phrases that indicate debug/troubleshooting observations (not decisions)
    debug_patterns = [
        r'^the (?:error|issue|problem|bug) (?:is|was|seems|appears)',
        r'^this (?:error|issue|problem) (?:is|was|occurs|happens)',
        r'^the (?:file|process|thread|watcher|server) (?:error|is|was)',
        r'^it (?:looks like|seems|appears)',
        r'just (?:because|from|due to)',
        r'error (?:persists|occurs|happens)',
        r'^(?:the )?hnsw',  # ChromaDB debug
        r'^(?:the )?chromadb (?:error|issue)',
        r'surfaces previous',  # from numbered lists
        r'^\d+\.',  # Numbered list items
    ]
    debug_re = re.compile('|'.join(debug_patterns), re.IGNORECASE)

    for msg in messages:
        if msg.get('role') != 'assistant':
            continue

        content = msg.get('content', '')

        for priority, pattern in patterns_with_priority:
            matches = pattern.findall(content)
            for match in matches[:3]:  # Limit per pattern per message
                # Clean up the match
                fact = match.strip() if isinstance(match, str) else match[0].strip()
                fact = ' '.join(fact.split())  # Normalize whitespace

                # Skip if too short, too long, or already seen
                if len(fact) < 20 or len(fact) > 200:
                    continue

                # Skip code snippets (contain brackets, braces, function syntax)
                if any(c in fact for c in ['{', '}', '[', ']', '()', '`', '==', '->']):
                    continue

                # Skip markdown syntax
                if '**' in fact or '```' in fact or '##' in fact:
                    continue

                # Skip debug/troubleshooting observations
                if debug_re.search(fact):
                    continue

                # Skip assistant reasoning/thinking phrases
                skip_phrases = [
                    'let me ', 'i\'ll ', 'i will ', 'i need to ', 'i should ',
                    'let\'s ', 'we can ', 'we should ', 'looking at ',
                    'checking ', 'searching ', 'reading ', 'now i\'ll ',
                    'first, ', 'next, ', 'then, ', 'but let me ',
                    'you\'re absolutely', 'that\'s ', 'i gave up',
                ]
                if any(fact.lower().startswith(p) for p in skip_phrases):
                    continue

                # Ensure fact is primarily text (>70% alphabetic)
                alpha_chars = sum(1 for c in fact if c.isalpha() or c.isspace())
                if alpha_chars / len(fact) < 0.7:
                    continue

                fact_key = fact[:50].lower()
                if fact_key in seen_facts:
                    continue

                seen_facts.add(fact_key)
                scored_facts.append((priority, fact))

    # Sort by priority (highest first) and return facts only
    scored_facts.sort(key=lambda x: x[0], reverse=True)
    return [fact for _, fact in scored_facts[:10]]


def clean_task_description(first_msg: str) -> str:
    """
    Clean up the task description from the first user message.

    Handles:
    - Greeting removal (Hi Claude, Hello, Hey, etc.)
    - Polite prefix removal (please, can you, etc.)
    - Multi-sentence handling
    - Proper capitalization
    """
    if not first_msg:
        return ""

    desc = first_msg.strip()

    # Remove common greetings at the start
    greeting_patterns = [
        r'^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))[\s,!.]*(?:claude)?[\s,!.]*',
        r'^claude[\s,!.]+',
        r'^thanks?\s+(?:for\s+)?(?:your\s+)?(?:help)?[\s,!.]*',
    ]
    for pattern in greeting_patterns:
        desc = re.sub(pattern, '', desc, flags=re.IGNORECASE).strip()

    # If message has multiple sentences, try to find the task sentence
    sentences = re.split(r'(?<=[.!?])\s+', desc)
    if len(sentences) > 1:
        # Skip short greeting-like sentences
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 15:
                continue
            if re.match(r'^(?:hi|hello|hey|thanks?|good\s+(?:morning|afternoon|evening))', sent, re.IGNORECASE):
                continue
            desc = sent
            break

    # Remove polite prefixes
    prefixes = [
        'please ', 'can you ', 'could you ', 'would you ', 'will you ',
        'i need ', 'i need you to ', 'i want ', 'i want you to ',
        'i would like ', 'i\'d like ', 'i\'d like you to ',
        'help me ', 'help me to ', 'help with ',
        'let\'s ', 'let me ', 'we need to ', 'we should ',
    ]

    # Keep removing prefixes until none match (handles chained prefixes)
    changed = True
    while changed:
        changed = False
        desc_lower = desc.lower()
        for prefix in prefixes:
            if desc_lower.startswith(prefix):
                desc = desc[len(prefix):]
                changed = True
                break  # Restart from beginning of prefix list

    # Capitalize first letter
    if desc:
        desc = desc[0].upper() + desc[1:]

    # Truncate at word boundary
    if len(desc) > 200:
        break_point = desc[:200].rfind(' ')
        if break_point > 100:
            desc = desc[:break_point] + '...'
        else:
            desc = desc[:200] + '...'

    return desc.strip()


def extract_todo_topics(todo_snapshots: list) -> list:
    """
    Extract unique task topics from TODO snapshots.

    This helps with searchability - if someone worked on "authentication"
    as a TODO item, it should be findable.
    """
    topics = set()
    for timestamp, todos in todo_snapshots:
        for todo in todos:
            task = todo.get('task', '')
            if task:
                topics.add(task)
    return sorted(topics)[:20]


def sample_messages_for_embedding(messages: list, max_tokens: int = 6000) -> list:
    """
    Intelligently sample messages from a conversation for embedding.

    Strategy for long conversations:
    1. Always include first few messages (establishes context)
    2. Always include last few messages (recent state)
    3. Detect "topic shifts" and include messages around them
    4. Sample evenly from remaining messages
    """
    if not messages:
        return []

    def estimate_tokens(text: str) -> int:
        return len(text) // CHARS_PER_TOKEN

    def get_message_text(msg: dict) -> str:
        return msg.get('content', '')[:1000]

    total_messages = len(messages)

    # For short conversations, include everything
    if total_messages <= 30:
        return messages

    # Topic shift patterns
    topic_shift_patterns = [
        'good morning', 'good afternoon', 'good evening',
        'hello again', 'back to', 'continuing', 'picking up',
        'let\'s switch', 'different topic', 'new task', 'another thing',
        'moving on', 'next up', 'let\'s work on', 'now let\'s',
        'that\'s done', 'that works', 'finished', 'completed',
        'unrelated', 'separate issue', 'different project', 'switching to',
    ]

    def is_topic_shift(msg: dict, prev_msg: dict = None) -> bool:
        if msg.get('role') != 'user':
            return False
        content = msg.get('content', '').lower()[:500]
        for pattern in topic_shift_patterns:
            if pattern in content:
                return True

        # Check for time gap
        if prev_msg:
            curr_ts = parse_timestamp(msg.get('timestamp', ''))
            prev_ts = parse_timestamp(prev_msg.get('timestamp', ''))
            if curr_ts and prev_ts:
                gap = (curr_ts - prev_ts).total_seconds()
                if gap > TIME_GAP_THRESHOLD:
                    return True

        return False

    # Identify important message indices
    important_indices = set()

    # First 5 messages
    for i in range(min(5, total_messages)):
        important_indices.add(i)

    # Last 10 messages
    for i in range(max(0, total_messages - 10), total_messages):
        important_indices.add(i)

    # Topic shift messages
    for i, msg in enumerate(messages):
        prev_msg = messages[i - 1] if i > 0 else None
        if is_topic_shift(msg, prev_msg):
            for j in range(max(0, i - 1), min(total_messages, i + 3)):
                important_indices.add(j)

    # Sample evenly from middle
    middle_start = 5
    middle_end = total_messages - 10
    if middle_end > middle_start:
        middle_indices = [i for i in range(middle_start, middle_end) if i not in important_indices]
        sample_interval = max(1, len(middle_indices) // 20)
        for i in range(0, len(middle_indices), sample_interval):
            important_indices.add(middle_indices[i])

    # Return messages in order
    return [messages[i] for i in sorted(important_indices)]


def build_document_content(conversation: dict, metadata: dict) -> str:
    """
    Build a text document for embedding from conversation.

    IMPORTANT: all-MiniLM-L6-v2 has a 256 token limit (~1000 chars).
    Strategy: Create a dense, high-signal document that fits within the limit.
    """
    MAX_CHARS = 900
    parts = []
    used_chars = 0

    def add_part(text: str, max_len: int = None) -> bool:
        nonlocal used_chars
        if max_len:
            text = text[:max_len]
        text_len = len(text)
        if used_chars + text_len + 2 <= MAX_CHARS:
            parts.append(text)
            used_chars += text_len + 2
            return True
        return False

    # Summary is highest priority
    summary = metadata.get('summary', '')
    if summary:
        add_part(f"Summary: {summary}", 300)

    # Task description
    task = metadata.get('task_description', '')
    if task and task != summary:
        add_part(f"Task: {task}", 150)

    # Keywords are critical for search
    keywords = metadata.get('keywords', [])
    if keywords:
        kw_text = ', '.join(keywords[:20])
        add_part(f"Keywords: {kw_text}", 200)

    # TODO topics
    todos = metadata.get('todo_topics', [])
    if todos:
        todo_text = '; '.join(todos[:5])
        add_part(f"Tasks: {todo_text}", 150)

    # Key facts
    facts = metadata.get('key_facts', [])
    if facts:
        fact_text = ' | '.join(facts[:3])
        add_part(f"Facts: {fact_text}", 150)

    # Git branch
    git_branch = metadata.get('git_branch', '')
    if git_branch:
        add_part(f"Branch: {git_branch}", 50)

    return '\n\n'.join(parts)


def extract_metadata(conversation: dict, file_info: dict) -> dict:
    """
    Extract metadata from a parsed conversation.

    Includes session-level metadata for richer search capabilities.
    """
    messages = conversation.get('messages', [])
    first_msg = conversation.get('first_user_message', '')
    summary = conversation.get('summary', '')
    session_meta = conversation.get('session_meta', {})
    todo_snapshots = conversation.get('todo_snapshots', [])

    # Build a better summary
    summary = build_summary(messages, first_msg, summary)

    # Extract keywords
    keywords = extract_keywords(messages)

    # Add file names to keywords
    files_touched = session_meta.get('files_touched', [])
    for fpath in files_touched[:20]:
        fname = Path(fpath).name if fpath else ''
        if fname and len(fname) > 2:
            base = fname.rsplit('.', 1)[0] if '.' in fname else fname
            if base.lower() not in [k.lower() for k in keywords]:
                keywords.append(base)

    # Extract key facts
    key_facts = extract_key_facts(messages)

    # Extract accomplishments (commits, fixes, implementations)
    accomplishments = extract_accomplishments(messages)

    # Build task description
    task_description = clean_task_description(first_msg)

    # Extract TODO topics
    todo_topics = extract_todo_topics(todo_snapshots)

    return {
        'session_id': file_info.get('session_id', ''),
        'project_path': file_info.get('project_path', ''),
        'summary': summary,
        'keywords': keywords[:50],
        'task_description': task_description,
        'key_facts': key_facts,
        'message_count': len(messages),
        'extracted_at': datetime.now().isoformat(),
        'source_file': file_info.get('file_path', ''),
        'last_modified': file_info.get('last_modified', ''),
        # Session metadata
        'slug': session_meta.get('slug', ''),
        'git_branch': session_meta.get('git_branch', ''),
        'cwd': session_meta.get('cwd', ''),
        'models_used': session_meta.get('models_used', []),
        # Convert tools_used from dict (tool->count) to list of tool names
        'tools_used': list(session_meta.get('tools_used', {}).keys()) if isinstance(session_meta.get('tools_used'), dict) else session_meta.get('tools_used', []),
        'files_touched': files_touched[:50],
        'todo_topics': todo_topics,
        'accomplishments': accomplishments,
    }
