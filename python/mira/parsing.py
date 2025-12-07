"""
MIRA3 Conversation Parsing Module

Handles parsing Claude Code conversation JSONL files.
"""

import json
from pathlib import Path

from .utils import log, extract_text_content


def extract_tool_usage(message: dict) -> tuple:
    """
    Extract tool usage statistics and file paths from assistant message.

    Returns (tools_dict, files_set) where:
    - tools_dict: {tool_name: count}
    - files_set: set of file paths touched
    """
    tools = {}
    files = set()

    if not isinstance(message, dict):
        return tools, files

    content = message.get('content', [])
    if not isinstance(content, list):
        return tools, files

    for item in content:
        if not isinstance(item, dict):
            continue

        if item.get('type') == 'tool_use':
            tool_name = item.get('name', 'unknown')
            tools[tool_name] = tools.get(tool_name, 0) + 1

            # Extract file paths from file-related tools
            tool_input = item.get('input', {})
            if isinstance(tool_input, dict):
                # Read, Edit, Write tools use file_path
                if 'file_path' in tool_input:
                    files.add(tool_input['file_path'])
                # Glob uses path
                if tool_name == 'Glob' and 'path' in tool_input:
                    files.add(tool_input['path'])

    return tools, files


def extract_todos_from_message(message: dict) -> list:
    """
    Extract TODO items from assistant message tool calls.

    Looks for TodoWrite tool usage and extracts the task descriptions.
    """
    if not isinstance(message, dict):
        return []

    content = message.get('content', [])
    if not isinstance(content, list):
        return []

    todos = []
    for item in content:
        if not isinstance(item, dict):
            continue

        # Look for tool_use blocks with TodoWrite
        if item.get('type') == 'tool_use' and item.get('name') == 'TodoWrite':
            tool_input = item.get('input', {})
            todo_list = tool_input.get('todos', [])
            for todo in todo_list:
                if isinstance(todo, dict):
                    # Extract task content and status
                    task = todo.get('content', '')
                    status = todo.get('status', 'pending')
                    if task:
                        todos.append({'task': task, 'status': status})

    return todos


def parse_conversation(file_path: Path) -> dict:
    """
    Parse a Claude Code conversation JSONL file.

    Extracts:
    - Messages with timestamps for time-gap detection
    - TODO lists for topic tracking
    - Session metadata (slug, git branch, model, tools used)
    - Summary if available
    """
    messages = []
    summary_text = ""
    first_user_message = ""
    todo_snapshots = []  # List of (timestamp, todos) tuples

    # Session-level metadata (extracted from first message with these fields)
    session_meta = {
        'slug': '',           # Human-readable session name (e.g., "velvet-hugging-reef")
        'git_branch': '',     # Git branch being worked on
        'cwd': '',            # Working directory
        'models_used': set(), # AI models used in this session
        'tools_used': {},     # Tool usage counts
        'files_touched': set(), # Files read/edited/written
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    msg_type = obj.get('type', '')
                    timestamp = obj.get('timestamp', '')  # ISO format: "2025-12-07T04:45:36.800Z"

                    # Extract session metadata from any message that has it
                    if not session_meta['slug'] and obj.get('slug'):
                        session_meta['slug'] = obj['slug']
                    if not session_meta['git_branch'] and obj.get('gitBranch'):
                        session_meta['git_branch'] = obj['gitBranch']
                    if not session_meta['cwd'] and obj.get('cwd'):
                        session_meta['cwd'] = obj['cwd']

                    if msg_type == 'user':
                        content = extract_text_content(obj.get('message', {}))
                        if content:
                            messages.append({
                                'role': 'user',
                                'content': content,
                                'timestamp': timestamp
                            })
                            if not first_user_message:
                                first_user_message = content

                    elif msg_type == 'assistant':
                        message_obj = obj.get('message', {})
                        content = extract_text_content(message_obj)

                        # Extract model used
                        model = message_obj.get('model', '')
                        if model:
                            session_meta['models_used'].add(model)

                        # Extract tool usage and file paths
                        tools, files = extract_tool_usage(message_obj)
                        for tool, count in tools.items():
                            session_meta['tools_used'][tool] = session_meta['tools_used'].get(tool, 0) + count
                        session_meta['files_touched'].update(files)

                        # Also extract TODO lists from tool calls
                        todos = extract_todos_from_message(message_obj)
                        if todos:
                            todo_snapshots.append((timestamp, todos))

                        if content:
                            messages.append({
                                'role': 'assistant',
                                'content': content,
                                'timestamp': timestamp,
                                'todos': todos  # Attach todos to message for context
                            })

                    elif msg_type == 'summary':
                        summary_text = obj.get('summary', '')

                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log(f"Error parsing {file_path}: {e}")
        return {}

    # Convert sets to lists for JSON serialization
    session_meta['models_used'] = list(session_meta['models_used'])
    session_meta['files_touched'] = list(session_meta['files_touched'])

    return {
        'messages': messages,
        'summary': summary_text,
        'first_user_message': first_user_message,
        'message_count': len(messages),
        'todo_snapshots': todo_snapshots,  # For topic extraction
        'session_meta': session_meta       # Additional session metadata
    }
