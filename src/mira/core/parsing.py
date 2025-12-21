"""
MIRA Conversation Parsing Module

Handles parsing Claude Code conversation JSONL files.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

from . import log
from .utils import extract_text_content


def extract_tool_usage(message: dict) -> Tuple[Dict[str, int], Set[str]]:
    """
    Extract tool usage statistics and file paths from assistant message.

    Returns (tools_dict, files_set) where:
    - tools_dict: {tool_name: count}
    - files_set: set of file paths touched
    """
    tools: Dict[str, int] = {}
    files: Set[str] = set()

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


def extract_todos_from_message(message: dict) -> List[Dict[str, str]]:
    """
    Extract TODO items from assistant message tool calls.

    Looks for TodoWrite tool usage and extracts the task descriptions.
    """
    if not isinstance(message, dict):
        return []

    content = message.get('content', [])
    if not isinstance(content, list):
        return []

    todos: List[Dict[str, str]] = []
    for item in content:
        if not isinstance(item, dict):
            continue

        # Look for tool_use blocks with TodoWrite
        if item.get('type') == 'tool_use' and item.get('name') == 'TodoWrite':
            tool_input = item.get('input', {})
            todo_list = tool_input.get('todos', [])
            for todo in todo_list:
                if isinstance(todo, dict):
                    task = todo.get('content', '')
                    status = todo.get('status', 'pending')
                    if task:
                        todos.append({'task': task, 'status': status})

    return todos


def parse_conversation(file_path: Path) -> Dict[str, Any]:
    """
    Parse a Claude Code conversation JSONL file.

    Extracts:
    - Messages with timestamps for time-gap detection
    - TODO lists for topic tracking
    - Session metadata (slug, git branch, model, tools used)
    - Summary if available
    """
    messages: List[Dict[str, Any]] = []
    summary_text = ""
    first_user_message = ""
    todo_snapshots: List[Tuple[str, List[Dict[str, str]]]] = []

    # Session-level metadata
    session_meta: Dict[str, Any] = {
        'slug': '',
        'git_branch': '',
        'cwd': '',
        'models_used': set(),
        'tools_used': {},
        'files_touched': set(),
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
                    timestamp = obj.get('timestamp', '')

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

                        model = message_obj.get('model', '')
                        if model:
                            session_meta['models_used'].add(model)

                        tools, files = extract_tool_usage(message_obj)
                        for tool, count in tools.items():
                            session_meta['tools_used'][tool] = session_meta['tools_used'].get(tool, 0) + count
                        session_meta['files_touched'].update(files)

                        todos = extract_todos_from_message(message_obj)
                        if todos:
                            todo_snapshots.append((timestamp, todos))

                        if content:
                            messages.append({
                                'role': 'assistant',
                                'content': content,
                                'timestamp': timestamp,
                                'todos': todos
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
        'todo_snapshots': todo_snapshots,
        'session_meta': session_meta
    }
