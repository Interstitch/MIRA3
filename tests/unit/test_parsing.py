"""Tests for mira.core.parsing module."""

import os
import json
import tempfile
from pathlib import Path

from mira.core import (
    parse_conversation, extract_tool_usage, extract_todos_from_message
)


class TestParsing:
    """Test conversation parsing functions."""

    def test_extract_tool_usage(self):
        message = {
            'content': [
                {'type': 'tool_use', 'name': 'Read', 'input': {'file_path': '/tmp/test.py'}},
                {'type': 'tool_use', 'name': 'Edit', 'input': {'file_path': '/tmp/test.py'}},
                {'type': 'text', 'text': 'Some text'}
            ]
        }
        tools, files = extract_tool_usage(message)
        assert tools.get('Read') == 1
        assert tools.get('Edit') == 1
        assert '/tmp/test.py' in files

    def test_extract_todos_from_message(self):
        message = {
            'content': [
                {
                    'type': 'tool_use',
                    'name': 'TodoWrite',
                    'input': {
                        'todos': [
                            {'content': 'Task 1', 'status': 'pending'},
                            {'content': 'Task 2', 'status': 'completed'}
                        ]
                    }
                }
            ]
        }
        todos = extract_todos_from_message(message)
        assert len(todos) == 2
        assert todos[0]['task'] == 'Task 1'


class TestConversationParsing:
    """Test parsing of actual conversation JSONL files."""

    def test_parse_conversation_file(self):
        # Create a temporary conversation file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        try:
            # Write sample conversation
            lines = [
                json.dumps({
                    'type': 'user',
                    'timestamp': '2025-12-07T10:00:00Z',
                    'message': {'content': 'Help me fix the bug'}
                }),
                json.dumps({
                    'type': 'assistant',
                    'timestamp': '2025-12-07T10:00:30Z',
                    'message': {
                        'content': [{'type': 'text', 'text': 'I\'ll help you fix that.'}],
                        'model': 'claude-3-opus'
                    }
                }),
                json.dumps({
                    'type': 'summary',
                    'summary': 'Bug fix conversation'
                })
            ]
            for line in lines:
                temp_file.write(line + '\n')
            temp_file.close()

            # Parse it
            result = parse_conversation(Path(temp_file.name))

            assert result['message_count'] == 2
            assert result['first_user_message'] == 'Help me fix the bug'
            assert result['summary'] == 'Bug fix conversation'
            assert 'claude-3-opus' in result['session_meta']['models_used']

        finally:
            os.unlink(temp_file.name)
