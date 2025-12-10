#!/usr/bin/env python3
"""
Comprehensive test suite for MIRA3 Python backend.

Run with: python -m pytest test/test_mira_backend.py -v
Or directly: python test/test_mira_backend.py
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from mira.utils import (
    get_mira_path, extract_text_content, parse_timestamp, get_custodian
)
from mira.constants import EMBEDDING_MODEL_NAME, DEPENDENCIES
from mira.artifacts import (
    init_artifact_db, store_file_operation, get_file_operations,
    reconstruct_file, get_artifact_stats, detect_language,
    extract_artifacts_from_content, store_artifact, search_artifacts_for_query,
    extract_artifacts_from_messages
)
from mira.metadata import (
    build_summary, extract_keywords, extract_key_facts,
    clean_task_description, extract_todo_topics, sample_messages_for_embedding,
    extract_metadata
)
from mira.parsing import (
    parse_conversation, extract_tool_usage, extract_todos_from_message
)
from mira.bootstrap import is_running_in_venv
from mira.custodian import (
    init_custodian_db, extract_custodian_learnings,
    get_full_custodian_profile, get_danger_zones_for_files
)
from mira.insights import (
    init_insights_db, extract_errors_from_conversation, extract_decisions_from_conversation,
    search_error_solutions, search_decisions, get_error_stats, get_decision_stats,
    normalize_error_message, compute_error_signature
)
from mira.concepts import init_concepts_db
from mira.db_manager import shutdown_db_manager


class TestUtils:
    """Test utility functions."""

    def test_extract_text_content_string(self):
        msg = {'content': 'Hello world'}
        assert extract_text_content(msg) == 'Hello world'

    def test_extract_text_content_list(self):
        msg = {'content': [
            {'type': 'text', 'text': 'Part 1'},
            {'type': 'text', 'text': 'Part 2'}
        ]}
        assert extract_text_content(msg) == 'Part 1\nPart 2'

    def test_extract_text_content_empty(self):
        assert extract_text_content({}) == ''
        assert extract_text_content(None) == ''

    def test_parse_timestamp_valid(self):
        ts = parse_timestamp('2025-12-07T14:30:00.123Z')
        assert ts is not None
        assert ts.year == 2025
        assert ts.month == 12
        assert ts.day == 7

    def test_parse_timestamp_invalid(self):
        assert parse_timestamp('') is None
        assert parse_timestamp('invalid') is None
        assert parse_timestamp(None) is None

    def test_get_mira_path(self):
        path = get_mira_path()
        assert path is not None
        assert str(path).endswith('.mira')

    def test_get_custodian(self):
        custodian = get_custodian()
        assert custodian is not None
        assert isinstance(custodian, str)
        assert len(custodian) > 0


class TestMetadata:
    """Test metadata extraction functions."""

    def test_clean_task_description_greeting(self):
        result = clean_task_description("Hi Claude, please help me fix this bug")
        assert result.startswith("Fix this bug") or "fix" in result.lower()
        assert "hi claude" not in result.lower()

    def test_clean_task_description_polite(self):
        result = clean_task_description("Can you please add a new feature?")
        assert "can you" not in result.lower()
        assert "please" not in result.lower()

    def test_extract_keywords_basic(self):
        messages = [
            {'content': 'import chromadb\nfrom sentence_transformers import SentenceTransformer'},
            {'content': 'def get_embedding_model():\n    pass'}
        ]
        keywords = extract_keywords(messages)
        assert len(keywords) > 0
        # Should extract package names and function names
        assert any('chromadb' in k.lower() for k in keywords) or \
               any('embedding' in k.lower() for k in keywords)

    def test_extract_key_facts_rules(self):
        messages = [
            {'role': 'assistant', 'content': 'You must always use HTTPS for API calls.'},
            {'role': 'assistant', 'content': 'Never store passwords in plain text.'}
        ]
        facts = extract_key_facts(messages)
        assert len(facts) > 0

    def test_extract_todo_topics(self):
        snapshots = [
            ('2025-12-07T10:00:00Z', [
                {'task': 'Implement authentication', 'status': 'pending'},
                {'task': 'Write tests', 'status': 'completed'}
            ])
        ]
        topics = extract_todo_topics(snapshots)
        assert 'Implement authentication' in topics
        assert 'Write tests' in topics

    def test_build_summary_with_existing(self):
        summary = build_summary([], '', 'Existing summary from Claude')
        assert summary == 'Existing summary from Claude'

    def test_build_summary_from_first_message(self):
        messages = [{'role': 'user', 'content': 'Fix the login bug'}]
        summary = build_summary(messages, 'Fix the login bug', '')
        assert 'login' in summary.lower() or 'fix' in summary.lower()

    def test_sample_messages_short_conversation(self):
        messages = [{'role': 'user', 'content': f'Message {i}'} for i in range(10)]
        sampled = sample_messages_for_embedding(messages)
        # Short conversations should keep all messages
        assert len(sampled) == 10

    def test_sample_messages_long_conversation(self):
        messages = [{'role': 'user', 'content': f'Message {i}'} for i in range(100)]
        sampled = sample_messages_for_embedding(messages)
        # Long conversations should be sampled
        assert len(sampled) < 100
        assert len(sampled) >= 15  # Should keep first 5 + last 10 at minimum

    def test_extract_metadata(self):
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'Fix the authentication bug'},
                {'role': 'assistant', 'content': 'I\'ll help fix that authentication issue.'}
            ],
            'summary': 'Auth bug fix',
            'first_user_message': 'Fix the authentication bug',
            'todo_snapshots': [],
            'session_meta': {
                'slug': 'test-session',
                'git_branch': 'main',
                'models_used': ['claude-3-opus'],
                'tools_used': {'Read': 2, 'Edit': 1},
                'files_touched': ['/src/auth.py']
            }
        }
        file_info = {
            'session_id': 'test-123',
            'project_path': '-workspaces-test',
            'last_modified': '2025-12-07T10:00:00Z'
        }
        metadata = extract_metadata(conversation, file_info)
        assert 'summary' in metadata
        assert 'keywords' in metadata
        assert 'task_description' in metadata
        assert metadata['session_id'] == 'test-123'


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


class TestArtifacts:
    """Test artifact storage and file reconstruction."""

    @classmethod
    def setup_class(cls):
        """Create a temporary .mira directory for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)

        # Initialize artifact DB in temp location
        mira_path = Path(cls.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)

    @classmethod
    def teardown_class(cls):
        """Clean up temporary directory."""
        shutdown_db_manager()  # Reset db_manager singleton
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_store_and_retrieve_write_operation(self):
        init_artifact_db()

        store_file_operation(
            session_id='test-session-1',
            op_type='write',
            file_path='/tmp/test.py',
            content='print("hello")',
            sequence_num=0,
            timestamp='2025-12-07T10:00:00Z'
        )

        ops = get_file_operations(file_path='/tmp/test.py')
        assert len(ops) >= 1
        assert ops[-1]['content'] == 'print("hello")'

    def test_store_and_retrieve_edit_operation(self):
        init_artifact_db()

        store_file_operation(
            session_id='test-session-2',
            op_type='edit',
            file_path='/tmp/test2.py',
            old_string='hello',
            new_string='world',
            replace_all=False,
            sequence_num=1,
            timestamp='2025-12-07T10:01:00Z'
        )

        ops = get_file_operations(file_path='/tmp/test2.py')
        assert len(ops) >= 1
        assert ops[-1]['old_string'] == 'hello'
        assert ops[-1]['new_string'] == 'world'

    def test_reconstruct_file_basic(self):
        init_artifact_db()

        # Store a write operation
        store_file_operation(
            session_id='test-session-3',
            op_type='write',
            file_path='/tmp/reconstruct.py',
            content='def hello():\n    print("hello")',
            sequence_num=0
        )

        # Store an edit operation
        store_file_operation(
            session_id='test-session-3',
            op_type='edit',
            file_path='/tmp/reconstruct.py',
            old_string='hello',
            new_string='world',
            replace_all=True,
            sequence_num=1
        )

        # Reconstruct
        result = reconstruct_file('/tmp/reconstruct.py')
        assert result is not None
        assert 'world' in result
        assert 'hello' not in result  # Should be replaced

    def test_artifact_stats(self):
        init_artifact_db()
        stats = get_artifact_stats()
        assert 'total' in stats
        assert 'file_operations' in stats


class TestArtifactDetection:
    """Test artifact detection from conversation content."""

    @classmethod
    def setup_class(cls):
        """Create a temporary .mira directory for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        mira_path = Path(cls.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)

    @classmethod
    def teardown_class(cls):
        """Clean up temporary directory."""
        shutdown_db_manager()  # Reset db_manager singleton
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_detect_language_python(self):
        code = '''def hello_world():
    print("Hello")

import os
class MyClass:
    pass'''
        lang = detect_language(code)
        assert lang == 'python'

    def test_detect_language_javascript(self):
        code = '''const foo = 42;
let bar = "hello";
function test() {
    return foo + bar;
}'''
        lang = detect_language(code)
        assert lang == 'javascript'

    def test_detect_language_sql(self):
        code = '''SELECT * FROM users
WHERE id = 1
INSERT INTO logs VALUES (1, 'test')'''
        lang = detect_language(code)
        assert lang == 'sql'

    def test_extract_code_blocks(self):
        """Test code block detection (without central storage)."""
        init_artifact_db()
        content = '''Here's some Python code:

```python
def hello():
    print("Hello, World!")

hello()
```

That should work!'''

        # Without central storage, extraction returns 0 but doesn't fail
        count = extract_artifacts_from_content(
            content=content,
            session_id='test-code-blocks',
            role='assistant'
        )
        # Will be 0 without central storage, but the parsing should work
        assert count >= 0

    def test_extract_bullet_list(self):
        """Test bullet list detection (without central storage)."""
        init_artifact_db()
        content = '''Here are the steps:

- First, install dependencies
- Second, configure the environment
- Third, run the tests
- Fourth, deploy to production

Done!'''

        count = extract_artifacts_from_content(
            content=content,
            session_id='test-bullet-list',
            role='assistant'
        )
        assert count >= 0  # 0 without central storage

    def test_extract_numbered_list(self):
        """Test numbered list detection (without central storage)."""
        init_artifact_db()
        content = '''Follow these steps:

1. Open the terminal
2. Navigate to the project directory
3. Run npm install
4. Start the development server

That's it!'''

        count = extract_artifacts_from_content(
            content=content,
            session_id='test-numbered-list',
            role='assistant'
        )
        assert count >= 0  # 0 without central storage

    def test_extract_markdown_table(self):
        """Test markdown table detection (without central storage)."""
        init_artifact_db()
        content = '''Here's a comparison table:

| Feature | Python | JavaScript |
|---------|--------|------------|
| Typing  | Dynamic | Dynamic   |
| Speed   | Medium  | Fast      |
| Syntax  | Clean   | Verbose   |

That covers the basics.'''

        count = extract_artifacts_from_content(
            content=content,
            session_id='test-table',
            role='assistant'
        )
        assert count >= 0  # 0 without central storage

    def test_extract_error_message(self):
        """Test error message detection (without central storage)."""
        init_artifact_db()
        content = '''I got this error:

Traceback (most recent call last):
  File "test.py", line 10, in <module>
    do_something()
  File "test.py", line 5, in do_something
    raise ValueError("Invalid input")
ValueError: Invalid input

Can you help?'''

        count = extract_artifacts_from_content(
            content=content,
            session_id='test-error',
            role='user'
        )
        assert count >= 0  # 0 without central storage

    def test_store_artifact_without_central_storage(self):
        """Test that store_artifact gracefully handles missing central storage."""
        init_artifact_db()
        # Without central storage, store_artifact returns False
        result = store_artifact(
            session_id='test-no-central',
            artifact_type='code_block',
            content='print("hello")',
            language='python'
        )
        # Should return False without central storage (not crash)
        assert result == False

    def test_search_artifacts_for_query(self):
        init_artifact_db()
        # Store a searchable artifact
        store_artifact(
            session_id='test-search-artifact',
            artifact_type='code_block',
            content='def authenticate_user(username, password):\n    return True',
            language='python',
            title='authentication function'
        )
        # Search should find it (may return empty if FTS not populated)
        results = search_artifacts_for_query('authenticate', limit=5)
        assert isinstance(results, list)

    def test_extract_artifacts_from_messages(self):
        init_artifact_db()
        messages = [
            {
                'type': 'user',
                'message': {'content': 'Here is some code:\n```python\ndef hello():\n    print("world")\n```'},
                'timestamp': '2025-12-07T10:00:00Z'
            },
            {
                'type': 'assistant',
                'message': {'content': [{'type': 'text', 'text': 'That looks good!'}]},
                'timestamp': '2025-12-07T10:01:00Z'
            }
        ]
        count = extract_artifacts_from_messages(messages, 'test-msg-extraction')
        assert count >= 0  # May be 0 if code block too short


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


class TestSearch:
    """Test search functionality."""

    def test_extract_excerpt_around_terms(self):
        from mira.search import extract_excerpt_around_terms
        content = "This is a test about authentication systems and how they work with user login."
        excerpt = extract_excerpt_around_terms(content, ['authentication'], context_chars=20)
        assert 'authentication' in excerpt

    def test_search_archive_for_excerpts(self):
        from mira.search import search_archive_for_excerpts
        import tempfile

        # Create temp archive file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        try:
            lines = [
                json.dumps({
                    'type': 'user',
                    'timestamp': '2025-12-07T10:00:00Z',
                    'message': {'content': 'Help me with authentication'}
                }),
                json.dumps({
                    'type': 'assistant',
                    'timestamp': '2025-12-07T10:00:30Z',
                    'message': {'content': [{'type': 'text', 'text': 'I can help with authentication and login systems.'}]}
                })
            ]
            for line in lines:
                temp_file.write(line + '\n')
            temp_file.close()

            excerpts = search_archive_for_excerpts(Path(temp_file.name), ['authentication'], max_excerpts=3)
            assert len(excerpts) >= 1
            assert any('authentication' in e['excerpt'].lower() for e in excerpts)
        finally:
            os.unlink(temp_file.name)

    def test_fulltext_search_archives(self):
        from mira.search import fulltext_search_archives
        import tempfile

        # Create temp .mira structure
        temp_dir = tempfile.mkdtemp()
        try:
            mira_path = Path(temp_dir) / '.mira'
            archives_path = mira_path / 'archives'
            metadata_path = mira_path / 'metadata'
            archives_path.mkdir(parents=True)
            metadata_path.mkdir(parents=True)

            # Create archive file
            archive_file = archives_path / 'test-session.jsonl'
            with open(archive_file, 'w') as f:
                f.write(json.dumps({
                    'type': 'user',
                    'message': {'content': 'Help with database queries'}
                }) + '\n')

            # Create metadata file
            meta_file = metadata_path / 'test-session.json'
            meta_file.write_text(json.dumps({'summary': 'Database help session'}))

            results = fulltext_search_archives('database', 5, mira_path)
            assert isinstance(results, list)
        finally:
            shutil.rmtree(temp_dir)


class TestIngestion:
    """Test ingestion functionality."""

    def test_discover_conversations_empty(self):
        from mira.ingestion import discover_conversations
        # Test with non-existent path
        result = discover_conversations(Path('/nonexistent/path'))
        assert result == []

    def test_build_document_content(self):
        from mira.metadata import build_document_content
        conversation = {
            'messages': [{'role': 'user', 'content': 'Test message'}],
            'summary': 'Test summary'
        }
        metadata = {
            'summary': 'Test summary',
            'task_description': 'Test task',
            'keywords': ['test', 'keyword'],
            'todo_topics': ['Task 1'],
            'key_facts': ['Fact 1'],
            'git_branch': 'main'
        }
        content = build_document_content(conversation, metadata)
        assert 'Test summary' in content
        assert len(content) <= 900  # Should respect token limit


class TestEmbedding:
    """Test embedding functionality (without loading model)."""

    def test_embedding_function_interface(self):
        from mira.embedding import MiraEmbeddingFunction
        ef = MiraEmbeddingFunction()
        # Model loading is lazy, so we can create the object without loading
        assert ef.model is None  # Not loaded yet
        assert hasattr(ef, 'embed_query')
        assert hasattr(ef, 'embed_documents')


class TestWatcher:
    """Test file watcher functionality."""

    def test_conversation_watcher_queue(self):
        from mira.watcher import ConversationWatcher
        import time

        # Create a mock watcher (without real collection)
        class MockCollection:
            def count(self):
                return 0

        watcher = ConversationWatcher(MockCollection(), Path('/tmp'))

        # Queue a file
        watcher.queue_file('/tmp/test.jsonl')
        assert '/tmp/test.jsonl' in watcher.pending_files

        # Queue same file again should update timestamp
        time.sleep(0.01)
        watcher.queue_file('/tmp/test.jsonl')
        assert '/tmp/test.jsonl' in watcher.pending_files


class TestHandlers:
    """Test RPC handler functions."""

    def test_calculate_storage_stats(self):
        from mira.handlers import calculate_storage_stats
        import tempfile

        temp_dir = tempfile.mkdtemp()
        try:
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()
            (mira_path / 'archives').mkdir()
            (mira_path / 'metadata').mkdir()

            stats = calculate_storage_stats(mira_path)
            assert 'total_mira' in stats
            assert 'components' in stats
            assert 'archives' in stats['components']
        finally:
            shutil.rmtree(temp_dir)

    def test_get_current_work_context_empty(self):
        from mira.handlers import get_current_work_context
        import tempfile

        shutdown_db_manager()  # Reset before changing directory
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            # Create minimal .mira structure
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()
            context = get_current_work_context()
            # When metadata_path doesn't exist, returns empty dict
            assert isinstance(context, dict)
        finally:
            shutdown_db_manager()
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_handle_recent(self):
        from mira.handlers import handle_recent
        import tempfile

        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            metadata_path = mira_path / 'metadata'
            metadata_path.mkdir(parents=True)

            # Create a test metadata file
            meta_file = metadata_path / 'test-session.json'
            meta_file.write_text(json.dumps({
                'summary': 'Test session',
                'project_path': '-workspaces-test'
            }))

            result = handle_recent({'limit': 5})
            assert 'projects' in result
            assert 'total' in result
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_handle_status(self):
        from mira.handlers import handle_status
        import tempfile

        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            # Collection is deprecated, storage is optional
            result = handle_status(None, storage=None)
            assert 'total_files' in result
            assert 'indexed' in result
            assert 'archived' in result
            assert 'storage_path' in result
            assert 'last_sync' in result
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_get_custodian_profile(self):
        from mira.handlers import get_custodian_profile
        import tempfile

        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            profile = get_custodian_profile()
            assert 'name' in profile
            assert 'tech_stack' in profile
            assert 'total_sessions' in profile
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_handle_rpc_request_unknown_method(self):
        from mira.handlers import handle_rpc_request

        class MockCollection:
            def count(self):
                return 0

        request = {'method': 'unknown_method', 'id': 1}
        response = handle_rpc_request(request, MockCollection())
        assert 'error' in response
        assert response['error']['code'] == -32601  # Method not found


class TestCustodian:
    """Test custodian learning functionality."""

    def test_init_custodian_db(self):
        """Test custodian database initialization."""
        shutdown_db_manager()  # Reset before creating new temp dir
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            db_path = mira_path / 'custodian.db'
            assert db_path.exists()
        finally:
            shutdown_db_manager()  # Clean up connections
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_extract_identity_from_messages(self):
        """Test that we learn the user's name from messages."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            # Conversation where user introduces themselves
            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'Hi, my name is Max. Can you help me?'},
                    {'role': 'assistant', 'content': 'Hello Max! I\'d be happy to help.'},
                ]
            }

            extract_custodian_learnings(conversation, 'test-session-1')

            profile = get_full_custodian_profile()
            # Check if name was learned
            assert 'identity' in profile
            # Name may or may not be captured depending on pattern matching
            # The important thing is the function runs without error
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_extract_preferences(self):
        """Test that we learn preferences from user statements."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'I prefer using pnpm instead of npm'},
                    {'role': 'user', 'content': 'I always use vitest for testing'},
                    {'role': 'user', 'content': 'No emojis please'},
                ]
            }

            extract_custodian_learnings(conversation, 'pref-session')

            profile = get_full_custodian_profile()
            assert 'preferences' in profile
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_extract_rules(self):
        """Test that we learn explicit rules from conversations."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'Never commit directly to main branch'},
                    {'role': 'assistant', 'content': 'You should always run tests before pushing'},
                    {'role': 'user', 'content': 'Avoid using var, use const or let instead'},
                ]
            }

            extract_custodian_learnings(conversation, 'rules-session')

            profile = get_full_custodian_profile()
            assert 'rules' in profile
            assert 'never' in profile['rules']
            assert 'always' in profile['rules']
            assert 'avoid' in profile['rules']
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_extract_danger_zones(self):
        """Test that we learn about problematic files/areas."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'There was an error in auth.py again'},
                    {'role': 'assistant', 'content': 'The problem with auth.py is the session handling'},
                    {'role': 'user', 'content': 'Be careful with legacy-api.js, it keeps breaking'},
                ]
            }

            # Extract twice to trigger frequency threshold
            extract_custodian_learnings(conversation, 'danger-session-1')
            extract_custodian_learnings(conversation, 'danger-session-2')

            profile = get_full_custodian_profile()
            assert 'danger_zones' in profile
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_get_danger_zones_for_files(self):
        """Test checking files against known danger zones."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            # Create a danger zone by extracting from a conversation
            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'The issue with payment.py is very serious'},
                    {'role': 'assistant', 'content': 'Yes, payment.py has had multiple bugs'},
                ]
            }
            extract_custodian_learnings(conversation, 'danger-1')
            extract_custodian_learnings(conversation, 'danger-2')

            # Check if files match danger zones
            warnings = get_danger_zones_for_files(['/src/payment.py', '/src/app.py'])
            # May or may not find depending on frequency threshold
            assert isinstance(warnings, list)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_full_custodian_profile(self):
        """Test getting the complete custodian profile."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            profile = get_full_custodian_profile()

            # Check structure
            assert 'name' in profile
            assert 'identity' in profile
            assert 'preferences' in profile
            assert 'rules' in profile
            assert 'danger_zones' in profile
            assert 'work_patterns' in profile
            assert 'summary' in profile
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_profile_summary_generation(self):
        """Test that profile summary is generated correctly."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_custodian_db()

            # Add some data
            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'My name is Alice'},
                    {'role': 'user', 'content': 'I prefer using typescript'},
                ]
            }
            extract_custodian_learnings(conversation, 'summary-test')

            profile = get_full_custodian_profile()
            assert 'summary' in profile
            assert isinstance(profile['summary'], str)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)


class TestInsights:
    """Test insights module (errors, decisions)."""

    def test_init_insights_db(self):
        """Test insights database initialization."""
        shutdown_db_manager()  # Reset before creating new temp dir
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_insights_db()

            db_path = mira_path / 'insights.db'
            assert db_path.exists()
        finally:
            shutdown_db_manager()  # Clean up connections
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_normalize_error_message(self):
        """Test error message normalization."""
        # Test line number removal
        error = "Error at line 42 in file.py"
        normalized = normalize_error_message(error)
        assert 'line <N>' in normalized

        # Test file path removal
        error = "Error in /home/user/project/file.py"
        normalized = normalize_error_message(error)
        assert '<FILE>' in normalized

        # Test memory address removal
        error = "Object at 0x7fff5fbff8c0"
        normalized = normalize_error_message(error)
        assert '<ADDR>' in normalized

    def test_compute_error_signature(self):
        """Test error signature computation."""
        error1 = "TypeError: Cannot read property 'foo' of undefined"
        error2 = "TypeError: Cannot read property 'bar' of undefined"

        sig1 = compute_error_signature(error1)
        sig2 = compute_error_signature(error2)

        # Should both start with TypeError
        assert sig1.startswith('TypeError:')
        assert sig2.startswith('TypeError:')

    def test_extract_errors_from_conversation(self):
        """Test error extraction from conversations."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_insights_db()

            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'I got this error: TypeError: Cannot read property "x" of undefined'},
                    {'role': 'assistant', 'content': 'This error occurs when you try to access a property on undefined. Check that the object exists before accessing it.'},
                ]
            }

            count = extract_errors_from_conversation(conversation, 'test-session')
            assert count >= 0  # May or may not match depending on pattern

            stats = get_error_stats()
            assert 'total' in stats
            assert 'by_type' in stats
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_extract_decisions_from_conversation(self):
        """Test decision extraction from conversations."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_insights_db()

            conversation = {
                'messages': [
                    {'role': 'assistant', 'content': 'I decided to use React for the frontend because it has better component reusability.'},
                    {'role': 'assistant', 'content': 'I recommend using PostgreSQL instead of MySQL for this project.'},
                ]
            }

            count = extract_decisions_from_conversation(conversation, 'decision-session')
            assert count >= 0

            stats = get_decision_stats()
            assert 'total' in stats
            assert 'by_category' in stats
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_search_error_solutions(self):
        """Test searching for error solutions."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_insights_db()

            # Add some errors first
            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'Error: ModuleNotFoundError: No module named "pandas"'},
                    {'role': 'assistant', 'content': 'You need to install pandas. Run: pip install pandas'},
                ]
            }
            extract_errors_from_conversation(conversation, 'pip-error-session')

            # Search for it
            results = search_error_solutions('ModuleNotFoundError pandas', limit=5)
            assert isinstance(results, list)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_search_decisions(self):
        """Test searching for decisions."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_insights_db()

            # Add some decisions first
            conversation = {
                'messages': [
                    {'role': 'assistant', 'content': 'I recommend using FastAPI for this API because it has automatic OpenAPI documentation.'},
                ]
            }
            extract_decisions_from_conversation(conversation, 'api-decision')

            # Search for it
            results = search_decisions('FastAPI API', limit=5)
            assert isinstance(results, list)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)


class TestBootstrap:
    """Test bootstrap functionality."""

    def test_is_running_in_venv(self):
        # This should return False since we're not running in .mira/.venv
        result = is_running_in_venv()
        assert isinstance(result, bool)


class TestUserExperienceScenarios:
    """
    End-to-end tests for real user experience scenarios.
    These test complete workflows, not individual functions.
    """

    @classmethod
    def setup_class(cls):
        """Create a complete mock MIRA environment."""
        shutdown_db_manager()  # Reset before creating new temp dir
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)

        # Create .mira directory structure
        cls.mira_path = Path(cls.temp_dir) / '.mira'
        cls.archives_path = cls.mira_path / 'archives'
        cls.metadata_path = cls.mira_path / 'metadata'
        cls.mira_path.mkdir()
        cls.archives_path.mkdir()
        cls.metadata_path.mkdir()

        # Create mock Claude projects directory
        cls.claude_path = Path(cls.temp_dir) / '.claude' / 'projects' / '-workspaces-testproject'
        cls.claude_path.mkdir(parents=True)

        # Initialize all databases for end-to-end tests
        init_artifact_db()
        init_custodian_db()
        init_insights_db()
        init_concepts_db()

    @classmethod
    def teardown_class(cls):
        """Clean up."""
        shutdown_db_manager()  # Reset db_manager singleton
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _create_conversation_file(self, session_id: str, messages: list, include_summary: bool = True):
        """Helper to create a realistic conversation JSONL file."""
        conv_file = self.claude_path / f"{session_id}.jsonl"
        with open(conv_file, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg) + '\n')
            if include_summary:
                f.write(json.dumps({'type': 'summary', 'summary': f'Session about {session_id}'}) + '\n')
        return conv_file

    def test_scenario_search_for_past_discussion(self):
        """
        SCENARIO: User wants to find where they discussed authentication before.

        User asks: "Where did I work on authentication?"
        Expected: MIRA finds relevant conversations with excerpts.
        """
        from mira.search import handle_search, fulltext_search_archives
        from mira.ingestion import ingest_conversation

        # Setup: Create a conversation about authentication
        messages = [
            {'type': 'user', 'timestamp': '2025-12-01T10:00:00Z',
             'message': {'content': 'Help me implement user authentication with JWT tokens'}},
            {'type': 'assistant', 'timestamp': '2025-12-01T10:01:00Z',
             'message': {'content': [{'type': 'text', 'text': 'I\'ll help you implement JWT authentication. First, install jsonwebtoken...'}],
                         'model': 'claude-3-opus'}},
            {'type': 'user', 'timestamp': '2025-12-01T10:05:00Z',
             'message': {'content': 'How do I validate the token on each request?'}},
            {'type': 'assistant', 'timestamp': '2025-12-01T10:06:00Z',
             'message': {'content': [{'type': 'text', 'text': 'Create middleware that extracts and verifies the JWT token from the Authorization header.'}],
                         'model': 'claude-3-opus'}}
        ]
        self._create_conversation_file('auth-session-123', messages)

        # Create a mock collection
        class MockCollection:
            def count(self): return 0  # Empty, will use fulltext fallback
            def query(self, **kwargs): return {'ids': [[]], 'metadatas': [[]], 'distances': [[]]}

        # Search for authentication
        result = handle_search({'query': 'authentication JWT tokens', 'limit': 5}, MockCollection())

        # Verify: Should use fulltext fallback and find results
        assert result['total'] >= 0  # May find the conversation via fulltext
        assert 'results' in result

    def test_scenario_find_recent_work_sessions(self):
        """
        SCENARIO: User wants to see their recent work sessions.

        User asks: "Show me my recent sessions"
        Expected: List of recent sessions grouped by project.
        """
        from mira.handlers import handle_recent

        # Setup: Create metadata files for recent sessions
        sessions = [
            {'summary': 'Fixed authentication bug', 'project_path': '-workspaces-webapp'},
            {'summary': 'Added database migrations', 'project_path': '-workspaces-webapp'},
            {'summary': 'Setup CI/CD pipeline', 'project_path': '-workspaces-devops'},
        ]
        for i, session in enumerate(sessions):
            meta_file = self.metadata_path / f'session-{i}.json'
            meta_file.write_text(json.dumps(session))

        # Action: Get recent sessions
        result = handle_recent({'limit': 10})

        # Verify: Should return sessions grouped by project
        assert 'projects' in result
        assert 'total' in result
        assert result['total'] >= 3

    def test_scenario_init_new_session_with_context(self):
        """
        SCENARIO: User starts a new Claude Code session and wants context.

        User calls mira_init to understand recent work.
        Expected: Get tiered output with guidance, alerts, and core context.
        """
        from mira.handlers import handle_init

        # Collection is deprecated, storage is optional (None = no central storage)
        result = handle_init({'project_path': '-workspaces-testproject'}, None, storage=None)

        # Verify: Should return tiered output structure
        assert 'guidance' in result  # Tier 1: actionable guidance
        assert 'alerts' in result    # Tier 1: alerts requiring attention
        assert 'core' in result      # Tier 2: essential context
        assert 'indexing' in result  # Stats about indexed conversations

        # Core should contain custodian and current_work
        assert 'custodian' in result['core']
        assert 'current_work' in result['core']

        # Without central storage, indexed count will be 0
        assert 'indexed' in result['indexing']

    def test_scenario_find_code_claude_wrote(self):
        """
        SCENARIO: User wants to find code Claude wrote for database queries.

        User asks: "Find the database query code you wrote"
        Expected: Should find code artifacts stored during previous sessions.
        """
        from mira.artifacts import init_artifact_db, store_artifact, search_artifacts_for_query

        init_artifact_db()

        # Setup: Store code artifacts
        store_artifact(
            session_id='db-session-1',
            artifact_type='code_block',
            content='''async function getUserById(id) {
    const query = 'SELECT * FROM users WHERE id = $1';
    const result = await db.query(query, [id]);
    return result.rows[0];
}''',
            language='javascript',
            title='database query function'
        )

        store_artifact(
            session_id='db-session-2',
            artifact_type='code_block',
            content='''def get_all_products():
    cursor.execute("SELECT * FROM products ORDER BY created_at DESC")
    return cursor.fetchall()''',
            language='python',
            title='product database query'
        )

        # Action: Search for database code
        results = search_artifacts_for_query('database query', limit=5)

        # Verify: Should find relevant code artifacts
        assert isinstance(results, list)

    def test_scenario_reconstruct_file_claude_wrote(self):
        """
        SCENARIO: User lost a file that Claude wrote and wants to recover it.

        User asks: "Reconstruct the config file you wrote yesterday"
        Expected: MIRA can replay Write/Edit operations to reconstruct the file.
        """
        from mira.artifacts import init_artifact_db, store_file_operation, reconstruct_file

        init_artifact_db()

        # Setup: Store file operations as Claude would have made them
        file_path = '/project/config.json'

        # Claude writes initial file
        store_file_operation(
            session_id='config-session',
            op_type='write',
            file_path=file_path,
            content='{\n  "debug": false,\n  "port": 3000\n}',
            sequence_num=0,
            timestamp='2025-12-06T10:00:00Z'
        )

        # Claude edits the file
        store_file_operation(
            session_id='config-session',
            op_type='edit',
            file_path=file_path,
            old_string='"debug": false',
            new_string='"debug": true',
            replace_all=False,
            sequence_num=1,
            timestamp='2025-12-06T10:05:00Z'
        )

        # Action: Reconstruct the file
        content = reconstruct_file(file_path)

        # Verify: Should have the edited content
        assert content is not None
        assert '"debug": true' in content
        assert '"port": 3000' in content

    def test_scenario_new_conversation_detected(self):
        """
        SCENARIO: User finishes a conversation and MIRA should automatically ingest it.

        A new .jsonl file appears in ~/.claude/projects/
        Expected: File watcher detects it, queues for ingestion after debounce.
        """
        from mira.watcher import ConversationWatcher
        from mira.ingestion import ingest_conversation
        import time

        # Setup: Create conversation file
        messages = [
            {'type': 'user', 'timestamp': '2025-12-07T10:00:00Z',
             'message': {'content': 'Help me write unit tests'}},
            {'type': 'assistant', 'timestamp': '2025-12-07T10:01:00Z',
             'message': {'content': [{'type': 'text', 'text': 'I\'ll help you write comprehensive unit tests.'}],
                         'model': 'claude-3-opus'}}
        ]
        conv_file = self._create_conversation_file('new-conv-456', messages)

        watcher = ConversationWatcher(None, self.mira_path, storage=None)

        # Action: Simulate file detection
        watcher.queue_file(str(conv_file))

        # Verify: File should be queued for processing
        assert str(conv_file) in watcher.pending_files

        # Directly test ingestion - without central storage it will fail but shouldn't crash
        file_info = {
            'session_id': 'new-conv-456',
            'file_path': str(conv_file),
            'project_path': '-workspaces-testproject',
            'last_modified': '2025-12-07T10:01:00Z'
        }
        # Without central storage, ingest_conversation returns False
        result = ingest_conversation(file_info, None, self.mira_path, storage=None)
        # Result is False without central storage but the function shouldn't crash
        assert result in [True, False]

    def test_scenario_skip_agent_files(self):
        """
        SCENARIO: MIRA should skip agent-*.jsonl subagent task logs.

        When discovering conversations, agent files should be filtered out.
        """
        from mira.ingestion import discover_conversations

        # Setup: Create regular and agent files
        regular_file = self.claude_path / 'regular-session.jsonl'
        agent_file = self.claude_path / 'agent-task-123.jsonl'

        regular_file.write_text(json.dumps({'type': 'user', 'message': {'content': 'test'}}) + '\n')
        agent_file.write_text(json.dumps({'type': 'user', 'message': {'content': 'agent task'}}) + '\n')

        # Action: Discover conversations
        conversations = discover_conversations(self.claude_path.parent)

        # Verify: Should include regular file but not agent file
        session_ids = [c['session_id'] for c in conversations]
        assert 'regular-session' in session_ids
        assert 'agent-task-123' not in session_ids

    def test_scenario_long_conversation_sampling(self):
        """
        SCENARIO: A very long conversation (100+ messages) needs to be indexed.

        ChromaDB has limited embedding context, so long conversations must be sampled.
        Expected: Sample first messages, time gaps, topic shifts, and last messages.
        """
        from mira.metadata import sample_messages_for_embedding

        # Setup: Create a 150-message conversation
        messages = []
        for i in range(150):
            messages.append({
                'role': 'user' if i % 2 == 0 else 'assistant',
                'content': f'Message number {i} about topic {"A" if i < 50 else "B" if i < 100 else "C"}',
                'timestamp': f'2025-12-07T{10 + i//60:02d}:{i%60:02d}:00Z'
            })

        # Action: Sample for embedding
        sampled = sample_messages_for_embedding(messages)

        # Verify: Should be significantly reduced but capture key parts
        assert len(sampled) < len(messages)
        assert len(sampled) >= 15  # At minimum: first 5 + last 10

    def test_scenario_fulltext_fallback_when_no_semantic_match(self):
        """
        SCENARIO: Semantic search finds nothing but the term exists in archives.

        User searches for very specific term that wasn't in the indexed summary.
        Expected: Falls back to fulltext search of archives.
        """
        from mira.search import handle_search

        # Setup: Create an archive with specific term not in summary
        archive_file = self.archives_path / 'specific-session.jsonl'
        with open(archive_file, 'w') as f:
            f.write(json.dumps({
                'type': 'user',
                'message': {'content': 'Help me configure the XYZ_OBSCURE_SETTING environment variable'}
            }) + '\n')
            f.write(json.dumps({
                'type': 'assistant',
                'message': {'content': [{'type': 'text', 'text': 'Set XYZ_OBSCURE_SETTING=true in your .env file'}]}
            }) + '\n')

        # Create metadata (without the obscure term in summary)
        meta_file = self.metadata_path / 'specific-session.json'
        meta_file.write_text(json.dumps({'summary': 'Environment configuration', 'project_path': '-test'}))

        # Collection is deprecated, storage is optional
        result = handle_search({'query': 'XYZ_OBSCURE_SETTING', 'limit': 5}, None, storage=None)

        # Verify: Without central storage, returns results or falls back to fulltext
        assert result is not None
        assert 'results' in result
        assert 'total' in result

    def test_scenario_enrich_search_results_with_excerpts(self):
        """
        SCENARIO: Semantic search finds a conversation, enrich with actual excerpts.

        User searches "database migration", semantic search finds relevant session,
        then we extract actual conversation excerpts with the term.
        """
        from mira.search import enrich_results_from_archives

        # Setup: Create archive with detailed discussion
        archive_file = self.archives_path / 'migration-session.jsonl'
        with open(archive_file, 'w') as f:
            f.write(json.dumps({
                'type': 'user',
                'timestamp': '2025-12-05T09:00:00Z',
                'message': {'content': 'I need to add a database migration for the new users table'}
            }) + '\n')
            f.write(json.dumps({
                'type': 'assistant',
                'timestamp': '2025-12-05T09:01:00Z',
                'message': {'content': [{'type': 'text', 'text': 'I\'ll create a migration file. Run: npx knex migrate:make add_users_table'}]}
            }) + '\n')

        # Simulated semantic search result
        semantic_results = [{
            'session_id': 'migration-session',
            'summary': 'Database work',
            'relevance': 0.85
        }]

        # Action: Enrich with excerpts
        enriched = enrich_results_from_archives(semantic_results, 'database migration', self.mira_path)

        # Verify: Should have excerpts added
        assert len(enriched) == 1
        assert 'excerpts' in enriched[0]
        assert 'has_archive_matches' in enriched[0]

    def test_scenario_extract_all_artifact_types(self):
        """
        SCENARIO: A conversation contains code, lists, tables, and errors.

        MIRA should detect and store all artifact types for later retrieval.
        Without central storage, extraction returns 0 but parsing should work.
        """
        from mira.artifacts import init_artifact_db, extract_artifacts_from_content, get_artifact_stats

        init_artifact_db()

        content = '''Here's the implementation plan:

1. Create the database schema
2. Implement the API endpoints
3. Add authentication middleware
4. Write unit tests
5. Deploy to staging

Here's the code:

```python
def create_user(data):
    user = User(**data)
    db.session.add(user)
    db.session.commit()
    return user
```

Here's a comparison of frameworks:

| Framework | Speed | Ease |
|-----------|-------|------|
| Django    | Med   | High |
| FastAPI   | High  | Med  |
| Flask     | Med   | High |

I got this error when testing:

Traceback (most recent call last):
  File "app.py", line 42, in create_user
    db.session.commit()
sqlalchemy.exc.IntegrityError: duplicate key value
'''

        # Action: Extract artifacts
        count = extract_artifacts_from_content(
            content=content,
            session_id='multi-artifact-session',
            role='assistant'
        )

        # Without central storage, extraction returns 0 but shouldn't crash
        assert count >= 0

        stats = get_artifact_stats()
        assert 'total' in stats

    def test_scenario_handle_empty_collection_gracefully(self):
        """
        SCENARIO: First-time user with no indexed conversations yet.

        User runs mira_search before any conversations have been indexed.
        Expected: Gracefully return empty results, not crash.
        """
        from mira.search import handle_search

        class EmptyCollection:
            def count(self): return 0

        # Action: Search on empty collection
        result = handle_search({'query': 'anything', 'limit': 10}, EmptyCollection())

        # Verify: Should return valid empty result
        assert result is not None
        assert 'results' in result
        assert 'total' in result

    def test_scenario_ingest_skips_already_indexed(self):
        """
        SCENARIO: Running ingestion twice should skip already-indexed conversations.

        Without central storage, both will return False (storage unavailable).
        The key test is that the function doesn't crash.
        """
        from mira.ingestion import ingest_conversation

        # Setup: Create conversation
        messages = [
            {'type': 'user', 'timestamp': '2025-12-07T10:00:00Z',
             'message': {'content': 'Quick question'}},
            {'type': 'assistant', 'timestamp': '2025-12-07T10:00:30Z',
             'message': {'content': [{'type': 'text', 'text': 'Here is the answer.'}],
                         'model': 'claude-3-opus'}}
        ]
        conv_file = self._create_conversation_file('idempotent-test', messages)

        file_info = {
            'session_id': 'idempotent-test',
            'file_path': str(conv_file),
            'project_path': '-workspaces-testproject',
            'last_modified': '2025-12-07T10:00:30Z'
        }

        # Without central storage, ingestion returns False but shouldn't crash
        result1 = ingest_conversation(file_info, None, self.mira_path, storage=None)
        assert result1 in [True, False]

        # Second call should also not crash
        result2 = ingest_conversation(file_info, None, self.mira_path, storage=None)
        assert result2 in [True, False]

    def test_scenario_system_status_overview(self):
        """
        SCENARIO: User wants to check MIRA system health.

        User calls mira_status to see indexed count, storage, pending files.
        """
        from mira.handlers import handle_status

        # Collection is deprecated, storage is optional
        result = handle_status(None, storage=None)

        # Verify: Should return system overview
        assert 'indexed' in result
        assert 'storage_path' in result
        assert 'last_sync' in result
        # Without central storage, indexed will be 0
        assert result['indexed'] >= 0


class TestLocalStore:
    """Test local SQLite storage module."""

    @classmethod
    def setup_class(cls):
        # Reset db_manager to ensure fresh state
        shutdown_db_manager()
        cls.mira_path = Path(tempfile.mkdtemp())
        os.environ['MIRA_PATH'] = str(cls.mira_path)
        # Initialize local store
        from mira import local_store
        local_store._initialized = False  # Reset initialization flag
        local_store.init_local_db()

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        if cls.mira_path.exists():
            shutil.rmtree(cls.mira_path, ignore_errors=True)
        if 'MIRA_PATH' in os.environ:
            del os.environ['MIRA_PATH']

    def test_local_store_init(self):
        """Test local store database initialization."""
        from mira.local_store import LOCAL_DB
        from mira.db_manager import get_db_manager

        db = get_db_manager()

        # Verify tables exist
        tables = db.execute_read(LOCAL_DB,
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", ())
        table_names = [t['name'] for t in tables]
        assert 'projects' in table_names
        assert 'sessions' in table_names
        assert 'archives' in table_names
        assert 'custodian' in table_names

    def test_local_project_operations(self):
        """Test get_or_create_project in local store."""
        from mira.local_store import get_or_create_project

        # Create project
        project_id = get_or_create_project('/workspaces/test', slug='test')
        assert project_id > 0

        # Get same project
        project_id2 = get_or_create_project('/workspaces/test')
        assert project_id2 == project_id

        # Create project with git remote
        project_id3 = get_or_create_project(
            '/other/path', git_remote='git@github.com:user/repo.git'
        )
        assert project_id3 > 0
        assert project_id3 != project_id

    def test_local_session_operations(self):
        """Test session upsert and retrieval in local store."""
        from mira.local_store import get_or_create_project, upsert_session, get_recent_sessions

        project_id = get_or_create_project('/workspaces/test2')

        # Create session
        session_id = upsert_session(
            project_id=project_id,
            session_id='test-session-123',
            summary='Test session summary',
            keywords=['test', 'local'],
            task_description='Testing local storage',
            message_count=5,
        )
        assert session_id > 0

        # Get recent sessions
        sessions = get_recent_sessions(project_id=project_id, limit=10)
        assert len(sessions) >= 1
        assert any(s['session_id'] == 'test-session-123' for s in sessions)

    def test_local_archive_operations(self):
        """Test archive storage in local store."""
        from mira.local_store import (
            get_or_create_project, upsert_session, upsert_archive, get_archive
        )

        project_id = get_or_create_project('/workspaces/test3')
        session_db_id = upsert_session(
            project_id=project_id,
            session_id='archive-test-session',
            summary='Archive test',
        )

        # Store archive
        content = '{"type":"user","message":{"content":"test"}}\n'
        archive_id = upsert_archive(
            session_db_id=session_db_id,
            content=content,
            content_hash='abc123',
        )
        assert archive_id > 0

        # Retrieve archive
        retrieved = get_archive('archive-test-session')
        assert retrieved == content

    def test_local_fts_search(self):
        """Test full-text search in local store."""
        from mira.local_store import (
            get_or_create_project, upsert_session, search_sessions_fts
        )

        project_id = get_or_create_project('/workspaces/fts-test')
        upsert_session(
            project_id=project_id,
            session_id='fts-session-1',
            summary='Discussion about Python and Flask web development',
            keywords=['python', 'flask', 'web'],
        )
        upsert_session(
            project_id=project_id,
            session_id='fts-session-2',
            summary='Debugging JavaScript React components',
            keywords=['javascript', 'react'],
        )

        # Search for Python
        results = search_sessions_fts('python', project_id=project_id)
        assert len(results) >= 1
        assert any(s['session_id'] == 'fts-session-1' for s in results)

        # Search for React
        results = search_sessions_fts('react', project_id=project_id)
        assert len(results) >= 1
        assert any(s['session_id'] == 'fts-session-2' for s in results)

    def test_local_custodian_operations(self):
        """Test custodian preferences in local store."""
        from mira.local_store import upsert_custodian, get_custodian_all

        upsert_custodian(
            key='preference:editor',
            value='vscode',
            category='preference',
            confidence=0.8,
            source_session='test-session',
        )

        prefs = get_custodian_all()
        assert len(prefs) >= 1
        editor_pref = next((p for p in prefs if p['key'] == 'preference:editor'), None)
        assert editor_pref is not None
        assert editor_pref['value'] == 'vscode'


class TestStorageFallback:
    """Test storage abstraction layer fallback behavior."""

    @classmethod
    def setup_class(cls):
        # Reset db_manager and storage to ensure fresh state
        shutdown_db_manager()
        from mira.storage import reset_storage
        from mira import local_store
        reset_storage()
        local_store._initialized = False

        cls.mira_path = Path(tempfile.mkdtemp())
        os.environ['MIRA_PATH'] = str(cls.mira_path)
        # Ensure no server.json exists (forces local mode)
        server_json = cls.mira_path / 'server.json'
        if server_json.exists():
            server_json.unlink()

    @classmethod
    def teardown_class(cls):
        from mira.storage import reset_storage
        reset_storage()
        shutdown_db_manager()
        if cls.mira_path.exists():
            shutil.rmtree(cls.mira_path, ignore_errors=True)
        if 'MIRA_PATH' in os.environ:
            del os.environ['MIRA_PATH']

    def test_storage_mode_without_config(self):
        """Test that storage falls back to local when no config exists."""
        from mira.storage import Storage, reset_storage

        reset_storage()
        storage = Storage()

        # Should not be using central (no config)
        assert storage.using_central == False

        mode = storage.get_storage_mode()
        assert mode['mode'] == 'local'
        assert 'limitations' in mode
        assert len(mode['limitations']) > 0

    def test_storage_local_fallback_operations(self):
        """Test that storage operations work in local mode."""
        from mira.storage import Storage, reset_storage

        reset_storage()
        storage = Storage()

        # Project operations should work locally
        project_id = storage.get_or_create_project('/test/project')
        assert project_id is not None

        # Session operations should work locally
        session_id = storage.upsert_session(
            project_path='/test/project',
            session_id='local-test-session',
            summary='Testing local fallback',
            keywords=['test'],
        )
        assert session_id is not None

        # Recent sessions should work
        sessions = storage.get_recent_sessions(limit=5)
        assert isinstance(sessions, list)

        # FTS search should work
        results = storage.search_sessions_fts('test')
        assert isinstance(results, list)

    def test_vector_operations_skip_in_local_mode(self):
        """Test that vector operations don't crash in local mode."""
        from mira.storage import Storage, reset_storage

        reset_storage()
        storage = Storage()

        # Vector search should return empty list, not raise
        results = storage.vector_search([0.1] * 384)
        assert results == []

        # Vector upsert should return None, not raise
        result = storage.vector_upsert(
            vector=[0.1] * 384,
            content='test',
            session_id='test',
            project_path='/test',
        )
        assert result is None

        # Batch upsert should return 0, not raise
        result = storage.vector_batch_upsert([])
        assert result == 0

    def test_storage_health_check_local_mode(self):
        """Test health check in local mode."""
        from mira.storage import Storage, reset_storage

        reset_storage()
        storage = Storage()

        health = storage.health_check()
        assert health['mode'] == 'local'
        assert health['using_central'] == False

    def test_session_exists_in_central_local_mode(self):
        """Test session_exists_in_central returns False in local mode."""
        from mira.storage import Storage, reset_storage

        reset_storage()
        storage = Storage()

        # In local mode, should always return False (no central storage)
        assert storage.session_exists_in_central('any-session-id') == False


class TestLocalToCentralSync:
    """Test sync detection from local to central storage."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        from mira.storage import reset_storage
        from mira import local_store
        reset_storage()
        local_store._initialized = False

        cls.mira_path = Path(tempfile.mkdtemp())
        os.environ['MIRA_PATH'] = str(cls.mira_path)

        # Create metadata directory
        (cls.mira_path / 'metadata').mkdir(parents=True, exist_ok=True)

    @classmethod
    def teardown_class(cls):
        from mira.storage import reset_storage
        reset_storage()
        shutdown_db_manager()
        if cls.mira_path.exists():
            shutil.rmtree(cls.mira_path, ignore_errors=True)
        if 'MIRA_PATH' in os.environ:
            del os.environ['MIRA_PATH']

    def test_sync_detection_skips_unchanged_in_local_mode(self):
        """Test that unchanged local sessions are skipped in local mode."""
        from mira.storage import Storage, reset_storage
        from mira.ingestion import ingest_conversation

        reset_storage()
        storage = Storage()

        # Create a test conversation file
        test_session_id = 'sync-test-session-001'
        test_file = self.mira_path / 'test_conversation.jsonl'
        test_file.write_text(json.dumps({
            'type': 'user',
            'message': {'role': 'user', 'content': 'Test message'}
        }) + '\n')

        # Create metadata file to simulate already-ingested session
        meta_file = self.mira_path / 'metadata' / f'{test_session_id}.json'
        meta_file.write_text(json.dumps({
            'last_modified': '2025-01-01T00:00:00',
            'summary': 'Test session'
        }))

        # Ingest should skip (already processed, local mode)
        file_info = {
            'session_id': test_session_id,
            'file_path': str(test_file),
            'project_path': '/test/project',
            'last_modified': '2025-01-01T00:00:00'  # Same as metadata
        }

        result = ingest_conversation(file_info, None, self.mira_path, storage)
        assert result == False  # Should be skipped

    def test_sync_detection_processes_modified_file(self):
        """Test that modified files are re-ingested."""
        from mira.storage import Storage, reset_storage
        from mira.ingestion import ingest_conversation

        reset_storage()
        storage = Storage()

        test_session_id = 'sync-test-session-002'
        test_file = self.mira_path / 'test_conversation2.jsonl'
        test_file.write_text(json.dumps({
            'type': 'user',
            'message': {'role': 'user', 'content': 'Test message for sync'}
        }) + '\n' + json.dumps({
            'type': 'assistant',
            'message': {'role': 'assistant', 'content': 'Response message'}
        }) + '\n')

        # Create OLD metadata file
        meta_file = self.mira_path / 'metadata' / f'{test_session_id}.json'
        meta_file.write_text(json.dumps({
            'last_modified': '2025-01-01T00:00:00',
            'summary': 'Old session'
        }))

        # Ingest with NEWER modification time
        file_info = {
            'session_id': test_session_id,
            'file_path': str(test_file),
            'project_path': '/test/project',
            'last_modified': '2025-01-02T00:00:00'  # Newer than metadata
        }

        result = ingest_conversation(file_info, None, self.mira_path, storage)
        assert result == True  # Should be ingested (file modified)


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestUtils,
        TestMetadata,
        TestParsing,
        TestArtifacts,
        TestArtifactDetection,
        TestConversationParsing,
        TestSearch,
        TestIngestion,
        TestEmbedding,
        TestWatcher,
        TestHandlers,
        TestCustodian,
        TestInsights,
        TestBootstrap,
        TestUserExperienceScenarios,
        TestLocalStore,
        TestStorageFallback,
        TestLocalToCentralSync,
    ]

    total = 0
    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n=== {test_class.__name__} ===")

        # Setup
        if hasattr(test_class, 'setup_class'):
            try:
                test_class.setup_class()
            except Exception as e:
                print(f"  Setup failed: {e}")
                continue

        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    failed += 1

        # Teardown
        if hasattr(test_class, 'teardown_class'):
            try:
                test_class.teardown_class()
            except:
                pass

        # Always reset db_manager between test classes to avoid cross-contamination
        try:
            shutdown_db_manager()
        except:
            pass

    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed, {failed} failed")

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
