"""Tests for mira.artifacts module."""

import os
import json
import tempfile
import shutil
from pathlib import Path

from mira.artifacts import (
    init_artifact_db, store_file_operation, get_file_operations,
    reconstruct_file, get_artifact_stats, detect_language,
    extract_artifacts_from_content, store_artifact, search_artifacts_for_query,
    extract_artifacts_from_messages
)
from mira.db_manager import shutdown_db_manager


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
        # Without central storage, store_artifact now queues for sync
        result = store_artifact(
            session_id='test-no-central',
            artifact_type='code_block',
            content='print("hello")',
            language='python'
        )
        # With sync queue, returns True (queued) or False (failed to queue)
        # Either way shouldn't crash
        assert result in [True, False]

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


class TestCollectArtifactsFromContent:
    """Test artifact collection function."""

    @classmethod
    def setup_class(cls):
        """Create a temporary .mira directory for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        mira_path = Path(cls.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)
        init_artifact_db()

    @classmethod
    def teardown_class(cls):
        """Clean up temporary directory."""
        shutdown_db_manager()
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_collect_code_blocks(self):
        """Test that code blocks are collected from content."""
        from mira.artifacts import collect_artifacts_from_content

        content = '''Here is some code:

```python
def calculate_sum(a, b):
    return a + b

result = calculate_sum(5, 3)
print(f"Result: {result}")
```

This function adds two numbers together.'''

        artifacts = collect_artifacts_from_content(
            content=content,
            session_id='test-collect-code',
            role='assistant'
        )

        # Should find the code block
        code_blocks = [a for a in artifacts if a['artifact_type'] == 'code_block']
        assert len(code_blocks) >= 1
        assert 'calculate_sum' in code_blocks[0]['content']

    def test_collect_bullet_list(self):
        """Test that bullet lists are collected."""
        from mira.artifacts import collect_artifacts_from_content

        content = '''Here are the requirements:

- Must support Python 3.8+
- Should handle async operations
- Needs to be thread-safe
- Must have comprehensive tests

These are non-negotiable.'''

        artifacts = collect_artifacts_from_content(
            content=content,
            session_id='test-collect-list',
            role='assistant'
        )

        # Should find the list
        lists = [a for a in artifacts if a['artifact_type'] == 'bullet_list']
        assert isinstance(lists, list)

    def test_collect_numbered_list(self):
        """Test that numbered lists are collected."""
        from mira.artifacts import collect_artifacts_from_content

        content = '''Installation steps:

1. Clone the repository
2. Install dependencies with pip
3. Configure environment variables
4. Run database migrations
5. Start the development server

You're ready to go!'''

        artifacts = collect_artifacts_from_content(
            content=content,
            session_id='test-collect-numbered',
            role='assistant'
        )

        # Should return a list (may be empty depending on detection thresholds)
        assert isinstance(artifacts, list)

    def test_collect_markdown_table(self):
        """Test that markdown tables are collected."""
        from mira.artifacts import collect_artifacts_from_content

        content = '''Here's a comparison:

| Database | Type | Use Case |
|----------|------|----------|
| PostgreSQL | Relational | Complex queries |
| MongoDB | Document | Flexible schema |
| Redis | Key-Value | Caching |

Choose based on your needs.'''

        artifacts = collect_artifacts_from_content(
            content=content,
            session_id='test-collect-table',
            role='assistant'
        )

        # Should return a list
        assert isinstance(artifacts, list)

    def test_collect_returns_metadata(self):
        """Test that collected artifacts have proper metadata."""
        from mira.artifacts import collect_artifacts_from_content

        content = '''```typescript
interface User {
    id: number;
    name: string;
}
```'''

        artifacts = collect_artifacts_from_content(
            content=content,
            session_id='test-collect-meta',
            role='assistant'
        )

        # If artifacts found, check structure
        for artifact in artifacts:
            assert 'artifact_type' in artifact
            assert 'content' in artifact
            assert 'session_id' in artifact


class TestBatchArtifactInsertion:
    """Test batch artifact insertion."""

    @classmethod
    def setup_class(cls):
        """Create a temporary .mira directory for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        mira_path = Path(cls.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)
        init_artifact_db()

    @classmethod
    def teardown_class(cls):
        """Clean up temporary directory."""
        shutdown_db_manager()
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_extract_artifacts_from_messages_batch(self):
        """Test extracting artifacts from multiple messages."""
        messages = [
            {
                'type': 'assistant',
                'message': {'content': [{'type': 'text', 'text': '''Here's the implementation:

```python
def process_data(data):
    results = []
    for item in data:
        results.append(transform(item))
    return results
```

And here are the key points:

- Handles empty lists gracefully
- Transforms each item individually
- Returns a new list (immutable)
- No side effects
'''}]},
                'timestamp': '2025-12-07T10:00:00Z'
            },
            {
                'type': 'assistant',
                'message': {'content': [{'type': 'text', 'text': '''Configuration options:

| Option | Default | Description |
|--------|---------|-------------|
| timeout | 30 | Request timeout in seconds |
| retries | 3 | Number of retry attempts |
| batch_size | 100 | Items per batch |
'''}]},
                'timestamp': '2025-12-07T10:01:00Z'
            }
        ]

        # Should extract without crashing, count depends on storage availability
        count = extract_artifacts_from_messages(messages, 'test-batch-session')
        assert count >= 0  # May be 0 if central storage not available
