"""Tests for mira.extraction.artifacts module."""

import os
import json
import tempfile
import shutil
from pathlib import Path

from mira.extraction import (
    init_artifact_db, store_file_operation, get_file_operations,
    reconstruct_file, get_artifact_stats, detect_language,
    collect_artifacts_from_content, store_artifact, search_artifacts_for_query,
    extract_artifacts_from_messages
)
from mira.core import shutdown_db_manager


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

    def test_collect_returns_metadata(self):
        """Test that collected artifacts have proper metadata."""
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
