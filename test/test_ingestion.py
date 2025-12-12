"""Tests for mira.ingestion module."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.db_manager import shutdown_db_manager


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


class TestActiveIngestionTracking:
    """Test active ingestion tracking functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        # Create .mira directory
        mira_path = Path(self.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutdown_db_manager()
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_active_ingestions_empty(self):
        """Test that empty state returns empty list."""
        from mira.ingestion import get_active_ingestions

        result = get_active_ingestions()
        assert isinstance(result, list)

    def test_mark_and_clear_ingestion(self):
        """Test marking and clearing active ingestions."""
        from mira.ingestion import get_active_ingestions, _mark_ingestion_active, _mark_ingestion_done

        # Mark as active
        _mark_ingestion_active(
            'test-session-123',
            '/test/path/file.jsonl',
            '-test-project',
            'TestWorker'
        )

        # Check it's tracked
        active = get_active_ingestions()
        assert len(active) == 1
        assert active[0]['session_id'] == 'test-session-123'
        assert active[0]['worker'] == 'TestWorker'

        # Clear it
        _mark_ingestion_done('test-session-123')

        # Should be gone
        active = get_active_ingestions()
        assert len(active) == 0

    def test_watcher_tracks_active_ingestions(self):
        """Test that ConversationWatcher tracks active ingestions."""
        from mira.watcher import ConversationWatcher
        from mira.ingestion import get_active_ingestions, _mark_ingestion_active, _mark_ingestion_done

        class MockCollection:
            def count(self):
                return 0

        watcher = ConversationWatcher(MockCollection(), Path('/tmp'))

        # Simulate starting an ingestion (would normally be in _process_file)
        _mark_ingestion_active(
            'watcher-test',
            '/test/file.jsonl',
            '-test',
            'Watcher'
        )

        active = get_active_ingestions()
        assert any(i['session_id'] == 'watcher-test' for i in active)

        _mark_ingestion_done('watcher-test')
