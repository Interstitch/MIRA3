"""Tests for mira.watcher module."""

import time
from pathlib import Path


class TestWatcher:
    """Test file watcher functionality."""

    def test_conversation_watcher_queue(self):
        from mira.watcher import ConversationWatcher

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
