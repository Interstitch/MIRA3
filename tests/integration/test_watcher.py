"""Integration tests for file watcher."""

import tempfile
from pathlib import Path

from mira.ingestion.watcher import ConversationWatcher


class TestWatcher:
    """Test file watcher functionality."""

    def test_watcher_import(self):
        """Test that watcher imports correctly."""
        assert ConversationWatcher is not None

    def test_watcher_instantiation(self):
        """Test that watcher can be instantiated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()
            watcher = ConversationWatcher(
                collection=None,  # Deprecated, ignored
                mira_path=mira_path,
                storage=None
            )
            assert watcher is not None
