"""Integration tests for mira.ingestion module."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.core import shutdown_db_manager


class TestIngestion:
    """Test ingestion functionality."""

    def test_discover_conversations_empty(self):
        """Test discover_conversations with non-existent path."""
        from mira.ingestion import discover_conversations
        result = discover_conversations(Path('/nonexistent/path'))
        assert result == []

    def test_ingestion_imports(self):
        """Test that ingestion module imports correctly."""
        from mira.ingestion import (
            ingest_conversation,
            run_full_ingestion,
            discover_conversations,
        )
        assert callable(ingest_conversation)
        assert callable(run_full_ingestion)
        assert callable(discover_conversations)


class TestActiveIngestionTracking:
    """Test active ingestion tracking functionality."""

    def setup_method(self):
        shutdown_db_manager()
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        mira_path = Path(self.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)

    def teardown_method(self):
        shutdown_db_manager()
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_active_ingestions_import(self):
        """Test that active ingestion functions are importable."""
        from mira.ingestion import get_active_ingestions
        result = get_active_ingestions()
        assert isinstance(result, list)
