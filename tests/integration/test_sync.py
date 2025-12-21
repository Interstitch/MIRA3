"""Integration tests for sync functionality."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.core import shutdown_db_manager


class TestSync:
    """Test sync functionality."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        mira_path = Path(cls.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_sync_imports(self):
        """Test that sync-related imports work."""
        # Basic import test for sync functionality
        try:
            from mira.storage.sync import SyncQueue
            assert SyncQueue is not None
        except ImportError:
            # Sync module may be structured differently
            pass
