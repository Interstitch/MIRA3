"""Integration tests for mira.storage module."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.core import shutdown_db_manager


class TestLocalStore:
    """Test local SQLite storage module."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        cls.mira_path = Path(tempfile.mkdtemp()) / '.mira'
        cls.mira_path.mkdir(parents=True)
        os.environ['MIRA_PATH'] = str(cls.mira_path)

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        if cls.mira_path.exists():
            shutil.rmtree(cls.mira_path.parent, ignore_errors=True)
        if 'MIRA_PATH' in os.environ:
            del os.environ['MIRA_PATH']

    def test_storage_imports(self):
        """Test that storage module imports correctly."""
        from mira.storage import Storage
        assert Storage is not None

    def test_local_store_imports(self):
        """Test that local_store imports correctly."""
        from mira.storage.local_store import init_local_db
        assert callable(init_local_db)
