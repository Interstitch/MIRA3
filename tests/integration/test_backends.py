"""Integration tests for storage backends."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.core import shutdown_db_manager


class TestStorageBackends:
    """Test storage backend functionality."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        cls.temp_dir = tempfile.mkdtemp()
        cls.mira_path = Path(cls.temp_dir) / '.mira'
        cls.mira_path.mkdir(parents=True)

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_storage_imports(self):
        """Test that storage backends import correctly."""
        from mira.storage import Storage
        assert Storage is not None

    def test_local_store_imports(self):
        """Test that local_store imports correctly."""
        from mira.storage.local_store import init_local_db
        assert callable(init_local_db)
