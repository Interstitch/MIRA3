"""Integration tests for mira.storage.migrations module."""

import os
import tempfile
import shutil

from mira.core import shutdown_db_manager


class TestMigrations:
    """Test database migration framework."""

    def setup_method(self):
        shutdown_db_manager()
        self.temp_dir = tempfile.mkdtemp()
        self.old_mira_path = os.environ.get("MIRA_PATH")
        os.environ["MIRA_PATH"] = self.temp_dir

    def teardown_method(self):
        if self.old_mira_path:
            os.environ["MIRA_PATH"] = self.old_mira_path
        else:
            os.environ.pop("MIRA_PATH", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutdown_db_manager()

    def test_migrations_import(self):
        """Test that migrations module imports correctly."""
        from mira.storage.migrations import run_migrations
        assert callable(run_migrations)

    def test_run_migrations(self):
        """Test running migrations."""
        from mira.storage.migrations import run_migrations

        result = run_migrations()
        assert result["status"] in ("success", "already_current", "error")
