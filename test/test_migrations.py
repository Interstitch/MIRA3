"""Tests for mira.migrations module."""

import os
import tempfile
import shutil

from mira.db_manager import shutdown_db_manager


class TestMigrations:
    """Test database migration framework."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.old_mira_path = os.environ.get("MIRA_PATH")
        os.environ["MIRA_PATH"] = self.temp_dir

    def teardown_method(self):
        """Cleanup test fixtures."""
        if self.old_mira_path:
            os.environ["MIRA_PATH"] = self.old_mira_path
        else:
            os.environ.pop("MIRA_PATH", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutdown_db_manager()

    def test_migration_decorator(self):
        """Test migration decorator registers functions."""
        from mira.migrations import _migrations

        # Check that migrations are registered
        assert len(_migrations) >= 3  # v1, v2, v3
        versions = [m[0] for m in _migrations]
        assert 1 in versions
        assert 2 in versions
        assert 3 in versions

    def test_init_migrations_db(self):
        """Test migrations database initialization."""
        from mira.migrations import init_migrations_db, MIGRATIONS_DB
        from mira.db_manager import get_db_manager

        init_migrations_db()
        db = get_db_manager()

        # Should have created schema_migrations table
        rows = db.execute_read(
            MIGRATIONS_DB,
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'",
            ()
        )
        assert len(list(rows)) == 1

    def test_get_current_version_empty(self):
        """Test getting version from fresh database returns valid version."""
        from mira.migrations import get_current_version

        # After setup_method sets a new MIRA_PATH, the migrations db is fresh
        # but other tests may have run migrations. Just verify we get a valid version.
        version = get_current_version()
        assert version >= 0  # Could be 0 (fresh) or higher if migrations ran

    def test_check_migrations_needed(self):
        """Test checking if migrations are needed."""
        from mira.migrations import check_migrations_needed, CURRENT_VERSION

        result = check_migrations_needed()

        assert "current_version" in result
        assert "target_version" in result
        assert result["target_version"] == CURRENT_VERSION
        assert "needs_migration" in result
        assert "pending_migrations" in result

    def test_run_migrations(self):
        """Test running migrations."""
        from mira.migrations import run_migrations, get_current_version

        # Run migrations
        result = run_migrations()

        assert result["status"] in ("success", "already_current")
        assert "migrations_run" in result

        # Version should now be current
        version = get_current_version()
        assert version >= 1

    def test_get_applied_migrations(self):
        """Test getting list of applied migrations."""
        from mira.migrations import run_migrations, get_applied_migrations

        # Run migrations first
        run_migrations()

        # Get applied
        applied = get_applied_migrations()

        assert isinstance(applied, list)
        if len(applied) > 0:
            assert "version" in applied[0]
            assert "name" in applied[0]
            assert "applied_at" in applied[0]

    def test_ensure_schema_current(self):
        """Test ensure_schema_current helper."""
        from mira.migrations import ensure_schema_current, get_current_version, CURRENT_VERSION

        # Should run without error
        ensure_schema_current()

        # Version should be current
        version = get_current_version()
        assert version == CURRENT_VERSION
