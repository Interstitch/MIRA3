"""Tests for mira.bootstrap and mira.main singleton functionality."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.bootstrap import is_running_in_venv


class TestBootstrap:
    """Test bootstrap functionality."""

    def test_is_running_in_venv(self):
        # This should return False since we're not running in .mira/.venv
        result = is_running_in_venv()
        assert isinstance(result, bool)


class TestSingletonLock:
    """Test singleton lock mechanism for preventing duplicate MIRA instances."""

    @classmethod
    def setup_class(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.mira_path = Path(cls.temp_dir) / '.mira'
        cls.mira_path.mkdir(exist_ok=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_acquire_singleton_lock_first_time(self):
        """Test that first lock acquisition succeeds."""
        from mira.main import acquire_singleton_lock

        # Use a unique subdir for this test
        test_path = self.mira_path / "test1"
        test_path.mkdir(exist_ok=True)

        result = acquire_singleton_lock(test_path)
        assert result is True

        # PID file should exist
        pid_file = test_path / "mira.pid"
        assert pid_file.exists()
        assert int(pid_file.read_text().strip()) == os.getpid()

        # Lock file should exist
        lock_file = test_path / "mira.lock"
        assert lock_file.exists()

    def test_pid_file_contains_current_pid(self):
        """Test that PID file contains the current process ID."""
        from mira.main import acquire_singleton_lock

        test_path = self.mira_path / "test2"
        test_path.mkdir(exist_ok=True)

        acquire_singleton_lock(test_path)

        pid_file = test_path / "mira.pid"
        stored_pid = int(pid_file.read_text().strip())
        assert stored_pid == os.getpid()

    def test_lock_file_created(self):
        """Test that lock file is created on acquisition."""
        from mira.main import acquire_singleton_lock

        test_path = self.mira_path / "test3"
        test_path.mkdir(exist_ok=True)

        # Lock file shouldn't exist yet
        lock_file = test_path / "mira.lock"
        assert not lock_file.exists()

        acquire_singleton_lock(test_path)

        # Now it should exist
        assert lock_file.exists()
