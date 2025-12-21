"""Tests for mira_status functionality."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.extraction import init_artifact_db
from mira.core import shutdown_db_manager


class TestMiraStatus:
    """Test mira_status tool functionality."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        mira_path = Path(cls.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)
        init_artifact_db()

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_status_returns_dict(self):
        """Test that handle_status returns a dict."""
        from mira.tools import handle_status

        result = handle_status({})
        assert isinstance(result, dict)

    def test_status_includes_storage_path(self):
        """Test that status includes storage_path."""
        from mira.tools import handle_status

        result = handle_status({})
        assert 'storage_path' in result

    def test_status_includes_storage_mode(self):
        """Test that status includes storage_mode."""
        from mira.tools import handle_status

        result = handle_status({})
        assert 'storage_mode' in result
