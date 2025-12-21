"""Tests for mira.tools handlers."""

import os
import json
import tempfile
import shutil
from pathlib import Path
import pytest

from mira.extraction import init_artifact_db, init_insights_db
from mira.core import shutdown_db_manager


class TestToolHandlers:
    """Test MCP tool handler functions."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        mira_path = Path(cls.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)
        (mira_path / 'archives').mkdir(exist_ok=True)
        (mira_path / 'metadata').mkdir(exist_ok=True)
        init_artifact_db()
        init_insights_db()

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_tools_import(self):
        """Test that tools package imports correctly."""
        from mira.tools import (
            handle_init,
            handle_search,
            handle_recent,
            handle_error_lookup,
            handle_decisions,
            handle_status,
        )
        assert callable(handle_init)
        assert callable(handle_search)
        assert callable(handle_recent)
        assert callable(handle_error_lookup)
        assert callable(handle_decisions)
        assert callable(handle_status)

    def test_handle_status_basic(self):
        """Test handle_status returns expected structure."""
        from mira.tools import handle_status

        result = handle_status({})
        assert isinstance(result, dict)
        # Status should include storage_path
        assert 'storage_path' in result

    def test_handle_recent_basic(self):
        """Test handle_recent returns expected structure."""
        from mira.tools import handle_recent

        result = handle_recent({'limit': 5})
        assert isinstance(result, dict)
        assert 'projects' in result

    def test_handle_search_basic(self):
        """Test handle_search returns expected structure."""
        from mira.tools import handle_search

        result = handle_search({'query': 'test', 'limit': 5})
        assert isinstance(result, dict)
        assert 'results' in result
        assert 'total' in result

    def test_handle_error_lookup_basic(self):
        """Test handle_error_lookup returns expected structure."""
        from mira.tools import handle_error_lookup

        result = handle_error_lookup({'query': 'TypeError', 'limit': 5})
        assert isinstance(result, dict)
        assert 'solutions' in result
        assert 'total' in result

    def test_handle_decisions_basic(self):
        """Test handle_decisions returns expected structure."""
        from mira.tools import handle_decisions

        result = handle_decisions({'query': 'architecture', 'limit': 5})
        assert isinstance(result, dict)
        assert 'decisions' in result
        assert 'total' in result
