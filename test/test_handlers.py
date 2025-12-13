"""Tests for mira.handlers module."""

import os
import json
import tempfile
import shutil
from pathlib import Path

from mira.artifacts import init_artifact_db
from mira.db_manager import shutdown_db_manager


class TestHandlers:
    """Test RPC handler functions."""

    def test_calculate_storage_stats(self):
        from mira.handlers import calculate_storage_stats

        temp_dir = tempfile.mkdtemp()
        try:
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()
            (mira_path / 'archives').mkdir()
            (mira_path / 'metadata').mkdir()

            stats = calculate_storage_stats(mira_path)
            assert 'total_mira' in stats
            assert 'components' in stats
            assert 'archives' in stats['components']
        finally:
            shutil.rmtree(temp_dir)

    def test_get_current_work_context_empty(self):
        from mira.handlers import get_current_work_context

        shutdown_db_manager()  # Reset before changing directory
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            # Create minimal .mira structure
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()
            context = get_current_work_context()
            # When metadata_path doesn't exist, returns empty dict
            assert isinstance(context, dict)
        finally:
            shutdown_db_manager()
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_handle_recent(self):
        from mira.handlers import handle_recent

        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            metadata_path = mira_path / 'metadata'
            metadata_path.mkdir(parents=True)

            # Create a test metadata file
            meta_file = metadata_path / 'test-session.json'
            meta_file.write_text(json.dumps({
                'summary': 'Test session',
                'project_path': '-workspaces-test'
            }))

            result = handle_recent({'limit': 5})
            assert 'projects' in result
            assert 'total' in result
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_handle_status(self):
        from mira.handlers import handle_status
        from mira.sync_queue import SyncQueue

        shutdown_db_manager()
        SyncQueue._initialized = False
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()
            # Initialize artifact db for status call
            init_artifact_db()

            # params, collection (deprecated), storage
            result = handle_status({}, None, storage=None)
            # Global stats are now nested under 'global' key
            assert 'global' in result
            assert 'files' in result['global']
            assert 'total' in result['global']['files']
            assert 'ingestion' in result['global']
            assert 'archived' in result['global']
            assert 'storage_path' in result
            assert 'last_sync' in result
        finally:
            shutdown_db_manager()
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_get_custodian_profile(self):
        from mira.handlers import get_custodian_profile

        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            profile = get_custodian_profile()
            assert 'name' in profile
            assert 'tech_stack' in profile
            assert 'total_sessions' in profile
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_handle_rpc_request_unknown_method(self):
        from mira.handlers import handle_rpc_request

        class MockCollection:
            def count(self):
                return 0

        request = {'method': 'unknown_method', 'id': 1}
        response = handle_rpc_request(request, MockCollection())
        assert 'error' in response
        assert response['error']['code'] == -32601  # Method not found


class TestRPCHandlersComplete:
    """Test all RPC handlers including error_lookup and decisions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        os.environ["MIRA_PATH"] = self.temp_dir

        # Initialize databases
        from mira.insights import init_insights_db
        init_insights_db()

    def teardown_method(self):
        """Cleanup test fixtures."""
        os.environ.pop("MIRA_PATH", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutdown_db_manager()

    def test_handle_error_lookup_basic(self):
        """Test error_lookup handler with basic query."""
        from mira.handlers import handle_error_lookup

        result = handle_error_lookup(
            params={"query": "TypeError", "limit": 5},
            storage=None
        )

        # Handler returns 'solutions' list (matches TypeScript expectation) and 'total' count
        assert "solutions" in result
        assert "total" in result
        assert isinstance(result["solutions"], list)

    def test_handle_error_lookup_with_project(self):
        """Test error_lookup handler with project filter."""
        from mira.handlers import handle_error_lookup

        result = handle_error_lookup(
            params={
                "query": "connection error",
                "project_path": "/test/project",
                "limit": 10
            },
            storage=None
        )

        assert "solutions" in result
        assert "total" in result

    def test_handle_decisions_basic(self):
        """Test decisions handler with basic query."""
        from mira.handlers import handle_decisions

        result = handle_decisions(
            params={"query": "architecture", "limit": 5},
            storage=None
        )

        # Handler returns 'decisions' list (matches TypeScript expectation) and 'total' count
        assert "decisions" in result
        assert "total" in result
        assert isinstance(result["decisions"], list)

    def test_handle_decisions_with_category(self):
        """Test decisions handler with category filter."""
        from mira.handlers import handle_decisions

        result = handle_decisions(
            params={
                "query": "database",
                "category": "architecture",
                "limit": 10
            },
            storage=None
        )

        assert "decisions" in result
        assert "total" in result

    def test_handle_rpc_request_error_lookup(self):
        """Test RPC dispatch for error_lookup method."""
        from mira.handlers import handle_rpc_request

        result = handle_rpc_request(
            request={
                "method": "error_lookup",
                "params": {"query": "test error"}
            },
            collection=None,
            storage=None
        )

        # RPC wraps result in jsonrpc format
        assert "result" in result
        assert "solutions" in result["result"]

    def test_handle_rpc_request_decisions(self):
        """Test RPC dispatch for decisions method."""
        from mira.handlers import handle_rpc_request

        result = handle_rpc_request(
            request={
                "method": "decisions",
                "params": {"query": "test decision"}
            },
            collection=None,
            storage=None
        )

        # RPC wraps result in jsonrpc format
        assert "result" in result
        assert "decisions" in result["result"]
