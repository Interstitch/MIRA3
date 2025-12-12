"""Tests for mira_status functionality and project ID operations."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.artifacts import init_artifact_db
from mira.db_manager import shutdown_db_manager


class TestMiraStatusArtifacts:
    """Test that mira_status includes artifact stats."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        from mira.sync_queue import SyncQueue
        SyncQueue._initialized = False
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        mira_path = Path(cls.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)
        # Initialize required databases
        init_artifact_db()

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_status_includes_artifacts(self):
        """Test that handle_status includes artifact count."""
        from mira.handlers import handle_status

        result = handle_status({}, None, storage=None)

        # Should include artifact stats under global
        assert 'global' in result
        assert 'artifacts' in result['global']
        artifacts = result['global']['artifacts']
        assert 'total' in artifacts

    def test_status_includes_sync_queue(self):
        """Test that handle_status includes sync queue stats."""
        from mira.handlers import handle_status

        result = handle_status({}, None, storage=None)

        # Should include sync_queue stats
        assert 'sync_queue' in result

    def test_status_includes_active_ingestions(self):
        """Test that handle_status includes active ingestions list."""
        from mira.handlers import handle_status

        result = handle_status({}, None, storage=None)

        # Should include active_ingestions list (empty when nothing ingesting)
        assert 'active_ingestions' in result
        assert isinstance(result['active_ingestions'], list)

    def test_status_includes_file_operations(self):
        """Test that handle_status includes file operations stats."""
        from mira.handlers import handle_status

        result = handle_status({}, None, storage=None)

        # Should include file_operations stats under global
        assert 'global' in result
        assert 'file_operations' in result['global']
        file_ops = result['global']['file_operations']
        assert 'total_operations' in file_ops
        assert 'unique_files' in file_ops

    def test_status_with_project_path(self):
        """Test that handle_status returns project-scoped stats when project_path is provided."""
        from mira.handlers import handle_status

        # Call with project_path - should include 'project' section in response
        result = handle_status({'project_path': '/workspaces/MIRA3'}, None, storage=None)

        # Should have global stats
        assert 'global' in result
        assert 'files' in result['global']
        assert 'total' in result['global']['files']
        assert 'ingestion' in result['global']

        # Should have project-specific stats when project_path is provided
        assert 'project' in result
        project_stats = result['project']
        assert 'path' in project_stats
        assert project_stats['path'] == '/workspaces/MIRA3'
        assert 'files' in project_stats
        assert 'ingestion' in project_stats

        # Should have storage mode info
        assert 'storage_mode' in result


class TestGetProjectId:
    """Test get_project_id functionality."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        from mira import local_store
        local_store._initialized = False
        cls.temp_dir = tempfile.mkdtemp()
        os.environ['MIRA_PATH'] = str(Path(cls.temp_dir) / '.mira')
        (Path(cls.temp_dir) / '.mira').mkdir(exist_ok=True)
        local_store.init_local_db()

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        if 'MIRA_PATH' in os.environ:
            del os.environ['MIRA_PATH']
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_get_project_id_exists(self):
        """Test getting project ID for existing project."""
        from mira.local_store import get_or_create_project, get_project_id

        # Create project
        project_id = get_or_create_project('/test/project/path')
        assert project_id > 0

        # Get project ID (should match)
        retrieved_id = get_project_id('/test/project/path')
        assert retrieved_id == project_id

    def test_get_project_id_not_exists(self):
        """Test getting project ID for non-existent project."""
        from mira.local_store import get_project_id

        # Should return None for non-existent project
        result = get_project_id('/nonexistent/path/12345')
        assert result is None
