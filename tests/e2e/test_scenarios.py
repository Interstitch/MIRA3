"""End-to-end tests for user scenarios."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.extraction import init_artifact_db, init_insights_db, init_concepts_db
from mira.custodian import init_custodian_db
from mira.core import shutdown_db_manager


class TestUserScenarios:
    """Test complete user workflows."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        cls.mira_path = Path(cls.temp_dir) / '.mira'
        cls.mira_path.mkdir()
        (cls.mira_path / 'archives').mkdir()
        (cls.mira_path / 'metadata').mkdir()

        # Initialize databases
        init_artifact_db()
        init_custodian_db()
        init_insights_db()
        init_concepts_db()

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_mira_init_scenario(self):
        """Test mira_init returns expected context."""
        from mira.tools import handle_init

        result = handle_init({'project_path': '/test/project'})
        assert isinstance(result, dict)

    def test_search_scenario(self):
        """Test search returns expected structure."""
        from mira.tools import handle_search

        result = handle_search({'query': 'authentication', 'limit': 5})
        assert 'results' in result
        assert 'total' in result

    def test_status_scenario(self):
        """Test status returns system info."""
        from mira.tools import handle_status

        result = handle_status({})
        assert 'storage_path' in result
        assert 'storage_mode' in result
