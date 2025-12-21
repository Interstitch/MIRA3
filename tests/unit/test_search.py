"""Tests for mira.search module."""

import os
import json
import tempfile
import shutil
from pathlib import Path

from mira.core import shutdown_db_manager


class TestSearch:
    """Test search functionality."""

    def test_search_imports(self):
        """Test that search module imports correctly."""
        from mira.search import handle_search, fulltext_search_archives
        assert callable(handle_search)
        assert callable(fulltext_search_archives)

    def test_fulltext_search_archives(self):
        """Test fulltext search in archives."""
        from mira.search import fulltext_search_archives

        # Create temp .mira structure
        temp_dir = tempfile.mkdtemp()
        try:
            mira_path = Path(temp_dir) / '.mira'
            archives_path = mira_path / 'archives'
            metadata_path = mira_path / 'metadata'
            archives_path.mkdir(parents=True)
            metadata_path.mkdir(parents=True)

            # Create archive file
            archive_file = archives_path / 'test-session.jsonl'
            with open(archive_file, 'w') as f:
                f.write(json.dumps({
                    'type': 'user',
                    'message': {'content': 'Help with database queries'}
                }) + '\n')

            # Create metadata file
            meta_file = metadata_path / 'test-session.json'
            meta_file.write_text(json.dumps({'summary': 'Database help session'}))

            # Note: fulltext_search_archives takes storage=None for local search
            results = fulltext_search_archives('database', 5, storage=None)
            assert isinstance(results, list)
        finally:
            shutil.rmtree(temp_dir)


class TestSearchHandlers:
    """Test search handler functions."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        mira_path = Path(cls.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)
        (mira_path / 'archives').mkdir()
        (mira_path / 'metadata').mkdir()

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_handle_search_basic(self):
        """Test handle_search with basic query."""
        from mira.search import handle_search

        result = handle_search(
            params={"query": "test query", "limit": 5}
        )

        assert "results" in result
        assert "total" in result

    def test_handle_search_with_project_path(self):
        """Test handle_search with project_path filter."""
        from mira.search import handle_search

        result = handle_search(
            params={
                "query": "test",
                "limit": 5,
                "project_path": "/some/project"
            }
        )

        assert "results" in result
        assert "total" in result
