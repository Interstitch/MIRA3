"""Tests for mira.search module."""

import os
import json
import tempfile
import shutil
from pathlib import Path

from mira.db_manager import shutdown_db_manager


class TestSearch:
    """Test search functionality."""

    def test_extract_excerpt_around_terms(self):
        from mira.search import extract_excerpt_around_terms
        content = "This is a test about authentication systems and how they work with user login."
        excerpt = extract_excerpt_around_terms(content, ['authentication'], context_chars=20)
        assert 'authentication' in excerpt

    def test_search_archive_for_excerpts(self):
        from mira.search import search_archive_for_excerpts

        # Create temp archive file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        try:
            lines = [
                json.dumps({
                    'type': 'user',
                    'timestamp': '2025-12-07T10:00:00Z',
                    'message': {'content': 'Help me with authentication'}
                }),
                json.dumps({
                    'type': 'assistant',
                    'timestamp': '2025-12-07T10:00:30Z',
                    'message': {'content': [{'type': 'text', 'text': 'I can help with authentication and login systems.'}]}
                })
            ]
            for line in lines:
                temp_file.write(line + '\n')
            temp_file.close()

            excerpts = search_archive_for_excerpts(Path(temp_file.name), ['authentication'], max_excerpts=3)
            assert len(excerpts) >= 1
            assert any('authentication' in e['excerpt'].lower() for e in excerpts)
        finally:
            os.unlink(temp_file.name)

    def test_fulltext_search_archives(self):
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

            results = fulltext_search_archives('database', 5, mira_path)
            assert isinstance(results, list)
        finally:
            shutil.rmtree(temp_dir)


class TestSearchHandlers:
    """Test search handler functions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mira_path = Path(self.temp_dir)
        self.archives_path = self.mira_path / "archives"
        self.archives_path.mkdir(parents=True)

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_handle_search_basic(self):
        """Test handle_search with basic query."""
        from mira.search import handle_search

        # Test with compact=False to get verbose output
        result = handle_search(
            params={"query": "test query", "limit": 5, "compact": False},
            collection=None,
            storage=None
        )

        assert "results" in result
        assert "total" in result
        assert "search_type" in result

    def test_handle_search_with_project_path(self):
        """Test handle_search with project_path filter."""
        from mira.search import handle_search

        result = handle_search(
            params={
                "query": "test",
                "limit": 5,
                "project_path": "/some/project"
            },
            collection=None,
            storage=None
        )

        assert "results" in result
        assert "total" in result

    def test_extract_excerpts_function(self):
        """Test _extract_excerpts helper."""
        from mira.search import _extract_excerpts

        content = "This is a test message about authentication. The auth module handles login and logout."
        excerpts = _extract_excerpts(content, "auth", max_excerpts=2)

        assert isinstance(excerpts, list)
        # Should find excerpts with "auth" in them
        if len(excerpts) > 0:
            assert any("auth" in e.lower() for e in excerpts)
