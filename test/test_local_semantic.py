"""Tests for mira.local_semantic module."""

import os
import sys
import tempfile
import shutil
import struct
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest


class TestLocalSemanticSearch:
    """Test LocalSemanticSearch class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_mira_path = os.environ.get('MIRA_PATH')
        os.environ['MIRA_PATH'] = self.temp_dir

        # Reset singleton for each test
        import mira.local_semantic as ls_module
        ls_module._local_semantic = None
        ls_module.LocalSemanticSearch._instance = None
        ls_module.LocalSemanticSearch._initialized = False
        ls_module.LocalSemanticSearch._model = None
        ls_module.LocalSemanticSearch._model_loading = False
        ls_module.LocalSemanticSearch._sqlite_vec_available = None

    def teardown_method(self):
        """Cleanup test fixtures."""
        if self.original_mira_path:
            os.environ['MIRA_PATH'] = self.original_mira_path
        else:
            os.environ.pop('MIRA_PATH', None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Reset DB manager
        from mira.db_manager import shutdown_db_manager
        shutdown_db_manager()

    def test_singleton_pattern(self):
        """Test that LocalSemanticSearch is a singleton."""
        from mira.local_semantic import LocalSemanticSearch

        instance1 = LocalSemanticSearch()
        instance2 = LocalSemanticSearch()

        assert instance1 is instance2

    def test_get_local_semantic_returns_singleton(self):
        """Test get_local_semantic returns singleton instance."""
        from mira.local_semantic import get_local_semantic, LocalSemanticSearch

        ls1 = get_local_semantic()
        ls2 = get_local_semantic()

        assert ls1 is ls2
        assert isinstance(ls1, LocalSemanticSearch)

    def test_model_not_ready_initially(self):
        """Test that model is not ready before loading."""
        from mira.local_semantic import get_local_semantic

        ls = get_local_semantic()
        assert ls.is_model_ready() is False

    def test_sqlite_vec_availability_check(self):
        """Test sqlite-vec availability detection."""
        from mira.local_semantic import get_local_semantic

        ls = get_local_semantic()
        # Should return a boolean (True or False depending on system)
        result = ls.sqlite_vec_available
        assert isinstance(result, bool)

    def test_get_status_structure(self):
        """Test get_status returns expected structure."""
        from mira.local_semantic import get_local_semantic

        ls = get_local_semantic()
        status = ls.get_status()

        assert 'available' in status
        assert 'sqlite_vec' in status
        assert 'model_ready' in status
        assert 'download_in_progress' in status
        assert 'indexed_sessions' in status
        assert 'total_vectors' in status

    def test_is_local_semantic_available_false_initially(self):
        """Test is_local_semantic_available returns False before model download."""
        from mira.local_semantic import is_local_semantic_available

        # Without model or sqlite-vec, should be False
        result = is_local_semantic_available()
        assert result is False

    @patch('mira.local_semantic.LocalSemanticSearch.sqlite_vec_available', True)
    def test_trigger_download_when_not_ready(self):
        """Test trigger_local_semantic_download starts download."""
        from mira.local_semantic import get_local_semantic, trigger_local_semantic_download

        ls = get_local_semantic()

        # Mock the background download to avoid actual network call
        with patch.object(ls, 'start_background_download') as mock_download:
            result = trigger_local_semantic_download()

            assert 'notice' in result
            assert 'downloading' in result['notice'].lower() or 'enabling' in result['notice'].lower()
            mock_download.assert_called_once()

    @patch('mira.local_semantic.LocalSemanticSearch.sqlite_vec_available', False)
    def test_trigger_download_when_sqlite_vec_unavailable(self):
        """Test trigger returns appropriate message when sqlite-vec unavailable."""
        from mira.local_semantic import trigger_local_semantic_download

        result = trigger_local_semantic_download()

        assert 'notice' in result
        assert 'unavailable' in result['notice'].lower()

    def test_download_not_triggered_if_model_ready(self):
        """Test that download is not triggered if model already ready."""
        from mira.local_semantic import get_local_semantic, trigger_local_semantic_download

        ls = get_local_semantic()

        # Pretend both sqlite_vec and model are ready using PropertyMock
        with patch.object(type(ls), 'sqlite_vec_available', new_callable=lambda: property(lambda self: True)):
            with patch.object(ls, 'is_model_ready', return_value=True):
                with patch.object(ls, 'start_background_download') as mock_download:
                    result = trigger_local_semantic_download()

                    # Should return empty dict (no notice needed)
                    assert result == {}
                    mock_download.assert_not_called()

    def test_is_download_in_progress(self):
        """Test is_download_in_progress tracking."""
        from mira.local_semantic import get_local_semantic

        ls = get_local_semantic()

        # Initially not downloading
        assert ls.is_download_in_progress() is False

    def test_chunk_content_small(self):
        """Test chunking small content returns single chunk."""
        from mira.local_semantic import get_local_semantic

        ls = get_local_semantic()

        content = "This is a small piece of content."
        chunks = ls._chunk_content(content)

        assert len(chunks) == 1
        assert chunks[0] == content

    def test_chunk_content_large(self):
        """Test chunking large content returns multiple chunks."""
        from mira.local_semantic import get_local_semantic

        ls = get_local_semantic()

        # Create content larger than CHUNK_SIZE (4000)
        content = "x" * 10000
        chunks = ls._chunk_content(content)

        assert len(chunks) > 1
        # Each chunk should be <= CHUNK_SIZE
        for chunk in chunks:
            assert len(chunk) <= 4000


class TestSearchIntegration:
    """Test search integration with local semantic."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_mira_path = os.environ.get('MIRA_PATH')
        os.environ['MIRA_PATH'] = self.temp_dir

        # Reset modules
        import mira.local_semantic as ls_module
        ls_module._local_semantic = None
        ls_module.LocalSemanticSearch._instance = None
        ls_module.LocalSemanticSearch._initialized = False
        ls_module.LocalSemanticSearch._model = None

    def teardown_method(self):
        """Cleanup test fixtures."""
        if self.original_mira_path:
            os.environ['MIRA_PATH'] = self.original_mira_path
        else:
            os.environ.pop('MIRA_PATH', None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        from mira.db_manager import shutdown_db_manager
        shutdown_db_manager()

    def test_search_triggers_local_semantic_when_remote_unavailable(self):
        """Test that search triggers local semantic download when remote unavailable."""
        from mira.search import handle_search
        import mira.local_semantic as ls_module

        # Create a mock storage that says central is not available
        mock_storage = Mock()
        mock_storage.using_central = False
        mock_storage.search_sessions_fts = Mock(return_value=[])

        # Patch at module level before import in function
        with patch.object(ls_module, 'is_local_semantic_available', return_value=False):
            with patch.object(ls_module, 'trigger_local_semantic_download') as mock_trigger:
                mock_trigger.return_value = {"notice": "Enabling local semantic..."}

                result = handle_search(
                    params={"query": "test", "limit": 5},
                    collection=None,
                    storage=mock_storage
                )

                # Should have triggered download
                mock_trigger.assert_called_once()

                # Should include notice in response
                assert "notice" in result

    def test_search_uses_local_semantic_when_available(self):
        """Test that search uses local semantic when it's available."""
        from mira.search import handle_search
        import mira.local_semantic as ls_module

        mock_storage = Mock()
        mock_storage.using_central = False

        # Mock local semantic as available
        mock_ls = Mock()
        mock_ls.search.return_value = [
            {"session_id": "test-123", "score": 0.9, "chunk_preview": "test content"}
        ]

        with patch.object(ls_module, 'is_local_semantic_available', return_value=True):
            with patch.object(ls_module, 'get_local_semantic', return_value=mock_ls):
                with patch('mira.search.enrich_results_from_archives', return_value=[
                    {"session_id": "test-123", "summary": "test", "relevance": 0.9}
                ]):
                    result = handle_search(
                        params={"query": "test", "limit": 5, "compact": False},
                        collection=None,
                        storage=mock_storage
                    )

                    # Should have called local semantic search
                    mock_ls.search.assert_called_once()

                    # Should return local_semantic as search type
                    assert result.get("search_type") == "local_semantic"

    def test_search_falls_through_to_fts5_when_local_semantic_not_ready(self):
        """Test search falls back to FTS5 when local semantic not ready."""
        from mira.search import handle_search
        import mira.local_semantic as ls_module

        mock_storage = Mock()
        mock_storage.using_central = False
        mock_storage.search_sessions_fts = Mock(return_value=[
            {"session_id": "fts-123", "summary": "fts result", "rank": 0.5}
        ])

        with patch.object(ls_module, 'is_local_semantic_available', return_value=False):
            with patch.object(ls_module, 'trigger_local_semantic_download', return_value={"notice": "Downloading..."}):
                with patch('mira.search.enrich_results_from_archives', side_effect=lambda r, *args, **kwargs: r):
                    result = handle_search(
                        params={"query": "test", "limit": 5, "compact": False},
                        collection=None,
                        storage=mock_storage
                    )

                    # Should fall through to FTS5
                    mock_storage.search_sessions_fts.assert_called()

                    # Should include download notice
                    assert "notice" in result

    def test_no_download_triggered_when_remote_available(self):
        """Test that no local semantic download happens when remote works."""
        from mira.search import handle_search
        import mira.local_semantic as ls_module
        import mira.embedding_client as embed_module

        mock_storage = Mock()
        mock_storage.using_central = True  # Remote IS available
        mock_storage.search_sessions_fts = Mock(return_value=[])
        mock_storage.search_archives_fts = Mock(return_value=[])
        mock_storage.postgres = Mock()
        mock_storage.postgres.get_project_id = Mock(return_value=None)

        with patch.object(embed_module, 'get_embedding_client', return_value=None):
            with patch.object(ls_module, 'trigger_local_semantic_download') as mock_trigger:
                result = handle_search(
                    params={"query": "test", "limit": 5},
                    collection=None,
                    storage=mock_storage
                )

                # Should NOT trigger local semantic download
                mock_trigger.assert_not_called()


class TestStatusIntegration:
    """Test mira_status integration with local semantic."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_mira_path = os.environ.get('MIRA_PATH')
        os.environ['MIRA_PATH'] = self.temp_dir

    def teardown_method(self):
        """Cleanup test fixtures."""
        if self.original_mira_path:
            os.environ['MIRA_PATH'] = self.original_mira_path
        else:
            os.environ.pop('MIRA_PATH', None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        from mira.db_manager import shutdown_db_manager
        shutdown_db_manager()

    def test_status_includes_local_semantic(self):
        """Test that handle_status includes local_semantic section."""
        from mira.handlers import handle_status
        import mira.sync_queue as sq_module
        import mira.ingestion as ing_module
        import mira.insights as ins_module
        import mira.artifacts as art_module

        mock_storage = Mock()
        mock_storage.using_central = False
        mock_storage.health_check = Mock(return_value={})

        # Create mock queue
        mock_queue = Mock()
        mock_queue.get_stats.return_value = {}

        with patch.object(sq_module, 'get_sync_queue', return_value=mock_queue):
            with patch.object(ing_module, 'get_active_ingestions', return_value=[]):
                with patch.object(ins_module, 'get_error_stats', return_value={}):
                    with patch.object(ins_module, 'get_decision_stats', return_value={}):
                        with patch.object(art_module, 'get_artifact_stats', return_value={}):
                            result = handle_status({}, None, mock_storage)

                            # Should include local_semantic section
                            assert 'local_semantic' in result
                            assert 'available' in result['local_semantic']
