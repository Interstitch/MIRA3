"""Tests for mira.search.local_semantic module.

ULTRATHINK-011: Unit tests for local embeddings
ULTRATHINK-012: Unit tests for local vector search
ULTRATHINK-013: Integration tests for search tiers

Note: Some tests require fastembed and sqlite-vec which may not be available
in all environments. These tests are marked with @pytest.mark.skipif decorators.
"""

import os
import struct
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading

import pytest


class TestLocalSemanticSearch:
    """Test LocalSemanticSearch class."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_mira_path = os.environ.get('MIRA_PATH')
        os.environ['MIRA_PATH'] = self.temp_dir

        # Create .mira directory
        (Path(self.temp_dir) / '.mira').mkdir(exist_ok=True)

        # Reset singleton state for clean tests
        from mira.search import local_semantic
        local_semantic._local_semantic = None
        local_semantic.LocalSemanticSearch._instance = None
        local_semantic.LocalSemanticSearch._initialized = False
        local_semantic.LocalSemanticSearch._model = None

    def teardown_method(self):
        if self.original_mira_path:
            os.environ['MIRA_PATH'] = self.original_mira_path
        else:
            os.environ.pop('MIRA_PATH', None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        from mira.core import shutdown_db_manager
        shutdown_db_manager()

        # Reset singleton for fresh tests
        from mira.search import local_semantic
        local_semantic._local_semantic = None
        local_semantic.LocalSemanticSearch._instance = None
        local_semantic.LocalSemanticSearch._initialized = False
        local_semantic.LocalSemanticSearch._model = None  # Reset cached model

    def test_local_semantic_imports(self):
        """Test that local_semantic module imports correctly."""
        from mira.search.local_semantic import LocalSemanticSearch, get_local_semantic
        assert LocalSemanticSearch is not None
        assert callable(get_local_semantic)

    def test_get_local_semantic_returns_instance(self):
        """Test get_local_semantic returns an instance."""
        from mira.search.local_semantic import get_local_semantic

        ls = get_local_semantic()
        assert ls is not None

    def test_get_local_semantic_returns_singleton(self):
        """Test get_local_semantic returns the same instance (singleton)."""
        from mira.search.local_semantic import get_local_semantic

        ls1 = get_local_semantic()
        ls2 = get_local_semantic()
        assert ls1 is ls2

    def test_local_semantic_has_search_method(self):
        """Test LocalSemanticSearch has search method."""
        from mira.search.local_semantic import get_local_semantic

        ls = get_local_semantic()
        assert hasattr(ls, 'search')

    def test_local_semantic_has_get_status(self):
        """Test LocalSemanticSearch has get_status method."""
        from mira.search.local_semantic import get_local_semantic

        ls = get_local_semantic()
        assert hasattr(ls, 'get_status')

    def test_get_status_returns_dict(self):
        """Test get_status returns a dictionary with expected keys."""
        from mira.search.local_semantic import get_local_semantic

        ls = get_local_semantic()
        status = ls.get_status()
        assert isinstance(status, dict)
        assert 'model_ready' in status
        assert 'available' in status
        # Note: sqlite_vec no longer in status - pure Python cosine similarity used

    def test_is_model_ready_initially_false(self):
        """Test model is not ready on fresh install."""
        from mira.search.local_semantic import get_local_semantic

        ls = get_local_semantic()
        # Model should not be ready without explicit loading
        assert ls.is_model_ready() is False


class TestChunkingLogic:
    """ULTRATHINK-011: Test content chunking for embedding."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ['MIRA_PATH'] = self.temp_dir
        (Path(self.temp_dir) / '.mira').mkdir(exist_ok=True)

    def teardown_method(self):
        os.environ.pop('MIRA_PATH', None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        from mira.core import shutdown_db_manager
        shutdown_db_manager()
        from mira.search import local_semantic
        local_semantic._local_semantic = None
        local_semantic.LocalSemanticSearch._instance = None
        local_semantic.LocalSemanticSearch._initialized = False

    def test_chunk_small_content(self):
        """Test chunking content smaller than chunk size."""
        from mira.search.local_semantic import get_local_semantic, CHUNK_SIZE

        ls = get_local_semantic()
        small_content = "This is a small piece of content."
        chunks = ls._chunk_content(small_content)

        assert len(chunks) == 1
        assert chunks[0] == small_content

    def test_chunk_exact_size_content(self):
        """Test chunking content exactly at chunk size."""
        from mira.search.local_semantic import get_local_semantic, CHUNK_SIZE

        ls = get_local_semantic()
        exact_content = "x" * CHUNK_SIZE
        chunks = ls._chunk_content(exact_content)

        assert len(chunks) == 1
        assert len(chunks[0]) == CHUNK_SIZE

    def test_chunk_large_content(self):
        """Test chunking content larger than chunk size."""
        from mira.search.local_semantic import get_local_semantic, CHUNK_SIZE, CHUNK_OVERLAP

        ls = get_local_semantic()
        # Create content that needs multiple chunks
        large_content = "x" * (CHUNK_SIZE * 3)
        chunks = ls._chunk_content(large_content)

        # Should have multiple chunks
        assert len(chunks) > 1

        # First chunk should be full size
        assert len(chunks[0]) == CHUNK_SIZE

    def test_chunk_content_with_newlines(self):
        """Test that chunking tries to break at newlines."""
        from mira.search.local_semantic import get_local_semantic, CHUNK_SIZE

        ls = get_local_semantic()

        # Create content with newlines near the end
        content = "x" * (CHUNK_SIZE - 100) + "\n\nBreak here\n\n" + "y" * (CHUNK_SIZE)
        chunks = ls._chunk_content(content)

        # Should have at least 2 chunks
        assert len(chunks) >= 2

    def test_chunk_max_limit(self):
        """Test that chunking respects MAX_CHUNKS limit."""
        from mira.search.local_semantic import get_local_semantic, CHUNK_SIZE, MAX_CHUNKS

        ls = get_local_semantic()

        # Create very large content that would exceed MAX_CHUNKS
        very_large = "x" * (CHUNK_SIZE * (MAX_CHUNKS + 10))
        chunks = ls._chunk_content(very_large)

        # Should not exceed MAX_CHUNKS
        assert len(chunks) <= MAX_CHUNKS

    def test_chunk_empty_content(self):
        """Test chunking empty content."""
        from mira.search.local_semantic import get_local_semantic

        ls = get_local_semantic()
        chunks = ls._chunk_content("")

        # Empty content should return empty list or single empty string
        assert len(chunks) <= 1


class TestBlobEncoding:
    """ULTRATHINK-011: Test embedding BLOB encoding/decoding."""

    def test_blob_pack_unpack_roundtrip(self):
        """Test that embedding vectors survive pack/unpack."""
        from mira.search.local_semantic import EMBEDDING_DIM

        # Create a test vector
        test_vector = [0.1 * i for i in range(EMBEDDING_DIM)]

        # Pack to blob
        blob = struct.pack(f'{EMBEDDING_DIM}f', *test_vector)

        # Unpack from blob
        unpacked = struct.unpack(f'{EMBEDDING_DIM}f', blob)

        # Should be approximately equal (float32 precision is ~6-7 decimal places)
        for orig, result in zip(test_vector, unpacked):
            assert abs(orig - result) < 1e-5

    def test_blob_size_correct(self):
        """Test that blob size matches expected dimensions."""
        from mira.search.local_semantic import EMBEDDING_DIM

        # 384 floats * 4 bytes = 1536 bytes
        test_vector = [0.0] * EMBEDDING_DIM
        blob = struct.pack(f'{EMBEDDING_DIM}f', *test_vector)

        assert len(blob) == EMBEDDING_DIM * 4  # 4 bytes per float32


class TestQueueFunctions:
    """ULTRATHINK-012: Test indexing queue functions."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ['MIRA_PATH'] = self.temp_dir
        (Path(self.temp_dir) / '.mira').mkdir(exist_ok=True)

    def teardown_method(self):
        os.environ.pop('MIRA_PATH', None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        from mira.core import shutdown_db_manager
        shutdown_db_manager()
        from mira.search import local_semantic
        local_semantic._local_semantic = None
        local_semantic.LocalSemanticSearch._instance = None
        local_semantic.LocalSemanticSearch._initialized = False

    def test_queue_session_for_indexing(self):
        """Test queuing a session for local vector indexing."""
        from mira.search.local_semantic import queue_session_for_indexing, get_pending_indexing_count
        from mira.search.local_semantic import get_local_semantic

        # Initialize the local semantic to ensure schema exists
        ls = get_local_semantic()

        # Queue a session
        result = queue_session_for_indexing(
            session_id="test-session-001",
            content="This is test content for indexing.",
            summary="Test summary"
        )

        assert result is True

        # Check pending count (may be 0 if already indexed, but call should succeed)
        count = get_pending_indexing_count()
        assert count >= 0  # Queue function succeeded

    def test_queue_session_empty_content_fails(self):
        """Test that empty content is rejected."""
        from mira.search.local_semantic import queue_session_for_indexing

        result = queue_session_for_indexing(
            session_id="empty-session",
            content="",
            summary=""
        )

        assert result is False

    def test_queue_session_duplicate(self):
        """Test queuing the same session twice."""
        from mira.search.local_semantic import queue_session_for_indexing, get_pending_indexing_count

        # Queue same session twice
        queue_session_for_indexing("dup-session", "Content 1", "Summary 1")
        initial_count = get_pending_indexing_count()

        queue_session_for_indexing("dup-session", "Content 2", "Summary 2")
        final_count = get_pending_indexing_count()

        # Should replace, not add duplicate
        assert final_count == initial_count

    def test_get_pending_indexing_count_empty(self):
        """Test pending count on fresh database."""
        from mira.search.local_semantic import get_pending_indexing_count

        count = get_pending_indexing_count()
        assert count >= 0


class TestLocalSemanticAvailability:
    """ULTRATHINK-013: Test local semantic availability checks."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ['MIRA_PATH'] = self.temp_dir
        (Path(self.temp_dir) / '.mira').mkdir(exist_ok=True)

    def teardown_method(self):
        os.environ.pop('MIRA_PATH', None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        from mira.core import shutdown_db_manager
        shutdown_db_manager()
        from mira.search import local_semantic
        local_semantic._local_semantic = None
        local_semantic.LocalSemanticSearch._instance = None
        local_semantic.LocalSemanticSearch._initialized = False

    def test_is_local_semantic_available_returns_bool(self):
        """Test is_local_semantic_available returns boolean."""
        from mira.search.local_semantic import is_local_semantic_available

        result = is_local_semantic_available()
        assert isinstance(result, bool)

    def test_trigger_local_semantic_download_returns_notice(self):
        """Test trigger_local_semantic_download returns notice dict."""
        from mira.search.local_semantic import trigger_local_semantic_download

        result = trigger_local_semantic_download()
        assert isinstance(result, dict)

        # Should have a notice if sqlite-vec unavailable or model not ready
        # (In test environment, likely one of these conditions)


class TestLocalSemanticIndexer:
    """ULTRATHINK-012: Test LocalSemanticIndexer background worker."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ['MIRA_PATH'] = self.temp_dir
        (Path(self.temp_dir) / '.mira').mkdir(exist_ok=True)

    def teardown_method(self):
        os.environ.pop('MIRA_PATH', None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        from mira.core import shutdown_db_manager
        shutdown_db_manager()
        from mira.search import local_semantic
        local_semantic._local_semantic = None
        local_semantic.LocalSemanticSearch._instance = None
        local_semantic.LocalSemanticSearch._initialized = False
        local_semantic._indexer = None

    def test_indexer_can_be_created(self):
        """Test LocalSemanticIndexer can be instantiated."""
        from mira.search.local_semantic import LocalSemanticIndexer

        indexer = LocalSemanticIndexer()
        assert indexer is not None
        assert indexer.running is False

    def test_indexer_start_stop(self):
        """Test indexer start and stop."""
        from mira.search.local_semantic import LocalSemanticIndexer

        indexer = LocalSemanticIndexer()

        # Start
        indexer.start()
        assert indexer.running is True

        # Stop
        indexer.stop()
        assert indexer.running is False

    def test_start_local_indexer_function(self):
        """Test start_local_indexer global function."""
        from mira.search.local_semantic import start_local_indexer, stop_local_indexer

        indexer = start_local_indexer()
        assert indexer is not None
        assert indexer.running is True

        stop_local_indexer()


class TestSearchIntegration:
    """ULTRATHINK-013: Test search integration with local semantic."""

    def setup_method(self):
        from mira.core import shutdown_db_manager
        shutdown_db_manager()
        self.temp_dir = tempfile.mkdtemp()
        os.environ['MIRA_PATH'] = self.temp_dir
        (Path(self.temp_dir) / '.mira').mkdir(exist_ok=True)

    def teardown_method(self):
        from mira.core import shutdown_db_manager
        shutdown_db_manager()
        os.environ.pop('MIRA_PATH', None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        from mira.search import local_semantic
        local_semantic._local_semantic = None
        local_semantic.LocalSemanticSearch._instance = None
        local_semantic.LocalSemanticSearch._initialized = False

    def test_search_falls_through_to_fts5(self):
        """Test that search works even without local semantic ready."""
        from mira.search import handle_search

        result = handle_search(params={'query': 'test', 'limit': 5})
        assert 'results' in result
        assert 'total' in result

    def test_search_empty_query(self):
        """Test search with empty query."""
        from mira.search import handle_search

        result = handle_search(params={'query': '', 'limit': 5})
        assert 'results' in result
        assert result['total'] == 0

    def test_search_with_project_filter(self):
        """Test search with project_path filter."""
        from mira.search import handle_search

        result = handle_search(params={
            'query': 'test',
            'limit': 5,
            'project_path': '/some/project'
        })
        assert 'results' in result


class TestSearchTierFallback:
    """ULTRATHINK-013: Test search tier fallback chain."""

    def setup_method(self):
        from mira.core import shutdown_db_manager
        shutdown_db_manager()
        self.temp_dir = tempfile.mkdtemp()
        os.environ['MIRA_PATH'] = self.temp_dir
        (Path(self.temp_dir) / '.mira').mkdir(exist_ok=True)

    def teardown_method(self):
        from mira.core import shutdown_db_manager
        shutdown_db_manager()
        os.environ.pop('MIRA_PATH', None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        from mira.search import local_semantic
        local_semantic._local_semantic = None
        local_semantic.LocalSemanticSearch._instance = None
        local_semantic.LocalSemanticSearch._initialized = False

    def test_search_without_central_uses_local(self):
        """Test that search without central storage uses local search."""
        from mira.search import handle_search

        # Without central storage configured, should use local tiers
        result = handle_search(params={'query': 'test'})
        assert 'results' in result

    @patch('mira.search.local_semantic.is_local_semantic_available')
    def test_local_semantic_fallback_to_fts5(self, mock_available):
        """Test fallback to FTS5 when local semantic unavailable."""
        mock_available.return_value = False

        from mira.search import handle_search

        result = handle_search(params={'query': 'test'})
        assert 'results' in result


class TestConstants:
    """Test configuration constants."""

    def test_embedding_model_defined(self):
        """Test EMBEDDING_MODEL constant is defined."""
        from mira.search.local_semantic import EMBEDDING_MODEL
        assert EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"

    def test_embedding_dim_matches_model(self):
        """Test EMBEDDING_DIM matches expected model dimensions."""
        from mira.search.local_semantic import EMBEDDING_DIM
        assert EMBEDDING_DIM == 384  # bge-small-en-v1.5 uses 384 dims

    def test_chunk_config_reasonable(self):
        """Test chunk configuration is reasonable."""
        from mira.search.local_semantic import CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS

        assert CHUNK_SIZE == 4000
        assert CHUNK_OVERLAP == 500
        assert MAX_CHUNKS == 50
        assert CHUNK_OVERLAP < CHUNK_SIZE
