"""Tests for mira.embedding_client module."""

from mira.config import EmbeddingConfig


class TestEmbeddingClient:
    """Test remote embedding client functionality."""

    def test_embedding_client_interface(self):
        """Test that embedding_client module has expected interface."""
        from mira.embedding_client import EmbeddingClient, get_embedding_client

        # Test EmbeddingClient class exists and has expected methods
        config = EmbeddingConfig(host="localhost", port=8200)
        client = EmbeddingClient(config)

        # Client only does search - embedding happens automatically on GCP
        assert hasattr(client, 'search')
        assert hasattr(client, 'health_check')
        assert hasattr(client, 'stats')
        assert client.base_url == "http://localhost:8200"
