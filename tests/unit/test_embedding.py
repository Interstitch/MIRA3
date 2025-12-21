"""Tests for embedding configuration."""

from mira.core import EmbeddingConfig


class TestEmbeddingConfig:
    """Test embedding configuration."""

    def test_embedding_config_creation(self):
        """Test that EmbeddingConfig can be created."""
        config = EmbeddingConfig(host="localhost", port=8200)
        assert config.host == "localhost"
        assert config.port == 8200

    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig default values."""
        # host is required, but port has a default
        config = EmbeddingConfig(host="localhost")
        assert config.host == "localhost"
        assert config.port == 8200
