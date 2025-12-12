"""Tests for mira.postgres_backend, mira.qdrant_backend, and mira.llm_extractor modules."""


class TestPostgresBackendMock:
    """Test PostgresBackend with mocked connections."""

    def test_postgres_backend_init(self):
        """Test PostgresBackend initialization."""
        from mira.postgres_backend import PostgresBackend

        backend = PostgresBackend(
            host="localhost",
            port=5432,
            database="testdb",
            user="testuser",
            password="testpass",
            pool_size=5,
            timeout=30
        )

        assert backend.host == "localhost"
        assert backend.port == 5432
        assert backend.database == "testdb"
        assert backend.user == "testuser"
        assert backend.password == "testpass"
        assert backend.pool_size == 5
        assert backend.timeout == 30
        assert backend._pool is None  # Lazy init
        assert backend._healthy is False

    def test_project_dataclass(self):
        """Test Project dataclass."""
        from mira.postgres_backend import Project

        project = Project(id=1, path="/test/path", slug="test-slug")
        assert project.id == 1
        assert project.path == "/test/path"
        assert project.slug == "test-slug"

    def test_session_dataclass(self):
        """Test Session dataclass."""
        from mira.postgres_backend import Session

        session = Session(
            id=1,
            project_id=1,
            session_id="abc-123",
            summary="Test session",
            keywords=["test", "example"],
            facts=["fact1"],
            task_description="Do something",
            git_branch="main",
            models_used=["claude-3"],
            tools_used=["Read", "Write"],
            files_touched=["file1.py"],
            message_count=10,
            started_at="2025-01-01",
            ended_at="2025-01-02"
        )

        assert session.session_id == "abc-123"
        assert len(session.keywords) == 2
        assert session.message_count == 10

    def test_project_cache_initialization(self):
        """Test that project cache is initialized empty."""
        from mira.postgres_backend import PostgresBackend

        backend = PostgresBackend(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test"
        )

        assert backend._project_cache == {}
        assert backend._project_cache_time == {}
        assert backend._project_cache_ttl == 3600


class TestQdrantBackendMock:
    """Test QdrantBackend with mocked connections."""

    def test_qdrant_backend_init(self):
        """Test QdrantBackend initialization."""
        from mira.qdrant_backend import QdrantBackend

        backend = QdrantBackend(
            host="localhost",
            port=6333,
            collection="test_collection",
            timeout=30,
            api_key="test_key"
        )

        assert backend.host == "localhost"
        assert backend.port == 6333
        assert backend.collection == "test_collection"
        assert backend.timeout == 30
        assert backend.api_key == "test_key"
        assert backend._client is None  # Lazy init
        assert backend._healthy is False

    def test_search_result_dataclass(self):
        """Test SearchResult dataclass."""
        from mira.qdrant_backend import SearchResult

        result = SearchResult(
            id="result-1",
            score=0.95,
            content="Test content",
            session_id="session-123",
            project_path="/test/path",
            chunk_type="message",
            role="assistant",
            metadata={"key": "value"}
        )

        assert result.id == "result-1"
        assert result.score == 0.95
        assert result.session_id == "session-123"
        assert result.metadata["key"] == "value"

    def test_health_check_caching(self):
        """Test that health check interval is set."""
        from mira.qdrant_backend import QdrantBackend

        backend = QdrantBackend(
            host="localhost",
            port=6333,
            collection="test"
        )

        assert backend._health_check_interval == 60
        assert backend._last_health_check == 0


class TestLLMExtractor:
    """Test LLM extractor client."""

    def test_llm_extractor_client_init(self):
        """Test LLMExtractorClient initialization."""
        from mira.llm_extractor import LLMExtractorClient

        client = LLMExtractorClient(base_url="http://localhost:8300")
        assert client.base_url == "http://localhost:8300"
        assert client.timeout == 120.0
        assert client._client is None  # Lazy init

    def test_llm_extractor_default_url(self):
        """Test LLMExtractorClient handles default URL from config."""
        from mira.llm_extractor import LLMExtractorClient

        # Without explicit URL, uses config (may be None or configured host)
        client = LLMExtractorClient()
        # base_url is either None (no config) or a string (from config)
        assert client.base_url is None or isinstance(client.base_url, str)

    def test_get_extractor_client_singleton(self):
        """Test get_extractor_client returns singleton."""
        from mira.llm_extractor import get_extractor_client

        client1 = get_extractor_client()
        client2 = get_extractor_client()

        # Should be same instance (or both None if not configured)
        assert client1 is client2
