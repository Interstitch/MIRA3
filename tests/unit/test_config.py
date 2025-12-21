"""Tests for mira.core.config module."""

import os
import json
import tempfile
import shutil
from pathlib import Path


class TestConfig:
    """Test configuration loading and validation."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "server.json"

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_dataclasses(self):
        """Test config dataclass creation."""
        from mira.core.config import QdrantConfig, PostgresConfig, ServerConfig

        qdrant = QdrantConfig(host="localhost", port=6333)
        assert qdrant.host == "localhost"
        assert qdrant.port == 6333

        postgres = PostgresConfig(host="localhost", password="secret")
        assert postgres.host == "localhost"
        assert postgres.password == "secret"

    def test_load_config_no_file(self):
        """Test loading config when file doesn't exist."""
        from mira.core.config import load_config

        old_env = os.environ.get("MIRA_CONFIG_PATH")
        os.environ["MIRA_CONFIG_PATH"] = "/nonexistent/path/server.json"

        try:
            config = load_config()
            assert config.version == 1
            assert config.central is None
        finally:
            if old_env:
                os.environ["MIRA_CONFIG_PATH"] = old_env
            else:
                os.environ.pop("MIRA_CONFIG_PATH", None)


class TestModuleImports:
    """Test that all mira modules can be imported without errors."""

    def test_import_config(self):
        """Test mira.core.config imports successfully."""
        from mira.core import config
        assert hasattr(config, 'load_config')

    def test_import_storage(self):
        """Test mira.storage imports successfully."""
        from mira import storage
        assert hasattr(storage, 'Storage')

    def test_import_handlers(self):
        """Test mira.tools imports successfully (handlers are now tools)."""
        from mira import tools
        assert tools is not None

    def test_import_local_store(self):
        """Test mira.storage.local_store imports successfully."""
        from mira.storage import local_store
        assert hasattr(local_store, 'init_local_db')

    def test_import_search(self):
        """Test mira.search imports successfully."""
        from mira import search
        assert search is not None

    def test_import_ingestion(self):
        """Test mira.ingestion imports successfully."""
        from mira import ingestion
        assert ingestion is not None

    def test_import_metadata(self):
        """Test mira.extraction.metadata imports successfully."""
        from mira.extraction import metadata
        assert hasattr(metadata, 'extract_metadata')

    def test_import_custodian(self):
        """Test mira.custodian imports successfully."""
        from mira import custodian
        assert hasattr(custodian, 'init_custodian_db')

    def test_import_insights(self):
        """Test mira.extraction.insights imports successfully."""
        from mira.extraction import insights
        # insights.py is the coordinator - init_insights_db is in errors.py
        assert hasattr(insights, 'extract_insights_from_conversation')

    def test_import_artifacts(self):
        """Test mira.extraction.artifacts imports successfully."""
        from mira.extraction import artifacts
        assert hasattr(artifacts, 'init_artifact_db')

    def test_import_concepts(self):
        """Test mira.extraction.concepts imports successfully."""
        from mira.extraction import concepts
        assert concepts is not None

    def test_import_watcher(self):
        """Test mira.ingestion.watcher imports successfully."""
        from mira.ingestion import watcher
        assert hasattr(watcher, 'ConversationWatcher')
