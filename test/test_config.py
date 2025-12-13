"""Tests for mira.config module."""

import os
import json
import tempfile
import shutil
from pathlib import Path


class TestConfig:
    """Test configuration loading and validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "server.json"

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_dataclasses(self):
        """Test config dataclass creation."""
        from mira.config import QdrantConfig, PostgresConfig, ServerConfig

        qdrant = QdrantConfig(host="localhost", port=6333)
        assert qdrant.host == "localhost"
        assert qdrant.port == 6333
        assert qdrant.collection == "mira_sessions"  # default (matches server)
        assert qdrant.api_key is None  # default

        postgres = PostgresConfig(host="localhost", password="secret")
        assert postgres.host == "localhost"
        assert postgres.password == "secret"
        assert postgres.database == "mira"  # default
        assert postgres.pool_size == 12  # default

    def test_postgres_connection_string(self):
        """Test PostgresConfig connection string generation."""
        from mira.config import PostgresConfig

        pg = PostgresConfig(
            host="db.example.com",
            port=5432,
            database="mydb",
            user="myuser",
            password="secret123"
        )

        # Unmasked
        conn_str = pg.connection_string()
        assert "secret123" in conn_str
        assert "myuser" in conn_str
        assert "db.example.com" in conn_str

        # Masked
        masked_str = pg.connection_string(mask_password=True)
        assert "secret123" not in masked_str
        assert "***MASKED***" in masked_str

    def test_server_config_central_enabled(self):
        """Test ServerConfig.central_enabled property."""
        from mira.config import ServerConfig, CentralConfig, QdrantConfig, PostgresConfig

        # No central config
        config = ServerConfig(version=1, central=None)
        assert config.central_enabled is False

        # Central config but disabled
        central = CentralConfig(
            enabled=False,
            qdrant=QdrantConfig(host="localhost"),
            postgres=PostgresConfig(host="localhost", password="x")
        )
        config = ServerConfig(version=1, central=central)
        assert config.central_enabled is False

        # Central config and enabled
        central.enabled = True
        assert config.central_enabled is True

    def test_load_config_no_file(self):
        """Test loading config when file doesn't exist."""
        from mira.config import load_config

        # Set env var to non-existent path
        old_env = os.environ.get("MIRA_CONFIG_PATH")
        os.environ["MIRA_CONFIG_PATH"] = "/nonexistent/path/server.json"

        try:
            config = load_config()
            assert config.version == 1
            assert config.central is None
            assert config.central_enabled is False
        finally:
            if old_env:
                os.environ["MIRA_CONFIG_PATH"] = old_env
            else:
                os.environ.pop("MIRA_CONFIG_PATH", None)

    def test_load_config_valid_file(self):
        """Test loading a valid config file."""
        from mira.config import load_config

        # Write valid config
        config_data = {
            "version": 1,
            "central": {
                "enabled": True,
                "qdrant": {
                    "host": "10.0.0.1",
                    "port": 6333,
                    "collection": "test_mira"
                },
                "postgres": {
                    "host": "10.0.0.1",
                    "port": 5432,
                    "database": "test_db",
                    "user": "test_user",
                    "password": "test_pass"
                }
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)

        old_env = os.environ.get("MIRA_CONFIG_PATH")
        os.environ["MIRA_CONFIG_PATH"] = str(self.config_path)

        try:
            config = load_config()
            assert config.version == 1
            assert config.central_enabled is True
            assert config.central.qdrant.host == "10.0.0.1"
            assert config.central.qdrant.collection == "test_mira"
            assert config.central.postgres.database == "test_db"
        finally:
            if old_env:
                os.environ["MIRA_CONFIG_PATH"] = old_env
            else:
                os.environ.pop("MIRA_CONFIG_PATH", None)

    def test_load_config_invalid_json(self):
        """Test loading config with invalid JSON."""
        from mira.config import load_config

        # Write invalid JSON
        with open(self.config_path, 'w') as f:
            f.write("{ invalid json }")

        old_env = os.environ.get("MIRA_CONFIG_PATH")
        os.environ["MIRA_CONFIG_PATH"] = str(self.config_path)

        try:
            config = load_config()
            # Should return default config on error
            assert config.version == 1
            assert config.central is None
        finally:
            if old_env:
                os.environ["MIRA_CONFIG_PATH"] = old_env
            else:
                os.environ.pop("MIRA_CONFIG_PATH", None)

    def test_env_password_override(self):
        """Test that MIRA_POSTGRES_PASSWORD env var overrides config."""
        from mira.config import load_config

        # Write config with one password
        config_data = {
            "version": 1,
            "central": {
                "enabled": True,
                "qdrant": {"host": "localhost"},
                "postgres": {
                    "host": "localhost",
                    "password": "config_password"
                }
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)

        old_config_env = os.environ.get("MIRA_CONFIG_PATH")
        old_pass_env = os.environ.get("MIRA_POSTGRES_PASSWORD")
        os.environ["MIRA_CONFIG_PATH"] = str(self.config_path)
        os.environ["MIRA_POSTGRES_PASSWORD"] = "env_password"

        try:
            config = load_config()
            # Env var should override
            assert config.central.postgres.password == "env_password"
        finally:
            if old_config_env:
                os.environ["MIRA_CONFIG_PATH"] = old_config_env
            else:
                os.environ.pop("MIRA_CONFIG_PATH", None)
            if old_pass_env:
                os.environ["MIRA_POSTGRES_PASSWORD"] = old_pass_env
            else:
                os.environ.pop("MIRA_POSTGRES_PASSWORD", None)

    def test_validate_file_permissions(self):
        """Test file permission validation."""
        from mira.config import validate_file_permissions

        # Create file with default permissions
        test_file = Path(self.temp_dir) / "test.json"
        test_file.write_text("{}")

        # Should return True (we warn but don't block)
        result = validate_file_permissions(test_file)
        assert result is True

    def test_api_key_loaded_from_config(self):
        """Test that Qdrant api_key is properly loaded from config file."""
        from mira.config import load_config

        # Write config with api_key
        config_data = {
            "version": 1,
            "central": {
                "enabled": True,
                "qdrant": {
                    "host": "10.0.0.1",
                    "port": 6333,
                    "api_key": "test_api_key_123"
                },
                "postgres": {
                    "host": "10.0.0.1",
                    "password": "test_pass"
                }
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)

        old_env = os.environ.get("MIRA_CONFIG_PATH")
        os.environ["MIRA_CONFIG_PATH"] = str(self.config_path)

        try:
            config = load_config()
            assert config.central.qdrant.api_key == "test_api_key_123"
        finally:
            if old_env:
                os.environ["MIRA_CONFIG_PATH"] = old_env
            else:
                os.environ.pop("MIRA_CONFIG_PATH", None)


class TestModuleImports:
    """Test that all mira modules can be imported without errors.

    This catches missing imports (like forgetting 'from datetime import datetime')
    at test time rather than at runtime.
    """

    def test_import_config(self):
        """Test mira.config imports successfully."""
        from mira import config
        assert hasattr(config, 'load_config')

    def test_import_storage(self):
        """Test mira.storage imports successfully."""
        from mira import storage
        assert hasattr(storage, 'Storage')

    def test_import_handlers(self):
        """Test mira.handlers imports successfully."""
        from mira import handlers
        # The module imports successfully - that's what we're testing
        assert handlers is not None

    def test_import_local_store(self):
        """Test mira.local_store imports successfully."""
        from mira import local_store
        assert hasattr(local_store, 'LOCAL_DB')

    def test_import_postgres_backend(self):
        """Test mira.postgres_backend imports successfully."""
        from mira import postgres_backend
        assert hasattr(postgres_backend, 'PostgresBackend')

    def test_import_search(self):
        """Test mira.search imports successfully."""
        from mira import search
        # The module imports successfully - that's what we're testing
        assert search is not None

    def test_import_ingestion(self):
        """Test mira.ingestion imports successfully."""
        from mira import ingestion
        # The module imports successfully - that's what we're testing
        assert ingestion is not None

    def test_import_metadata(self):
        """Test mira.metadata imports successfully."""
        from mira import metadata
        assert hasattr(metadata, 'extract_metadata')

    def test_import_custodian(self):
        """Test mira.custodian imports successfully."""
        from mira import custodian
        assert hasattr(custodian, 'init_custodian_db')

    def test_import_insights(self):
        """Test mira.insights imports successfully."""
        from mira import insights
        assert hasattr(insights, 'init_insights_db')

    def test_import_artifacts(self):
        """Test mira.artifacts imports successfully."""
        from mira import artifacts
        assert hasattr(artifacts, 'init_artifact_db')

    def test_import_concepts(self):
        """Test mira.concepts imports successfully."""
        from mira import concepts
        # The module imports successfully - that's what we're testing
        assert concepts is not None

    def test_import_watcher(self):
        """Test mira.watcher imports successfully."""
        from mira import watcher
        assert hasattr(watcher, 'ConversationWatcher')

    def test_import_main(self):
        """Test mira.main imports successfully."""
        from mira import main
        assert hasattr(main, 'main')
