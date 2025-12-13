"""
MIRA Central Server Configuration Loader

Security:
- Validates file permissions (600 on Unix)
- Supports environment variable overrides for sensitive values
- Never logs passwords or connection strings with credentials
"""

import json
import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Embedding service connection settings."""
    host: str
    port: int = 8200
    timeout_seconds: int = 60  # Embedding can take time


@dataclass
class QdrantConfig:
    """Qdrant vector database connection settings."""
    host: str
    port: int = 6333
    collection: str = "mira_sessions"
    timeout_seconds: int = 30
    api_key: Optional[str] = None  # Optional API key for auth


@dataclass
class PostgresConfig:
    """PostgreSQL database connection settings."""
    host: str
    port: int = 5432
    database: str = "mira"
    user: str = "mira"
    password: str = ""
    pool_size: int = 12  # max_workers=4 × 3 connections/worker for nested operations
    timeout_seconds: int = 30

    def connection_string(self, mask_password: bool = False) -> str:
        """Generate connection string, optionally masking password for logging."""
        pwd = "***MASKED***" if mask_password else self.password
        return f"postgresql://{self.user}:{pwd}@{self.host}:{self.port}/{self.database}"


@dataclass
class FallbackConfig:
    """Fallback behavior when central server is unavailable."""
    enabled: bool = True
    warn_on_fallback: bool = True


@dataclass
class CacheConfig:
    """Caching settings for stable data."""
    custodian_ttl_seconds: int = 300
    project_id_ttl_seconds: int = 3600


@dataclass
class CentralConfig:
    """Central server configuration."""
    enabled: bool
    qdrant: QdrantConfig
    postgres: PostgresConfig
    embedding: Optional[EmbeddingConfig] = None


@dataclass
class ServerConfig:
    """Complete MIRA server configuration."""
    version: int
    central: Optional[CentralConfig] = None
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @property
    def central_enabled(self) -> bool:
        """Check if central storage is configured and enabled."""
        return self.central is not None and self.central.enabled


def get_config_path() -> Path:
    """
    Get the path to server.json configuration file.

    Search order:
    1. MIRA_CONFIG_PATH environment variable (explicit override)
    2. <workspace>/.mira/server.json (project-local config)
    3. ~/.mira/server.json (user-global config)

    Project-local config allows different projects to use different
    central storage configurations, or for the same config to be
    shared in the workspace .mira directory.
    """
    # Check environment override first
    env_path = os.environ.get("MIRA_CONFIG_PATH")
    if env_path:
        return Path(env_path)

    # Check for project-local config in workspace .mira directory
    # This is set by get_mira_path() based on current working directory
    from .utils import get_mira_path
    workspace_config = get_mira_path() / "server.json"
    if workspace_config.exists():
        return workspace_config

    # Fallback to user-global config: ~/.mira/server.json
    return Path.home() / ".mira" / "server.json"


def validate_file_permissions(path: Path) -> bool:
    """
    Validate that config file has secure permissions.

    On Unix: Should be 600 (owner read/write only)
    On Windows: Skipped (NTFS ACLs are different)

    Returns True if permissions are acceptable, False otherwise.
    """
    if os.name == "nt":
        # Windows - skip permission check (NTFS ACLs work differently)
        # TODO: Could check NTFS ACLs but it's complex
        return True

    try:
        mode = path.stat().st_mode & 0o777
        if mode != 0o600:
            log.warning(
                f"server.json has insecure permissions {oct(mode)}, "
                f"should be 600. Run: chmod 600 {path}"
            )
            # Still allow loading, but warn
            return True
        return True
    except OSError as e:
        log.error(f"Cannot check permissions on {path}: {e}")
        return False


def load_config() -> ServerConfig:
    """
    Load MIRA server configuration.

    Search order:
    1. MIRA_CONFIG_PATH environment variable
    2. <workspace>/.mira/server.json (project-local)
    3. ~/.mira/server.json (user-global)

    If not found, returns config with central disabled (local-only mode).

    Environment variable overrides:
    - MIRA_CONFIG_PATH: Path to config file
    - MIRA_POSTGRES_PASSWORD: Override postgres password
    - MIRA_CENTRAL_ENABLED: "true"/"false" to override enabled state

    Security:
    - Validates file permissions on Unix (warns if not 600)
    - Password can be overridden via env var (avoids file storage)
    """
    config_path = get_config_path()

    if not config_path.exists():
        log.info(f"No server.json found at {config_path}, using local-only mode")
        return ServerConfig(version=1, central=None)

    # Validate permissions
    validate_file_permissions(config_path)

    try:
        with open(config_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in {config_path}: {e}")
        return ServerConfig(version=1, central=None)
    except OSError as e:
        log.error(f"Cannot read {config_path}: {e}")
        return ServerConfig(version=1, central=None)

    # Validate version
    version = data.get("version", 1)
    if version != 1:
        log.warning(f"Unknown config version {version}, attempting to load anyway")

    # Parse central config
    central_data = data.get("central", {})
    central = None

    if central_data.get("enabled", False):
        # Parse Qdrant config
        qdrant_data = central_data.get("qdrant", {})
        qdrant = QdrantConfig(
            host=qdrant_data.get("host", ""),
            port=qdrant_data.get("port", 6333),
            collection=qdrant_data.get("collection", "mira_sessions"),
            timeout_seconds=qdrant_data.get("timeout_seconds", 30),
            api_key=qdrant_data.get("api_key"),
        )

        # Parse Postgres config
        pg_data = central_data.get("postgres", {})
        postgres = PostgresConfig(
            host=pg_data.get("host", ""),
            port=pg_data.get("port", 5432),
            database=pg_data.get("database", "mira"),
            user=pg_data.get("user", "mira"),
            password=pg_data.get("password", ""),
            pool_size=pg_data.get("pool_size", 3),
            timeout_seconds=pg_data.get("timeout_seconds", 30),
        )

        # Environment variable override for password
        env_password = os.environ.get("MIRA_POSTGRES_PASSWORD")
        if env_password:
            postgres.password = env_password
            log.debug("Using MIRA_POSTGRES_PASSWORD from environment")

        # Parse Embedding service config (uses same host as qdrant by default)
        embed_data = central_data.get("embedding", {})
        embedding = None
        if embed_data:
            embedding = EmbeddingConfig(
                host=embed_data.get("host", qdrant_data.get("host", "")),
                port=embed_data.get("port", 8200),
                timeout_seconds=embed_data.get("timeout_seconds", 60),
            )
        elif qdrant_data.get("host"):
            # Default: embedding service runs on same host as qdrant
            embedding = EmbeddingConfig(
                host=qdrant_data.get("host"),
                port=8200,
            )

        central = CentralConfig(
            enabled=True,
            qdrant=qdrant,
            postgres=postgres,
            embedding=embedding,
        )

        # Log connection info (without password)
        embed_info = f", Embedding={embedding.host}:{embedding.port}" if embedding else ""
        log.info(f"Central storage configured: Qdrant={qdrant.host}:{qdrant.port}, "
                 f"Postgres={postgres.connection_string(mask_password=True)}{embed_info}")

    # Environment variable override for enabled state
    env_enabled = os.environ.get("MIRA_CENTRAL_ENABLED")
    if env_enabled is not None:
        if env_enabled.lower() in ("false", "0", "no"):
            if central:
                central.enabled = False
                log.info("Central storage disabled via MIRA_CENTRAL_ENABLED=false")
        elif env_enabled.lower() in ("true", "1", "yes"):
            if central:
                central.enabled = True

    # Parse fallback config
    fallback_data = data.get("fallback", {})
    fallback = FallbackConfig(
        enabled=fallback_data.get("enabled", True),
        warn_on_fallback=fallback_data.get("warn_on_fallback", True),
    )

    # Parse cache config
    cache_data = data.get("cache", {})
    cache = CacheConfig(
        custodian_ttl_seconds=cache_data.get("custodian_ttl_seconds", 300),
        project_id_ttl_seconds=cache_data.get("project_id_ttl_seconds", 3600),
    )

    return ServerConfig(
        version=version,
        central=central,
        fallback=fallback,
        cache=cache,
    )


# Global cached config (loaded once per process)
_config: Optional[ServerConfig] = None


def get_config() -> ServerConfig:
    """Get the cached server configuration, loading if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> ServerConfig:
    """Force reload of server configuration."""
    global _config
    _config = load_config()
    return _config
