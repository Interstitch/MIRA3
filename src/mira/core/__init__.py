"""
MIRA Core Module

Fundamental infrastructure: constants, config, database, bootstrap, utilities.
"""

from .constants import (
    VERSION,
    MIRA_PATH,
    get_mira_path,
    get_global_mira_path,
    get_project_mira_path,
    get_db_path,
    GLOBAL_DATABASES,
    PROJECT_DATABASES,
    DEPENDENCIES,
    DEPENDENCIES_SEMANTIC,
    DB_LOCAL_STORE,
    DB_ARTIFACTS,
    DB_CUSTODIAN,
    DB_INSIGHTS,
    DB_CONCEPTS,
    DB_CODE_HISTORY,
    DB_SYNC_QUEUE,
    DB_MIGRATIONS,
    DB_LOCAL_VECTORS,
)

from .config import (
    get_config,
    load_config,
    ServerConfig,
    CentralConfig,
    PostgresConfig,
    QdrantConfig,
    EmbeddingConfig,
)

from .database import (
    DatabaseManager,
    get_db_manager,
    shutdown_db_manager,
)

from .utils import (
    log,
    get_venv_path,
    get_venv_python,
    get_venv_pip,
    get_mira_config,
    get_custodian,
    get_git_remote,
    normalize_git_remote,
    parse_timestamp,
    extract_text_content,
    get_claude_projects_path,
    get_project_filter,
)

from .bootstrap import (
    is_running_in_venv,
    ensure_venv_and_deps,
    activate_venv_deps,
    get_venv_site_packages,
)

from .parsing import (
    parse_conversation,
    extract_tool_usage,
    extract_todos_from_message,
)

__all__ = [
    # Constants
    "VERSION",
    "MIRA_PATH",
    "get_mira_path",
    "get_global_mira_path",
    "get_project_mira_path",
    "get_db_path",
    "GLOBAL_DATABASES",
    "PROJECT_DATABASES",
    "DEPENDENCIES",
    "DEPENDENCIES_SEMANTIC",
    "DB_LOCAL_STORE",
    "DB_ARTIFACTS",
    "DB_CUSTODIAN",
    "DB_INSIGHTS",
    "DB_CONCEPTS",
    "DB_CODE_HISTORY",
    "DB_SYNC_QUEUE",
    "DB_MIGRATIONS",
    "DB_LOCAL_VECTORS",
    # Config
    "get_config",
    "load_config",
    "ServerConfig",
    "CentralConfig",
    "PostgresConfig",
    "QdrantConfig",
    "EmbeddingConfig",
    # Database
    "DatabaseManager",
    "get_db_manager",
    "shutdown_db_manager",
    # Utils
    "log",
    "get_venv_path",
    "get_venv_python",
    "get_venv_pip",
    "get_mira_config",
    "get_custodian",
    "get_git_remote",
    "normalize_git_remote",
    "parse_timestamp",
    "extract_text_content",
    "get_claude_projects_path",
    "get_project_filter",
    # Bootstrap
    "is_running_in_venv",
    "ensure_venv_and_deps",
    "activate_venv_deps",
    "get_venv_site_packages",
    # Parsing
    "parse_conversation",
    "extract_tool_usage",
    "extract_todos_from_message",
]
