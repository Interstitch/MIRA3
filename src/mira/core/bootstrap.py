"""
MIRA Bootstrap Module

Handles initial setup: directory creation and Claude Code configuration.

Dependencies are managed by pip via pyproject.toml:
  - Core: pip install claude-mira3
  - Semantic: pip install claude-mira3[semantic]
"""

import sys
import json
from pathlib import Path

from .constants import get_global_mira_path, get_project_mira_path
from .utils import log


def _get_mira_command() -> str:
    """Get the mira command for Claude Code configuration."""
    # When installed via pip, 'mira' is available as a command
    # On Windows it's mira.exe, on Unix it's just mira
    # But we use 'python -m mira' for reliability
    return sys.executable


def _configure_claude_code():
    """
    Auto-configure Claude Code to use MIRA as MCP server.

    Updates ~/.claude.json and ~/.claude/settings.json if needed.
    """
    home = Path.home()
    config_paths = [
        home / ".claude.json",
        home / ".claude" / "settings.json",
    ]

    # Use python -m mira for maximum compatibility
    new_mira_config = {
        "command": sys.executable,
        "args": ["-m", "mira"],
    }

    for config_path in config_paths:
        try:
            config = {}
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))

            if "mcpServers" not in config:
                config["mcpServers"] = {}

            # Check for old Node.js config and remove it
            for old_key in ["mira3"]:
                if old_key in config["mcpServers"]:
                    old_cfg = config["mcpServers"][old_key]
                    if old_cfg.get("command") == "node" or "npx" in str(old_cfg.get("args", [])):
                        log(f"Removing old Node.js config from {config_path}")
                        del config["mcpServers"][old_key]

            # Check if update needed
            existing = config["mcpServers"].get("mira")
            if existing == new_mira_config:
                continue  # Already configured correctly

            # Update config
            config["mcpServers"]["mira"] = new_mira_config
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            log(f"Configured Claude Code: {config_path}")

        except Exception as e:
            log(f"Could not update {config_path}: {e}")


def is_running_in_venv() -> bool:
    """Check if we're running inside a virtualenv (any venv, not just ours)."""
    return hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )


def get_venv_site_packages() -> Path:
    """
    Get the site-packages directory.

    DEPRECATED: No longer uses a separate venv.
    Returns the current interpreter's site-packages for compatibility.
    """
    import site
    paths = site.getsitepackages()
    if paths:
        return Path(paths[0])
    # Fallback
    return Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def activate_venv_deps():
    """
    DEPRECATED: No-op for backwards compatibility.

    Dependencies are now installed via pip alongside claude-mira3.
    No separate venv activation needed.
    """
    pass  # No-op - deps are already available


def has_semantic_deps() -> bool:
    """Check if semantic search dependencies are installed (now in core)."""
    try:
        import fastembed  # noqa: F401
        # numpy is a fastembed dependency, used for vector similarity
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def has_remote_deps() -> bool:
    """Check if remote storage dependencies are installed (optional)."""
    try:
        import qdrant_client  # noqa: F401
        import psycopg2  # noqa: F401
        return True
    except ImportError:
        return False


def ensure_venv_and_deps() -> bool:
    """
    Ensure MIRA directories exist and Claude Code is configured.

    Returns False (no re-exec needed - deps installed via pip).

    What this does:
    1. Creates ~/.mira/ and <project>/.mira/ directories
    2. Configures Claude Code to use MIRA as MCP server
    3. Checks for optional semantic deps (informational only)
    """
    global_mira_path = get_global_mira_path()
    project_mira_path = get_project_mira_path()
    config_path = global_mira_path / "config.json"

    # Create both global and project .mira directories
    global_mira_path.mkdir(parents=True, exist_ok=True)
    project_mira_path.mkdir(parents=True, exist_ok=True)

    # Check if already initialized
    already_configured = False
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            already_configured = config.get("configured", False)
        except (json.JSONDecodeError, IOError, OSError):
            pass

    if not already_configured:
        # First run - configure Claude Code
        try:
            _configure_claude_code()
        except Exception as e:
            log(f"Claude Code config failed (non-fatal): {e}")

        # Check semantic deps (should be in core now)
        if not has_semantic_deps():
            log("Warning: Semantic search deps missing - reinstall with: pip install claude-mira3")

        # Check for server.json (remote storage config)
        project_server_config = project_mira_path / "server.json"
        global_server_config = global_mira_path / "server.json"
        has_server_config = project_server_config.exists() or global_server_config.exists()

        if has_server_config and not has_remote_deps():
            log("Remote storage configured but deps not installed")
            log("Install with: pip install claude-mira3[remote]")

        # Mark as configured
        config = {
            "configured": True,
            "semantic_available": has_semantic_deps(),
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    return False  # No re-exec needed
