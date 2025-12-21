"""
MIRA Bootstrap Module

Handles virtualenv creation, dependency installation, and re-execution.

Strategy:
1. Create venv with python -m venv (built-in, always works)
2. Install uv first via pip (one slow install, ~3 seconds)
3. Use uv pip for all other deps (10-100x faster)
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

from .constants import DEPENDENCIES, DEPENDENCIES_SEMANTIC, get_global_mira_path, get_project_mira_path
from .utils import get_venv_path, get_venv_python, get_venv_pip, get_venv_uv, get_venv_mira, log
from mira._version import __version__ as CURRENT_VERSION


def _configure_claude_code():
    """
    Auto-configure Claude Code to use MIRA as MCP server.

    Updates ~/.claude.json and ~/.claude/settings.json if needed.
    Replaces any old Node.js-based MIRA config with Python version.
    """
    home = Path.home()
    config_paths = [
        home / ".claude.json",
        home / ".claude" / "settings.json",
    ]

    # Get the mira command path (cross-platform)
    mira_bin = get_venv_mira()

    new_mira_config = {
        "command": mira_bin,
        "args": [],
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
    """Check if we're running inside our virtualenv."""
    venv_path = get_venv_path()
    # Check if sys.prefix points to our venv
    if Path(sys.prefix).resolve() == venv_path.resolve():
        return True
    # Check VIRTUAL_ENV environment variable
    virtual_env = os.environ.get("VIRTUAL_ENV", "")
    if virtual_env and Path(virtual_env).resolve() == venv_path.resolve():
        return True
    return False


def _get_uv_path(venv_path: Path) -> str:
    """Get path to uv binary in venv (cross-platform)."""
    return get_venv_uv()


def get_venv_site_packages() -> Path:
    """Get the site-packages directory in the venv (cross-platform)."""
    venv_path = get_venv_path()
    if sys.platform == "win32":
        return venv_path / "Lib" / "site-packages"
    else:
        # Unix: lib/python3.x/site-packages
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        return venv_path / "lib" / python_version / "site-packages"


def activate_venv_deps():
    """
    Add venv's site-packages to sys.path to use its dependencies.

    This allows the global mira install to use dependencies from the venv
    without re-executing. The mira code stays in the global install (no file
    locking issues on Windows), but dependencies come from the venv.
    """
    site_packages = get_venv_site_packages()
    if site_packages.exists():
        site_packages_str = str(site_packages)
        if site_packages_str not in sys.path:
            # Insert at beginning so venv packages take priority
            sys.path.insert(0, site_packages_str)
            log(f"Activated venv dependencies from {site_packages}")
            return True
    return False


def _install_with_uv(venv_path: Path, deps: list, optional: bool = False) -> bool:
    """
    Install dependencies using uv pip.

    Returns True on success, False on failure.
    """
    uv = _get_uv_path(venv_path)
    python = get_venv_python()

    try:
        dep_list = ", ".join(deps)
        if optional:
            log(f"Installing optional: {dep_list}")
        else:
            log(f"Installing: {dep_list}")

        result = subprocess.run(
            [uv, "pip", "install", "--python", python] + deps + ["-q"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes for large packages like fastembed
        )
        if result.returncode != 0:
            if optional:
                log(f"Optional install failed (non-fatal): {result.stderr[:200] if result.stderr else 'unknown'}")
            else:
                log(f"Install failed: {result.stderr[:200] if result.stderr else 'unknown'}")
            return False
        return True
    except Exception as e:
        log(f"uv install error: {e}")
        return False


def ensure_venv_and_deps() -> bool:
    """
    Ensure virtualenv exists and dependencies are installed.
    Returns True if we need to re-exec in the venv.

    Storage layout:
    - Global (~/.mira/): venv, global config, user-wide databases
    - Project (<cwd>/.mira/): project-specific databases, logs, archives

    Strategy:
    1. Create venv with python -m venv (built-in)
    2. Install uv first via pip
    3. Use uv for remaining deps (fast)
    """
    global_mira_path = get_global_mira_path()
    project_mira_path = get_project_mira_path()
    venv_path = get_venv_path()  # Now returns global path
    config_path = global_mira_path / "config.json"  # Global config for venv state

    # Create both global and project .mira directories
    global_mira_path.mkdir(parents=True, exist_ok=True)
    project_mira_path.mkdir(parents=True, exist_ok=True)

    # Check if venv exists and is set up
    deps_installed = False
    deps_version = 0

    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            deps_installed = config.get("deps_installed", False)
            deps_version = config.get("deps_version", 0)
        except (json.JSONDecodeError, IOError, OSError):
            pass

    # Current dependency version - increment when adding new required packages
    # v1: Added qdrant-client and psycopg2-binary as required (not optional)
    # v2: Upgrade pip, setuptools, and wheel for security (setuptools vulnerability fix)
    # v3: Added optional semantic search deps (fastembed, sqlite-vec)
    # v4: Pure Python restructure (mcp package)
    # v5: Added uv support for faster installation
    # v6: Fixed conditional logging for uv
    # v7: Simplified - install uv as dep, use for all installs
    # v8: Added sync worker module (0.3.5)
    # v9: Auto-configure Claude Code, fix MCP SDK 1.25.0 compatibility (0.3.8)
    # v10: Renamed to claude-mira3, Windows compatibility (0.4.0)
    # v11: Hybrid storage model - global venv, project-local data (0.4.0)
    # v12: Removed claude-mira3 from venv (runs from global install) - fixes Windows upgrade locking
    CURRENT_DEPS_VERSION = 12

    # Force reinstall if deps version is outdated
    if deps_version < CURRENT_DEPS_VERSION:
        deps_installed = False
        log(f"Dependency version outdated ({deps_version} < {CURRENT_DEPS_VERSION}), will reinstall")

    if not venv_path.exists():
        # Find Python with sqlite extension support for sqlite-vec
        python_to_use = sys.executable

        # On Unix, try to use system Python which typically has sqlite extension support
        # Windows doesn't have /usr/bin/python3, so skip this check
        if sys.platform != "win32":
            try:
                # Check if /usr/bin/python3 has extension support (system Python usually does)
                result = subprocess.run(
                    ["/usr/bin/python3", "-c",
                     "import sqlite3; sqlite3.connect(':memory:').enable_load_extension(True); print('ok')"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and "ok" in result.stdout:
                    python_to_use = "/usr/bin/python3"
                    log("Using system Python (has sqlite extension support)")
            except Exception:
                pass  # Fall back to sys.executable

        log(f"Creating virtualenv at {venv_path}")
        subprocess.run(
            [python_to_use, "-m", "venv", str(venv_path)],
            check=True,
            capture_output=True
        )
        deps_installed = False

    if not deps_installed:
        pip = get_venv_pip()

        # Step 1: Install uv first (one slow pip install)
        log("Installing uv (for fast dependency installation)...")
        subprocess.run(
            [pip, "install", "uv", "-q"],
            check=True,
            capture_output=True
        )

        # Step 2: Use uv for all remaining deps (fast!)
        semantic_deps_installed = False
        core_deps_installed = _install_with_uv(venv_path, DEPENDENCIES)

        if not core_deps_installed:
            # Core dependencies failed - do NOT mark as installed
            # This ensures bootstrap will retry on next run
            log("ERROR: Core dependencies failed to install. Will retry on next run.")
            # Write partial config so we know venv exists but deps failed
            config = {
                "deps_installed": False,
                "deps_version": 0,
                "install_failed_at": datetime.now().isoformat()
            }
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            return True  # Still try to re-exec, maybe partial install works

        # Core deps succeeded, try optional semantic deps
        semantic_deps_installed = _install_with_uv(
            venv_path, DEPENDENCIES_SEMANTIC, optional=True
        )
        if semantic_deps_installed:
            log("Semantic search dependencies installed")
        else:
            log("Local semantic search unavailable - using keyword search")

        # Verify server.json exists (required for remote storage)
        # Check both project-local and global locations
        project_server_config = project_mira_path / "server.json"
        global_server_config = global_mira_path / "server.json"
        if not project_server_config.exists() and not global_server_config.exists():
            log("Note: server.json not found - running in local-only mode")

        # Auto-configure Claude Code MCP settings
        try:
            _configure_claude_code()
        except Exception as e:
            log(f"Claude Code config update failed (non-fatal): {e}")

        # Mark as installed with version (only if core deps succeeded)
        config = {
            "deps_installed": True,
            "deps_version": CURRENT_DEPS_VERSION,
            "semantic_deps_installed": semantic_deps_installed,
            "installed_at": datetime.now().isoformat()
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        log("Dependencies installed successfully")

    # Activate venv dependencies (add to sys.path)
    # No re-exec needed - mira runs from global install, deps from venv
    activate_venv_deps()

    return False  # No re-exec needed
