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

from .constants import DEPENDENCIES, DEPENDENCIES_SEMANTIC
from .utils import get_venv_path, get_venv_python, get_venv_pip, log
from .constants import get_mira_path


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
    """Get path to uv binary in venv."""
    return str(venv_path / "bin" / "uv")


def _install_with_uv(venv_path: Path, deps: list, optional: bool = False) -> bool:
    """
    Install dependencies using uv pip.

    Returns True on success, False on failure.
    """
    uv = _get_uv_path(venv_path)
    python = str(venv_path / "bin" / "python")

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

    Strategy:
    1. Create venv with python -m venv (built-in)
    2. Install uv first via pip
    3. Use uv for remaining deps (fast)
    """
    mira_path = get_mira_path()
    venv_path = get_venv_path()
    config_path = mira_path / "config.json"

    # Create .mira directory if needed
    mira_path.mkdir(parents=True, exist_ok=True)

    # Check if venv exists and is set up
    deps_installed = False
    deps_version = 0

    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
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
    CURRENT_DEPS_VERSION = 7

    # Force reinstall if deps version is outdated
    if deps_version < CURRENT_DEPS_VERSION:
        deps_installed = False
        log(f"Dependency version outdated ({deps_version} < {CURRENT_DEPS_VERSION}), will reinstall")

    if not venv_path.exists():
        # Find Python with sqlite extension support for sqlite-vec
        python_to_use = sys.executable
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

        if _install_with_uv(venv_path, DEPENDENCIES):
            # Try optional semantic deps
            semantic_deps_installed = _install_with_uv(
                venv_path, DEPENDENCIES_SEMANTIC, optional=True
            )
            if semantic_deps_installed:
                log("Semantic search dependencies installed")
            else:
                log("Local semantic search unavailable - using keyword search")

        # Verify server.json exists (required for remote storage)
        server_config_path = mira_path / "server.json"
        if not server_config_path.exists():
            log("Note: server.json not found - running in local-only mode")

        # Mark as installed with version
        config = {
            "deps_installed": True,
            "deps_version": CURRENT_DEPS_VERSION,
            "semantic_deps_installed": semantic_deps_installed,
            "installed_at": datetime.now().isoformat()
        }
        config_path.write_text(json.dumps(config, indent=2))
        log("Dependencies installed successfully")

    # Check if we need to re-exec in the venv
    if not is_running_in_venv():
        return True

    return False


def reexec_in_venv():
    """Re-execute this script inside the virtualenv."""
    venv_python = get_venv_python()

    # Build command: always use -m mira to ensure proper module execution
    # sys.argv[0] might be a file path, so we replace it with module form
    # Keep all other args (--init, --raw, etc.)
    cmd = [venv_python, "-m", "mira"] + sys.argv[1:]

    # Preserve environment (including PYTHONPATH for dev mode)
    result = subprocess.run(
        cmd,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ.copy()
    )
    sys.exit(result.returncode)
