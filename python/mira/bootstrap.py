"""
MIRA3 Bootstrap Module

Handles virtualenv creation, dependency installation, and re-execution.
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

from .utils import (
    get_mira_path, get_venv_path, get_venv_python, get_venv_pip, log
)
from .constants import DEPENDENCIES


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


def ensure_venv_and_deps() -> bool:
    """
    Ensure virtualenv exists and dependencies are installed.
    Returns True if we need to re-exec in the venv.
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
    CURRENT_DEPS_VERSION = 1

    # Force reinstall if deps version is outdated
    if deps_version < CURRENT_DEPS_VERSION:
        deps_installed = False
        log(f"Dependency version outdated ({deps_version} < {CURRENT_DEPS_VERSION}), will reinstall")

    if not venv_path.exists():
        log("Creating virtualenv at " + str(venv_path))
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            check=True,
            capture_output=True
        )
        deps_installed = False

    if not deps_installed:
        dep_list = ", ".join(DEPENDENCIES)
        log(f"Installing dependencies: {dep_list}")
        pip = get_venv_pip()

        # Upgrade pip first
        subprocess.run(
            [pip, "install", "--upgrade", "pip", "-q"],
            check=True,
            capture_output=True
        )

        # Install CPU-only PyTorch first to avoid 6GB+ of CUDA libraries
        # sentence-transformers will use this instead of downloading GPU version
        log("Installing CPU-only PyTorch (saves ~6GB vs GPU version)...")
        subprocess.run(
            [pip, "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu", "-q"],
            check=True,
            capture_output=True
        )

        # Install all dependencies (Qdrant + Postgres included by default)
        subprocess.run(
            [pip, "install"] + DEPENDENCIES + ["-q"],
            check=True,
            capture_output=True
        )

        # Verify server.json exists (required for remote storage)
        server_config_path = mira_path / "server.json"
        if not server_config_path.exists():
            log("WARNING: ~/.mira/server.json not found!")
            log("MIRA requires server.json to connect to remote Qdrant + Postgres.")
            log("See docs/CENTRAL_SETUP.md for configuration instructions.")

        # Mark as installed with version
        config = {
            "deps_installed": True,
            "deps_version": CURRENT_DEPS_VERSION,
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
    # Use subprocess instead of execv for better compatibility
    result = subprocess.run(
        [venv_python] + sys.argv,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    sys.exit(result.returncode)
