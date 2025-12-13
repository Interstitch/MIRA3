"""
MIRA3 Utility Functions

Core utilities for paths, logging, and common operations.
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_mira_path() -> Path:
    """Get the .mira storage directory path."""
    return Path.cwd() / ".mira"


def get_venv_path() -> Path:
    """Get the virtualenv path."""
    return get_mira_path() / ".venv"


def get_venv_python() -> str:
    """Get the Python executable inside the virtualenv."""
    venv = get_venv_path()
    if sys.platform == "win32":
        return str(venv / "Scripts" / "python.exe")
    return str(venv / "bin" / "python")


def get_venv_pip() -> str:
    """Get the pip executable inside the virtualenv."""
    venv = get_venv_path()
    if sys.platform == "win32":
        return str(venv / "Scripts" / "pip.exe")
    return str(venv / "bin" / "pip")


def get_mira_config() -> dict:
    """
    Load MIRA config.json settings.

    Returns dict with optional keys:
    - project_path: Restrict MIRA to only index this project path
    """
    config_path = get_mira_path() / "config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_project_filter() -> Optional[str]:
    """
    Get the project_path filter from config, if set.

    When set, MIRA only indexes conversations from this project.
    Returns the encoded project path (e.g., "-workspaces-MIRA3") or None.
    """
    config = get_mira_config()
    project_path = config.get("project_path")
    if not project_path:
        return None

    # Convert filesystem path to Claude's encoded format
    # /workspaces/MIRA3 -> -workspaces-MIRA3
    encoded = project_path.replace("/", "-").lstrip("-")
    return encoded


def get_claude_projects_path() -> Path:
    """Get the path where Claude Code stores conversations."""
    return Path.home() / ".claude" / "projects"


def log(message: str):
    """Log a message to stderr and to a log file for monitoring.

    Respects MIRA_QUIET environment variable - when set, suppresses stderr output
    but still writes to log file.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[MIRA {timestamp}] {message}"

    # Only print to stderr if not in quiet mode
    if not os.environ.get('MIRA_QUIET'):
        print(formatted, file=sys.stderr, flush=True)

    # Always write to log file for debugging
    try:
        log_path = get_mira_path() / "mira.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{formatted}\n")
    except Exception:
        pass  # Don't fail if logging fails


def parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime object."""
    if not ts:
        return None
    try:
        # Handle ISO format: "2025-12-07T04:45:36.800Z"
        ts = ts.rstrip('Z')
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def extract_text_content(message: dict) -> str:
    """Extract text content from a message object."""
    if not isinstance(message, dict):
        return ""

    content = message.get('content', '')

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                texts.append(item.get('text', ''))
            elif isinstance(item, str):
                texts.append(item)
        return '\n'.join(texts)

    return ""


# Cached custodian name
_custodian_cache: Optional[str] = None


def get_custodian() -> str:
    """Try to discover the user's name (cached)."""
    global _custodian_cache
    if _custodian_cache is not None:
        return _custodian_cache

    # Try git config
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            _custodian_cache = result.stdout.strip()
            return _custodian_cache
    except (subprocess.SubprocessError, OSError, TimeoutError):
        pass

    # Try environment
    for var in ["USER", "USERNAME", "LOGNAME"]:
        if var in os.environ:
            _custodian_cache = os.environ[var]
            return _custodian_cache

    _custodian_cache = "Unknown"
    return _custodian_cache


import re

# Pre-compiled pattern for extracting query terms (alphanumeric, 3+ chars)
_QUERY_TERM_PATTERN = re.compile(r'\b[a-zA-Z0-9]{3,}\b')


def extract_query_terms(query: str, max_terms: int = 10) -> list:
    """
    Extract search terms from a query string.

    Returns lowercase alphanumeric terms with 3+ characters.
    """
    if not query:
        return []
    terms = _QUERY_TERM_PATTERN.findall(query.lower())
    return terms[:max_terms]


# Cached git remote lookups
_git_remote_cache: dict = {}


def get_git_remote(project_path: str) -> Optional[str]:
    """
    Get the git remote URL for a project path.

    This is the canonical identifier for cross-machine project matching.
    Returns None if not a git repo or no remote configured.

    The remote URL is normalized to handle SSH vs HTTPS variants:
    - git@github.com:user/repo.git -> github.com/user/repo
    - https://github.com/user/repo.git -> github.com/user/repo
    """
    # Check cache first
    if project_path in _git_remote_cache:
        return _git_remote_cache[project_path]

    result = None
    try:
        # Run git remote get-url origin from the project directory
        proc = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project_path
        )
        if proc.returncode == 0 and proc.stdout.strip():
            result = normalize_git_remote(proc.stdout.strip())
    except (subprocess.SubprocessError, OSError, TimeoutError, FileNotFoundError):
        pass

    # Cache the result (even None, to avoid repeated lookups)
    _git_remote_cache[project_path] = result
    return result


def normalize_git_remote(url: str) -> str:
    """
    Normalize a git remote URL to a canonical form.

    Converts SSH and HTTPS URLs to a common format:
    - git@github.com:user/repo.git -> github.com/user/repo
    - https://github.com/user/repo.git -> github.com/user/repo
    - ssh://git@github.com/user/repo.git -> github.com/user/repo

    This allows matching the same repo regardless of clone method.
    """
    if not url:
        return url

    # Remove trailing .git
    if url.endswith('.git'):
        url = url[:-4]

    # Handle SSH format: git@github.com:user/repo
    if url.startswith('git@'):
        # git@github.com:user/repo -> github.com/user/repo
        url = url[4:]  # Remove git@
        url = url.replace(':', '/', 1)  # Replace first : with /
        return url

    # Handle ssh:// format: ssh://git@github.com/user/repo
    if url.startswith('ssh://'):
        url = url[6:]  # Remove ssh://
        if url.startswith('git@'):
            url = url[4:]  # Remove git@
        return url

    # Handle HTTPS format: https://github.com/user/repo
    if url.startswith('https://'):
        return url[8:]  # Remove https://

    # Handle HTTP format: http://github.com/user/repo
    if url.startswith('http://'):
        return url[7:]  # Remove http://

    return url


def get_git_remote_for_claude_path(encoded_path: str) -> Optional[str]:
    """
    Get git remote for a Claude Code encoded project path.

    Claude stores projects in ~/.claude/projects/{encoded-path}/
    where encoded-path is like "-workspaces-MIRA3" (dashes replace slashes).

    This function decodes the path and gets the git remote.
    """
    if not encoded_path:
        return None

    # Decode: "-workspaces-MIRA3" -> "/workspaces/MIRA3"
    # Handle leading dash (represents root /)
    if encoded_path.startswith('-'):
        decoded = '/' + encoded_path[1:].replace('-', '/')
    else:
        decoded = encoded_path.replace('-', '/')

    # Check if path exists
    if not os.path.isdir(decoded):
        return None

    return get_git_remote(decoded)
