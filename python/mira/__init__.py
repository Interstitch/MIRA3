"""
MIRA3 Python Backend Package

Modular backend for central Qdrant + Postgres storage, conversation ingestion, and search.
"""

import json
from pathlib import Path


def _get_version() -> str:
    """Read version from package.json (single source of truth)."""
    try:
        # Look for package.json relative to this file
        pkg_path = Path(__file__).parent.parent.parent / "package.json"
        if pkg_path.exists():
            with open(pkg_path) as f:
                return json.load(f).get("version", "0.0.0")
    except Exception:
        pass
    return "0.0.0"


__version__ = _get_version()
