"""
Pytest configuration and shared fixtures for MIRA tests.

This file is automatically loaded by pytest and provides:
- Path setup for importing mira modules
- Shared fixtures for temp directories and database cleanup
- Common test utilities
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

import pytest

# Add src directory to path for all test modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test use."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def mira_temp_path(temp_dir):
    """Set up a temporary MIRA_PATH and clean up after."""
    old_path = os.environ.get("MIRA_PATH")
    os.environ["MIRA_PATH"] = temp_dir

    yield Path(temp_dir)

    if old_path:
        os.environ["MIRA_PATH"] = old_path
    else:
        os.environ.pop("MIRA_PATH", None)

    # Clean up db_manager
    from mira.core import shutdown_db_manager
    shutdown_db_manager()


@pytest.fixture
def archives_path(mira_temp_path):
    """Create archives directory in temp MIRA path."""
    path = mira_temp_path / "archives"
    path.mkdir(parents=True)
    return path


@pytest.fixture
def metadata_path(mira_temp_path):
    """Create metadata directory in temp MIRA path."""
    path = mira_temp_path / "metadata"
    path.mkdir(parents=True)
    return path


def cleanup_db_manager():
    """Helper to clean up db_manager state between tests."""
    from mira.core import shutdown_db_manager
    shutdown_db_manager()
