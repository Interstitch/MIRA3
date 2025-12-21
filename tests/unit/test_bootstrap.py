"""Tests for mira.core.bootstrap functionality."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.core import is_running_in_venv


class TestBootstrap:
    """Test bootstrap functionality."""

    def test_is_running_in_venv(self):
        # This should return False since we're not running in .mira/.venv
        result = is_running_in_venv()
        assert isinstance(result, bool)
