"""
MIRA - Memory Information Retriever and Archiver

Persistent memory for Claude Code: semantic search, cross-session context,
learned preferences, error tracking, and decision journaling.
"""

from ._version import __version__

__author__ = "Max"

from .core.config import get_config
from .core.constants import MIRA_PATH

__all__ = [
    "__version__",
    "get_config",
    "MIRA_PATH",
]
