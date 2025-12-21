"""
MIRA Search Tool

Semantic and keyword search across conversation history.
Re-exports the main search handler from mira.search.core.
"""

from mira.search.core import handle_search

__all__ = ["handle_search"]
