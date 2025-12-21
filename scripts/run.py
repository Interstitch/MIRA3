#!/usr/bin/env python3
"""
MIRA MCP Server Runner

Entry point for running the MIRA MCP server.
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mira.server import main

if __name__ == "__main__":
    main()
