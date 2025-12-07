#!/usr/bin/env python3
"""
MIRA3 Python Backend Entry Point

This is the main entry point called by the Node.js MCP server.
All functionality is now in the mira/ package.
"""

import sys
import os

# Add the python directory to the path so we can import the mira package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mira.main import main

if __name__ == "__main__":
    main()
