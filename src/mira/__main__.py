"""
MIRA entry point.

Usage:
    python -m mira              # Start MCP server
    python -m mira --install    # Configure Claude Code to use MIRA
    python -m mira --init       # Run mira_init and output JSON (for hooks)
    python -m mira --version
    python -m mira --help
"""

import os
import sys
import argparse

from . import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="mira",
        description="MIRA - Persistent memory for Claude Code"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"mira {__version__}"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress startup messages"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Configure Claude Code to use MIRA (MCP server + SessionStart hook)"
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove MIRA from Claude Code configuration"
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Run mira_init and output JSON (for hooks)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="",
        help="Project path for init context"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw JSON instead of hookSpecificOutput format"
    )

    args = parser.parse_args()

    # Handle install/uninstall before bootstrap (no deps needed)
    if args.install:
        from .install import install
        install()
        return

    if args.uninstall:
        from .install import uninstall
        uninstall()
        return

    # Set quiet mode early
    if args.quiet:
        os.environ['MIRA_QUIET'] = '1'

    # Bootstrap: ensure venv deps are installed and activated BEFORE importing server
    # (server.py has top-level imports that require installed deps)
    # Note: mira runs from global install, only dependencies come from venv
    from .core.bootstrap import ensure_venv_and_deps
    ensure_venv_and_deps()

    # Now safe to import server (deps are installed)
    if args.init:
        # Init mode: run mira_init and output JSON for hooks
        from .server import run_init_cli
        run_init_cli(args.project, quiet=args.quiet, raw=args.raw)
    else:
        # Default: run MCP server
        from .server import main as server_main
        server_main()


if __name__ == "__main__":
    main()
