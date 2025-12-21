"""
MIRA MCP Server

Pure Python MCP server using the official Python MCP SDK.
Provides 7 tools for Claude Code session memory and context.
"""

import asyncio
import signal
import sys
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Optional

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel.server import NotificationOptions

from mira.core import (
    VERSION,
    get_mira_path,
    log,
    get_config,
    shutdown_db_manager,
)
# WINDOWS FIX: Set MCP mode to skip git subprocess calls that interfere with stdio
os.environ['MIRA_MCP_MODE'] = '1'

# Global state
_storage = None
_initialized = False


def _ensure_directories():
    """Ensure required directories exist."""
    mira_path = get_mira_path()
    archives_path = mira_path / "archives"
    metadata_path = mira_path / "metadata"

    for path in [mira_path, archives_path, metadata_path]:
        path.mkdir(parents=True, exist_ok=True)


def _initialize():
    """Initialize MIRA components (called once on first tool use)."""
    global _storage, _initialized

    if _initialized:
        return

    log("Initializing MIRA components...")
    _ensure_directories()

    # Run schema migrations
    try:
        from mira.storage.migrations import ensure_schema_current
        ensure_schema_current()
    except Exception as e:
        log(f"Schema migration warning: {e}")

    # Initialize storage (lazy - don't block on central connection)
    try:
        from mira.storage import get_storage
        _storage = get_storage()
        # Note: Central connection is lazy - checked on first use, not here
        # This prevents cold-start timeout when central is slow/unreachable
        log("Storage: initialized (mode determined on first use)")
    except Exception as e:
        log(f"Storage init error: {e}")
        _storage = None

    # Start background workers
    _start_background_workers()

    _initialized = True
    log("MIRA initialization complete")


def _start_background_workers():
    """Start background ingestion and sync workers."""
    global _storage

    mira_path = get_mira_path()

    # Initial ingestion
    def run_ingestion():
        try:
            from mira.ingestion import run_full_ingestion
            stats = run_full_ingestion(collection=None, mira_path=mira_path, storage=_storage)
            log(f"Initial ingestion: {stats.get('ingested', 0)} new conversations indexed")
        except Exception as e:
            log(f"Initial ingestion error: {e}")

    threading.Thread(target=run_ingestion, daemon=True, name="InitialIngestion").start()

    # File watcher
    def run_watcher():
        try:
            from mira.ingestion.watcher import run_file_watcher
            run_file_watcher(collection=None, mira_path=mira_path, storage=_storage)
        except Exception as e:
            log(f"File watcher error: {e}")

    threading.Thread(target=run_watcher, daemon=True, name="FileWatcher").start()

    # Sync worker (if central storage)
    if _storage and _storage.using_central:
        try:
            from mira.storage.sync import start_sync_worker
            start_sync_worker(_storage)
        except Exception as e:
            log(f"Sync worker error: {e}")

    # Local semantic indexer
    try:
        from mira.search.local_semantic import start_local_indexer
        start_local_indexer()
    except Exception as e:
        log(f"Local indexer error: {e}")


# Create the MCP server
server = Server("mira")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available MIRA tools."""
    return [
        types.Tool(
            name="mira_init",
            description="Initialize MIRA context for this session. Returns user profile, recent work, and usage guidance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Current project path"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="mira_search",
            description="Search past Claude Code conversations. Returns relevant sessions and artifacts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="mira_recent",
            description="Get summaries of recent conversation sessions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum sessions (default 5)",
                        "default": 5
                    },
                    "project_only": {
                        "type": "boolean",
                        "description": "Filter to current project",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="mira_error_lookup",
            description="Search past error patterns and their solutions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Error message or pattern to search"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="mira_decisions",
            description="Search architectural and design decisions with reasoning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Decision topic to search"
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (architecture, technology, etc.)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 10)",
                        "default": 10
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="mira_code_history",
            description="Get the history of changes to a file across sessions. Supports three modes: timeline (default), snapshot (reconstruct file at date), and changes (list of edits).",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path or pattern (e.g., 'handlers.py', 'src/*.py')"
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Function or class name to search (alternative to path)"
                    },
                    "mode": {
                        "type": "string",
                        "description": "Query mode: 'timeline' (default), 'snapshot', or 'changes'",
                        "enum": ["timeline", "snapshot", "changes"],
                        "default": "timeline"
                    },
                    "date": {
                        "type": "string",
                        "description": "Target date for snapshot mode (ISO format, e.g., '2025-12-01')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="mira_status",
            description="Get MIRA system status, health, and statistics.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls."""
    _initialize()

    try:
        result = await _dispatch_tool(name, arguments)
        import json
        response = [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        log(f"Tool {name} error: {e}")
        import traceback
        response = [types.TextContent(type="text", text=f"Error: {str(e)}\n{traceback.format_exc()}")]

    # WINDOWS FIX: Flush stdout after tool responses
    # The MCP SDK's stdio transport can lose responses in Windows buffering.
    # Flushing ensures the response is written before the SDK returns.
    if sys.platform == 'win32':
        try:
            sys.stdout.flush()
        except Exception:
            pass

    return response


async def _dispatch_tool(name: str, arguments: dict[str, Any]) -> dict:
    """Dispatch to the appropriate tool handler."""
    global _storage

    # Import handlers lazily to avoid circular imports
    from mira.tools import (
        handle_init,
        handle_search,
        handle_recent,
        handle_error_lookup,
        handle_decisions,
        handle_code_history,
        handle_status,
    )

    handlers = {
        "mira_init": handle_init,
        "mira_search": handle_search,
        "mira_recent": handle_recent,
        "mira_error_lookup": handle_error_lookup,
        "mira_decisions": handle_decisions,
        "mira_code_history": handle_code_history,
        "mira_status": handle_status,
    }

    handler = handlers.get(name)
    if not handler:
        return {"error": f"Unknown tool: {name}"}

    # Call the handler (they're sync functions, but we can await if needed)
    return handler(arguments, _storage)


def _shutdown():
    """
    Clean shutdown with graceful worker termination.

    Shutdown order matters:
    1. Signal all workers to stop (they'll drain current operations)
    2. Wait for workers to finish
    3. Close storage connections
    4. Shutdown database manager (waits for write queue to drain)
    """
    global _storage

    log("MIRA shutting down...")

    # Stop file watcher first (signals shutdown event)
    try:
        from mira.ingestion.watcher import stop_file_watcher
        stop_file_watcher()
    except Exception as e:
        log(f"File watcher stop error: {e}")

    # Stop sync worker
    try:
        from mira.storage.sync import stop_sync_worker
        stop_sync_worker()
    except Exception as e:
        log(f"Sync worker stop error: {e}")

    # Stop local indexer
    try:
        from mira.search.local_semantic import stop_local_indexer
        stop_local_indexer()
    except Exception as e:
        log(f"Local indexer stop error: {e}")

    # Close storage connections
    if _storage:
        try:
            _storage.close()
        except Exception as e:
            log(f"Storage close error: {e}")

    # Shutdown database manager last (waits for write queue to drain)
    shutdown_db_manager()

    log("MIRA shutdown complete")


async def run_server():
    """Run the MCP server."""
    log(f"Starting MIRA MCP server v{VERSION}")

    # WINDOWS FIX: Force binary mode on stdio
    # The MCP SDK's stdio transport has buffering issues on Windows that cause
    # responses to be lost. Binary mode prevents newline translation issues.
    if sys.platform == 'win32':
        try:
            import msvcrt
            msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
            msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
            log("Windows: stdio set to binary mode")
        except Exception as e:
            log(f"Windows binary mode warning: {e}")

    # Track if shutdown was requested via signal
    shutdown_requested = threading.Event()

    # Set up signal handlers
    def signal_handler(signum, frame):
        log(f"Received signal {signum}, initiating graceful shutdown...")
        shutdown_requested.set()
        # Don't call sys.exit() - let the server complete its current operation
        # and shutdown gracefully in the finally block

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Run using stdio transport
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mira",
                    server_version=VERSION,
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    finally:
        _shutdown()


def run_init_cli(project_path: str, quiet: bool = False, raw: bool = False):
    """
    Run mira_init directly and output JSON.

    This is a lightweight mode for SessionStart hooks.
    Note: Bootstrap already handled by __main__.py before this is called.
    """
    import json

    try:
        _ensure_directories()

        # Run schema migrations
        try:
            from mira.storage.migrations import ensure_schema_current
            ensure_schema_current()
        except Exception as e:
            log(f"Schema migration warning: {e}")

        # Initialize storage
        storage = None
        try:
            from mira.storage import get_storage
            storage = get_storage()
        except Exception as e:
            log(f"Storage init error: {e}")

        # Call handle_init
        from mira.tools import handle_init
        result = handle_init({'project_path': project_path}, storage)

        # Output based on mode
        if raw:
            print(json.dumps(result, indent=2, default=str))
        else:
            # Format for Claude Code hook
            formatted_context = _format_mira_context(result)
            hook_output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": formatted_context
                }
            }
            print(json.dumps(hook_output))

        # WINDOWS FIX: Flush stdout to ensure hook output is received
        sys.stdout.flush()

        # Clean exit
        if storage:
            try:
                storage.close()
            except Exception:
                pass

        sys.exit(0)

    except Exception as e:
        import json
        error_context = f"MIRA initialization failed: {str(e)}"
        if raw:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            hook_output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": error_context
                }
            }
            print(json.dumps(hook_output))
        # WINDOWS FIX: Flush stdout to ensure error output is received
        sys.stdout.flush()
        sys.exit(1)


def _format_mira_context(result: dict) -> str:
    """Format MIRA init result as a readable string for Claude Code hooks."""
    lines = []

    lines.append("=== MIRA Session Context ===")
    lines.append("")

    # Guidance section
    guidance = result.get("guidance", {})
    if guidance:
        lines.append("## How to Use This Context")
        lines.append(guidance.get("how_to_use_this", ""))
        lines.append("")

        # Actions
        actions = guidance.get("actions", [])
        if actions:
            lines.append("### Immediate Actions")
            for action in actions:
                lines.append(f"- {action}")
            lines.append("")

        # Usage triggers
        triggers = guidance.get("mira_usage_triggers", [])
        if triggers:
            lines.append("### When to Consult MIRA")
            for trigger in triggers:
                priority = trigger.get("priority", "optional")
                situation = trigger.get("situation", "")
                action = trigger.get("action", "")
                reason = trigger.get("reason", "")
                lines.append(f"- [{priority.upper()}] {situation}")
                lines.append(f"  Action: {action}")
                if reason:
                    lines.append(f"  Reason: {reason}")
            lines.append("")

        # Tool quick reference
        tools = guidance.get("tool_quick_reference", {})
        if tools:
            lines.append("### MIRA Tools Quick Reference")
            for tool_name, tool_info in tools.items():
                purpose = tool_info.get("purpose", "")
                when = tool_info.get("when", "")
                lines.append(f"- {tool_name}: {purpose}")
                lines.append(f"  Use when: {when}")
            lines.append("")

    # Alerts
    alerts = result.get("alerts", [])
    if alerts:
        lines.append("## Alerts")
        for alert in alerts:
            priority = alert.get("priority", "info")
            message = alert.get("message", "")
            suggestion = alert.get("suggestion", "")
            lines.append(f"- [{priority.upper()}] {message}")
            if suggestion:
                lines.append(f"  Suggestion: {suggestion}")
        lines.append("")

    # Core context
    core = result.get("core", {})
    if core:
        custodian = core.get("custodian", {})
        if custodian:
            lines.append("## User Profile (Custodian)")
            name = custodian.get("name", "Unknown")
            lines.append(f"Name: {name}")
            summary = custodian.get("summary", "")
            if summary:
                lines.append(f"Summary: {summary}")
            lifecycle = custodian.get("development_lifecycle", "")
            if lifecycle:
                lines.append(f"Development Lifecycle: {lifecycle}")
            tips = custodian.get("interaction_tips", [])
            if tips:
                lines.append("Interaction Tips:")
                for tip in tips[:5]:
                    lines.append(f"  - {tip}")
            danger_zones = custodian.get("danger_zones", [])
            if danger_zones:
                lines.append("Danger Zones (proceed carefully):")
                for zone in danger_zones:
                    lines.append(f"  - {zone}")
            lines.append("")

    # Storage mode
    storage_info = result.get("storage", {})
    if storage_info:
        mode = storage_info.get("mode", "unknown")
        description = storage_info.get("description", "")
        lines.append(f"## Storage Mode: {mode}")
        if description:
            lines.append(description)
        lines.append("")

    # Indexing status
    indexing = result.get("indexing", {})
    if indexing:
        indexed = indexing.get("indexed", 0)
        total = indexing.get("total", 0)
        pending = indexing.get("pending", 0)
        lines.append(f"## Indexing Status: {indexed}/{total} sessions ({pending} pending)")
        lines.append("")

    return "\n".join(lines)


def main():
    """
    Main entry point for MCP server.

    Note: Bootstrap already handled by __main__.py before this is called.
    """
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        # Shutdown already handled in run_server's finally block
        log("Interrupted")
    except Exception as e:
        log(f"Fatal error: {e}")
        _shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
