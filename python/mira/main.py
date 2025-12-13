"""
MIRA3 Main Entry Point

Initializes all components and runs the main JSON-RPC loop.
Supports both central storage (Qdrant + Postgres) and local fallback (SQLite).

CLI Modes:
  - Default: Run MCP JSON-RPC server
  - --init: Run mira_init and output JSON (for SessionStart hooks)
"""

import sys
import json
import os
import signal
import fcntl
import threading
import argparse
from pathlib import Path

from .utils import log, get_mira_path
from .bootstrap import ensure_venv_and_deps, reexec_in_venv
from .config import get_config
from .storage import get_storage, Storage
from .handlers import handle_rpc_request, handle_init
from .watcher import run_file_watcher
from .ingestion import run_full_ingestion
from .migrations import ensure_schema_current, check_migrations_needed


# Global lock file handle - must stay open for duration of process
_lock_file = None


def acquire_singleton_lock(mira_path: Path) -> bool:
    """
    Acquire exclusive lock to ensure only one MIRA instance runs.

    Uses flock() which automatically releases on process death.
    If another instance holds the lock, kills it and takes over.

    Returns True if lock acquired, False on failure.
    """
    global _lock_file

    lock_path = mira_path / "mira.lock"
    pid_path = mira_path / "mira.pid"

    # Open lock file (create if doesn't exist)
    _lock_file = open(lock_path, "w")

    try:
        # Try non-blocking exclusive lock
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Got the lock - write our PID
        pid_path.write_text(str(os.getpid()))
        log(f"Acquired singleton lock (PID {os.getpid()})")
        return True

    except BlockingIOError:
        # Another process holds the lock - try to kill it
        log("Another MIRA instance detected, attempting takeover...")

        old_pid = None
        if pid_path.exists():
            try:
                old_pid = int(pid_path.read_text().strip())
            except (ValueError, OSError):
                pass

        if old_pid:
            try:
                # Send SIGTERM first (graceful)
                os.kill(old_pid, signal.SIGTERM)
                log(f"Sent SIGTERM to old MIRA process {old_pid}")

                # Wait briefly for it to die
                import time
                for _ in range(10):  # 1 second max
                    time.sleep(0.1)
                    try:
                        # Check if still alive (signal 0 = just check)
                        os.kill(old_pid, 0)
                    except ProcessLookupError:
                        break  # Dead
                else:
                    # Still alive - force kill
                    try:
                        os.kill(old_pid, signal.SIGKILL)
                        log(f"Sent SIGKILL to stubborn process {old_pid}")
                    except ProcessLookupError:
                        pass

            except ProcessLookupError:
                log(f"Old process {old_pid} already dead")
            except PermissionError:
                log(f"Cannot kill process {old_pid} - permission denied")
                return False

        # Try to acquire lock again
        try:
            fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pid_path.write_text(str(os.getpid()))
            log(f"Acquired singleton lock after takeover (PID {os.getpid()})")
            return True
        except BlockingIOError:
            log("Failed to acquire lock even after kill attempt")
            return False


def send_notification(method: str, params: dict):
    """Send a JSON-RPC notification (no id, no response expected)."""
    notification = {"jsonrpc": "2.0", "method": method, "params": params}
    print(json.dumps(notification), flush=True)


def run_backend():
    """Main backend loop - handles JSON-RPC requests."""

    # Initialize storage paths
    mira_path = get_mira_path()
    archives_path = mira_path / "archives"
    metadata_path = mira_path / "metadata"

    # Ensure directories exist
    for path in [archives_path, metadata_path]:
        path.mkdir(parents=True, exist_ok=True)

    # Acquire singleton lock - kills any existing MIRA instance
    if not acquire_singleton_lock(mira_path):
        log("FATAL: Could not acquire singleton lock")
        sys.exit(1)

    # Run schema migrations before anything else
    log("Checking database schema...")
    try:
        ensure_schema_current()
    except Exception as e:
        log(f"Schema migration warning: {e}")

    # Load configuration and initialize storage
    config = get_config()
    log("Initializing storage...")
    storage = get_storage()

    # Health check and mode reporting
    health = storage.health_check()
    if storage.using_central:
        log(f"Storage: CENTRAL (Qdrant={health['qdrant_healthy']}, Postgres={health['postgres_healthy']})")

        # Check embedding service health
        from .embedding_client import get_embedding_client
        embed_client = get_embedding_client()
        if embed_client:
            embed_health = embed_client.health_check()
            if embed_health.get("status") == "healthy":
                log(f"Embedding service: HEALTHY (model={embed_health.get('model', 'unknown')})")
            else:
                log(f"Embedding service: UNHEALTHY ({embed_health.get('error', 'unknown error')})")
        else:
            log("Embedding service: NOT CONFIGURED")
    else:
        log("Storage: LOCAL (SQLite with FTS - keyword search only)")
        log("To enable semantic search, configure ~/.mira/server.json")

    # Run initial ingestion in background with watchdog
    ingestion_thread = [None]  # Use list to allow mutation in nested functions
    ingestion_complete = [False]
    last_ingestion_stats = [None]

    def initial_ingestion():
        try:
            # Pass storage explicitly (collection param is deprecated/ignored)
            stats = run_full_ingestion(collection=None, mira_path=mira_path, storage=storage)
            last_ingestion_stats[0] = stats
            log(f"Initial ingestion: {stats['ingested']} new conversations indexed")
        except Exception as e:
            log(f"Initial ingestion error: {e}")
        finally:
            ingestion_complete[0] = True

    ingestion_thread[0] = threading.Thread(target=initial_ingestion, daemon=True, name="InitialIngestion")
    ingestion_thread[0].start()

    # Send ready signal to Node.js
    send_notification("ready", {})

    # Start file watcher in background thread
    # Pass None for collection (deprecated), storage for central Qdrant + Postgres
    watcher_thread = threading.Thread(
        target=run_file_watcher,
        kwargs={"collection": None, "mira_path": mira_path, "storage": storage},
        daemon=True
    )
    watcher_thread.start()

    # Start sync worker - flushes local queue to central storage
    sync_worker = None
    try:
        from .sync_worker import start_sync_worker, stop_sync_worker
        sync_worker = start_sync_worker(storage)
    except Exception as e:
        log(f"Failed to start sync worker: {e}")

    # Watchdog: restart ingestion if stuck (pending > 0, no active jobs, thread dead)
    def ingestion_watchdog():
        import time
        from .ingestion import get_active_ingestions, discover_conversations

        WATCHDOG_INTERVAL = 60  # Check every 60 seconds
        RESTART_DELAY = 30      # Wait 30s after thread dies before restarting

        while True:
            time.sleep(WATCHDOG_INTERVAL)

            try:
                # Skip if ingestion is still running
                if ingestion_thread[0] and ingestion_thread[0].is_alive():
                    continue

                # Skip if there are active ingestions (watcher might be processing)
                active = get_active_ingestions()
                if active:
                    continue

                # Check if there are pending files
                conversations = discover_conversations()
                if not conversations:
                    continue

                # Count how many are actually pending (not in central)
                pending_count = 0
                for conv in conversations:
                    session_id = conv['session_id']
                    meta_file = mira_path / "metadata" / f"{session_id}.json"

                    if not meta_file.exists():
                        # No metadata = needs processing (might have no messages, but worth checking)
                        pending_count += 1
                    elif storage.using_central:
                        # Has metadata but check if in central
                        if not storage.session_exists_in_central(session_id):
                            pending_count += 1

                if pending_count > 0:
                    log(f"[Watchdog] Detected {pending_count} pending sessions, ingestion thread dead. Restarting...")
                    time.sleep(RESTART_DELAY)

                    # Double-check nothing started in the meantime
                    if not get_active_ingestions():
                        ingestion_complete[0] = False
                        ingestion_thread[0] = threading.Thread(
                            target=initial_ingestion,
                            daemon=True,
                            name="WatchdogIngestion"
                        )
                        ingestion_thread[0].start()
                        log("[Watchdog] Ingestion restarted")

            except Exception as e:
                log(f"[Watchdog] Error: {e}")

    watchdog_thread = threading.Thread(target=ingestion_watchdog, daemon=True, name="IngestionWatchdog")
    watchdog_thread.start()
    log("Ingestion watchdog started")

    # Main JSON-RPC loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = None
        try:
            request = json.loads(line)
            response = handle_rpc_request(request, None, storage)
            print(json.dumps(response), flush=True)

            # Handle shutdown
            if request.get("method") == "shutdown":
                if sync_worker:
                    stop_sync_worker()  # Use module function for clean shutdown
                storage.close()
                break

        except json.JSONDecodeError as e:
            log(f"Invalid JSON: {e}")
        except Exception as e:
            log(f"Error handling request: {e}")
            if request is not None:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {"code": -32603, "message": "Internal error processing request"}
                }
                print(json.dumps(error_response), flush=True)


def _format_mira_context(result: dict) -> str:
    """
    Format MIRA init result as a readable string for Claude Code hooks.

    Claude Code injects additionalContext as a string, so we need to format
    the structured data into a readable format that Claude can understand.
    """
    lines = []

    # Header
    lines.append("=== MIRA Session Context ===")
    lines.append("")

    # Guidance section (most important)
    guidance = result.get("guidance", {})
    if guidance:
        lines.append("## How to Use This Context")
        lines.append(guidance.get("how_to_use_this", ""))
        lines.append("")

        # Actions (immediate priorities)
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

    # Alerts section
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
        # Custodian (user profile)
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

        # Current work context
        work = core.get("current_work", {})
        if work:
            recent_topics = work.get("recent_topics", [])
            active_tasks = work.get("active_tasks", [])
            if recent_topics or active_tasks:
                lines.append("## Current Work Context")
                if recent_topics:
                    lines.append("Recent Topics:")
                    for topic in recent_topics[:5]:
                        lines.append(f"  - {topic}")
                if active_tasks:
                    lines.append("Active Tasks:")
                    for task in active_tasks[:5]:
                        lines.append(f"  - {task}")
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


def run_init_cli(project_path: str, quiet: bool = False, raw: bool = False):
    """
    Run mira_init directly and output JSON.

    This is a lightweight mode that:
    1. Bootstraps venv if needed (already done by main())
    2. Initializes storage (read-only is fine)
    3. Runs schema migrations
    4. Calls handle_init
    5. Outputs JSON to stdout (wrapped in hookSpecificOutput format)
    6. Exits

    Designed for use in SessionStart hooks.
    Note: MIRA_QUIET is set in main() before this is called.

    Args:
        project_path: The project path for context
        quiet: Suppress log output
        raw: Output raw JSON instead of hookSpecificOutput format
    """
    try:
        # Ensure directories exist
        mira_path = get_mira_path()
        archives_path = mira_path / "archives"
        metadata_path = mira_path / "metadata"
        for path in [archives_path, metadata_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Run schema migrations (lightweight, needed for custodian data)
        try:
            ensure_schema_current()
        except Exception as e:
            log(f"Schema migration warning: {e}")

        # Initialize storage (lightweight, read-only is fine)
        storage = get_storage()

        # Call handle_init
        result = handle_init({'project_path': project_path}, None, storage)

        # Output based on mode
        if raw:
            # Raw mode: output plain JSON (for debugging/testing)
            print(json.dumps(result, indent=2))
        else:
            # Hook mode: wrap in hookSpecificOutput for Claude Code
            formatted_context = _format_mira_context(result)
            hook_output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": formatted_context
                }
            }
            print(json.dumps(hook_output))

        # Clean exit
        try:
            storage.close()
        except Exception:
            pass

        sys.exit(0)

    except Exception as e:
        # Output error in hookSpecificOutput format
        error_context = f"""=== MIRA Session Context ===

## Error
MIRA initialization failed: {str(e)}

## Fallback Guidance
MIRA context is unavailable for this session. You can still use MIRA tools manually if the MCP server is running.
"""
        if raw:
            error_response = {
                "error": str(e),
                "guidance": {
                    "how_to_use_this": "MIRA init failed - proceeding without context",
                    "mira_usage_triggers": [],
                    "tool_quick_reference": {},
                    "actions": ["MIRA context unavailable - check server logs"]
                }
            }
            print(json.dumps(error_response, indent=2))
        else:
            hook_output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": error_context
                }
            }
            print(json.dumps(hook_output))
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MIRA3 - Memory Information Retriever and Archiver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m mira                    Run MCP JSON-RPC server (default)
  python -m mira --init             Run mira_init and output JSON
  python -m mira --init --project=/workspaces/MyProject --quiet
"""
    )

    parser.add_argument(
        '--init',
        action='store_true',
        help='Run mira_init and output JSON (for SessionStart hooks)'
    )

    parser.add_argument(
        '--project',
        type=str,
        default='',
        help='Project path for init context'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress log output (JSON only)'
    )

    parser.add_argument(
        '--raw',
        action='store_true',
        help='Output raw JSON instead of hookSpecificOutput format'
    )

    return parser.parse_args()


def main():
    """Main entry point with bootstrap."""
    # Parse arguments first (before bootstrap might re-exec)
    args = parse_args()

    # Set quiet mode early, before any logging
    if args.quiet:
        os.environ['MIRA_QUIET'] = '1'

    try:
        # Bootstrap: ensure venv and deps
        if ensure_venv_and_deps():
            # Need to re-exec in venv - preserve args
            log("Re-executing in virtualenv...")
            reexec_in_venv()

        # Now running in venv with all deps available

        if args.init:
            # Init mode: run mira_init and output JSON
            run_init_cli(args.project, quiet=args.quiet, raw=args.raw)
        else:
            # Default: run MCP JSON-RPC server
            run_backend()

    except KeyboardInterrupt:
        log("Shutting down...")
    except Exception as e:
        log(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
