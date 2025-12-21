"""
MIRA Installation Helper

Configures Claude Code to use MIRA as an MCP server.
Updates both ~/.claude.json and ~/.claude/settings.json.
"""

import json
import os
import sys
from pathlib import Path


def get_claude_config_paths():
    """Get all possible Claude Code config file paths."""
    home = Path.home()
    return [
        home / ".claude.json",
        home / ".claude" / "settings.json",
    ]


def get_mira_command():
    """Get the command to run MIRA."""
    # If installed via pip, use the entry point
    # Otherwise, use python -m mira
    import shutil

    mira_bin = shutil.which("mira")
    if mira_bin:
        return mira_bin, []
    else:
        return sys.executable, ["-m", "mira"]


def get_session_start_hook():
    """
    Get the SessionStart hook configuration for MIRA context injection.

    CRITICAL ON WINDOWS: Due to MCP SDK stdio transport bug, mira_init times out
    via MCP but works fine via CLI. This hook bypasses MCP entirely.

    Hook format requirements:
    - Must use nested {"hooks": [...]} structure
    - matcher is optional; if used, must be string not object
    """
    import platform

    if platform.system() == "Windows":
        # Windows: Use full python path, no shell redirects
        # The --quiet flag suppresses output, making error redirect unnecessary
        command = f'{sys.executable} -m mira --init --project="%CLAUDE_PROJECT_DIR%" --quiet'
    else:
        # Unix: Use shell features for graceful fallback
        command = 'python -m mira --init --project="$CLAUDE_PROJECT_DIR" --quiet 2>/dev/null || echo \'{"guidance":{"actions":["MIRA unavailable"]}}\''

    return {
        "hooks": [
            {
                "type": "command",
                "command": command,
                "timeout": 30000
            }
        ]
    }


def update_claude_config(config_path: Path, dry_run: bool = False) -> bool:
    """
    Update a Claude Code config file with MIRA MCP server and SessionStart hook.

    Returns True if the config was updated, False if skipped.
    """
    config = {}

    # Read existing config
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            pass

    # Ensure mcpServers exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Get MIRA command
    command, args = get_mira_command()

    # Build new MIRA config
    new_mira_config = {
        "command": command,
        "args": args,
    }

    # Check if update is needed
    existing = config["mcpServers"].get("mira")
    existing_mira3 = config["mcpServers"].get("mira3")

    needs_update = False

    # Check if existing config points to old Node.js version
    if existing_mira3:
        if existing_mira3.get("command") == "node" or "npx" in str(existing_mira3.get("args", [])):
            print(f"  Found old Node.js config in {config_path}, updating...")
            del config["mcpServers"]["mira3"]
            needs_update = True
        elif existing_mira3.get("command") != command or existing_mira3.get("args") != args:
            print(f"  Updating mira3 config in {config_path}...")
            del config["mcpServers"]["mira3"]
            needs_update = True

    # Check if mira config exists and is correct
    if existing:
        if existing.get("command") != command or existing.get("args") != args:
            print(f"  Updating mira config in {config_path}...")
            needs_update = True
    else:
        needs_update = True

    # Check if SessionStart hook is configured
    hooks = config.get("hooks", {})
    session_start = hooks.get("SessionStart", [])
    has_mira_hook = any(
        "mira" in str(hook.get("hooks", [{}])[0].get("command", ""))
        for hook in session_start
        if isinstance(hook, dict) and hook.get("hooks")
    )

    if not has_mira_hook:
        needs_update = True
        print(f"  SessionStart hook not configured in {config_path}")

    if not needs_update:
        print(f"  {config_path}: Already configured correctly")
        return False

    # Update MCP server config
    config["mcpServers"]["mira"] = new_mira_config

    # Update SessionStart hook
    if not has_mira_hook:
        if "hooks" not in config:
            config["hooks"] = {}
        if "SessionStart" not in config["hooks"]:
            config["hooks"]["SessionStart"] = []

        # Add MIRA hook
        config["hooks"]["SessionStart"].append(get_session_start_hook())

    if dry_run:
        print(f"  Would update {config_path}")
        print(f"    MCP server: {command} {' '.join(args)}")
        print(f"    SessionStart hook: configured")
        return True

    # Write config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"  Updated {config_path}")

    return True


def install(dry_run: bool = False):
    """Install MIRA into Claude Code's MCP configuration."""
    print("Configuring MIRA for Claude Code...")
    print("  - MCP server (mira_search, mira_status, etc.)")
    print("  - SessionStart hook (auto-injects context)")
    print()

    updated = False
    for config_path in get_claude_config_paths():
        if update_claude_config(config_path, dry_run):
            updated = True

    if updated and not dry_run:
        print("\nMIRA configured successfully!")
        print("Restart Claude Code to activate MIRA.")
        print("\nOn first search, MIRA downloads a ~100MB embedding model for local semantic search.")
    elif not updated:
        print("\nMIRA already configured correctly.")

    return updated


def uninstall():
    """Remove MIRA from Claude Code's MCP configuration."""
    print("Removing MIRA from Claude Code...")

    for config_path in get_claude_config_paths():
        if not config_path.exists():
            continue

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))

            removed = False

            # Remove MCP server entries
            for key in ["mira", "mira3"]:
                if key in config.get("mcpServers", {}):
                    del config["mcpServers"][key]
                    removed = True
                    print(f"  Removed MCP server '{key}' from {config_path}")

            # Remove SessionStart hook
            session_start = config.get("hooks", {}).get("SessionStart", [])
            new_hooks = [
                hook for hook in session_start
                if not (isinstance(hook, dict) and
                        "mira" in str(hook.get("hooks", [{}])[0].get("command", "")))
            ]
            if len(new_hooks) != len(session_start):
                config["hooks"]["SessionStart"] = new_hooks
                removed = True
                print(f"  Removed SessionStart hook from {config_path}")

            if removed:
                config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Error updating {config_path}: {e}")

    print("\nMIRA removed from Claude Code.")


def main():
    """CLI entry point for install command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Configure MIRA for Claude Code"
    )
    parser.add_argument(
        "--uninstall", "-u",
        action="store_true",
        help="Remove MIRA from Claude Code"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    if args.uninstall:
        uninstall()
    else:
        install(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
