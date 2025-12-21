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


def update_claude_config(config_path: Path, dry_run: bool = False) -> bool:
    """
    Update a Claude Code config file with MIRA MCP server.

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

    if not needs_update:
        print(f"  {config_path}: Already configured correctly")
        return False

    # Update config
    config["mcpServers"]["mira"] = new_mira_config

    if dry_run:
        print(f"  Would update {config_path}")
        print(f"    command: {command}")
        print(f"    args: {args}")
        return True

    # Write config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"  Updated {config_path}")

    return True


def install(dry_run: bool = False):
    """Install MIRA into Claude Code's MCP configuration."""
    print("Configuring MIRA for Claude Code...")

    updated = False
    for config_path in get_claude_config_paths():
        if update_claude_config(config_path, dry_run):
            updated = True

    if updated and not dry_run:
        print("\nMIRA configured successfully!")
        print("Restart Claude Code to activate MIRA.")
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
            for key in ["mira", "mira3"]:
                if key in config.get("mcpServers", {}):
                    del config["mcpServers"][key]
                    removed = True

            if removed:
                config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                print(f"  Removed from {config_path}")
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
