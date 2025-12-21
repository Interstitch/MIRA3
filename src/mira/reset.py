"""
MIRA Reset Tool

Wipes MIRA storage and triggers fresh bootstrap.
Useful for:
- Clean slate after major version upgrades
- Troubleshooting bootstrap issues
- Windows users with path issues from old versions
"""

import shutil
import sys
from pathlib import Path

from .core.constants import get_global_mira_path, get_project_mira_path


def get_storage_info() -> dict:
    """Get information about current storage locations and sizes."""
    global_path = get_global_mira_path()
    project_path = get_project_mira_path()

    def dir_size_mb(path: Path) -> float:
        if not path.exists():
            return 0.0
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except (OSError, PermissionError):
            pass
        return round(total / (1024 * 1024), 2)

    return {
        "global_path": global_path,
        "global_exists": global_path.exists(),
        "global_size_mb": dir_size_mb(global_path),
        "project_path": project_path,
        "project_exists": project_path.exists(),
        "project_size_mb": dir_size_mb(project_path),
    }


def reset(
    wipe_global: bool = True,
    wipe_project: bool = True,
    reconfigure: bool = True,
    force: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Reset MIRA storage and optionally reconfigure Claude Code.

    Args:
        wipe_global: Delete ~/.mira/ (venv, global DBs)
        wipe_project: Delete <cwd>/.mira/ (project DBs, logs)
        reconfigure: Re-run mira-install after wiping
        force: Skip confirmation prompt
        dry_run: Show what would be done without doing it

    Returns:
        True if reset was performed, False if cancelled
    """
    info = get_storage_info()

    # Show what will be deleted
    print("MIRA Reset")
    print("=" * 50)

    if wipe_global:
        if info["global_exists"]:
            print(f"Global storage: {info['global_path']}")
            print(f"  Size: {info['global_size_mb']} MB")
            print(f"  Contains: venv, user preferences, error patterns")
        else:
            print(f"Global storage: {info['global_path']} (not found)")

    if wipe_project:
        if info["project_exists"]:
            print(f"Project storage: {info['project_path']}")
            print(f"  Size: {info['project_size_mb']} MB")
            print(f"  Contains: conversation index, artifacts, logs")
        else:
            print(f"Project storage: {info['project_path']} (not found)")

    if not info["global_exists"] and not info["project_exists"]:
        print("\nNothing to reset - no MIRA storage found.")
        return False

    total_mb = 0
    if wipe_global and info["global_exists"]:
        total_mb += info["global_size_mb"]
    if wipe_project and info["project_exists"]:
        total_mb += info["project_size_mb"]

    print(f"\nTotal to delete: {total_mb} MB")

    if dry_run:
        print("\n[DRY RUN] Would delete the above directories.")
        if reconfigure:
            print("[DRY RUN] Would reconfigure Claude Code MCP settings.")
        return True

    # Confirm
    if not force:
        print("\nThis will delete all MIRA data including:")
        if wipe_global:
            print("  - Learned user preferences")
            print("  - Error pattern history")
            print("  - Decision journal")
            print("  - Python virtualenv (will be reinstalled)")
        if wipe_project:
            print("  - Conversation index for this project")
            print("  - Code artifacts")
            print("  - Local logs")

        try:
            response = input("\nProceed? [y/N] ").strip().lower()
            if response not in ("y", "yes"):
                print("Cancelled.")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return False

    # Perform deletion
    print("\nDeleting...")

    if wipe_global and info["global_exists"]:
        try:
            shutil.rmtree(info["global_path"])
            print(f"  Deleted: {info['global_path']}")
        except (OSError, PermissionError) as e:
            print(f"  Error deleting {info['global_path']}: {e}")
            print("  You may need to close MIRA/Claude Code first.")
            return False

    if wipe_project and info["project_exists"]:
        try:
            shutil.rmtree(info["project_path"])
            print(f"  Deleted: {info['project_path']}")
        except (OSError, PermissionError) as e:
            print(f"  Error deleting {info['project_path']}: {e}")
            print("  You may need to close MIRA/Claude Code first.")
            return False

    print("\nReset complete!")

    # Reconfigure
    if reconfigure:
        print("\nReconfiguring Claude Code...")
        try:
            from .install import install
            install()
        except Exception as e:
            print(f"  Warning: Could not reconfigure: {e}")
            print("  Run 'mira-install' manually after restart.")

    print("\nNext steps:")
    print("  1. Restart Claude Code")
    print("  2. MIRA will automatically bootstrap on first use")
    print("  3. Your preferences will be re-learned over a few sessions")

    return True


def main():
    """CLI entry point for reset command."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mira-reset",
        description="Reset MIRA storage and trigger fresh bootstrap"
    )
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Only wipe global storage (~/.mira/)"
    )
    parser.add_argument(
        "--project-only",
        action="store_true",
        help="Only wipe project storage (<cwd>/.mira/)"
    )
    parser.add_argument(
        "--no-reconfigure",
        action="store_true",
        help="Don't reconfigure Claude Code after reset"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Show storage info without resetting"
    )

    args = parser.parse_args()

    if args.info:
        info = get_storage_info()
        print("MIRA Storage Info")
        print("=" * 50)
        print(f"Global: {info['global_path']}")
        print(f"  Exists: {info['global_exists']}")
        print(f"  Size: {info['global_size_mb']} MB")
        print(f"Project: {info['project_path']}")
        print(f"  Exists: {info['project_exists']}")
        print(f"  Size: {info['project_size_mb']} MB")
        return

    # Determine what to wipe
    wipe_global = not args.project_only
    wipe_project = not args.global_only

    if args.global_only and args.project_only:
        print("Error: Cannot specify both --global-only and --project-only")
        sys.exit(1)

    success = reset(
        wipe_global=wipe_global,
        wipe_project=wipe_project,
        reconfigure=not args.no_reconfigure,
        force=args.force,
        dry_run=args.dry_run
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
