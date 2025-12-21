#!/usr/bin/env python3
"""
Reset MIRA local databases for a fresh start.

Uses the hybrid storage model:
- Global (~/.mira/): custodian.db, insights.db
- Project (<cwd>/.mira/): local_store.db, artifacts.db, concepts.db, etc.

WARNING: This will delete indexed conversations and learned data.

Usage:
    python scripts/dev/reset_local_db.py [--force] [--global-only] [--project-only]
"""

import sys
import os
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def get_db_size(path: Path) -> str:
    """Get file size in human-readable format."""
    if not path.exists():
        return "0 B"
    size = path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Reset MIRA local databases (SQLite)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Reset all databases (global + project)
    python scripts/dev/reset_local_db.py

    # Reset only global databases (custodian, insights)
    python scripts/dev/reset_local_db.py --global-only

    # Reset only project databases (local_store, artifacts, etc.)
    python scripts/dev/reset_local_db.py --project-only

    # Skip confirmation prompt
    python scripts/dev/reset_local_db.py --force
        """
    )
    parser.add_argument("--force", "-f", action="store_true",
                        help="Skip confirmation prompt")
    parser.add_argument("--global-only", action="store_true",
                        help="Only reset global databases (~/.mira/)")
    parser.add_argument("--project-only", action="store_true",
                        help="Only reset project databases (<cwd>/.mira/)")
    parser.add_argument("--keep-metadata", action="store_true",
                        help="Keep metadata/ directory")
    parser.add_argument("--keep-archives", action="store_true",
                        help="Keep archives/ directory")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be deleted without deleting")

    args = parser.parse_args()

    if args.global_only and args.project_only:
        print("Error: Cannot specify both --global-only and --project-only")
        sys.exit(1)

    from mira.core.constants import (
        get_global_mira_path,
        get_project_mira_path,
        GLOBAL_DATABASES,
        PROJECT_DATABASES,
    )

    global_path = get_global_mira_path()
    project_path = get_project_mira_path()

    reset_global = not args.project_only
    reset_project = not args.global_only

    print("MIRA Local Database Reset")
    print("=" * 60)
    print()

    # Collect files to delete
    to_delete = []
    total_size = 0

    if reset_global:
        print(f"Global storage: {global_path}")
        for db_name in sorted(GLOBAL_DATABASES):
            db_path = global_path / db_name
            if db_path.exists():
                size = db_path.stat().st_size
                total_size += size
                to_delete.append(("db", db_path, db_name))
                print(f"  [{get_db_size(db_path):>8}] {db_name}")
            else:
                print(f"  [   n/a  ] {db_name} (not found)")
        print()

    if reset_project:
        print(f"Project storage: {project_path}")
        for db_name in sorted(PROJECT_DATABASES):
            db_path = project_path / db_name
            if db_path.exists():
                size = db_path.stat().st_size
                total_size += size
                to_delete.append(("db", db_path, db_name))
                print(f"  [{get_db_size(db_path):>8}] {db_name}")
            else:
                print(f"  [   n/a  ] {db_name} (not found)")

        # Metadata directory
        if not args.keep_metadata:
            metadata_path = project_path / "metadata"
            if metadata_path.exists():
                meta_size = sum(f.stat().st_size for f in metadata_path.rglob("*") if f.is_file())
                total_size += meta_size
                to_delete.append(("dir", metadata_path, "metadata/"))
                print(f"  [{get_db_size(Path('/')) if meta_size == 0 else f'{meta_size/1024:.1f} KB':>8}] metadata/")

        # Archives directory
        if not args.keep_archives:
            archives_path = project_path / "archives"
            if archives_path.exists():
                arch_size = sum(f.stat().st_size for f in archives_path.rglob("*") if f.is_file())
                total_size += arch_size
                to_delete.append(("dir", archives_path, "archives/"))
                count = sum(1 for _ in archives_path.glob("*.jsonl"))
                print(f"  [{get_db_size(Path('/')) if arch_size == 0 else f'{arch_size/1024/1024:.1f} MB':>8}] archives/ ({count} files)")
        print()

    if not to_delete:
        print("Nothing to delete - no databases found.")
        return

    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    print()

    if args.dry_run:
        print("[DRY RUN] Would delete the above files/directories.")
        return

    # Confirm
    if not args.force:
        print("This will delete:")
        if reset_global:
            print("  - User preferences (custodian)")
            print("  - Error patterns and decisions (insights)")
        if reset_project:
            print("  - Conversation index (local_store)")
            print("  - Code artifacts")
            print("  - Codebase concepts")
            if not args.keep_metadata:
                print("  - Session metadata")
            if not args.keep_archives:
                print("  - Conversation archives")

        try:
            confirm = input("\nProceed? [y/N] ").strip().lower()
            if confirm not in ("y", "yes"):
                print("Aborted.")
                return
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return

    # Perform deletion
    print("\nDeleting...")
    deleted_count = 0

    for item_type, path, name in to_delete:
        try:
            if item_type == "db":
                path.unlink()
                print(f"  Deleted: {name}")
                deleted_count += 1
            elif item_type == "dir":
                shutil.rmtree(path)
                print(f"  Deleted: {name}")
                deleted_count += 1
        except (OSError, PermissionError) as e:
            print(f"  Error deleting {name}: {e}")

    print(f"\nDeleted {deleted_count} items.")
    print("Run `python -m mira` to reinitialize databases.")


if __name__ == "__main__":
    main()
