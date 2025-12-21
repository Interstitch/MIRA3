#!/usr/bin/env python3
"""
Reset MIRA databases for a fresh start.

WARNING: This will delete all indexed conversations and learned data.

Usage:
    python scripts/dev/reset_db.py [--force]
"""

import sys
import os
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def main():
    parser = argparse.ArgumentParser(description="Reset MIRA databases")
    parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    from mira.core import get_mira_path

    mira_path = get_mira_path()

    db_files = [
        "local_store.db",
        "artifacts.db",
        "custodian.db",
        "insights.db",
        "concepts.db",
        "code_history.db",
        "sync_queue.db",
        "migrations.db",
        "local_vectors.db",
    ]

    print(f"MIRA path: {mira_path}")
    print(f"This will delete: {', '.join(db_files)}")
    print()

    if not args.force:
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return

    deleted = 0
    for db_file in db_files:
        db_path = mira_path / db_file
        if db_path.exists():
            db_path.unlink()
            print(f"  Deleted: {db_file}")
            deleted += 1

    # Also delete metadata
    metadata_path = mira_path / "metadata"
    if metadata_path.exists():
        shutil.rmtree(metadata_path)
        print(f"  Deleted: metadata/")

    print(f"\nDeleted {deleted} database files.")
    print("Run `python -m mira` to reinitialize.")


if __name__ == "__main__":
    main()
