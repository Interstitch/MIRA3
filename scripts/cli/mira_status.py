#!/usr/bin/env python3
"""
CLI wrapper for mira_status tool.

Check MIRA system health and statistics.

Usage:
    python scripts/cli/mira_status.py [--json]
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def main():
    parser = argparse.ArgumentParser(description="Check MIRA status")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    from mira.storage import get_storage
    from mira.tools import handle_status

    storage = get_storage()
    result = handle_status({}, storage)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print("=== MIRA Status ===\n")

        # Get storage mode from health check
        storage_health = result.get("storage_health", {})
        storage_mode = storage_health.get("mode", "unknown")
        using_central = storage_health.get("using_central", False)
        print(f"Storage Mode: {storage_mode}")
        if using_central:
            print(f"  Qdrant: {'healthy' if storage_health.get('qdrant_healthy') else 'unhealthy'}")
            print(f"  Postgres: {'healthy' if storage_health.get('postgres_healthy') else 'unhealthy'}")
        print(f"Storage Path: {result.get('storage_path', 'unknown')}")
        print()

        global_stats = result.get("global", {})
        files = global_stats.get("files", {})
        ingestion = global_stats.get("ingestion", {})
        print(f"Total Files: {files.get('total', 0)} ({files.get('indexable', 0)} indexable)")
        print(f"Indexed: {ingestion.get('indexed', 0)}/{files.get('indexable', 0)} ({ingestion.get('percent', 0)}%)")
        print(f"In Database: {ingestion.get('total_in_db', 0)}")
        print(f"Archived: {global_stats.get('archived', 0)}")
        print()

        # Show sync status if central
        if using_central:
            sync = global_stats.get("central_sync", {})
            print(f"Central Sync: {sync.get('status', 'unknown')} (pending: {sync.get('pending', 0)}, failed: {sync.get('failed', 0)})")
            print()

        print(f"Last Sync: {result.get('last_sync', 'Never')}")


if __name__ == "__main__":
    main()
