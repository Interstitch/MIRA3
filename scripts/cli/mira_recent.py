#!/usr/bin/env python3
"""
CLI wrapper for mira_recent tool.

View recent conversation sessions.

Usage:
    python scripts/cli/mira_recent.py [--limit N] [--project-only]
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def main():
    parser = argparse.ArgumentParser(description="View recent MIRA sessions")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Max sessions")
    parser.add_argument("--project-only", "-p", action="store_true", help="Current project only")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    from mira.storage import get_storage
    from mira.tools import handle_recent

    storage = get_storage()
    result = handle_recent({
        "limit": args.limit,
        "project_only": args.project_only
    }, storage)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"=== Recent Sessions (Total: {result.get('total', 0)}) ===\n")

        for project_obj in result.get("projects", []):
            project_path = project_obj.get("path", "unknown")
            sessions = project_obj.get("sessions", [])
            print(f"Project: {project_path}")
            for s in sessions:
                summary = s.get('summary', 'No summary')[:60]
                timestamp = s.get('timestamp', 'Unknown')
                # Extract date from timestamp if present
                date = timestamp[:10] if len(timestamp) >= 10 else timestamp
                print(f"  - {summary}")
                print(f"    Date: {date}")
            print()


if __name__ == "__main__":
    main()
