#!/usr/bin/env python3
"""
CLI wrapper for mira_error_lookup tool.

Search past error patterns and solutions.

Usage:
    python scripts/cli/mira_errors.py "TypeError" [--limit N]
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def main():
    parser = argparse.ArgumentParser(description="Search error patterns")
    parser.add_argument("query", help="Error message or pattern")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Max results")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    from mira.storage import get_storage
    from mira.tools import handle_error_lookup

    storage = get_storage()
    result = handle_error_lookup({"query": args.query, "limit": args.limit}, storage)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        solutions = result.get("solutions", [])
        print(f"Found {result.get('total', 0)} matching errors\n")

        for i, sol in enumerate(solutions, 1):
            print(f"{i}. {sol.get('error_type', 'Unknown')}")
            print(f"   Message: {sol.get('error_message', '')[:80]}")
            if sol.get('solution_summary'):
                print(f"   Solution: {sol.get('solution_summary')[:80]}")
            print(f"   Occurrences: {sol.get('occurrence_count', 1)}")
            print()


if __name__ == "__main__":
    main()
