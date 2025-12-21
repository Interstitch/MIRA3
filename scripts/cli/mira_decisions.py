#!/usr/bin/env python3
"""
CLI wrapper for mira_decisions tool.

Search architectural and design decisions.

Usage:
    python scripts/cli/mira_decisions.py "database" [--category architecture]
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def main():
    parser = argparse.ArgumentParser(description="Search decisions")
    parser.add_argument("query", nargs="?", default="", help="Decision topic")
    parser.add_argument("--category", "-c", help="Filter by category")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Max results")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    from mira.storage import get_storage
    from mira.tools import handle_decisions

    storage = get_storage()
    params = {"query": args.query, "limit": args.limit}
    if args.category:
        params["category"] = args.category

    result = handle_decisions(params, storage)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        decisions = result.get("decisions", [])
        print(f"Found {result.get('total', 0)} decisions\n")

        for i, dec in enumerate(decisions, 1):
            print(f"{i}. [{dec.get('category', 'general')}] {dec.get('decision_summary', '')[:70]}")
            if dec.get('reasoning'):
                print(f"   Reason: {dec.get('reasoning')[:70]}")
            print(f"   Confidence: {dec.get('confidence', 0.5):.0%}")
            print()


if __name__ == "__main__":
    main()
