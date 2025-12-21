#!/usr/bin/env python3
"""
CLI wrapper for mira_search tool.

Search past Claude Code conversations from the command line.

Usage:
    python scripts/cli/mira_search.py "search query" [--limit N]
"""

import sys
import os
import argparse
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def main():
    parser = argparse.ArgumentParser(
        description="Search MIRA conversation history"
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Max results")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    from mira.storage import get_storage
    from mira.tools import handle_search

    storage = get_storage()
    result = handle_search({"query": args.query, "limit": args.limit}, storage)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        results = result.get("results", [])
        print(f"Found {result.get('total', 0)} results")

        # Show corrections if any
        if result.get("corrections"):
            print(f"(Corrected: '{result.get('original_query')}' → '{result.get('query')}')")
        print()

        for i, r in enumerate(results, 1):
            summary = r.get('summary', 'No summary')[:80]
            date = r.get('date', 'Unknown')
            score = r.get('score', '')
            topics = r.get('topics', [])

            print(f"{i}. {summary}")
            print(f"   Date: {date}", end="")
            if score:
                print(f" | Score: {score}", end="")
            print()
            if topics:
                print(f"   Topics: {', '.join(topics[:5])}")

            # Show excerpt if available
            excerpt = r.get('excerpt', '')
            if excerpt:
                print(f"   Excerpt: {excerpt[:100]}...")
            print()


if __name__ == "__main__":
    main()
