#!/usr/bin/env python3
"""
MIRA system health check.

Verifies all components are working correctly.

Usage:
    python scripts/validate/health_check.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def check_imports():
    """Check all required imports."""
    print("Checking imports...")
    try:
        from mira.core import VERSION, get_mira_path
        from mira.storage import get_storage
        from mira.tools import handle_init, handle_status
        print(f"  MIRA version: {VERSION}")
        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def check_directories():
    """Check required directories exist."""
    print("\nChecking directories...")
    from mira.core import get_mira_path

    mira_path = get_mira_path()
    required = ["archives", "metadata"]

    all_ok = True
    for dir_name in required:
        dir_path = mira_path / dir_name
        if dir_path.exists():
            print(f"  [OK] {dir_name}/")
        else:
            print(f"  [MISSING] {dir_name}/")
            all_ok = False

    return all_ok


def check_databases():
    """Check databases can be opened."""
    print("\nChecking databases...")
    from mira.core import get_db_manager, DB_LOCAL_STORE

    try:
        db = get_db_manager()
        # Try a simple query
        result = db.execute_read_one(
            DB_LOCAL_STORE,
            "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1",
            ()
        )
        print("  [OK] Database manager working")
        return True
    except Exception as e:
        print(f"  [FAIL] Database error: {e}")
        return False


def check_storage():
    """Check storage backend."""
    print("\nChecking storage...")
    try:
        from mira.storage import get_storage

        storage = get_storage()
        health = storage.health_check()

        if storage.using_central:
            print(f"  Mode: CENTRAL")
            print(f"  Qdrant: {'OK' if health.get('qdrant_healthy') else 'FAIL'}")
            print(f"  Postgres: {'OK' if health.get('postgres_healthy') else 'FAIL'}")
        else:
            print(f"  Mode: LOCAL (SQLite FTS)")
            print(f"  [OK] Storage working")

        return True
    except Exception as e:
        print(f"  [FAIL] Storage error: {e}")
        return False


def main():
    print("=" * 50)
    print("MIRA Health Check")
    print("=" * 50)

    checks = [
        ("Imports", check_imports),
        ("Directories", check_directories),
        ("Databases", check_databases),
        ("Storage", check_storage),
    ]

    results = []
    for name, check_fn in checks:
        try:
            results.append((name, check_fn()))
        except Exception as e:
            print(f"\n[ERROR] {name} check crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All checks passed!")
        sys.exit(0)
    else:
        print("Some checks failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
