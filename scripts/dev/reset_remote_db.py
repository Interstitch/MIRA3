#!/usr/bin/env python3
"""
Reset MIRA remote databases (Postgres + Qdrant).

Connects to central storage and resets tables/collections.

WARNING: This will delete ALL data from central storage across ALL projects.
         This is a destructive operation intended for development/testing.

Usage:
    python scripts/dev/reset_remote_db.py [--force] [--postgres-only] [--qdrant-only]
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


# PostgreSQL tables in dependency order (children first for foreign keys)
POSTGRES_TABLES = [
    # Child tables (have foreign keys)
    "archives",
    "file_operations",
    "artifacts",
    "error_patterns",
    "decisions",
    "concepts",
    "sessions",
    # Parent tables
    "projects",
    # Standalone tables
    "custodian",
    "name_candidates",
    "lifecycle_patterns",
]


def load_server_config() -> dict:
    """Load server.json configuration."""
    from mira.core.constants import get_global_mira_path, get_project_mira_path

    # Check project path first, then global
    for base_path in [get_project_mira_path(), get_global_mira_path()]:
        config_path = base_path / "server.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

    return {}


def reset_postgres(config: dict, force: bool = False, dry_run: bool = False) -> bool:
    """Reset PostgreSQL database by truncating all tables."""
    pg_config = config.get("postgres", {})
    if not pg_config:
        print("PostgreSQL: No configuration found in server.json")
        return False

    host = pg_config.get("host")
    port = pg_config.get("port", 5432)
    database = pg_config.get("database")
    user = pg_config.get("user")
    password = pg_config.get("password")

    if not all([host, database, user, password]):
        print("PostgreSQL: Missing required configuration (host, database, user, password)")
        return False

    print(f"PostgreSQL: {host}:{port}/{database}")
    print(f"  Tables to truncate: {', '.join(POSTGRES_TABLES)}")

    if dry_run:
        print("  [DRY RUN] Would truncate all tables")
        return True

    if not force:
        confirm = input("  Truncate ALL PostgreSQL tables? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("  Skipped.")
            return False

    try:
        import psycopg2

        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            connect_timeout=10,
        )
        conn.autocommit = False
        cur = conn.cursor()

        # Check which tables exist
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        existing_tables = {row[0] for row in cur.fetchall()}

        # Truncate tables in order
        truncated = 0
        for table in POSTGRES_TABLES:
            if table in existing_tables:
                try:
                    cur.execute(f"TRUNCATE TABLE {table} CASCADE")
                    print(f"  Truncated: {table}")
                    truncated += 1
                except Exception as e:
                    print(f"  Error truncating {table}: {e}")
                    conn.rollback()

        conn.commit()
        cur.close()
        conn.close()

        print(f"  Truncated {truncated} tables.")
        return True

    except ImportError:
        print("  Error: psycopg2 not installed. Run: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"  Error connecting to PostgreSQL: {e}")
        return False


def reset_qdrant(config: dict, force: bool = False, dry_run: bool = False) -> bool:
    """Reset Qdrant by deleting and recreating the collection."""
    qdrant_config = config.get("qdrant", {})
    if not qdrant_config:
        print("Qdrant: No configuration found in server.json")
        return False

    host = qdrant_config.get("host")
    port = qdrant_config.get("port", 6333)
    collection = qdrant_config.get("collection", "mira_embeddings")
    api_key = qdrant_config.get("api_key")

    if not host:
        print("Qdrant: Missing required configuration (host)")
        return False

    print(f"Qdrant: {host}:{port}")
    print(f"  Collection: {collection}")

    if dry_run:
        print("  [DRY RUN] Would delete collection")
        return True

    if not force:
        confirm = input(f"  Delete collection '{collection}'? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("  Skipped.")
            return False

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            timeout=30,
        )

        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection not in collection_names:
            print(f"  Collection '{collection}' does not exist.")
            return True

        # Get collection info before deleting
        try:
            info = client.get_collection(collection)
            point_count = info.points_count
            vector_dim = info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else 384
            print(f"  Points: {point_count}, Vector dimension: {vector_dim}")
        except Exception:
            vector_dim = 384  # Default for fastembed

        # Delete collection
        client.delete_collection(collection)
        print(f"  Deleted collection: {collection}")

        # Recreate empty collection with same config
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE,
            ),
        )
        print(f"  Recreated empty collection: {collection}")

        return True

    except ImportError:
        print("  Error: qdrant-client not installed. Run: pip install qdrant-client")
        return False
    except Exception as e:
        print(f"  Error connecting to Qdrant: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Reset MIRA remote databases (Postgres + Qdrant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Reset all remote databases
    python scripts/dev/reset_remote_db.py

    # Reset only PostgreSQL
    python scripts/dev/reset_remote_db.py --postgres-only

    # Reset only Qdrant
    python scripts/dev/reset_remote_db.py --qdrant-only

    # Skip confirmation prompts
    python scripts/dev/reset_remote_db.py --force

    # Show what would be done without doing it
    python scripts/dev/reset_remote_db.py --dry-run
        """
    )
    parser.add_argument("--force", "-f", action="store_true",
                        help="Skip confirmation prompts")
    parser.add_argument("--postgres-only", action="store_true",
                        help="Only reset PostgreSQL")
    parser.add_argument("--qdrant-only", action="store_true",
                        help="Only reset Qdrant")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be done without doing it")
    parser.add_argument("--config", "-c", type=str,
                        help="Path to server.json config file")

    args = parser.parse_args()

    if args.postgres_only and args.qdrant_only:
        print("Error: Cannot specify both --postgres-only and --qdrant-only")
        sys.exit(1)

    print("MIRA Remote Database Reset")
    print("=" * 60)
    print()
    print("WARNING: This will delete ALL data from central storage!")
    print("         This affects ALL projects using this central server.")
    print()

    # Load config
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = load_server_config()

    if not config:
        print("Error: No server.json configuration found.")
        print("       Create ~/.mira/server.json or <project>/.mira/server.json")
        print("       Or specify --config path/to/server.json")
        sys.exit(1)

    reset_pg = not args.qdrant_only
    reset_qd = not args.postgres_only

    success = True

    if reset_pg:
        print()
        if not reset_postgres(config, force=args.force, dry_run=args.dry_run):
            success = False

    if reset_qd:
        print()
        if not reset_qdrant(config, force=args.force, dry_run=args.dry_run):
            success = False

    print()
    if args.dry_run:
        print("[DRY RUN] No changes made.")
    elif success:
        print("Reset complete. Remote databases are empty.")
    else:
        print("Reset completed with errors. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
