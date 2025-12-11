"""
MIRA3 Schema Migration Framework

Manages database schema versioning and migrations for:
- Local SQLite databases (artifacts, custodian, insights, concepts, local_store, audit)
- Central Postgres database

Migrations are idempotent and can be safely re-run.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .utils import log, get_mira_path
from .db_manager import get_db_manager

# Current schema version
CURRENT_VERSION = 2

# Migration registry
MIGRATIONS_DB = "migrations.db"

MIGRATIONS_SCHEMA = """
-- Track applied migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER UNIQUE NOT NULL,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    duration_ms INTEGER,
    checksum TEXT,
    status TEXT DEFAULT 'success'
);

-- Track individual database versions
CREATE TABLE IF NOT EXISTS database_versions (
    db_name TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# Migration definitions
# Each migration is a tuple of (version, name, up_function)
_migrations: List[tuple] = []


def migration(version: int, name: str):
    """Decorator to register a migration function."""
    def decorator(func: Callable):
        _migrations.append((version, name, func))
        return func
    return decorator


# ==================== Migration Definitions ====================

@migration(1, "initial_schema")
def migrate_v1(db_manager):
    """Initial schema - ensure all tables exist with base structure."""
    # This migration ensures base tables exist
    # Individual modules handle their own schema creation,
    # but we track that version 1 has been "applied"
    log("Migration v1: Verifying base schemas exist")

    # Import and initialize all database schemas
    from .artifacts import init_artifact_db
    from .custodian import init_custodian_db
    from .insights import init_insights_db
    from .concepts import init_concepts_db
    from .local_store import init_local_db
    from .audit import init_audit_db

    init_artifact_db()
    init_custodian_db()
    init_insights_db()
    init_concepts_db()
    init_local_db()
    init_audit_db()

    return True


@migration(2, "add_audit_indexes")
def migrate_v2(db_manager):
    """Add additional indexes to audit log for better query performance."""
    log("Migration v2: Adding audit log indexes")

    from .audit import AUDIT_DB

    # Add composite index for common query pattern
    try:
        db_manager.execute_write(
            AUDIT_DB,
            """CREATE INDEX IF NOT EXISTS idx_audit_action_status
               ON audit_log(action, status)""",
            ()
        )
        db_manager.execute_write(
            AUDIT_DB,
            """CREATE INDEX IF NOT EXISTS idx_audit_machine_timestamp
               ON audit_log(machine_id, timestamp DESC)""",
            ()
        )
    except Exception as e:
        log(f"Migration v2 index creation: {e}")

    return True


# ==================== Migration Runner ====================

def init_migrations_db():
    """Initialize the migrations tracking database."""
    db = get_db_manager()
    db.init_schema(MIGRATIONS_DB, MIGRATIONS_SCHEMA)


def get_current_version() -> int:
    """Get the current schema version from the database."""
    init_migrations_db()
    db = get_db_manager()

    row = db.execute_read_one(
        MIGRATIONS_DB,
        "SELECT MAX(version) as version FROM schema_migrations WHERE status = 'success'",
        ()
    )

    return row['version'] if row and row['version'] else 0


def get_applied_migrations() -> List[Dict[str, Any]]:
    """Get list of all applied migrations."""
    init_migrations_db()
    db = get_db_manager()

    rows = db.execute_read(
        MIGRATIONS_DB,
        "SELECT * FROM schema_migrations ORDER BY version",
        ()
    )

    return [dict(row) for row in rows]


def run_migrations(target_version: Optional[int] = None) -> Dict[str, Any]:
    """
    Run all pending migrations up to target_version.

    Args:
        target_version: Version to migrate to (default: CURRENT_VERSION)

    Returns:
        Dict with migration results
    """
    if target_version is None:
        target_version = CURRENT_VERSION

    init_migrations_db()
    db = get_db_manager()

    current = get_current_version()
    results = {
        "start_version": current,
        "target_version": target_version,
        "migrations_run": [],
        "status": "success",
    }

    if current >= target_version:
        log(f"Schema already at version {current}, target is {target_version}")
        results["status"] = "already_current"
        return results

    # Sort migrations by version
    sorted_migrations = sorted(_migrations, key=lambda x: x[0])

    for version, name, migrate_func in sorted_migrations:
        if version <= current:
            continue
        if version > target_version:
            break

        log(f"Running migration v{version}: {name}")
        start_time = datetime.utcnow()

        try:
            success = migrate_func(db)
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            if success:
                # Record successful migration
                db.execute_write(
                    MIGRATIONS_DB,
                    """INSERT INTO schema_migrations (version, name, applied_at, duration_ms, status)
                       VALUES (?, ?, ?, ?, 'success')""",
                    (version, name, datetime.utcnow().isoformat(), duration_ms)
                )
                results["migrations_run"].append({
                    "version": version,
                    "name": name,
                    "duration_ms": duration_ms,
                    "status": "success",
                })
                log(f"Migration v{version} completed in {duration_ms}ms")
            else:
                results["status"] = "failed"
                results["error"] = f"Migration v{version} returned False"
                break

        except Exception as e:
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            log(f"Migration v{version} failed: {e}")

            # Record failed migration
            db.execute_write(
                MIGRATIONS_DB,
                """INSERT INTO schema_migrations (version, name, applied_at, duration_ms, status)
                   VALUES (?, ?, ?, ?, 'failed')""",
                (version, name, datetime.utcnow().isoformat(), duration_ms)
            )

            results["status"] = "failed"
            results["error"] = str(e)
            results["migrations_run"].append({
                "version": version,
                "name": name,
                "duration_ms": duration_ms,
                "status": "failed",
                "error": str(e),
            })
            break

    results["end_version"] = get_current_version()
    return results


def check_migrations_needed() -> Dict[str, Any]:
    """
    Check if migrations are needed without running them.

    Returns:
        Dict with current version, target version, and pending migrations
    """
    current = get_current_version()

    pending = []
    for version, name, _ in sorted(_migrations, key=lambda x: x[0]):
        if version > current:
            pending.append({"version": version, "name": name})

    return {
        "current_version": current,
        "target_version": CURRENT_VERSION,
        "needs_migration": len(pending) > 0,
        "pending_migrations": pending,
    }


def ensure_schema_current():
    """
    Ensure the schema is up to date. Called on startup.

    This is safe to call multiple times - migrations are idempotent.
    """
    check = check_migrations_needed()

    if check["needs_migration"]:
        log(f"Schema needs migration: v{check['current_version']} -> v{check['target_version']}")
        result = run_migrations()
        if result["status"] != "success" and result["status"] != "already_current":
            log(f"Migration warning: {result.get('error', 'unknown error')}")
    else:
        log(f"Schema is current at v{check['current_version']}")


# ==================== Postgres Migrations (Central Storage) ====================

def run_postgres_migrations(postgres_backend) -> Dict[str, Any]:
    """
    Run migrations on central Postgres database.

    Args:
        postgres_backend: PostgresBackend instance

    Returns:
        Dict with migration results
    """
    results = {
        "status": "success",
        "migrations_run": [],
    }

    try:
        with postgres_backend._get_connection() as conn:
            with conn.cursor() as cur:
                # Check if schema_version table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'schema_version'
                    )
                """)
                exists = cur.fetchone()[0]

                if not exists:
                    log("Creating Postgres schema_version table")
                    cur.execute("""
                        CREATE TABLE schema_version (
                            version INTEGER PRIMARY KEY,
                            applied_at TIMESTAMPTZ DEFAULT NOW(),
                            description TEXT
                        )
                    """)
                    cur.execute(
                        "INSERT INTO schema_version (version, description) VALUES (1, 'Initial schema')"
                    )
                    conn.commit()
                    results["migrations_run"].append({"version": 1, "name": "initial"})

                # Get current Postgres version
                cur.execute("SELECT MAX(version) FROM schema_version")
                pg_version = cur.fetchone()[0] or 0

                # Postgres migration v2: Add file_operations table
                if pg_version < 2:
                    log("Postgres migration v2: Adding file_operations table")
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS file_operations (
                            id SERIAL PRIMARY KEY,
                            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                            operation_type TEXT NOT NULL,  -- 'write' or 'edit'
                            file_path TEXT NOT NULL,
                            content TEXT,                  -- Full content for writes
                            old_string TEXT,               -- Old text for edits
                            new_string TEXT,               -- New text for edits
                            replace_all BOOLEAN DEFAULT FALSE,
                            sequence_num INTEGER DEFAULT 0,
                            timestamp TEXT,
                            operation_hash TEXT UNIQUE,    -- For deduplication
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    # Add indexes for common queries
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_file_ops_session ON file_operations(session_id)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_file_ops_path ON file_operations(file_path)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_file_ops_created ON file_operations(created_at)")
                    cur.execute("INSERT INTO schema_version (version, description) VALUES (2, 'Add file_operations table')")
                    conn.commit()
                    results["migrations_run"].append({"version": 2, "name": "add_file_operations"})
                    pg_version = 2

                results["current_version"] = pg_version

    except Exception as e:
        log(f"Postgres migration error: {e}")
        results["status"] = "failed"
        results["error"] = str(e)

    return results
