"""
MIRA3 Schema Migration Framework

Manages database schema versioning and migrations for:
- Local SQLite databases (artifacts, custodian, insights, concepts, local_store)
- Central Postgres database

Migrations are idempotent and can be safely re-run.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .utils import log, get_mira_path
from .db_manager import get_db_manager

# Current schema version
CURRENT_VERSION = 6

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

# Custodian database schema
CUSTODIAN_SCHEMA = """
-- Identity table - who is the custodian
CREATE TABLE IF NOT EXISTS identity (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    source_session TEXT,
    learned_at TEXT
);

-- Preferences table - what they prefer
CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    preference TEXT NOT NULL,
    value TEXT,
    evidence TEXT,
    frequency INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.5,
    first_seen TEXT,
    last_seen TEXT,
    source_sessions TEXT
);

-- Rules table - explicit always/never rules with enhanced features
CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type TEXT NOT NULL,
    rule_text TEXT NOT NULL,
    normalized_text TEXT,
    context TEXT,
    scope TEXT,
    frequency INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.8,
    first_seen TEXT,
    last_seen TEXT,
    source_sessions TEXT,
    revoked INTEGER DEFAULT 0,
    revoked_at TEXT
);

-- Danger zones - files/modules that caused issues
CREATE TABLE IF NOT EXISTS danger_zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path_pattern TEXT NOT NULL,
    issue_description TEXT,
    issue_count INTEGER DEFAULT 1,
    last_issue TEXT,
    resolution TEXT,
    source_sessions TEXT
);

-- Work patterns - how they work
CREATE TABLE IF NOT EXISTS work_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,
    pattern_description TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.5,
    first_seen TEXT,
    last_seen TEXT
);

-- Environment-specific prerequisites learned from conversations
CREATE TABLE IF NOT EXISTS prerequisites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    environment TEXT NOT NULL,          -- User-defined env name (lowercase)
    action TEXT NOT NULL,               -- Human description of what to do
    command TEXT,                       -- Shell command to run
    check_command TEXT,                 -- Verification command (exit 0 = met)
    reason TEXT,                        -- Why it's needed
    confidence REAL DEFAULT 0.5,        -- 0.0-1.0 confidence score
    frequency INTEGER DEFAULT 1,        -- Times mentioned/confirmed
    source_session TEXT,                -- Session ID where first learned
    learned_at TEXT,                    -- ISO timestamp when learned
    last_triggered TEXT,                -- Last time shown as alert
    last_confirmed TEXT,                -- Last user confirmation
    suppressed INTEGER DEFAULT 0,       -- User said "don't remind me"
    UNIQUE(environment, action)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_pref_category ON preferences(category);
CREATE INDEX IF NOT EXISTS idx_rules_type ON rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_danger_path ON danger_zones(path_pattern);
CREATE INDEX IF NOT EXISTS idx_prereq_env ON prerequisites(environment);
CREATE INDEX IF NOT EXISTS idx_prereq_confidence ON prerequisites(confidence DESC);
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

    init_artifact_db()
    init_custodian_db()
    init_insights_db()
    init_concepts_db()
    init_local_db()

    return True


@migration(2, "add_audit_indexes")
def migrate_v2(db_manager):
    """No-op: audit.db has been removed."""
    log("Migration v2: Skipped (audit.db removed)")
    return True


@migration(3, "add_name_candidates")
def migrate_v3(db_manager):
    """Add name_candidates table for confidence-weighted name selection."""
    log("Migration v3: Adding name_candidates table to custodian.db")

    from .custodian import CUSTODIAN_DB

    try:
        # Create name_candidates table for tracking all name extractions
        db_manager.execute_write(
            CUSTODIAN_DB,
            """CREATE TABLE IF NOT EXISTS name_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                confidence REAL NOT NULL,
                pattern_type TEXT NOT NULL,
                source_session TEXT NOT NULL,
                context TEXT,
                extracted_at TEXT NOT NULL,
                UNIQUE(name, source_session)
            )""",
            ()
        )

        # Add indexes for efficient querying
        db_manager.execute_write(
            CUSTODIAN_DB,
            "CREATE INDEX IF NOT EXISTS idx_name_candidates_name ON name_candidates(name)",
            ()
        )
        db_manager.execute_write(
            CUSTODIAN_DB,
            "CREATE INDEX IF NOT EXISTS idx_name_candidates_conf ON name_candidates(confidence DESC)",
            ()
        )

        log("  Created name_candidates table")

    except Exception as e:
        log(f"Migration v3 error: {e}")
        # Don't fail - table might already exist
        pass

    return True


@migration(4, "add_missing_custodian_columns")
def migrate_v4(db_manager):
    """Add missing columns to custodian tables (rules, identity)."""
    log("Migration v4: Adding missing columns to custodian tables")

    from .custodian import CUSTODIAN_DB

    # Helper to add column if it doesn't exist
    def add_column_if_missing(table: str, column: str, definition: str):
        try:
            row = db_manager.execute_read_one(
                CUSTODIAN_DB,
                f"SELECT COUNT(*) as cnt FROM pragma_table_info('{table}') WHERE name='{column}'",
                ()
            )
            if row and row['cnt'] == 0:
                db_manager.execute_write(
                    CUSTODIAN_DB,
                    f"ALTER TABLE {table} ADD COLUMN {column} {definition}",
                    ()
                )
                log(f"  Added {column} to {table}")
                return True
        except Exception as e:
            log(f"  Failed to add {column} to {table}: {e}")
        return False

    # Add missing columns to rules table
    add_column_if_missing("rules", "normalized_text", "TEXT")
    add_column_if_missing("rules", "scope", "TEXT")
    add_column_if_missing("rules", "confidence", "REAL DEFAULT 0.8")
    add_column_if_missing("rules", "revoked", "INTEGER DEFAULT 0")
    add_column_if_missing("rules", "revoked_at", "TEXT")

    # Add confidence to identity table if missing
    add_column_if_missing("identity", "confidence", "REAL DEFAULT 0.5")

    return True


@migration(5, "add_vocabulary_table")
def migrate_v5(db_manager):
    """Add vocabulary table for fuzzy search typo correction."""
    log("Migration v5: Adding vocabulary table to local_store.db")

    from .local_store import LOCAL_DB

    try:
        # Create vocabulary table for typo correction
        db_manager.execute_write(
            LOCAL_DB,
            """CREATE TABLE IF NOT EXISTS vocabulary (
                term TEXT PRIMARY KEY,
                frequency INTEGER DEFAULT 1,
                source TEXT DEFAULT 'unknown',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
            ()
        )

        # Add index for efficient frequency-based lookups
        db_manager.execute_write(
            LOCAL_DB,
            "CREATE INDEX IF NOT EXISTS idx_vocabulary_frequency ON vocabulary(frequency DESC)",
            ()
        )

        log("  Created vocabulary table for fuzzy matching")

    except Exception as e:
        log(f"Migration v5 error: {e}")
        # Don't fail - table might already exist
        pass

    return True


@migration(6, "add_code_history_db")
def migrate_v6(db_manager):
    """Add code_history.db for file operation tracking and reconstruction."""
    log("Migration v6: Creating code_history database")

    try:
        from .code_history import init_code_history_db
        init_code_history_db()
        log("  Created code_history.db with file tracking tables")

    except Exception as e:
        log(f"Migration v6 error: {e}")
        # Don't fail - tables might already exist
        pass

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
        start_time = datetime.now(timezone.utc)

        try:
            success = migrate_func(db)
            duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            if success:
                # Record successful migration
                db.execute_write(
                    MIGRATIONS_DB,
                    """INSERT INTO schema_migrations (version, name, applied_at, duration_ms, status)
                       VALUES (?, ?, ?, ?, 'success')""",
                    (version, name, datetime.now(timezone.utc).isoformat(), duration_ms)
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
            duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            log(f"Migration v{version} failed: {e}")

            # Record failed migration
            db.execute_write(
                MIGRATIONS_DB,
                """INSERT INTO schema_migrations (version, name, applied_at, duration_ms, status)
                   VALUES (?, ?, ?, ?, 'failed')""",
                (version, name, datetime.now(timezone.utc).isoformat(), duration_ms)
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

    Creates base tables if needed, then runs incremental migrations.
    This ensures users only need to provide credentials - MIRA handles
    all schema setup automatically.

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
                    # Fresh install - create all base tables
                    log("Creating Postgres base schema (first run)")

                    # Schema version tracking
                    cur.execute("""
                        CREATE TABLE schema_version (
                            version INTEGER PRIMARY KEY,
                            applied_at TIMESTAMPTZ DEFAULT NOW(),
                            description TEXT
                        )
                    """)

                    # Projects table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS projects (
                            id SERIAL PRIMARY KEY,
                            project_path TEXT UNIQUE NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_path ON projects(project_path)")

                    # Sessions table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS sessions (
                            id SERIAL PRIMARY KEY,
                            session_id TEXT UNIQUE NOT NULL,
                            project_id INTEGER REFERENCES projects(id),
                            slug TEXT,
                            summary TEXT,
                            task_description TEXT,
                            git_branch TEXT,
                            keywords TEXT[],
                            message_count INTEGER DEFAULT 0,
                            file_hash TEXT,
                            indexed_at TIMESTAMPTZ DEFAULT NOW(),
                            llm_processed_at TIMESTAMPTZ
                        )
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id)")

                    # Archives table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS archives (
                            id SERIAL PRIMARY KEY,
                            session_id TEXT UNIQUE NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                            content TEXT,
                            archived_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)

                    # Artifacts table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS artifacts (
                            id SERIAL PRIMARY KEY,
                            project_id INTEGER REFERENCES projects(id),
                            session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
                            artifact_type TEXT NOT NULL,
                            content TEXT NOT NULL,
                            language TEXT,
                            context TEXT,
                            line_count INTEGER,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type)")

                    # Decisions table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS decisions (
                            id SERIAL PRIMARY KEY,
                            project_id INTEGER REFERENCES projects(id),
                            decision TEXT NOT NULL,
                            category TEXT DEFAULT 'general',
                            reasoning TEXT,
                            alternatives TEXT[],
                            confidence REAL DEFAULT 0.5,
                            source TEXT DEFAULT 'regex',
                            session_id TEXT,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project_id)")

                    # Error patterns table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS error_patterns (
                            id SERIAL PRIMARY KEY,
                            project_id INTEGER REFERENCES projects(id),
                            signature TEXT NOT NULL,
                            error_type TEXT,
                            error_text TEXT NOT NULL,
                            solution TEXT,
                            file_path TEXT,
                            occurrences INTEGER DEFAULT 1,
                            first_seen TIMESTAMPTZ DEFAULT NOW(),
                            last_seen TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_errors_project ON error_patterns(project_id)")

                    # Custodian table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS custodian (
                            id SERIAL PRIMARY KEY,
                            key TEXT UNIQUE NOT NULL,
                            value TEXT NOT NULL,
                            category TEXT,
                            confidence REAL DEFAULT 0.5,
                            frequency INTEGER DEFAULT 1,
                            source_sessions TEXT[],
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_custodian_key ON custodian(key)")

                    # Concepts table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS concepts (
                            id SERIAL PRIMARY KEY,
                            project_id INTEGER REFERENCES projects(id),
                            concept_type TEXT NOT NULL,
                            name TEXT NOT NULL,
                            description TEXT,
                            confidence REAL DEFAULT 0.5,
                            frequency INTEGER DEFAULT 1,
                            source_sessions TEXT[],
                            updated_at TIMESTAMPTZ DEFAULT NOW(),
                            UNIQUE(project_id, concept_type, name)
                        )
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_concepts_project ON concepts(project_id)")

                    cur.execute(
                        "INSERT INTO schema_version (version, description) VALUES (1, 'Initial schema with all base tables')"
                    )
                    conn.commit()
                    results["migrations_run"].append({"version": 1, "name": "initial_full_schema"})
                    log("  Created all base tables (projects, sessions, archives, artifacts, decisions, error_patterns, custodian, concepts)")

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

                # Postgres migration v3: Add unique indexes and dedupe artifacts/decisions
                if pg_version < 3:
                    log("Postgres migration v3: Adding unique indexes and deduping")

                    # First, dedupe artifacts - keep the oldest record of each duplicate
                    cur.execute("""
                        DELETE FROM artifacts a
                        USING artifacts b
                        WHERE a.id > b.id
                          AND a.session_id = b.session_id
                          AND a.artifact_type = b.artifact_type
                          AND a.content = b.content
                    """)
                    artifact_deleted = cur.rowcount
                    log(f"  Deleted {artifact_deleted} duplicate artifacts")

                    # Add unique INDEX (not constraint - Postgres doesn't allow md5() in constraints)
                    cur.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_session_type_content
                        ON artifacts (session_id, artifact_type, md5(content))
                    """)
                    log("  Added unique index to artifacts")

                    # Dedupe decisions - keep the oldest record of each duplicate
                    cur.execute("""
                        DELETE FROM decisions a
                        USING decisions b
                        WHERE a.id > b.id
                          AND a.session_id = b.session_id
                          AND a.decision = b.decision
                    """)
                    decision_deleted = cur.rowcount
                    log(f"  Deleted {decision_deleted} duplicate decisions")

                    # Add unique INDEX to decisions
                    cur.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_decisions_session_decision
                        ON decisions (session_id, md5(decision))
                    """)
                    log("  Added unique index to decisions")

                    cur.execute("INSERT INTO schema_version (version, description) VALUES (3, 'Add unique indexes, dedupe artifacts/decisions')")
                    conn.commit()
                    results["migrations_run"].append({"version": 3, "name": "dedupe_and_indexes"})
                    pg_version = 3

                # Postgres migration v4: Add name_candidates table
                if pg_version < 4:
                    log("Postgres migration v4: Adding name_candidates table")
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS name_candidates (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            confidence REAL NOT NULL,
                            pattern_type TEXT NOT NULL,
                            source_session TEXT NOT NULL,
                            context TEXT,
                            extracted_at TIMESTAMPTZ DEFAULT NOW(),
                            UNIQUE(name, source_session)
                        )
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_name_candidates_name ON name_candidates(name)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_name_candidates_conf ON name_candidates(confidence DESC)")
                    cur.execute("INSERT INTO schema_version (version, description) VALUES (4, 'Add name_candidates table for confidence-weighted name selection')")
                    conn.commit()
                    results["migrations_run"].append({"version": 4, "name": "add_name_candidates"})
                    pg_version = 4
                    log("  Created name_candidates table in Postgres")

                # Postgres migration v5: Add LLM extraction tracking columns
                if pg_version < 5:
                    log("Postgres migration v5: Adding LLM extraction tracking columns")

                    # Add llm_processed_at column to sessions table
                    cur.execute("""
                        ALTER TABLE sessions
                        ADD COLUMN IF NOT EXISTS llm_processed_at TIMESTAMPTZ
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_sessions_llm_processed
                        ON sessions(llm_processed_at)
                    """)
                    log("  Added llm_processed_at to sessions")

                    # Add source column to decisions table to track regex vs llm extraction
                    cur.execute("""
                        ALTER TABLE decisions
                        ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'regex'
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_decisions_source
                        ON decisions(source)
                    """)
                    log("  Added source column to decisions")

                    cur.execute("INSERT INTO schema_version (version, description) VALUES (5, 'Add llm_processed_at to sessions and source to decisions')")
                    conn.commit()
                    results["migrations_run"].append({"version": 5, "name": "llm_extraction_tracking"})
                    pg_version = 5

                # Postgres migration v6: Fix file_operations column name
                if pg_version < 6:
                    log("Postgres migration v6: Fixing file_operations column name")

                    # Check if 'operation' column exists (old schema) and rename to 'operation_type'
                    cur.execute("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = 'file_operations' AND column_name = 'operation'
                    """)
                    if cur.fetchone():
                        cur.execute("ALTER TABLE file_operations RENAME COLUMN operation TO operation_type")
                        log("  Renamed operation -> operation_type")

                    # Also add missing columns if they don't exist
                    cur.execute("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = 'file_operations' AND column_name = 'content'
                    """)
                    if not cur.fetchone():
                        cur.execute("ALTER TABLE file_operations ADD COLUMN content TEXT")
                        cur.execute("ALTER TABLE file_operations ADD COLUMN old_string TEXT")
                        cur.execute("ALTER TABLE file_operations ADD COLUMN new_string TEXT")
                        cur.execute("ALTER TABLE file_operations ADD COLUMN replace_all BOOLEAN DEFAULT FALSE")
                        cur.execute("ALTER TABLE file_operations ADD COLUMN sequence_num INTEGER")
                        cur.execute("ALTER TABLE file_operations ADD COLUMN operation_hash TEXT UNIQUE")
                        cur.execute("ALTER TABLE file_operations ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW()")
                        log("  Added missing columns to file_operations")

                    # Handle both old (version, applied_at) and new (version, description) schema_version formats
                    cur.execute("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = 'schema_version' AND column_name = 'description'
                    """)
                    if cur.fetchone():
                        cur.execute("INSERT INTO schema_version (version, description) VALUES (6, 'Fix file_operations schema')")
                    else:
                        cur.execute("INSERT INTO schema_version (version) VALUES (6)")
                    conn.commit()
                    results["migrations_run"].append({"version": 6, "name": "fix_file_operations_schema"})
                    pg_version = 6

                results["current_version"] = pg_version

    except Exception as e:
        log(f"Postgres migration error: {e}")
        results["status"] = "failed"
        results["error"] = str(e)

    return results
