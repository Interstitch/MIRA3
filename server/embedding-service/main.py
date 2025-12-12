"""
MIRA Embedding Service

A centralized service that:
1. Automatically polls Postgres for new/updated sessions
2. Computes embeddings and stores them in Qdrant
3. Provides semantic search API

The service runs a background worker that continuously indexes new content.
Clients only need to query the /search endpoint - no manual embedding calls needed.

Endpoints:
- POST /search    - Semantic search via Qdrant
- GET  /health    - Health check
- GET  /stats     - Collection statistics
- POST /reindex/all - Manual full reindex (admin)
"""

import os
import asyncio
import logging
import threading
import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

# ===========================================
# Configuration
# ===========================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mira_sessions")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "mira")
POSTGRES_USER = os.getenv("POSTGRES_USER", "mira")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional: enables auth
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Background worker settings
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))  # seconds between polls
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))  # sessions per batch

# Database schema - auto-created on startup
SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);

-- Projects
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_projects_path ON projects(path);

-- Sessions
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    summary TEXT,
    keywords TEXT[],
    facts TEXT[],
    task_description TEXT,
    git_branch TEXT,
    models_used TEXT[],
    tools_used TEXT[],
    files_touched TEXT[],
    message_count INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    llm_processed_at TIMESTAMPTZ,
    vector_indexed_at TIMESTAMPTZ,
    UNIQUE(project_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_llm_processed ON sessions(llm_processed_at);

-- Archives (conversation content)
CREATE TABLE IF NOT EXISTS archives (
    id SERIAL PRIMARY KEY,
    session_id INTEGER UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER,
    line_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_archives_session ON archives(session_id);

-- Artifacts
CREATE TABLE IF NOT EXISTS artifacts (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,
    content TEXT NOT NULL,
    language TEXT,
    line_count INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_session_type_content ON artifacts(session_id, artifact_type, md5(content));
CREATE INDEX IF NOT EXISTS idx_artifacts_fts ON artifacts USING gin(to_tsvector('english', content));

-- Decisions
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
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(project_id, decision)
);
CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project_id);
CREATE INDEX IF NOT EXISTS idx_decisions_category ON decisions(category);

-- Error Patterns
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
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(project_id, signature)
);
CREATE INDEX IF NOT EXISTS idx_errors_project ON error_patterns(project_id);
CREATE INDEX IF NOT EXISTS idx_errors_signature ON error_patterns(signature);

-- Custodian (user preferences)
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
);
CREATE INDEX IF NOT EXISTS idx_custodian_key ON custodian(key);
CREATE INDEX IF NOT EXISTS idx_custodian_category ON custodian(category);

-- Name Candidates
CREATE TABLE IF NOT EXISTS name_candidates (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    pattern_type TEXT,
    source_session TEXT,
    context TEXT,
    extracted_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, source_session)
);
CREATE INDEX IF NOT EXISTS idx_name_candidates_name ON name_candidates(name);

-- Lifecycle Patterns
CREATE TABLE IF NOT EXISTS lifecycle_patterns (
    id SERIAL PRIMARY KEY,
    pattern TEXT UNIQUE NOT NULL,
    confidence REAL DEFAULT 0.5,
    occurrences INTEGER DEFAULT 1,
    source_sessions TEXT[],
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Concepts
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
);
CREATE INDEX IF NOT EXISTS idx_concepts_project ON concepts(project_id);
CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts(concept_type);

-- File Operations
CREATE TABLE IF NOT EXISTS file_operations (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    session_id TEXT,
    file_path TEXT NOT NULL,
    operation TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_file_ops_project ON file_operations(project_id);
CREATE INDEX IF NOT EXISTS idx_file_ops_file ON file_operations(file_path);

-- Record schema version
INSERT INTO schema_version (version) VALUES (5) ON CONFLICT (version) DO NOTHING;
"""

# Logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("embedding-service")

# ===========================================
# FastAPI App
# ===========================================

app = FastAPI(
    title="MIRA Embedding Service",
    description="Automatic embedding computation and semantic search for MIRA",
    version="2.0.0"
)

# Global state
model: Optional[SentenceTransformer] = None
qdrant: Optional[QdrantClient] = None
pg_pool: Optional[pool.SimpleConnectionPool] = None
indexer_thread: Optional[threading.Thread] = None
indexer_running = False
indexer_stats = {
    "last_poll": None,
    "sessions_indexed": 0,
    "errors": 0,
    "last_error": None,
}


# ===========================================
# Request/Response Models
# ===========================================

class SearchRequest(BaseModel):
    """Request for semantic search."""
    query: str
    project_path: Optional[str] = None
    project_id: Optional[int] = None
    limit: int = 10


class SearchResult(BaseModel):
    """A single search result."""
    session_id: str
    score: float
    metadata: Dict[str, Any]


# ===========================================
# Database Connection Pool
# ===========================================

@contextmanager
def get_db_connection():
    """Get a connection from the pool."""
    conn = pg_pool.getconn()
    try:
        yield conn
    finally:
        pg_pool.putconn(conn)


# ===========================================
# Background Indexer
# ===========================================

def index_session(row: dict) -> bool:
    """Index a single session to Qdrant. Returns True on success."""
    try:
        # Get text to embed (prefer archive content, fall back to summary)
        text = ""
        if row.get('content'):
            text = row['content'][:8000]
        elif row.get('summary'):
            text = row['summary']

        if not text:
            logger.debug(f"Skipping {row['session_id'][:12]}: no content")
            return False

        # Compute embedding
        embedding = model.encode(text).tolist()

        # Generate stable ID from session_id
        point_id = hash(row['session_id']) & 0x7FFFFFFFFFFFFFFF

        # Upsert to Qdrant
        qdrant.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "session_id": row['session_id'],
                        "project_id": row['project_id'],
                        "project_path": row.get('project_path', ''),
                        "summary": row.get('summary', ''),
                        "keywords": row.get('keywords', []),
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                )
            ]
        )
        return True
    except Exception as e:
        logger.error(f"Failed to index {row.get('session_id', 'unknown')}: {e}")
        return False


def mark_sessions_indexed(conn, session_ids: List[int]):
    """Mark sessions as indexed in Postgres."""
    if not session_ids:
        return

    cur = conn.cursor()
    cur.execute("""
        UPDATE sessions
        SET vector_indexed_at = NOW()
        WHERE id = ANY(%s)
    """, (session_ids,))
    conn.commit()
    cur.close()


def poll_and_index():
    """Poll Postgres for unindexed sessions and index them."""
    global indexer_stats

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()

            # Find sessions that need indexing:
            # - No vector_indexed_at timestamp, OR
            # - Archive updated after vector_indexed_at
            cur.execute("""
                SELECT s.id, s.session_id, s.project_id, s.summary, s.keywords,
                       p.path as project_path, a.content, a.updated_at as archive_updated
                FROM sessions s
                JOIN projects p ON s.project_id = p.id
                LEFT JOIN archives a ON s.id = a.session_id
                WHERE s.vector_indexed_at IS NULL
                   OR (a.updated_at IS NOT NULL AND a.updated_at > s.vector_indexed_at)
                ORDER BY s.created_at DESC
                LIMIT %s
            """, (BATCH_SIZE,))

            rows = cur.fetchall()
            cur.close()

            if not rows:
                return 0

            logger.info(f"Found {len(rows)} sessions to index")

            indexed_ids = []
            for row in rows:
                if index_session(dict(row)):
                    indexed_ids.append(row['id'])
                    indexer_stats["sessions_indexed"] += 1

            # Mark as indexed
            if indexed_ids:
                mark_sessions_indexed(conn, indexed_ids)
                logger.info(f"Indexed {len(indexed_ids)} sessions")

            return len(indexed_ids)

    except Exception as e:
        logger.error(f"Poll error: {e}")
        indexer_stats["errors"] += 1
        indexer_stats["last_error"] = str(e)
        return 0


def indexer_worker():
    """Background worker that continuously polls and indexes."""
    global indexer_running, indexer_stats

    logger.info(f"Indexer started (poll_interval={POLL_INTERVAL}s, batch_size={BATCH_SIZE})")

    while indexer_running:
        try:
            indexer_stats["last_poll"] = datetime.utcnow().isoformat()
            indexed = poll_and_index()

            # If we indexed a full batch, poll again immediately
            if indexed >= BATCH_SIZE:
                continue

        except Exception as e:
            logger.error(f"Indexer error: {e}")
            indexer_stats["errors"] += 1
            indexer_stats["last_error"] = str(e)

        # Wait before next poll
        time.sleep(POLL_INTERVAL)

    logger.info("Indexer stopped")


# ===========================================
# Lifecycle Events
# ===========================================

@app.on_event("startup")
async def startup():
    """Initialize model, connections, and start background indexer."""
    global model, qdrant, pg_pool, indexer_thread, indexer_running

    # Load embedding model
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Connect to Qdrant
    logger.info(f"Connecting to Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    qdrant = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY  # None if not set (auth disabled)
    )

    # Ensure collection exists
    try:
        collections = qdrant.get_collections().collections
        collection_names = [c.name for c in collections]

        if QDRANT_COLLECTION not in collection_names:
            logger.info(f"Creating collection: {QDRANT_COLLECTION}")
            qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qdrant_models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=qdrant_models.Distance.COSINE
                )
            )
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")

    # Create Postgres connection pool
    logger.info(f"Connecting to Postgres: {POSTGRES_HOST}:{POSTGRES_PORT}")
    pg_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        cursor_factory=RealDictCursor
    )

    # Initialize database schema (skip if tables already exist)
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            # Check if sessions table exists (core table)
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'sessions'
                )
            """)
            tables_exist = cur.fetchone()['exists']

            if tables_exist:
                logger.info("Database schema already exists, skipping initialization")
            else:
                cur.execute(SCHEMA_SQL)
                conn.commit()
                logger.info("Database schema initialized")
            cur.close()
    except Exception as e:
        logger.error(f"Failed to initialize schema: {e}")

    # Start background indexer
    indexer_running = True
    indexer_thread = threading.Thread(target=indexer_worker, daemon=True)
    indexer_thread.start()

    logger.info("Startup complete - background indexer running")


@app.on_event("shutdown")
async def shutdown():
    """Stop background indexer and close connections."""
    global indexer_running, pg_pool

    logger.info("Shutting down...")
    indexer_running = False

    if indexer_thread:
        indexer_thread.join(timeout=5)

    if pg_pool:
        pg_pool.closeall()

    logger.info("Shutdown complete")


# ===========================================
# Endpoints
# ===========================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    qdrant_ok = False
    postgres_ok = False

    try:
        qdrant.get_collections()
        qdrant_ok = True
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        postgres_ok = True
    except Exception as e:
        logger.warning(f"Postgres health check failed: {e}")

    status = "healthy" if (qdrant_ok and postgres_ok) else "degraded"

    return {
        "status": status,
        "qdrant": "connected" if qdrant_ok else "disconnected",
        "postgres": "connected" if postgres_ok else "disconnected",
        "model": MODEL_NAME,
        "collection": QDRANT_COLLECTION,
        "indexer": {
            "running": indexer_running,
            "last_poll": indexer_stats["last_poll"],
            "sessions_indexed": indexer_stats["sessions_indexed"],
            "errors": indexer_stats["errors"],
        }
    }


@app.post("/search")
async def search(request: SearchRequest):
    """Semantic search via Qdrant."""
    try:
        # Compute query embedding
        query_embedding = model.encode(request.query).tolist()

        # Build filter conditions
        filter_conditions = []

        if request.project_id is not None:
            filter_conditions.append(
                qdrant_models.FieldCondition(
                    key="project_id",
                    match=qdrant_models.MatchValue(value=request.project_id)
                )
            )

        if request.project_path:
            filter_conditions.append(
                qdrant_models.FieldCondition(
                    key="project_path",
                    match=qdrant_models.MatchValue(value=request.project_path)
                )
            )

        query_filter = None
        if filter_conditions:
            query_filter = qdrant_models.Filter(must=filter_conditions)

        # Search Qdrant
        search_result = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            query_filter=query_filter,
            limit=request.limit
        )
        results = search_result.points

        logger.info(f"Search '{request.query[:50]}...' returned {len(results)} results")

        return {
            "results": [
                {
                    "session_id": r.payload.get("session_id"),
                    "score": r.score,
                    "metadata": r.payload
                }
                for r in results
            ],
            "query": request.query,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex/all")
async def reindex_all():
    """Manual full reindex - resets all vector_indexed_at timestamps."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()

            # Reset all indexed timestamps
            cur.execute("UPDATE sessions SET vector_indexed_at = NULL")
            conn.commit()

            # Count sessions
            cur.execute("SELECT COUNT(*) as count FROM sessions")
            count = cur.fetchone()['count']
            cur.close()

        logger.info(f"Reset vector_indexed_at for {count} sessions - indexer will reprocess")

        return {
            "message": "Reindex triggered",
            "sessions_queued": count,
            "note": "Background indexer will process sessions automatically"
        }

    except Exception as e:
        logger.error(f"Reindex trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    """Get collection and indexer statistics."""
    try:
        collection_info = qdrant.get_collection(QDRANT_COLLECTION)

        # Count unindexed sessions
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE vector_indexed_at IS NULL) as unindexed
                FROM sessions
            """)
            row = cur.fetchone()
            cur.close()

        return {
            "collection": QDRANT_COLLECTION,
            "points_count": collection_info.points_count,
            "status": str(collection_info.status),
            "postgres": {
                "total_sessions": row['total'],
                "unindexed_sessions": row['unindexed'],
            },
            "indexer": indexer_stats
        }
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# Main Entry Point
# ===========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)

