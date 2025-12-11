"""
MIRA PostgreSQL Backend

Provides structured data storage against a centralized PostgreSQL server.

Security:
- ALL queries use parameterized syntax ($1, $2, etc.) - NEVER string interpolation
- Connection only to Tailscale IP (network-level auth)
- No sensitive data in logs
- Uses plainto_tsquery() for FTS (sanitizes input)

Efficiency:
- Connection pooling
- Lazy connection (only connect when needed)
- Batch operations for inserts
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

log = logging.getLogger(__name__)

# psycopg2 - imported lazily to avoid dependency if not using central
_psycopg2 = None
_pool = None


def _get_psycopg2():
    """Lazy import of psycopg2 to avoid import errors if not installed."""
    global _psycopg2
    if _psycopg2 is not None:
        return _psycopg2
    try:
        import psycopg2
        import psycopg2.pool
        import psycopg2.extras
        _psycopg2 = {
            "module": psycopg2,
            "pool": psycopg2.pool,
            "extras": psycopg2.extras,
        }
        return _psycopg2
    except ImportError:
        log.warning("psycopg2 not installed, central structured storage unavailable")
        return None


@dataclass
class Project:
    """Project record."""
    id: int
    path: str
    slug: Optional[str]


@dataclass
class Session:
    """Session record."""
    id: int
    project_id: int
    session_id: str
    summary: Optional[str]
    keywords: List[str]
    facts: List[str]
    task_description: Optional[str]
    git_branch: Optional[str]
    models_used: List[str]
    tools_used: List[str]
    files_touched: List[str]
    message_count: int
    started_at: Optional[str]
    ended_at: Optional[str]


class PostgresBackend:
    """
    PostgreSQL database backend for MIRA.

    Provides:
    - Project management
    - Session storage
    - Artifact storage
    - Custodian preferences
    - Error patterns
    - Decisions
    - Concepts

    All queries use parameterized syntax for security.
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        pool_size: int = 6,
        timeout: int = 30,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self.timeout = timeout
        self._pool = None
        self._healthy = False
        self._last_health_check = 0
        self._health_check_interval = 60

        # Cache for project IDs (immutable lookups)
        self._project_cache: Dict[str, int] = {}
        self._project_cache_time: Dict[str, float] = {}
        self._project_cache_ttl = 3600  # 1 hour

        # Pool warmup config
        self._min_connections = 2  # Keep warm connections ready

    def _get_pool(self):
        """Get or create connection pool (lazy initialization)."""
        if self._pool is not None:
            return self._pool

        pg = _get_psycopg2()
        if pg is None:
            raise ImportError("psycopg2 not installed")

        log.info(f"Connecting to PostgreSQL at {self.host}:{self.port}/{self.database}")

        try:
            # Use minconn=2 to keep warm connections ready
            # This avoids cold-start latency on each query
            self._pool = pg["pool"].ThreadedConnectionPool(
                minconn=self._min_connections,
                maxconn=self.pool_size,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=self.timeout,
                # TCP keepalive to prevent Tailscale/firewall from closing idle connections
                keepalives=1,
                keepalives_idle=30,      # Start keepalive after 30s idle
                keepalives_interval=10,  # Send keepalive every 10s
                keepalives_count=3,      # Give up after 3 failed keepalives
            )
            self._healthy = True
            self._last_health_check = time.time()
            log.info(f"Connected to PostgreSQL (pool: min={self._min_connections}, max={self.pool_size})")
            return self._pool
        except Exception as e:
            log.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    @contextmanager
    def _get_connection(self) -> Generator:
        """Get a connection from the pool (context manager)."""
        pool = self._get_pool()
        conn = pool.getconn()
        close_conn = False
        try:
            # Validate connection is alive (catches stale Tailscale connections)
            if conn.closed:
                pool.putconn(conn, close=True)
                conn = pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            # If connection error, mark it bad so pool doesn't reuse
            if "connection" in str(e).lower() or "server closed" in str(e).lower():
                close_conn = True
            raise
        finally:
            pool.putconn(conn, close=close_conn)

    def is_healthy(self) -> bool:
        """Check if PostgreSQL connection is healthy."""
        now = time.time()
        if now - self._last_health_check < self._health_check_interval:
            return self._healthy

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            self._healthy = True
            self._last_health_check = now
            return True
        except Exception as e:
            log.warning(f"PostgreSQL health check failed: {e}")
            self._healthy = False
            self._last_health_check = now
            return False

    # ==================== Projects ====================

    def get_or_create_project(
        self,
        path: str,
        slug: Optional[str] = None,
        git_remote: Optional[str] = None
    ) -> int:
        """
        Get project ID, creating if necessary.

        Lookup priority:
        1. If git_remote provided, match by git_remote (canonical cross-machine identity)
        2. Fall back to path matching

        When creating new project, stores both path and git_remote.
        Uses cache for performance.
        """
        # Cache key is git_remote if available, otherwise path
        cache_key = git_remote or path

        # Check cache first
        now = time.time()
        if cache_key in self._project_cache:
            cache_time = self._project_cache_time.get(cache_key, 0)
            if now - cache_time < self._project_cache_ttl:
                return self._project_cache[cache_key]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                project_id = None

                # Try to find by git_remote first (cross-machine identity)
                if git_remote:
                    cur.execute(
                        "SELECT id FROM projects WHERE git_remote = %s",
                        (git_remote,)
                    )
                    row = cur.fetchone()
                    if row:
                        project_id = row[0]
                        # Update path if it changed (different machine)
                        cur.execute(
                            "UPDATE projects SET path = %s WHERE id = %s AND path != %s",
                            (path, project_id, path)
                        )

                # Fall back to path matching
                if project_id is None:
                    cur.execute(
                        "SELECT id FROM projects WHERE path = %s",
                        (path,)
                    )
                    row = cur.fetchone()
                    if row:
                        project_id = row[0]
                        # Update git_remote if we now have it
                        if git_remote:
                            cur.execute(
                                "UPDATE projects SET git_remote = %s WHERE id = %s AND git_remote IS NULL",
                                (git_remote, project_id)
                            )

                # Create new project if not found
                if project_id is None:
                    cur.execute(
                        "INSERT INTO projects (path, slug, git_remote) VALUES (%s, %s, %s) RETURNING id",
                        (path, slug, git_remote)
                    )
                    project_id = cur.fetchone()[0]

                # Cache it (by both keys if git_remote is provided)
                self._project_cache[cache_key] = project_id
                self._project_cache_time[cache_key] = now
                if git_remote and path != git_remote:
                    self._project_cache[path] = project_id
                    self._project_cache_time[path] = now

                return project_id

    # ==================== Sessions ====================

    def session_exists(self, project_id: int, session_id: str) -> bool:
        """Check if a session already exists."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM sessions WHERE project_id = %s AND session_id = %s",
                    (project_id, session_id)
                )
                return cur.fetchone() is not None

    def upsert_session(
        self,
        project_id: int,
        session_id: str,
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        facts: Optional[List[str]] = None,
        task_description: Optional[str] = None,
        git_branch: Optional[str] = None,
        models_used: Optional[List[str]] = None,
        tools_used: Optional[List[str]] = None,
        files_touched: Optional[List[str]] = None,
        message_count: int = 0,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
    ) -> int:
        """Insert or update a session. Returns session ID."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (
                        project_id, session_id, summary, keywords, facts,
                        task_description, git_branch, models_used, tools_used,
                        files_touched, message_count, started_at, ended_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (project_id, session_id) DO UPDATE SET
                        summary = EXCLUDED.summary,
                        keywords = EXCLUDED.keywords,
                        facts = EXCLUDED.facts,
                        task_description = EXCLUDED.task_description,
                        git_branch = EXCLUDED.git_branch,
                        models_used = EXCLUDED.models_used,
                        tools_used = EXCLUDED.tools_used,
                        files_touched = EXCLUDED.files_touched,
                        message_count = EXCLUDED.message_count,
                        started_at = EXCLUDED.started_at,
                        ended_at = EXCLUDED.ended_at
                    RETURNING id
                    """,
                    (
                        project_id, session_id, summary,
                        keywords or [], facts or [],
                        task_description, git_branch,
                        models_used or [], tools_used or [],
                        files_touched or [], message_count,
                        started_at, ended_at
                    )
                )
                return cur.fetchone()[0]

    def get_recent_sessions(
        self,
        project_id: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent sessions, optionally filtered by project."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if project_id:
                    cur.execute(
                        """
                        SELECT s.id, s.session_id, s.summary, s.keywords,
                               s.started_at, s.ended_at, p.path as project_path
                        FROM sessions s
                        JOIN projects p ON s.project_id = p.id
                        WHERE s.project_id = %s
                        ORDER BY s.started_at DESC NULLS LAST
                        LIMIT %s
                        """,
                        (project_id, limit)
                    )
                else:
                    cur.execute(
                        """
                        SELECT s.id, s.session_id, s.summary, s.keywords,
                               s.started_at, s.ended_at, p.path as project_path
                        FROM sessions s
                        JOIN projects p ON s.project_id = p.id
                        ORDER BY s.started_at DESC NULLS LAST
                        LIMIT %s
                        """,
                        (limit,)
                    )

                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    def search_sessions_fts(
        self,
        query: str,
        project_id: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full-text search on sessions."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Use plainto_tsquery for safe input handling
                if project_id:
                    cur.execute(
                        """
                        SELECT s.id, s.session_id, s.summary, s.keywords,
                               s.started_at, p.path as project_path,
                               ts_rank(
                                   to_tsvector('english', COALESCE(s.summary, '') || ' ' || COALESCE(s.task_description, '')),
                                   plainto_tsquery('english', %s)
                               ) as rank
                        FROM sessions s
                        JOIN projects p ON s.project_id = p.id
                        WHERE s.project_id = %s
                          AND to_tsvector('english', COALESCE(s.summary, '') || ' ' || COALESCE(s.task_description, ''))
                              @@ plainto_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s
                        """,
                        (query, project_id, query, limit)
                    )
                else:
                    cur.execute(
                        """
                        SELECT s.id, s.session_id, s.summary, s.keywords,
                               s.started_at, p.path as project_path,
                               ts_rank(
                                   to_tsvector('english', COALESCE(s.summary, '') || ' ' || COALESCE(s.task_description, '')),
                                   plainto_tsquery('english', %s)
                               ) as rank
                        FROM sessions s
                        JOIN projects p ON s.project_id = p.id
                        WHERE to_tsvector('english', COALESCE(s.summary, '') || ' ' || COALESCE(s.task_description, ''))
                              @@ plainto_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s
                        """,
                        (query, query, limit)
                    )

                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    # ==================== Artifacts ====================

    def insert_artifact(
        self,
        session_id: int,
        artifact_type: str,
        content: str,
        language: Optional[str] = None,
        line_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert an artifact. Returns artifact ID."""
        pg = _get_psycopg2()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO artifacts (session_id, artifact_type, content, language, line_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        session_id, artifact_type, content, language, line_count,
                        pg["extras"].Json(metadata) if metadata else None
                    )
                )
                return cur.fetchone()[0]

    def batch_insert_artifacts(
        self,
        artifacts: List[Dict[str, Any]],
    ) -> int:
        """
        Batch insert multiple artifacts in a single transaction.

        Much faster than individual inserts - reduces network round trips.

        Args:
            artifacts: List of artifact dicts with keys:
                - session_id (int): Postgres session ID
                - artifact_type (str): code_block, list, table, etc.
                - content (str): Artifact content
                - language (str, optional): Programming language
                - line_count (int, optional): Number of lines
                - metadata (dict, optional): Additional metadata

        Returns:
            Number of artifacts inserted
        """
        import time
        if not artifacts:
            return 0

        pg = _get_psycopg2()

        t_start = time.time()
        t_prep_start = time.time()

        with self._get_connection() as conn:
            t_conn = (time.time() - t_prep_start) * 1000
            with conn.cursor() as cur:
                # Use execute_values for efficient batch insert
                values = []
                for a in artifacts:
                    metadata_json = pg["extras"].Json(a.get("metadata")) if a.get("metadata") else None
                    values.append((
                        a.get("session_id"),
                        a.get("artifact_type"),
                        a.get("content"),
                        a.get("language"),
                        a.get("line_count"),
                        metadata_json,
                    ))

                t_prep = (time.time() - t_prep_start) * 1000 - t_conn

                # Batch insert with ON CONFLICT to skip duplicates
                t_exec_start = time.time()
                pg["extras"].execute_values(
                    cur,
                    """
                    INSERT INTO artifacts (session_id, artifact_type, content, language, line_count, metadata)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                    """,
                    values,
                    page_size=100  # Insert 100 rows per round-trip
                )
                t_exec = (time.time() - t_exec_start) * 1000

                t_total = (time.time() - t_start) * 1000
                pages = (len(artifacts) + 99) // 100
                from .utils import log as mira_log
                mira_log(f"Batch insert: {len(artifacts)} artifacts in {pages} pages | conn={t_conn:.0f}ms prep={t_prep:.0f}ms exec={t_exec:.0f}ms total={t_total:.0f}ms | {len(artifacts)*1000/max(1,t_total):.0f}/sec")

                return len(artifacts)

    def search_artifacts_fts(
        self,
        query: str,
        artifact_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full-text search on artifacts."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if artifact_type:
                    cur.execute(
                        """
                        SELECT a.id, a.artifact_type, a.content, a.language,
                               s.session_id, p.path as project_path,
                               ts_rank(to_tsvector('english', a.content), plainto_tsquery('english', %s)) as rank
                        FROM artifacts a
                        JOIN sessions s ON a.session_id = s.id
                        JOIN projects p ON s.project_id = p.id
                        WHERE a.artifact_type = %s
                          AND to_tsvector('english', a.content) @@ plainto_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s
                        """,
                        (query, artifact_type, query, limit)
                    )
                else:
                    cur.execute(
                        """
                        SELECT a.id, a.artifact_type, a.content, a.language,
                               s.session_id, p.path as project_path,
                               ts_rank(to_tsvector('english', a.content), plainto_tsquery('english', %s)) as rank
                        FROM artifacts a
                        JOIN sessions s ON a.session_id = s.id
                        JOIN projects p ON s.project_id = p.id
                        WHERE to_tsvector('english', a.content) @@ plainto_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s
                        """,
                        (query, query, limit)
                    )

                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    # ==================== Custodian ====================

    def get_custodian(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a custodian preference by key."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT key, value, category, confidence, frequency FROM custodian WHERE key = %s",
                    (key,)
                )
                row = cur.fetchone()
                if row:
                    return {
                        "key": row[0],
                        "value": row[1],
                        "category": row[2],
                        "confidence": row[3],
                        "frequency": row[4],
                    }
                return None

    def get_all_custodian(self) -> List[Dict[str, Any]]:
        """Get all custodian preferences."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT key, value, category, confidence, frequency FROM custodian ORDER BY confidence DESC"
                )
                return [
                    {
                        "key": row[0],
                        "value": row[1],
                        "category": row[2],
                        "confidence": row[3],
                        "frequency": row[4],
                    }
                    for row in cur.fetchall()
                ]

    def upsert_custodian(
        self,
        key: str,
        value: str,
        category: Optional[str] = None,
        confidence: float = 0.5,
        source_session: Optional[str] = None,
    ):
        """Insert or update a custodian preference."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO custodian (key, value, category, confidence, frequency, source_sessions)
                    VALUES (%s, %s, %s, %s, 1, ARRAY[%s])
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        category = COALESCE(EXCLUDED.category, custodian.category),
                        confidence = GREATEST(custodian.confidence, EXCLUDED.confidence),
                        frequency = custodian.frequency + 1,
                        source_sessions = array_append(custodian.source_sessions, %s),
                        updated_at = NOW()
                    """,
                    (key, value, category, confidence, source_session, source_session)
                )

    # ==================== Error Patterns ====================

    def upsert_error_pattern(
        self,
        project_id: int,
        signature: str,
        error_type: Optional[str],
        error_text: str,
        solution: Optional[str] = None,
        file_path: Optional[str] = None,
    ):
        """Insert or update an error pattern."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO error_patterns (project_id, signature, error_type, error_text, solution, file_path)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (project_id, signature) DO UPDATE SET
                        error_text = EXCLUDED.error_text,
                        solution = COALESCE(EXCLUDED.solution, error_patterns.solution),
                        file_path = COALESCE(EXCLUDED.file_path, error_patterns.file_path),
                        occurrences = error_patterns.occurrences + 1,
                        last_seen = NOW()
                    """,
                    (project_id, signature, error_type, error_text, solution, file_path)
                )

    def search_error_patterns(
        self,
        query: str,
        project_id: Optional[int] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search error patterns by text."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if project_id:
                    cur.execute(
                        """
                        SELECT e.error_type, e.error_text, e.solution, e.file_path,
                               e.occurrences, e.last_seen, p.path as project_path
                        FROM error_patterns e
                        JOIN projects p ON e.project_id = p.id
                        WHERE e.project_id = %s
                          AND to_tsvector('english', e.error_text || ' ' || COALESCE(e.solution, ''))
                              @@ plainto_tsquery('english', %s)
                        ORDER BY e.occurrences DESC, e.last_seen DESC
                        LIMIT %s
                        """,
                        (project_id, query, limit)
                    )
                else:
                    cur.execute(
                        """
                        SELECT e.error_type, e.error_text, e.solution, e.file_path,
                               e.occurrences, e.last_seen, p.path as project_path
                        FROM error_patterns e
                        JOIN projects p ON e.project_id = p.id
                        WHERE to_tsvector('english', e.error_text || ' ' || COALESCE(e.solution, ''))
                              @@ plainto_tsquery('english', %s)
                        ORDER BY e.occurrences DESC, e.last_seen DESC
                        LIMIT %s
                        """,
                        (query, limit)
                    )

                columns = [desc[0] for desc in cur.description]
                results = []
                for row in cur.fetchall():
                    item = dict(zip(columns, row))
                    # Convert datetime to ISO string for JSON serialization
                    if 'last_seen' in item and item['last_seen']:
                        item['last_seen'] = item['last_seen'].isoformat()
                    results.append(item)
                return results

    # ==================== Decisions ====================

    def insert_decision(
        self,
        project_id: int,
        decision: str,
        category: Optional[str] = None,
        reasoning: Optional[str] = None,
        alternatives: Optional[List[str]] = None,
        session_id: Optional[int] = None,
        confidence: float = 0.5,
    ) -> int:
        """Insert a decision. Returns decision ID."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO decisions (project_id, session_id, category, decision, reasoning, alternatives, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (project_id, session_id, category, decision, reasoning, alternatives or [], confidence)
                )
                return cur.fetchone()[0]

    def search_decisions(
        self,
        query: str,
        project_id: Optional[int] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search decisions by text and optional category."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                conditions = ["to_tsvector('english', d.decision || ' ' || COALESCE(d.reasoning, '')) @@ plainto_tsquery('english', %s)"]
                params = [query]

                if project_id:
                    conditions.append("d.project_id = %s")
                    params.append(project_id)
                if category:
                    conditions.append("d.category = %s")
                    params.append(category)

                params.extend([query, limit])

                cur.execute(
                    f"""
                    SELECT d.category, d.decision, d.reasoning, d.alternatives,
                           d.confidence, d.created_at, p.path as project_path,
                           ts_rank(
                               to_tsvector('english', d.decision || ' ' || COALESCE(d.reasoning, '')),
                               plainto_tsquery('english', %s)
                           ) as rank
                    FROM decisions d
                    JOIN projects p ON d.project_id = p.id
                    WHERE {' AND '.join(conditions)}
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    params
                )

                columns = [desc[0] for desc in cur.description]
                results = []
                for row in cur.fetchall():
                    item = dict(zip(columns, row))
                    # Convert datetime to ISO string for JSON serialization
                    if 'created_at' in item and item['created_at']:
                        item['created_at'] = item['created_at'].isoformat()
                    # Convert Decimal to float for JSON serialization
                    if 'rank' in item and hasattr(item['rank'], '__float__'):
                        item['rank'] = float(item['rank'])
                    if 'confidence' in item and hasattr(item['confidence'], '__float__'):
                        item['confidence'] = float(item['confidence'])
                    results.append(item)
                return results

    # ==================== Concepts ====================

    def upsert_concept(
        self,
        project_id: int,
        concept_type: str,
        name: str,
        description: Optional[str] = None,
        confidence: float = 0.5,
        source_session: Optional[str] = None,
    ):
        """Insert or update a concept."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO concepts (project_id, concept_type, name, description, confidence, frequency, source_sessions)
                    VALUES (%s, %s, %s, %s, %s, 1, ARRAY[%s])
                    ON CONFLICT (project_id, concept_type, name) DO UPDATE SET
                        description = COALESCE(EXCLUDED.description, concepts.description),
                        confidence = GREATEST(concepts.confidence, EXCLUDED.confidence),
                        frequency = concepts.frequency + 1,
                        source_sessions = array_append(concepts.source_sessions, %s),
                        updated_at = NOW()
                    """,
                    (project_id, concept_type, name, description, confidence, source_session, source_session)
                )

    def get_concepts(
        self,
        project_id: int,
        concept_type: Optional[str] = None,
        min_confidence: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Get concepts for a project."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if concept_type:
                    cur.execute(
                        """
                        SELECT concept_type, name, description, confidence, frequency
                        FROM concepts
                        WHERE project_id = %s AND concept_type = %s AND confidence >= %s
                        ORDER BY confidence DESC, frequency DESC
                        """,
                        (project_id, concept_type, min_confidence)
                    )
                else:
                    cur.execute(
                        """
                        SELECT concept_type, name, description, confidence, frequency
                        FROM concepts
                        WHERE project_id = %s AND confidence >= %s
                        ORDER BY confidence DESC, frequency DESC
                        """,
                        (project_id, min_confidence)
                    )

                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    # ==================== Lifecycle Patterns ====================

    def upsert_lifecycle_pattern(
        self,
        pattern: str,
        confidence: float = 0.5,
        source_session: Optional[str] = None,
    ):
        """Insert or update a lifecycle pattern."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO lifecycle_patterns (pattern, confidence, occurrences, source_sessions)
                    VALUES (%s, %s, 1, ARRAY[%s])
                    ON CONFLICT ON CONSTRAINT lifecycle_patterns_pkey DO NOTHING
                    """,
                    (pattern, confidence, source_session)
                )
                # If exists, update separately (no unique constraint on pattern)
                cur.execute(
                    """
                    UPDATE lifecycle_patterns
                    SET confidence = GREATEST(confidence, %s),
                        occurrences = occurrences + 1,
                        source_sessions = array_append(source_sessions, %s),
                        updated_at = NOW()
                    WHERE pattern = %s
                    """,
                    (confidence, source_session, pattern)
                )

    def get_lifecycle_patterns(self, min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """Get lifecycle patterns."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pattern, confidence, occurrences
                    FROM lifecycle_patterns
                    WHERE confidence >= %s
                    ORDER BY confidence DESC, occurrences DESC
                    """,
                    (min_confidence,)
                )
                return [
                    {"pattern": row[0], "confidence": row[1], "occurrences": row[2]}
                    for row in cur.fetchall()
                ]

    # ==================== Archives ====================

    def upsert_archive(
        self,
        session_id: int,
        content: str,
        content_hash: str,
    ) -> int:
        """
        Store or update a conversation archive.

        Args:
            session_id: The Postgres session ID (foreign key)
            content: Full JSONL content (newline-separated JSON objects)
            content_hash: SHA256 hash for deduplication

        Returns archive ID.
        """
        size_bytes = len(content.encode('utf-8'))
        line_count = content.count('\n') + 1 if content else 0

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO archives (session_id, content, content_hash, size_bytes, line_count)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        content_hash = EXCLUDED.content_hash,
                        size_bytes = EXCLUDED.size_bytes,
                        line_count = EXCLUDED.line_count,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    (session_id, content, content_hash, size_bytes, line_count)
                )
                return cur.fetchone()[0]

    def get_archive(self, session_id: int) -> Optional[str]:
        """
        Get archive content for a session.

        Args:
            session_id: The Postgres session ID

        Returns the JSONL content or None if not found.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM archives WHERE session_id = %s",
                    (session_id,)
                )
                row = cur.fetchone()
                return row[0] if row else None

    def get_archive_by_session_uuid(self, session_uuid: str) -> Optional[str]:
        """
        Get archive content by the session UUID (the filename).

        Args:
            session_uuid: The session ID string (from filename)

        Returns the JSONL content or None if not found.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT a.content
                    FROM archives a
                    JOIN sessions s ON a.session_id = s.id
                    WHERE s.session_id = %s
                    """,
                    (session_uuid,)
                )
                row = cur.fetchone()
                return row[0] if row else None

    def archive_exists(self, session_id: int) -> bool:
        """Check if an archive exists for a session."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM archives WHERE session_id = %s",
                    (session_id,)
                )
                return cur.fetchone() is not None

    def search_archives_fts(
        self,
        query: str,
        project_id: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search on archive content.

        Uses ILIKE for reliable matching (works with any size content),
        avoids tsvector which has a 1MB limit.

        Returns session info and matching excerpts.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Use ILIKE for reliable matching regardless of content size
                # tsvector has a 1MB limit which large archives exceed
                search_pattern = f'%{query}%'

                if project_id:
                    cur.execute(
                        """
                        SELECT s.session_id, s.summary, a.content,
                               p.path as project_path,
                               1.0 as rank
                        FROM archives a
                        JOIN sessions s ON a.session_id = s.id
                        JOIN projects p ON s.project_id = p.id
                        WHERE s.project_id = %s
                          AND a.content ILIKE %s
                        ORDER BY a.updated_at DESC
                        LIMIT %s
                        """,
                        (project_id, search_pattern, limit)
                    )
                else:
                    cur.execute(
                        """
                        SELECT s.session_id, s.summary, a.content,
                               p.path as project_path,
                               1.0 as rank
                        FROM archives a
                        JOIN sessions s ON a.session_id = s.id
                        JOIN projects p ON s.project_id = p.id
                        WHERE a.content ILIKE %s
                        ORDER BY a.updated_at DESC
                        LIMIT %s
                        """,
                        (search_pattern, limit)
                    )

                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_archive_stats(self) -> Dict[str, Any]:
        """Get statistics about stored archives."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) as total_archives,
                        COALESCE(SUM(size_bytes), 0) as total_bytes,
                        COALESCE(SUM(line_count), 0) as total_lines,
                        COALESCE(AVG(size_bytes), 0) as avg_bytes
                    FROM archives
                    """
                )
                row = cur.fetchone()
                return {
                    "total_archives": row[0],
                    "total_bytes": row[1],
                    "total_lines": row[2],
                    "avg_bytes": float(row[3]),
                }

    def close(self):
        """Close all connections in the pool."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            self._healthy = False
