"""
MIRA Storage Abstraction Layer

Provides unified interface to storage with automatic fallback:
- Primary: Central Qdrant + Postgres (full semantic search, cross-machine sync)
- Fallback: Local SQLite with FTS (keyword search only, single machine)

Design principles:
- Central storage preferred when available
- Graceful fallback to local SQLite when central is unavailable
- Lazy initialization (connect only when needed)
- Clear messaging about current storage mode
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from .config import get_config, ServerConfig

log = logging.getLogger(__name__)

# Import local store module (always available)
from . import local_store

# Backends - lazily initialized
_qdrant_backend = None
_postgres_backend = None


class StorageError(Exception):
    """Raised when storage operations fail."""
    pass


class Storage:
    """
    Unified storage interface for MIRA.

    Uses central Qdrant + Postgres storage exclusively.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or get_config()
        self._qdrant = None
        self._postgres = None
        self._using_central = False
        self._central_init_attempted = False

    def _init_central(self) -> bool:
        """
        Initialize central backends.

        Returns True if central is available, False otherwise.
        """
        if self._central_init_attempted:
            return self._using_central

        self._central_init_attempted = True

        if not self.config.central_enabled:
            log.error("Central storage not enabled in config")
            return False

        try:
            # Initialize Qdrant
            from .qdrant_backend import QdrantBackend
            qdrant_cfg = self.config.central.qdrant
            self._qdrant = QdrantBackend(
                host=qdrant_cfg.host,
                port=qdrant_cfg.port,
                collection=qdrant_cfg.collection,
                timeout=qdrant_cfg.timeout_seconds,
            )

            # Initialize Postgres
            from .postgres_backend import PostgresBackend
            pg_cfg = self.config.central.postgres
            self._postgres = PostgresBackend(
                host=pg_cfg.host,
                port=pg_cfg.port,
                database=pg_cfg.database,
                user=pg_cfg.user,
                password=pg_cfg.password,
                pool_size=pg_cfg.pool_size,
                timeout=pg_cfg.timeout_seconds,
            )

            # Health check both
            if self._qdrant.is_healthy() and self._postgres.is_healthy():
                self._using_central = True
                log.info("Central storage initialized successfully")
                return True
            else:
                log.error("Central storage health check failed")
                return False

        except ImportError as e:
            log.error(f"Central storage dependencies not installed: {e}")
            return False
        except Exception as e:
            log.error(f"Failed to initialize central storage: {e}")
            return False

    @property
    def using_central(self) -> bool:
        """Check if using central storage."""
        self._init_central()
        return self._using_central

    def _queue_for_sync(self, data_type: str, item_id: str, payload: Dict[str, Any]):
        """Queue an item for later sync to central storage."""
        try:
            from .sync_queue import get_sync_queue
            queue = get_sync_queue()

            # Generate hash for deduplication
            hash_input = f"{data_type}:{item_id}"
            item_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:32]

            queue.enqueue(data_type, item_hash, payload)
            log.debug(f"Queued {data_type} for sync: {item_id}")
        except Exception as e:
            log.error(f"Failed to queue {data_type} for sync: {e}")

    @property
    def qdrant(self):
        """Get Qdrant backend."""
        if self._init_central():
            return self._qdrant
        return None

    @property
    def postgres(self):
        """Get Postgres backend."""
        if self._init_central():
            return self._postgres
        return None

    # ==================== Vector Operations ====================

    def vector_search(
        self,
        query_vector: List[float],
        project_path: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Qdrant.

        Vector search is only available with central storage.
        Use search_sessions_fts for local fallback.
        """
        if not self._init_central() or not self._qdrant:
            # No local fallback for vector search - FTS must be used instead
            log.warning("Vector search unavailable in local mode, use FTS")
            return []

        try:
            results = self._qdrant.search(
                query_vector=query_vector,
                project_path=project_path,
                limit=limit,
            )
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "content": r.content,
                    "session_id": r.session_id,
                    "project_path": r.project_path,
                    "chunk_type": r.chunk_type,
                    "role": r.role,
                }
                for r in results
            ]
        except Exception as e:
            log.error(f"Vector search failed: {e}")
            return []  # Return empty instead of raising

    def vector_upsert(
        self,
        vector: List[float],
        content: str,
        session_id: str,
        project_path: str,
        chunk_type: str = "message",
        role: Optional[str] = None,
        point_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Upsert a vector with payload to Qdrant.

        Returns the point ID, or None if central storage unavailable.
        Local fallback stores metadata only (no vectors).
        """
        if not self._init_central() or not self._qdrant:
            # No local fallback for vector storage - skip silently
            log.debug("Vector upsert skipped in local mode")
            return None

        try:
            return self._qdrant.upsert(
                vector=vector,
                content=content,
                session_id=session_id,
                project_path=project_path,
                chunk_type=chunk_type,
                role=role,
                point_id=point_id,
            )
        except Exception as e:
            log.error(f"Vector upsert failed: {e}")
            return None  # Return None instead of raising

    def vector_batch_upsert(
        self,
        points: List[Dict[str, Any]],
    ) -> int:
        """
        Batch upsert vectors to Qdrant.

        Returns count of points upserted (0 if central unavailable).
        """
        if not points:
            return 0

        if not self._init_central() or not self._qdrant:
            log.debug("Batch vector upsert skipped in local mode")
            return 0

        try:
            return self._qdrant.batch_upsert(points)
        except Exception as e:
            log.error(f"Batch upsert failed: {e}")
            return 0

    # ==================== Project Operations ====================

    def get_project_id(self, project_path: str) -> Optional[int]:
        """
        Get project ID for a given path without creating if it doesn't exist.

        Args:
            project_path: Filesystem path to the project

        Returns project ID or None if not found.
        """
        if self._init_central() and self._postgres:
            try:
                with self._postgres._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT id FROM projects WHERE path = %s",
                            (project_path,)
                        )
                        row = cur.fetchone()
                        return row[0] if row else None
            except Exception as e:
                log.error(f"Central get_project_id failed: {e}")

        # Local fallback
        try:
            return local_store.get_project_id(project_path)
        except Exception as e:
            log.error(f"Local get_project_id failed: {e}")
            return None

    def get_or_create_project(
        self,
        path: str,
        slug: Optional[str] = None,
        git_remote: Optional[str] = None
    ) -> Optional[int]:
        """
        Get or create a project.

        Uses central Postgres if available, falls back to local SQLite.

        Args:
            path: Filesystem path to the project
            slug: Optional short name for the project
            git_remote: Normalized git remote URL (canonical cross-machine identity)

        Returns project ID.
        """
        if self._init_central() and self._postgres:
            try:
                return self._postgres.get_or_create_project(path, slug, git_remote)
            except Exception as e:
                log.error(f"Central get_or_create_project failed: {e}")
                # Fall through to local

        # Local fallback
        try:
            return local_store.get_or_create_project(path, slug, git_remote)
        except Exception as e:
            log.error(f"Local get_or_create_project failed: {e}")
            return None

    # ==================== Session Operations ====================

    def upsert_session(
        self,
        project_path: str,
        session_id: str,
        git_remote: Optional[str] = None,
        **kwargs,
    ) -> Optional[int]:
        """
        Upsert a session.

        Strategy:
        1. Try central Postgres first
        2. If central fails, queue for later sync AND store in local fallback

        Args:
            project_path: Filesystem path to the project
            session_id: Unique session identifier
            git_remote: Normalized git remote URL for cross-machine project matching

        Returns session ID.
        """
        if self._init_central() and self._postgres:
            try:
                project_id = self._postgres.get_or_create_project(project_path, git_remote=git_remote)
                return self._postgres.upsert_session(
                    project_id=project_id,
                    session_id=session_id,
                    **kwargs,
                )
            except Exception as e:
                log.error(f"Central upsert_session failed: {e}")
                # Queue for later sync
                self._queue_for_sync("session", session_id, {
                    "project_path": project_path,
                    "session_id": session_id,
                    "git_remote": git_remote,
                    **kwargs,
                })
                # Fall through to local

        # Local fallback (also used as temp storage while queued)
        try:
            project_id = local_store.get_or_create_project(project_path, git_remote=git_remote)
            return local_store.upsert_session(
                project_id=project_id,
                session_id=session_id,
                **kwargs,
            )
        except Exception as e:
            log.error(f"Local upsert_session failed: {e}")
            return None

    def session_exists_in_central(self, session_id: str) -> bool:
        """
        Check if a session exists in central storage.

        Used to detect local sessions that need to be synced to central.
        Returns False if not using central storage.
        """
        if not self._init_central() or not self._postgres:
            return False

        try:
            with self._postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM sessions WHERE session_id = %s LIMIT 1",
                        (session_id,)
                    )
                    return cur.fetchone() is not None
        except Exception as e:
            log.error(f"session_exists_in_central check failed: {e}")
            return False

    def get_recent_sessions(
        self,
        project_path: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent sessions. Uses central if available, falls back to local."""
        if self._init_central() and self._postgres:
            try:
                project_id = None
                if project_path:
                    project_id = self._postgres.get_or_create_project(project_path)
                return self._postgres.get_recent_sessions(
                    project_id=project_id,
                    limit=limit,
                )
            except Exception as e:
                log.error(f"Central get_recent_sessions failed: {e}")
                # Fall through to local

        # Local fallback
        try:
            project_id = None
            if project_path:
                project_id = local_store.get_or_create_project(project_path)
            return local_store.get_recent_sessions(
                project_id=project_id,
                limit=limit,
            )
        except Exception as e:
            log.error(f"Local get_recent_sessions failed: {e}")
            return []

    def search_sessions_fts(
        self,
        query: str,
        project_path: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full-text search on sessions. Uses central if available, falls back to local."""
        if self._init_central() and self._postgres:
            try:
                project_id = None
                if project_path:
                    project_id = self._postgres.get_or_create_project(project_path)
                return self._postgres.search_sessions_fts(
                    query=query,
                    project_id=project_id,
                    limit=limit,
                )
            except Exception as e:
                log.error(f"Central search_sessions_fts failed: {e}")
                # Fall through to local

        # Local fallback
        try:
            project_id = None
            if project_path:
                project_id = local_store.get_or_create_project(project_path)
            return local_store.search_sessions_fts(
                query=query,
                project_id=project_id,
                limit=limit,
            )
        except Exception as e:
            log.error(f"Local search_sessions_fts failed: {e}")
            return []

    # ==================== Custodian Operations ====================

    def get_custodian_all(self) -> List[Dict[str, Any]]:
        """Get all custodian preferences. Uses central if available, falls back to local."""
        if self._init_central() and self._postgres:
            try:
                return self._postgres.get_all_custodian()
            except Exception as e:
                log.error(f"Central get_custodian_all failed: {e}")
                # Fall through to local

        # Local fallback
        try:
            return local_store.get_custodian_all()
        except Exception as e:
            log.error(f"Local get_custodian_all failed: {e}")
            return []

    def upsert_custodian(
        self,
        key: str,
        value: str,
        category: Optional[str] = None,
        confidence: float = 0.5,
        source_session: Optional[str] = None,
    ):
        """
        Upsert a custodian preference.

        Strategy:
        1. Try central Postgres first
        2. If central fails, queue for later sync AND store locally
        """
        if self._init_central() and self._postgres:
            try:
                self._postgres.upsert_custodian(
                    key=key,
                    value=value,
                    category=category,
                    confidence=confidence,
                    source_session=source_session,
                )
                return
            except Exception as e:
                log.error(f"Central upsert_custodian failed: {e}")
                # Queue for later sync
                self._queue_for_sync("custodian", key, {
                    "key": key,
                    "value": value,
                    "category": category,
                    "confidence": confidence,
                    "source_session": source_session,
                })
                # Fall through to local

        # Local fallback
        try:
            local_store.upsert_custodian(
                key=key,
                value=value,
                category=category,
                confidence=confidence,
                source_session=source_session,
            )
        except Exception as e:
            log.error(f"Local upsert_custodian failed: {e}")

    # ==================== Error Pattern Operations ====================

    def search_error_patterns(
        self,
        query: str,
        project_path: Optional[str] = None,
        project_id: Optional[int] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search error patterns. Uses central if available, falls back to local.

        Args:
            query: Error message or description to search for
            project_path: Optional project path to filter by
            project_id: Optional project ID (takes precedence over project_path)
            limit: Maximum results
        """
        if self._init_central() and self._postgres:
            try:
                pid = project_id
                if pid is None and project_path:
                    pid = self._postgres.get_or_create_project(project_path)
                return self._postgres.search_error_patterns(
                    query=query,
                    project_id=pid,
                    limit=limit,
                )
            except Exception as e:
                log.error(f"Central search_error_patterns failed: {e}")
                # Fall through to local

        # Local fallback
        try:
            pid = project_id
            if pid is None and project_path:
                pid = local_store.get_or_create_project(project_path)
            return local_store.search_error_patterns(
                query=query,
                project_id=pid,
                limit=limit,
            )
        except Exception as e:
            log.error(f"Local search_error_patterns failed: {e}")
            return []

    def upsert_error_pattern(
        self,
        project_path: str,
        signature: str,
        error_type: Optional[str],
        error_text: str,
        solution: Optional[str] = None,
        file_path: Optional[str] = None,
    ):
        """
        Upsert an error pattern.

        Strategy:
        1. Try central Postgres first
        2. If central fails, queue for later sync AND store locally
        """
        if self._init_central() and self._postgres:
            try:
                project_id = self._postgres.get_or_create_project(project_path)
                self._postgres.upsert_error_pattern(
                    project_id=project_id,
                    signature=signature,
                    error_type=error_type,
                    error_text=error_text,
                    solution=solution,
                    file_path=file_path,
                )
                return
            except Exception as e:
                log.error(f"Central upsert_error_pattern failed: {e}")
                # Queue for later sync
                self._queue_for_sync("error", signature, {
                    "project_path": project_path,
                    "signature": signature,
                    "error_type": error_type,
                    "error_text": error_text,
                    "solution": solution,
                    "file_path": file_path,
                })
                # Fall through to local

        # Local fallback
        try:
            project_id = local_store.get_or_create_project(project_path)
            local_store.upsert_error_pattern(
                project_id=project_id,
                signature=signature,
                error_type=error_type,
                error_text=error_text,
                solution=solution,
                file_path=file_path,
            )
        except Exception as e:
            log.error(f"Local upsert_error_pattern failed: {e}")

    # ==================== Decision Operations ====================

    def search_decisions(
        self,
        query: str,
        project_path: Optional[str] = None,
        project_id: Optional[int] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search decisions. Uses central if available, falls back to local.

        Args:
            query: Search query
            project_path: Optional project path to filter by
            project_id: Optional project ID (takes precedence over project_path)
            category: Optional category filter
            limit: Maximum results
        """
        if self._init_central() and self._postgres:
            try:
                pid = project_id
                if pid is None and project_path:
                    pid = self._postgres.get_or_create_project(project_path)
                return self._postgres.search_decisions(
                    query=query,
                    project_id=pid,
                    category=category,
                    limit=limit,
                )
            except Exception as e:
                log.error(f"Central search_decisions failed: {e}")
                # Fall through to local

        # Local fallback
        try:
            pid = project_id
            if pid is None and project_path:
                pid = local_store.get_or_create_project(project_path)
            return local_store.search_decisions(
                query=query,
                project_id=pid,
                category=category,
                limit=limit,
            )
        except Exception as e:
            log.error(f"Local search_decisions failed: {e}")
            return []

    def upsert_decision(
        self,
        project_path: str,
        decision: str,
        category: Optional[str] = None,
        reasoning: Optional[str] = None,
        alternatives: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        confidence: float = 0.5,
    ):
        """
        Upsert a decision.

        Strategy:
        1. Try central Postgres first
        2. If central fails, queue for later sync AND store locally
        """
        if self._init_central() and self._postgres:
            try:
                project_id = self._postgres.get_or_create_project(project_path)
                self._postgres.insert_decision(
                    project_id=project_id,
                    decision=decision,
                    category=category,
                    reasoning=reasoning,
                    alternatives=alternatives,
                    confidence=confidence,
                )
                return
            except Exception as e:
                log.error(f"Central upsert_decision failed: {e}")
                # Queue for later sync
                decision_hash = hashlib.sha256(f"{project_path}:{decision[:100]}".encode()).hexdigest()[:16]
                self._queue_for_sync("decision", decision_hash, {
                    "project_path": project_path,
                    "decision": decision,
                    "category": category,
                    "reasoning": reasoning,
                    "alternatives": alternatives,
                    "session_id": session_id,
                    "confidence": confidence,
                })
                # Fall through to local

        # Local fallback
        try:
            project_id = local_store.get_or_create_project(project_path)
            local_store.upsert_decision(
                project_id=project_id,
                decision=decision,
                category=category,
                reasoning=reasoning,
                alternatives=alternatives,
                confidence=confidence,
            )
        except Exception as e:
            log.error(f"Local upsert_decision failed: {e}")

    # ==================== Concept Operations ====================

    def get_concepts(
        self,
        project_path: str,
        concept_type: Optional[str] = None,
        min_confidence: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Get concepts for a project. Uses central if available, returns empty if not."""
        if self._init_central() and self._postgres:
            try:
                project_id = self._postgres.get_or_create_project(project_path)
                return self._postgres.get_concepts(
                    project_id=project_id,
                    concept_type=concept_type,
                    min_confidence=min_confidence,
                )
            except Exception as e:
                log.error(f"Central get_concepts failed: {e}")

        # Local fallback - concepts not stored locally yet, return empty
        return []

    # ==================== Archive Operations ====================

    def upsert_archive(
        self,
        postgres_session_id: int,
        content: str,
        content_hash: str,
    ) -> Optional[int]:
        """
        Store or update a conversation archive.

        Uses central Postgres if available, falls back to local SQLite.

        Args:
            postgres_session_id: The session ID (foreign key)
            content: Full JSONL content
            content_hash: SHA256 hash for deduplication

        Returns archive ID or None if failed.
        """
        if self._init_central() and self._postgres:
            try:
                return self._postgres.upsert_archive(
                    session_id=postgres_session_id,
                    content=content,
                    content_hash=content_hash,
                )
            except Exception as e:
                log.error(f"Central upsert_archive failed: {e}")
                # Fall through to local

        # Local fallback
        try:
            return local_store.upsert_archive(
                session_db_id=postgres_session_id,
                content=content,
                content_hash=content_hash,
            )
        except Exception as e:
            log.error(f"Local upsert_archive failed: {e}")
            return None

    def update_archive(
        self,
        session_id: str,
        content: str,
        size_bytes: int,
        line_count: int,
    ) -> bool:
        """
        Update archive content for an existing session by UUID.

        Used for active session sync - updates the archive without full re-ingestion.

        Args:
            session_id: The session UUID string
            content: Full JSONL content
            size_bytes: File size in bytes
            line_count: Number of lines

        Returns True if successful, False otherwise.
        """
        if not self._init_central() or not self._postgres:
            return False

        try:
            with self._postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    # Get the internal session ID
                    cur.execute(
                        "SELECT id FROM sessions WHERE session_id = %s",
                        (session_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        return False

                    postgres_session_id = row[0]
                    content_hash = hashlib.sha256(content.encode()).hexdigest()

                    # Update archive
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
                        """,
                        (postgres_session_id, content, content_hash, size_bytes, line_count)
                    )
                    conn.commit()
                    return True
        except Exception as e:
            log.error(f"update_archive failed: {e}")
            return False

    def update_session_metadata(
        self,
        session_id: str,
        summary: str,
        keywords: List[str],
    ) -> bool:
        """
        Update session metadata (summary and keywords) for an existing session.

        Used for active session sync - updates metadata without full re-ingestion.

        Args:
            session_id: The session UUID string
            summary: Updated summary text
            keywords: Updated keywords list

        Returns True if successful, False otherwise.
        """
        if not self._init_central() or not self._postgres:
            return False

        try:
            with self._postgres._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE sessions
                        SET summary = %s, keywords = %s
                        WHERE session_id = %s
                        """,
                        (summary, keywords, session_id)
                    )
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            log.error(f"update_session_metadata failed: {e}")
            return False

    def get_archive(self, session_uuid: str) -> Optional[str]:
        """
        Get archive content by session UUID.

        Uses central Postgres if available, falls back to local SQLite.

        Args:
            session_uuid: The session ID string (filename without .jsonl)

        Returns the JSONL content or None if not found.
        """
        if self._init_central() and self._postgres:
            try:
                result = self._postgres.get_archive_by_session_uuid(session_uuid)
                if result:
                    return result
            except Exception as e:
                log.error(f"Central get_archive failed: {e}")
                # Fall through to local

        # Local fallback
        try:
            return local_store.get_archive(session_uuid)
        except Exception as e:
            log.error(f"Local get_archive failed: {e}")
            return None

    def search_archives_fts(
        self,
        query: str,
        project_path: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full-text search on archive content. Uses central if available, falls back to local."""
        if self._init_central() and self._postgres:
            try:
                project_id = None
                if project_path:
                    project_id = self._postgres.get_or_create_project(project_path)
                return self._postgres.search_archives_fts(
                    query=query,
                    project_id=project_id,
                    limit=limit,
                )
            except Exception as e:
                log.error(f"Central search_archives_fts failed: {e}")
                # Fall through to local

        # Local fallback
        try:
            project_id = None
            if project_path:
                project_id = local_store.get_or_create_project(project_path)
            return local_store.search_archives_fts(
                query=query,
                project_id=project_id,
                limit=limit,
            )
        except Exception as e:
            log.error(f"Local search_archives_fts failed: {e}")
            return []

    def get_archive_stats(self) -> Dict[str, Any]:
        """Get statistics about stored archives."""
        if self._init_central() and self._postgres:
            try:
                return self._postgres.get_archive_stats()
            except Exception as e:
                log.error(f"Central get_archive_stats failed: {e}")
                # Fall through to local

        # Local fallback
        try:
            return local_store.get_archive_stats()
        except Exception as e:
            log.error(f"Local get_archive_stats failed: {e}")
            return {"total_archives": 0, "total_bytes": 0, "total_lines": 0, "avg_bytes": 0}

    # ==================== Health & Status ====================

    def get_storage_mode(self) -> Dict[str, Any]:
        """
        Get current storage mode information for display to user.

        Returns a dict with:
        - mode: "central" or "local"
        - description: Human-readable description
        - limitations: What's not available in local mode
        - setup_instructions: How to enable central storage (if in local mode)
        """
        self._init_central()

        if self._using_central:
            return {
                "mode": "central",
                "description": "Using central Qdrant + Postgres storage (cross-machine sync enabled)",
                "qdrant_host": self.config.central.qdrant.host if self.config.central else None,
                "postgres_host": self.config.central.postgres.host if self.config.central else None,
            }
        else:
            # Get local session count
            try:
                session_count = local_store.get_session_count()
            except Exception:
                session_count = 0

            return {
                "mode": "local",
                "description": "Using local SQLite storage (keyword search only, single-machine)",
                "session_count": session_count,
                "limitations": [
                    "Keyword search only (no semantic/vector search)",
                    "History stays on this machine only",
                    "Codebase concepts not available",
                ],
                "setup": {
                    "summary": "To enable semantic search and cross-machine sync, set up central storage.",
                    "steps": [
                        "1. Set up a server with Qdrant (port 6333) and Postgres (port 5432) - Docker recommended",
                        "2. Install Tailscale on both the server and this machine for secure VPN access",
                        "3. Create ~/.mira/server.json with connection details (see template below)",
                        "4. Restart MIRA - it will auto-connect to central storage",
                    ],
                    "config_template": {
                        "version": 1,
                        "central": {
                            "enabled": True,
                            "qdrant": {"host": "YOUR_TAILSCALE_IP", "port": 6333},
                            "postgres": {
                                "host": "YOUR_TAILSCALE_IP",
                                "port": 5432,
                                "database": "mira",
                                "user": "mira",
                                "password": "YOUR_PASSWORD"
                            }
                        }
                    },
                    "note": "Central storage is optional. MIRA works in local mode with keyword search."
                }
            }

    def health_check(self) -> Dict[str, Any]:
        """Check health of central storage with detailed diagnostics."""
        status = {
            "central_configured": self.config.central_enabled,
            "central_available": False,
            "qdrant_healthy": False,
            "postgres_healthy": False,
            "using_central": False,
            "mode": "local",
        }

        if self._init_central():
            status["central_available"] = True
            status["using_central"] = self._using_central
            status["mode"] = "central" if self._using_central else "local"
            if self._qdrant:
                status["qdrant_healthy"] = self._qdrant.is_healthy()
            if self._postgres:
                status["postgres_healthy"] = self._postgres.is_healthy()
        elif self.config.central_enabled:
            # Central is configured but not available - add diagnostics
            status["diagnostics"] = self._get_connectivity_diagnostics()

        return status

    def _get_connectivity_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics when central storage is unreachable."""
        import socket
        import subprocess

        diag = {
            "issue": "Central storage configured but unreachable",
            "checks": [],
            "suggestions": [],
        }

        if not self.config.central or not self.config.central.qdrant:
            diag["checks"].append("Config: Missing central configuration")
            return diag

        qdrant_host = self.config.central.qdrant.host
        qdrant_port = self.config.central.qdrant.port
        pg_host = self.config.central.postgres.host
        pg_port = self.config.central.postgres.port

        # Check if this looks like a Tailscale IP
        is_tailscale_ip = qdrant_host.startswith("100.")

        # Check Tailscale status if it's a Tailscale IP
        if is_tailscale_ip:
            try:
                result = subprocess.run(
                    ["tailscale", "status", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    import json
                    ts_status = json.loads(result.stdout)
                    if ts_status.get("BackendState") == "Running":
                        diag["checks"].append(f"Tailscale: Running")
                        # Check if the target IP is in peers
                        peers = ts_status.get("Peer", {})
                        target_found = False
                        for peer_id, peer in peers.items():
                            peer_ips = peer.get("TailscaleIPs", [])
                            if qdrant_host in peer_ips:
                                target_found = True
                                online = peer.get("Online", False)
                                diag["checks"].append(
                                    f"Tailscale peer {qdrant_host}: {'online' if online else 'OFFLINE'}"
                                )
                                if not online:
                                    diag["suggestions"].append(
                                        "The target server appears to be offline in Tailscale"
                                    )
                                break
                        if not target_found:
                            diag["checks"].append(f"Tailscale peer {qdrant_host}: NOT FOUND in network")
                            diag["suggestions"].append(
                                "Target IP not found in Tailscale network. Verify the server is connected to Tailscale."
                            )
                    else:
                        diag["checks"].append(f"Tailscale: Not running (state: {ts_status.get('BackendState')})")
                        diag["suggestions"].append("Start Tailscale: sudo tailscale up")
                else:
                    diag["checks"].append("Tailscale: Not connected or not installed")
                    diag["suggestions"].append("Install/start Tailscale to connect to central storage")
            except FileNotFoundError:
                diag["checks"].append("Tailscale: NOT INSTALLED")
                diag["suggestions"].append(
                    "Tailscale is required to reach the central server. "
                    "Install from https://tailscale.com/download"
                )
            except subprocess.TimeoutExpired:
                diag["checks"].append("Tailscale: Command timed out")
            except Exception as e:
                diag["checks"].append(f"Tailscale: Error checking status - {e}")

        # Test TCP connectivity to Qdrant
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((qdrant_host, qdrant_port))
            sock.close()
            if result == 0:
                diag["checks"].append(f"Qdrant port {qdrant_host}:{qdrant_port}: REACHABLE")
            else:
                diag["checks"].append(f"Qdrant port {qdrant_host}:{qdrant_port}: UNREACHABLE (error {result})")
        except socket.timeout:
            diag["checks"].append(f"Qdrant port {qdrant_host}:{qdrant_port}: TIMEOUT")
        except Exception as e:
            diag["checks"].append(f"Qdrant port {qdrant_host}:{qdrant_port}: ERROR - {e}")

        # Test TCP connectivity to Postgres
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((pg_host, pg_port))
            sock.close()
            if result == 0:
                diag["checks"].append(f"Postgres port {pg_host}:{pg_port}: REACHABLE")
            else:
                diag["checks"].append(f"Postgres port {pg_host}:{pg_port}: UNREACHABLE (error {result})")
        except socket.timeout:
            diag["checks"].append(f"Postgres port {pg_host}:{pg_port}: TIMEOUT")
        except Exception as e:
            diag["checks"].append(f"Postgres port {pg_host}:{pg_port}: ERROR - {e}")

        # Add general suggestions if no specific ones yet
        if not diag["suggestions"]:
            if is_tailscale_ip:
                diag["suggestions"].append(
                    "Ensure Tailscale is installed and connected on this machine"
                )
                diag["suggestions"].append(
                    "Verify the central server is running and connected to Tailscale"
                )
            else:
                diag["suggestions"].append(
                    "Check network connectivity to the central server"
                )
                diag["suggestions"].append(
                    "Verify firewall rules allow connections to Qdrant and Postgres ports"
                )

        return diag

    def close(self):
        """Close all connections."""
        if self._qdrant:
            self._qdrant.close()
            self._qdrant = None
        if self._postgres:
            self._postgres.close()
            self._postgres = None
        self._using_central = False
        self._central_init_attempted = False


# Global storage instance (lazy singleton)
_storage: Optional[Storage] = None


def get_storage() -> Storage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage


def reset_storage():
    """Reset the global storage instance (for testing)."""
    global _storage
    if _storage:
        _storage.close()
    _storage = None
