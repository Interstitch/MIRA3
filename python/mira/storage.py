"""
MIRA Storage Abstraction Layer - LOCAL-FIRST Architecture

All writes go to local SQLite FIRST, then sync to central in background.
This ensures:
- Fast writes (no network latency)
- Offline capability (works without central)
- Reliable data (local write always succeeds)

Data flow:
1. Write → Local SQLite (always, fast)
2. Queue → Sync queue (if central configured)
3. Background → Sync worker flushes to Central Postgres
4. Embedding Service → Polls Postgres, indexes to Qdrant

For reads:
- Try central first (to get cross-machine data)
- Fall back to local if central unavailable
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

    LOCAL-FIRST: All writes go to local SQLite first, then sync to central.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or get_config()
        self._qdrant = None
        self._postgres = None
        self._using_central = False
        self._central_init_attempted = False

    def _init_central(self) -> bool:
        """
        Initialize central backends (lazy, for reads).

        Returns True if central is available, False otherwise.
        """
        if self._central_init_attempted:
            return self._using_central

        self._central_init_attempted = True

        if not self.config.central_enabled:
            log.debug("Central storage not enabled in config")
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
                api_key=getattr(qdrant_cfg, 'api_key', None),
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

                # Run Postgres schema migrations
                try:
                    from .migrations import run_postgres_migrations
                    migration_result = run_postgres_migrations(self._postgres)
                    if migration_result.get("migrations_run"):
                        for m in migration_result["migrations_run"]:
                            log.info(f"Postgres migration: {m['name']}")
                except Exception as e:
                    log.error(f"Postgres migration error: {e}")

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

    @property
    def central_configured(self) -> bool:
        """Check if central storage is configured (even if not connected)."""
        return self.config.central_enabled

    def _queue_for_sync(self, data_type: str, item_id: str, payload: Dict[str, Any]):
        """Queue an item for later sync to central storage."""
        if not self.config.central_enabled:
            return  # No central configured, nothing to queue

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
        """Get Qdrant backend (for reads)."""
        if self._init_central():
            return self._qdrant
        return None

    @property
    def postgres(self):
        """Get Postgres backend (for reads and direct sync operations)."""
        if self._init_central():
            return self._postgres
        return None

    # ==================== Vector Operations ====================
    # Vector search only available via embedding service (queries Qdrant)
    # No local vector operations - client uses embedding_client.search()

    # ==================== Project Operations ====================

    def get_project_id(self, project_path: str) -> Optional[int]:
        """
        Get project ID for a given path without creating.

        Checks local first (fast), then central for cross-machine data.
        """
        # Try local first (fast)
        try:
            local_id = local_store.get_project_id(project_path)
            if local_id:
                return local_id
        except Exception as e:
            log.error(f"Local get_project_id failed: {e}")

        # Try central for cross-machine data
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

        return None

    def get_or_create_project(
        self,
        path: str,
        slug: Optional[str] = None,
        git_remote: Optional[str] = None
    ) -> Optional[int]:
        """
        Get or create a project - LOCAL-FIRST.

        Always writes to local SQLite, queues for central sync.
        """
        # Always write to local first
        try:
            local_id = local_store.get_or_create_project(path, slug, git_remote)

            # Queue for central sync
            self._queue_for_sync("project", path, {
                "path": path,
                "slug": slug,
                "git_remote": git_remote,
            })

            return local_id
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
        Upsert a session - LOCAL-FIRST.

        Always writes to local SQLite, queues for central sync.
        """
        # Always write to local first
        try:
            project_id = local_store.get_or_create_project(project_path, git_remote=git_remote)
            local_id = local_store.upsert_session(
                project_id=project_id,
                session_id=session_id,
                **kwargs,
            )

            # Queue for central sync
            self._queue_for_sync("session", session_id, {
                "project_path": project_path,
                "session_id": session_id,
                "git_remote": git_remote,
                **kwargs,
            })

            return local_id
        except Exception as e:
            log.error(f"Local upsert_session failed: {e}")
            return None

    def session_exists_in_central(self, session_id: str) -> bool:
        """
        Check if a session exists in central storage.

        Used to detect what needs to be synced.
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
        """
        Get recent sessions.

        Tries central first (cross-machine data), falls back to local.
        """
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
        """
        Full-text search on sessions.

        Tries central first (cross-machine data), falls back to local.
        """
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
        """
        Get all custodian preferences.

        Tries central first (cross-machine data), falls back to local.
        """
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
        Upsert a custodian preference - LOCAL-FIRST.

        Always writes to local, queues for central sync.
        """
        # Always write to local first
        try:
            local_store.upsert_custodian(
                key=key,
                value=value,
                category=category,
                confidence=confidence,
                source_session=source_session,
            )

            # Queue for central sync
            self._queue_for_sync("custodian", key, {
                "key": key,
                "value": value,
                "category": category,
                "confidence": confidence,
                "source_session": source_session,
            })
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
        Search error patterns.

        Tries central first (cross-machine data), falls back to local.
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
        Upsert an error pattern - LOCAL-FIRST.

        Always writes to local, queues for central sync.
        """
        # Always write to local first
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

            # Queue for central sync
            self._queue_for_sync("error", signature, {
                "project_path": project_path,
                "signature": signature,
                "error_type": error_type,
                "error_text": error_text,
                "solution": solution,
                "file_path": file_path,
            })
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
        Search decisions.

        Tries central first (cross-machine data), falls back to local.
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
        Upsert a decision - LOCAL-FIRST.

        Always writes to local, queues for central sync.
        """
        # Always write to local first
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

            # Queue for central sync
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
        except Exception as e:
            log.error(f"Local upsert_decision failed: {e}")

    # ==================== Concept Operations ====================

    def get_concepts(
        self,
        project_path: str,
        concept_type: Optional[str] = None,
        min_confidence: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Get concepts for a project. Uses central if available."""
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

        # Concepts not stored locally yet
        return []

    # ==================== Archive Operations ====================

    def upsert_archive(
        self,
        postgres_session_id: int,
        content: str,
        content_hash: str,
    ) -> Optional[int]:
        """
        Store or update a conversation archive - LOCAL-FIRST.

        Always writes to local SQLite, queues for central sync.
        """
        # Always write to local first
        try:
            local_id = local_store.upsert_archive(
                session_db_id=postgres_session_id,
                content=content,
                content_hash=content_hash,
            )

            # Queue for central sync (archive content is large, use hash as reference)
            self._queue_for_sync("archive", content_hash[:16], {
                "session_db_id": postgres_session_id,
                "content": content,
                "content_hash": content_hash,
            })

            return local_id
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

        Used for active session sync.
        Writes to local first, queues for central.
        """
        # Write to local first
        try:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            # Get local session ID
            local_session = local_store.get_session_by_uuid(session_id)
            if local_session:
                local_store.upsert_archive(
                    session_db_id=local_session['id'],
                    content=content,
                    content_hash=content_hash,
                )

            # Queue for central sync
            self._queue_for_sync("archive_update", session_id, {
                "session_id": session_id,
                "content": content,
                "size_bytes": size_bytes,
                "line_count": line_count,
            })

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
        Update session metadata for an existing session.

        Used for active session sync.
        Writes to local first, queues for central.
        """
        # Write to local first
        try:
            local_store.update_session_metadata(session_id, summary, keywords)

            # Queue for central sync
            self._queue_for_sync("session_metadata", session_id, {
                "session_id": session_id,
                "summary": summary,
                "keywords": keywords,
            })

            return True
        except Exception as e:
            log.error(f"update_session_metadata failed: {e}")
            return False

    def get_archive(self, session_uuid: str) -> Optional[str]:
        """
        Get archive content by session UUID.

        Tries central first (cross-machine data), falls back to local.
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
        """
        Full-text search on archive content.

        Tries central first (cross-machine data), falls back to local.
        """
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
        """
        self._init_central()

        # Get sync queue stats
        try:
            from .sync_queue import get_sync_queue
            queue_stats = get_sync_queue().get_stats()
        except Exception:
            queue_stats = {"total_pending": 0}

        if self._using_central:
            return {
                "mode": "central",
                "description": "Local-first with central sync (cross-machine sync enabled)",
                "qdrant_host": self.config.central.qdrant.host if self.config.central else None,
                "postgres_host": self.config.central.postgres.host if self.config.central else None,
                "pending_sync": queue_stats.get("total_pending", 0),
            }
        else:
            # Get local session count
            try:
                session_count = local_store.get_session_count()
            except Exception:
                session_count = 0

            return {
                "mode": "local",
                "description": "Local SQLite only (keyword search, single-machine)",
                "session_count": session_count,
                "pending_sync": queue_stats.get("total_pending", 0),
                "limitations": [
                    "Keyword search only (no semantic/vector search)",
                    "History stays on this machine only",
                ],
                "setup": {
                    "summary": "To enable semantic search and cross-machine sync, set up central storage.",
                    "config_template": {
                        "version": 1,
                        "central": {
                            "enabled": True,
                            "qdrant": {"host": "YOUR_SERVER_IP", "port": 6333},
                            "postgres": {
                                "host": "YOUR_SERVER_IP",
                                "port": 5432,
                                "database": "mira",
                                "user": "mira",
                                "password": "YOUR_PASSWORD"
                            }
                        }
                    }
                }
            }

    def health_check(self) -> Dict[str, Any]:
        """Check health of storage systems."""
        status = {
            "central_configured": self.config.central_enabled,
            "central_available": False,
            "qdrant_healthy": False,
            "postgres_healthy": False,
            "using_central": False,
            "mode": "local",
            "local_healthy": True,  # Local SQLite is always assumed healthy
        }

        # Check local store
        try:
            local_store.get_session_count()
            status["local_healthy"] = True
        except Exception:
            status["local_healthy"] = False

        # Check central
        if self._init_central():
            status["central_available"] = True
            status["using_central"] = self._using_central
            status["mode"] = "central" if self._using_central else "local"
            if self._qdrant:
                status["qdrant_healthy"] = self._qdrant.is_healthy()
            if self._postgres:
                status["postgres_healthy"] = self._postgres.is_healthy()

        # Get sync queue stats
        try:
            from .sync_queue import get_sync_queue
            status["sync_queue"] = get_sync_queue().get_stats()
        except Exception:
            status["sync_queue"] = {"error": "Failed to get stats"}

        return status

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
