"""
MIRA Storage Abstraction Layer

Provides unified interface to central Qdrant + Postgres storage.

Design principles:
- Central storage only (no local fallback)
- Lazy initialization (connect only when needed)
- Clear error reporting when central is unavailable
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .config import get_config, ServerConfig

log = logging.getLogger(__name__)

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
        """
        if not self._init_central() or not self._qdrant:
            raise StorageError("Central storage not available")

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
            raise StorageError(f"Vector search failed: {e}")

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

        Returns the point ID.
        """
        if not self._init_central() or not self._qdrant:
            raise StorageError("Central storage not available")

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
            raise StorageError(f"Vector upsert failed: {e}")

    def vector_batch_upsert(
        self,
        points: List[Dict[str, Any]],
    ) -> int:
        """
        Batch upsert vectors to Qdrant.

        Returns count of points upserted.
        """
        if not points:
            return 0

        if not self._init_central() or not self._qdrant:
            raise StorageError("Central storage not available")

        try:
            return self._qdrant.batch_upsert(points)
        except Exception as e:
            log.error(f"Batch upsert failed: {e}")
            raise StorageError(f"Batch upsert failed: {e}")

    # ==================== Project Operations ====================

    def get_or_create_project(
        self,
        path: str,
        slug: Optional[str] = None,
        git_remote: Optional[str] = None
    ) -> Optional[int]:
        """
        Get or create a project in Postgres.

        Args:
            path: Filesystem path to the project
            slug: Optional short name for the project
            git_remote: Normalized git remote URL (canonical cross-machine identity)

        Returns project ID.
        """
        if not self._init_central() or not self._postgres:
            raise StorageError("Central storage not available")

        try:
            return self._postgres.get_or_create_project(path, slug, git_remote)
        except Exception as e:
            log.error(f"get_or_create_project failed: {e}")
            raise StorageError(f"Project operation failed: {e}")

    # ==================== Session Operations ====================

    def upsert_session(
        self,
        project_path: str,
        session_id: str,
        git_remote: Optional[str] = None,
        **kwargs,
    ) -> Optional[int]:
        """
        Upsert a session to Postgres.

        Args:
            project_path: Filesystem path to the project
            session_id: Unique session identifier
            git_remote: Normalized git remote URL for cross-machine project matching

        Returns session ID.
        """
        if not self._init_central() or not self._postgres:
            raise StorageError("Central storage not available")

        try:
            project_id = self._postgres.get_or_create_project(project_path, git_remote=git_remote)
            return self._postgres.upsert_session(
                project_id=project_id,
                session_id=session_id,
                **kwargs,
            )
        except Exception as e:
            log.error(f"upsert_session failed: {e}")
            raise StorageError(f"Session operation failed: {e}")

    def get_recent_sessions(
        self,
        project_path: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent sessions from Postgres."""
        if not self._init_central() or not self._postgres:
            raise StorageError("Central storage not available")

        try:
            project_id = None
            if project_path:
                project_id = self._postgres.get_or_create_project(project_path)
            return self._postgres.get_recent_sessions(
                project_id=project_id,
                limit=limit,
            )
        except Exception as e:
            log.error(f"get_recent_sessions failed: {e}")
            raise StorageError(f"Session query failed: {e}")

    def search_sessions_fts(
        self,
        query: str,
        project_path: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full-text search on sessions in Postgres."""
        if not self._init_central() or not self._postgres:
            raise StorageError("Central storage not available")

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
            log.error(f"search_sessions_fts failed: {e}")
            raise StorageError(f"FTS query failed: {e}")

    # ==================== Custodian Operations ====================

    def get_custodian_all(self) -> List[Dict[str, Any]]:
        """Get all custodian preferences from Postgres."""
        if not self._init_central() or not self._postgres:
            raise StorageError("Central storage not available")

        try:
            return self._postgres.get_all_custodian()
        except Exception as e:
            log.error(f"get_custodian_all failed: {e}")
            raise StorageError(f"Custodian query failed: {e}")

    def upsert_custodian(
        self,
        key: str,
        value: str,
        category: Optional[str] = None,
        confidence: float = 0.5,
        source_session: Optional[str] = None,
    ):
        """Upsert a custodian preference to Postgres."""
        if not self._init_central() or not self._postgres:
            raise StorageError("Central storage not available")

        try:
            self._postgres.upsert_custodian(
                key=key,
                value=value,
                category=category,
                confidence=confidence,
                source_session=source_session,
            )
        except Exception as e:
            log.error(f"upsert_custodian failed: {e}")
            raise StorageError(f"Custodian update failed: {e}")

    # ==================== Error Pattern Operations ====================

    def search_error_patterns(
        self,
        query: str,
        project_path: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search error patterns in Postgres."""
        if not self._init_central() or not self._postgres:
            raise StorageError("Central storage not available")

        try:
            project_id = None
            if project_path:
                project_id = self._postgres.get_or_create_project(project_path)
            return self._postgres.search_error_patterns(
                query=query,
                project_id=project_id,
                limit=limit,
            )
        except Exception as e:
            log.error(f"search_error_patterns failed: {e}")
            raise StorageError(f"Error search failed: {e}")

    # ==================== Decision Operations ====================

    def search_decisions(
        self,
        query: str,
        project_path: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search decisions in Postgres."""
        if not self._init_central() or not self._postgres:
            raise StorageError("Central storage not available")

        try:
            project_id = None
            if project_path:
                project_id = self._postgres.get_or_create_project(project_path)
            return self._postgres.search_decisions(
                query=query,
                project_id=project_id,
                category=category,
                limit=limit,
            )
        except Exception as e:
            log.error(f"search_decisions failed: {e}")
            raise StorageError(f"Decision search failed: {e}")

    # ==================== Concept Operations ====================

    def get_concepts(
        self,
        project_path: str,
        concept_type: Optional[str] = None,
        min_confidence: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Get concepts for a project from Postgres."""
        if not self._init_central() or not self._postgres:
            raise StorageError("Central storage not available")

        try:
            project_id = self._postgres.get_or_create_project(project_path)
            return self._postgres.get_concepts(
                project_id=project_id,
                concept_type=concept_type,
                min_confidence=min_confidence,
            )
        except Exception as e:
            log.error(f"get_concepts failed: {e}")
            raise StorageError(f"Concept query failed: {e}")

    # ==================== Archive Operations ====================

    def upsert_archive(
        self,
        postgres_session_id: int,
        content: str,
        content_hash: str,
    ) -> Optional[int]:
        """
        Store or update a conversation archive in central storage.

        Args:
            postgres_session_id: The Postgres session ID (foreign key)
            content: Full JSONL content
            content_hash: SHA256 hash for deduplication

        Returns archive ID or None if failed.
        """
        if not self._init_central() or not self._postgres:
            raise StorageError("Central storage not available")

        try:
            return self._postgres.upsert_archive(
                session_id=postgres_session_id,
                content=content,
                content_hash=content_hash,
            )
        except Exception as e:
            log.error(f"upsert_archive failed: {e}")
            raise StorageError(f"Archive operation failed: {e}")

    def get_archive(self, session_uuid: str) -> Optional[str]:
        """
        Get archive content by session UUID.

        Args:
            session_uuid: The session ID string (filename without .jsonl)

        Returns the JSONL content or None if not found.
        """
        if not self._init_central() or not self._postgres:
            raise StorageError("Central storage not available")

        try:
            return self._postgres.get_archive_by_session_uuid(session_uuid)
        except Exception as e:
            log.error(f"get_archive failed: {e}")
            raise StorageError(f"Archive retrieval failed: {e}")

    def search_archives_fts(
        self,
        query: str,
        project_path: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full-text search on archive content."""
        if not self._init_central() or not self._postgres:
            raise StorageError("Central storage not available")

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
            log.error(f"search_archives_fts failed: {e}")
            raise StorageError(f"Archive search failed: {e}")

    def get_archive_stats(self) -> Dict[str, Any]:
        """Get statistics about stored archives."""
        if not self._init_central() or not self._postgres:
            return {"total_archives": 0, "total_bytes": 0, "total_lines": 0, "avg_bytes": 0}

        try:
            return self._postgres.get_archive_stats()
        except Exception as e:
            log.error(f"get_archive_stats failed: {e}")
            return {"total_archives": 0, "total_bytes": 0, "total_lines": 0, "avg_bytes": 0}

    # ==================== Health & Status ====================

    def get_storage_mode(self) -> Dict[str, Any]:
        """
        Get current storage mode information for display to user.

        Returns a dict with:
        - mode: "central" or "local"
        - description: Human-readable description
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
            return {
                "mode": "local",
                "description": "Using local SQLite storage only (single-machine)",
                "setup": {
                    "summary": "To sync across machines, you need a central server running Qdrant + Postgres accessible via Tailscale VPN.",
                    "steps": [
                        "1. Set up a server with Qdrant (port 6333) and Postgres (port 5432) - can use Docker",
                        "2. Install Tailscale on both the server and this machine",
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
                    "note": "Central storage is optional. MIRA works fully in local mode."
                }
            }

    def health_check(self) -> Dict[str, Any]:
        """Check health of central storage."""
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
