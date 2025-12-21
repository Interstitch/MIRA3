"""
MIRA Storage Abstraction Layer - LOCAL-FIRST Architecture

All writes go to local SQLite FIRST, then sync to central in background.
This ensures:
- Fast writes (no network latency)
- Offline capability (works without central)
- Reliable data (local write always succeeds)
"""

import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from mira.core import get_config

log = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when storage operations fail."""
    pass


class Storage:
    """
    Unified storage interface for MIRA.

    LOCAL-FIRST: All writes go to local SQLite first, then sync to central.
    """

    CENTRAL_RETRY_INTERVAL = 60  # seconds

    def __init__(self, config=None):
        from mira.core.config import ServerConfig
        self.config = config or get_config()
        self._qdrant = None
        self._postgres = None
        self._using_central = False
        self._central_init_attempted = False
        self._central_init_failed_at: Optional[float] = None

    def _init_central(self) -> bool:
        """Initialize central backends (lazy, for reads)."""
        if self._using_central:
            return True

        if self._central_init_attempted and not self._using_central:
            if self._central_init_failed_at is not None:
                elapsed = time.time() - self._central_init_failed_at
                if elapsed < self.CENTRAL_RETRY_INTERVAL:
                    return False
                self._central_init_attempted = False
                self._central_init_failed_at = None

        if self._central_init_attempted:
            return self._using_central

        self._central_init_attempted = True

        if not self.config.central_enabled:
            return False

        try:
            from mira.storage.qdrant_backend import QdrantBackend
            from mira.storage.postgres_backend import PostgresBackend

            qdrant_cfg = self.config.central.qdrant
            self._qdrant = QdrantBackend(
                host=qdrant_cfg.host,
                port=qdrant_cfg.port,
                collection=qdrant_cfg.collection,
                timeout=qdrant_cfg.timeout_seconds,
                api_key=getattr(qdrant_cfg, 'api_key', None),
            )

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

            if self._qdrant.is_healthy() and self._postgres.is_healthy():
                self._using_central = True
                log.info("Central storage initialized successfully")
                return True
            else:
                self._central_init_failed_at = time.time()
                return False

        except ImportError as e:
            log.error(f"Central storage dependencies not installed: {e}")
            self._central_init_failed_at = time.time()
            return False
        except Exception as e:
            log.error(f"Failed to initialize central storage: {e}")
            self._central_init_failed_at = time.time()
            return False

    @property
    def using_central(self) -> bool:
        """Check if using central storage."""
        self._init_central()
        return self._using_central

    @property
    def central_configured(self) -> bool:
        """Check if central storage is configured."""
        return self.config.central_enabled

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

    def _queue_for_sync(self, data_type: str, item_id: str, payload: Dict[str, Any]):
        """Queue an item for later sync to central storage."""
        if not self.config.central_enabled:
            return

        try:
            from mira.storage.sync.queue import get_sync_queue
            queue = get_sync_queue()
            hash_input = f"{data_type}:{item_id}"
            item_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:32]
            queue.enqueue(data_type, item_hash, payload)
        except Exception as e:
            log.error(f"Failed to queue {data_type} for sync: {e}")

    # ==================== Project Operations ====================

    def get_project_id(self, project_path: str) -> Optional[int]:
        """Get project ID for a given path."""
        from mira.storage.local_store import get_project_id as local_get_project_id
        try:
            local_id = local_get_project_id(project_path)
            if local_id:
                return local_id
        except Exception as e:
            log.error(f"Local get_project_id failed: {e}")
        return None

    def get_or_create_project(self, path: str, slug: Optional[str] = None, git_remote: Optional[str] = None) -> Optional[int]:
        """Get or create a project - LOCAL-FIRST."""
        from mira.storage.local_store import get_or_create_project as local_get_or_create
        try:
            local_id = local_get_or_create(path, slug, git_remote)
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

    def upsert_session(self, project_path: str, session_id: str, git_remote: Optional[str] = None, **kwargs) -> Optional[int]:
        """Upsert a session - LOCAL-FIRST."""
        from mira.storage import local_store
        try:
            project_id = local_store.get_or_create_project(project_path, git_remote=git_remote)
            local_id = local_store.upsert_session(project_id=project_id, session_id=session_id, **kwargs)
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

    def upsert_archive(self, postgres_session_id: Optional[int], content: str, content_hash: str, session_id: str) -> Optional[int]:
        """Upsert conversation archive - CENTRAL ONLY (local stores raw files)."""
        # Archives are only stored in central postgres, not locally
        # Local storage uses the raw .jsonl files directly
        if self._using_central and self._postgres:
            try:
                return self._postgres.upsert_archive(
                    session_id=postgres_session_id,
                    content=content,
                    content_hash=content_hash,
                )
            except Exception as e:
                log.debug(f"Central upsert_archive failed (non-fatal): {e}")
                return None
        return None  # No-op when running local-only

    def get_recent_sessions(self, project_path: Optional[str] = None, limit: int = 10, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get recent sessions."""
        from mira.storage import local_store
        try:
            project_id = None
            if project_path:
                project_id = local_store.get_or_create_project(project_path)
            return local_store.get_recent_sessions(project_id=project_id, limit=limit, since=since)
        except Exception as e:
            log.error(f"Local get_recent_sessions failed: {e}")
            return []

    def search_sessions_fts(self, query: str, project_path: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search sessions - uses central if available, else local."""
        # Try central first
        if self._using_central and self._postgres:
            try:
                project_id = None
                if project_path:
                    project_id = self._postgres.get_project_id(project_path)
                return self._postgres.search_sessions_fts(query, project_id, limit)
            except Exception as e:
                log.error(f"Central FTS search failed: {e}")

        # Fall back to local
        from mira.storage.local_store import search_sessions_fts as local_fts
        try:
            project_id = None
            if project_path:
                from mira.storage.local_store import get_project_id
                project_id = get_project_id(project_path)
            return local_fts(query, project_id, limit)
        except Exception as e:
            log.error(f"Local FTS search failed: {e}")
            return []

    def search_archives_fts(self, query: str, project_path: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search archive content - central only."""
        if self._using_central and self._postgres:
            try:
                project_id = None
                if project_path:
                    project_id = self._postgres.get_project_id(project_path)
                return self._postgres.search_archives_fts(query, project_id, limit)
            except Exception as e:
                log.error(f"Central archive FTS search failed: {e}")
        return []

    def get_archive(self, session_id: str) -> Optional[str]:
        """Get archive content for a session - central only."""
        if self._using_central and self._postgres:
            try:
                return self._postgres.get_archive_by_session_uuid(session_id)
            except Exception as e:
                log.error(f"Central get_archive failed: {e}")
        return None

    # ==================== Health & Status ====================

    def get_storage_mode(self) -> Dict[str, Any]:
        """Get current storage mode information."""
        self._init_central()

        if self._using_central:
            return {
                "mode": "central",
                "description": "Local-first with central sync (cross-machine sync enabled)",
            }
        else:
            from mira.storage.local_store import get_session_count
            try:
                session_count = get_session_count()
            except Exception:
                session_count = 0

            return {
                "mode": "local",
                "description": "Local SQLite only (keyword search, single-machine)",
                "session_count": session_count,
                "limitations": [
                    "Keyword search only (no semantic/vector search)",
                    "History stays on this machine only",
                ],
            }

    def health_check(self) -> Dict[str, Any]:
        """Check health of storage systems."""
        from mira.storage.local_store import get_session_count
        status = {
            "central_configured": self.config.central_enabled,
            "central_available": False,
            "qdrant_healthy": False,
            "postgres_healthy": False,
            "using_central": False,
            "mode": "local",
            "local_healthy": True,
        }

        try:
            get_session_count()
            status["local_healthy"] = True
        except Exception:
            status["local_healthy"] = False

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


# Global storage instance
_storage: Optional[Storage] = None


def get_storage() -> Storage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage


def reset_storage():
    """Reset the global storage instance."""
    global _storage
    if _storage:
        _storage.close()
    _storage = None
