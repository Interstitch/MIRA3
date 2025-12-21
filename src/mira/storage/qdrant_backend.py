"""
MIRA Qdrant Backend

Provides vector search operations against a centralized Qdrant server.

Security:
- Connection only to Tailscale IP (network-level auth)
- No sensitive data in logs

Efficiency:
- Lazy connection (only connect when needed)
- Connection reuse (singleton client)
- Batch operations for upserts
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

log = logging.getLogger(__name__)

# Qdrant client - imported lazily to avoid dependency if not using central
_qdrant_client = None


def _get_qdrant_module():
    """Lazy import of qdrant_client to avoid import errors if not installed."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance,
            PointStruct,
            VectorParams,
            Filter,
            FieldCondition,
            MatchValue,
        )
        return {
            "QdrantClient": QdrantClient,
            "Distance": Distance,
            "PointStruct": PointStruct,
            "VectorParams": VectorParams,
            "Filter": Filter,
            "FieldCondition": FieldCondition,
            "MatchValue": MatchValue,
        }
    except ImportError:
        log.warning("qdrant-client not installed, central vector search unavailable")
        return None


@dataclass
class SearchResult:
    """Result from a vector search."""
    id: str
    score: float
    content: str
    session_id: str
    project_path: str
    chunk_type: str
    role: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QdrantBackend:
    """
    Qdrant vector database backend for MIRA.

    Provides:
    - Vector similarity search
    - Batch upsert for embeddings
    - Health checks
    - Project-scoped queries
    """

    def __init__(self, host: str, port: int, collection: str, timeout: int = 30, api_key: Optional[str] = None):
        self.host = host
        self.port = port
        self.collection = collection
        self.timeout = timeout
        self.api_key = api_key
        self._client = None
        self._healthy = False
        self._last_health_check = 0
        self._health_check_interval = 60  # seconds

    def _quick_tcp_check(self, timeout: float = 0.5) -> bool:
        """
        Quick TCP reachability check before attempting expensive client creation.

        Returns True if the port is reachable, False otherwise.
        This prevents 30-second timeouts when the host is unreachable
        (e.g., Tailscale not running).
        """
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _get_client(self):
        """Get or create Qdrant client (lazy initialization)."""
        if self._client is not None:
            return self._client

        # Quick TCP check FIRST - before expensive qdrant-client import
        # This avoids 1+ second import time when host is unreachable
        if not self._quick_tcp_check():
            raise ConnectionError(f"Host {self.host}:{self.port} is not reachable (quick check failed)")

        qdrant = _get_qdrant_module()
        if qdrant is None:
            raise ImportError("qdrant-client not installed")

        # Use explicit URL to force HTTP (QdrantClient defaults to HTTPS when api_key is set)
        qdrant_url = f"http://{self.host}:{self.port}"
        log.info(f"Connecting to Qdrant at {qdrant_url}")
        self._client = qdrant["QdrantClient"](
            url=qdrant_url,
            timeout=self.timeout,
            api_key=self.api_key,  # None if not set (auth disabled)
        )

        # Verify connection and collection exists
        try:
            collections = self._client.get_collections()
            collection_names = [c.name for c in collections.collections]
            if self.collection not in collection_names:
                log.warning(f"Collection '{self.collection}' not found, creating...")
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qdrant["VectorParams"](
                        size=384,  # all-MiniLM-L6-v2
                        distance=qdrant["Distance"].COSINE,
                    ),
                )
            self._healthy = True
            self._last_health_check = time.time()
            log.info(f"Connected to Qdrant, collection: {self.collection}")
        except Exception as e:
            log.error(f"Failed to connect to Qdrant: {e}")
            self._client = None
            raise

        return self._client

    def is_healthy(self) -> bool:
        """
        Check if Qdrant connection is healthy.

        Caches health check result for efficiency.
        """
        now = time.time()
        if now - self._last_health_check < self._health_check_interval:
            return self._healthy

        try:
            client = self._get_client()
            client.get_collection(self.collection)
            self._healthy = True
            self._last_health_check = now
            return True
        except Exception as e:
            log.warning(f"Qdrant health check failed: {e}")
            self._healthy = False
            self._last_health_check = now
            return False

    def search(
        self,
        query_vector: List[float],
        project_path: Optional[str] = None,
        limit: int = 10,
        chunk_types: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: 384-dimensional embedding vector
            project_path: Optional project path to filter by
            limit: Maximum number of results
            chunk_types: Optional list of chunk types to filter by

        Returns:
            List of SearchResult objects sorted by similarity
        """
        qdrant = _get_qdrant_module()
        if qdrant is None:
            return []

        try:
            client = self._get_client()

            # Build filter conditions
            filter_conditions = []
            if project_path:
                filter_conditions.append(
                    qdrant["FieldCondition"](
                        key="project_path",
                        match=qdrant["MatchValue"](value=project_path),
                    )
                )
            if chunk_types:
                # Match any of the specified chunk types
                for chunk_type in chunk_types:
                    filter_conditions.append(
                        qdrant["FieldCondition"](
                            key="chunk_type",
                            match=qdrant["MatchValue"](value=chunk_type),
                        )
                    )

            query_filter = None
            if filter_conditions:
                query_filter = qdrant["Filter"](must=filter_conditions)

            results = client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

            return [
                SearchResult(
                    id=str(r.id),
                    score=r.score,
                    content=r.payload.get("content", ""),
                    session_id=r.payload.get("session_id", ""),
                    project_path=r.payload.get("project_path", ""),
                    chunk_type=r.payload.get("chunk_type", "message"),
                    role=r.payload.get("role"),
                    metadata=r.payload.get("metadata"),
                )
                for r in results
            ]

        except Exception as e:
            log.error(f"Qdrant search failed: {e}")
            self._healthy = False
            raise

    def upsert(
        self,
        vector: List[float],
        content: str,
        session_id: str,
        project_path: str,
        chunk_type: str = "message",
        role: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        point_id: Optional[str] = None,
    ) -> str:
        """
        Upsert a single vector with payload.

        Returns the point ID.
        """
        qdrant = _get_qdrant_module()
        if qdrant is None:
            raise ImportError("qdrant-client not installed")

        point_id = point_id or str(uuid4())

        payload = {
            "content": content,
            "session_id": session_id,
            "project_path": project_path,
            "chunk_type": chunk_type,
            "indexed_at": time.time(),
        }
        if role:
            payload["role"] = role
        if metadata:
            payload["metadata"] = metadata

        try:
            client = self._get_client()
            client.upsert(
                collection_name=self.collection,
                points=[
                    qdrant["PointStruct"](
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            return point_id
        except Exception as e:
            log.error(f"Qdrant upsert failed: {e}")
            self._healthy = False
            raise

    def batch_upsert(
        self,
        points: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Batch upsert multiple vectors.

        Args:
            points: List of dicts with keys:
                - vector: List[float]
                - content: str
                - session_id: str
                - project_path: str
                - chunk_type: str (optional)
                - role: str (optional)
                - metadata: dict (optional)
            batch_size: Number of points per batch

        Returns:
            Number of points upserted
        """
        qdrant = _get_qdrant_module()
        if qdrant is None:
            raise ImportError("qdrant-client not installed")

        if not points:
            return 0

        try:
            client = self._get_client()
            total = 0

            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                qdrant_points = []

                for p in batch:
                    point_id = p.get("id") or str(uuid4())
                    payload = {
                        "content": p["content"],
                        "session_id": p["session_id"],
                        "project_path": p["project_path"],
                        "chunk_type": p.get("chunk_type", "message"),
                        "indexed_at": time.time(),
                    }
                    if p.get("role"):
                        payload["role"] = p["role"]
                    if p.get("metadata"):
                        payload["metadata"] = p["metadata"]

                    qdrant_points.append(
                        qdrant["PointStruct"](
                            id=point_id,
                            vector=p["vector"],
                            payload=payload,
                        )
                    )

                client.upsert(
                    collection_name=self.collection,
                    points=qdrant_points,
                )
                total += len(qdrant_points)
                log.debug(f"Upserted batch of {len(qdrant_points)} points")

            return total

        except Exception as e:
            log.error(f"Qdrant batch upsert failed: {e}")
            self._healthy = False
            raise

    def delete_by_session(self, session_id: str) -> int:
        """Delete all points for a session (for re-indexing)."""
        qdrant = _get_qdrant_module()
        if qdrant is None:
            raise ImportError("qdrant-client not installed")

        try:
            client = self._get_client()
            result = client.delete(
                collection_name=self.collection,
                points_selector=qdrant["Filter"](
                    must=[
                        qdrant["FieldCondition"](
                            key="session_id",
                            match=qdrant["MatchValue"](value=session_id),
                        )
                    ]
                ),
            )
            return result.status
        except Exception as e:
            log.error(f"Qdrant delete failed: {e}")
            raise

    def count(self, project_path: Optional[str] = None) -> int:
        """Count points, optionally filtered by project."""
        try:
            client = self._get_client()
            info = client.get_collection(self.collection)
            # Note: This returns total count, not filtered
            # For filtered count, would need to do a count query
            return info.points_count
        except Exception as e:
            log.error(f"Qdrant count failed: {e}")
            return 0

    def close(self):
        """Close the Qdrant connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._healthy = False
