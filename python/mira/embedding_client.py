"""
MIRA Embedding Client

Client for semantic search via the remote embedding service.
The embedding service handles all vector computation and storage automatically.
Client only needs to query for search results.
"""

import logging
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error
import json

from .config import EmbeddingConfig

log = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Client for semantic search via the MIRA embedding service.

    The embedding service automatically polls Postgres for new sessions
    and indexes them to Qdrant. This client only queries for search.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.base_url = f"http://{config.host}:{config.port}"
        self.timeout = config.timeout_seconds

    def _request(self, method: str, endpoint: str, data: Optional[dict] = None) -> dict:
        """Make HTTP request to embedding service."""
        url = f"{self.base_url}{endpoint}"

        headers = {"Content-Type": "application/json"}
        body = json.dumps(data).encode("utf-8") if data else None

        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            log.error(f"Embedding service error {e.code}: {error_body}")
            raise RuntimeError(f"Embedding service error: {e.code}")
        except urllib.error.URLError as e:
            log.error(f"Cannot connect to embedding service: {e.reason}")
            raise RuntimeError(f"Embedding service unavailable: {e.reason}")

    def health_check(self) -> Dict[str, Any]:
        """Check embedding service health."""
        try:
            return self._request("GET", "/health")
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def is_healthy(self) -> bool:
        """Check if embedding service is healthy."""
        health = self.health_check()
        return health.get("status") == "healthy"

    def search(
        self,
        query: str,
        project_id: Optional[int] = None,
        project_path: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Semantic search via Qdrant.

        The embedding service computes the query vector and searches Qdrant.

        Args:
            query: Search query text
            project_id: Optional project ID to filter by
            project_path: Optional project path to filter by
            limit: Maximum results

        Returns:
            Dict with results list, query, count
        """
        data = {"query": query, "limit": limit}
        if project_id is not None:
            data["project_id"] = project_id
        if project_path:
            data["project_path"] = project_path

        return self._request("POST", "/search", data)

    def stats(self) -> Dict[str, Any]:
        """
        Get Qdrant collection statistics.

        Returns:
            Dict with collection name, vectors_count, points_count, status
        """
        return self._request("GET", "/stats")


# Module-level singleton
_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client(config: Optional[EmbeddingConfig] = None) -> Optional[EmbeddingClient]:
    """
    Get or create the embedding client singleton.

    Returns None if embedding service is not configured.
    """
    global _embedding_client

    if _embedding_client is not None:
        return _embedding_client

    if config is None:
        from .config import get_config
        server_config = get_config()
        if server_config.central and server_config.central.embedding:
            config = server_config.central.embedding

    if config is None:
        return None

    _embedding_client = EmbeddingClient(config)
    return _embedding_client
