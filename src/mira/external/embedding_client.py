"""
MIRA Embedding Client

Client for the remote embedding service that provides:
- Query embedding generation
- Vector search against Qdrant

The embedding service runs on the same host as Qdrant (default port 8200).
"""

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Global client instance
_client: Optional["EmbeddingClient"] = None


class EmbeddingClient:
    """
    Client for MIRA's embedding service.

    The service provides:
    - POST /search - Embed query and search Qdrant
    - GET /health - Service health check
    """

    def __init__(self, host: str, port: int = 8200, timeout: int = 60):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"
        self._httpx = None
        self._available = None

    def _get_client(self):
        """Lazy load httpx client."""
        if self._httpx is None:
            try:
                import httpx
                self._httpx = httpx.Client(timeout=self.timeout)
            except ImportError:
                log.warning("httpx not installed, embedding client unavailable")
                return None
        return self._httpx

    def is_available(self) -> bool:
        """Check if the embedding service is reachable."""
        if self._available is not None:
            return self._available

        client = self._get_client()
        if not client:
            self._available = False
            return False

        try:
            response = client.get(f"{self.base_url}/health", timeout=5)
            self._available = response.status_code == 200
            if self._available:
                log.info(f"Embedding service available at {self.base_url}")
            return self._available
        except Exception as e:
            log.debug(f"Embedding service not available: {e}")
            self._available = False
            return False

    def search(
        self,
        query: str,
        project_id: Optional[int] = None,
        project_path: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Search using the embedding service.

        Args:
            query: Search query text
            project_id: Optional project ID filter
            project_path: Optional project path filter
            limit: Maximum results

        Returns:
            Dict with 'results' list containing session matches
        """
        client = self._get_client()
        if not client:
            return {"results": [], "error": "httpx not available"}

        try:
            payload = {
                "query": query,
                "limit": limit,
            }
            if project_id is not None:
                payload["project_id"] = project_id
            if project_path:
                payload["project_path"] = project_path

            response = client.post(
                f"{self.base_url}/search",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            log.error(f"Embedding search failed: {e}")
            return {"results": [], "error": str(e)}

    def close(self):
        """Close the HTTP client."""
        if self._httpx:
            self._httpx.close()
            self._httpx = None


def get_embedding_client() -> Optional[EmbeddingClient]:
    """
    Get the global embedding client instance.

    Returns None if embedding service is not configured or unavailable.
    """
    global _client

    if _client is not None:
        return _client if _client.is_available() else None

    try:
        from mira.core.config import get_config
        config = get_config()

        if not config.central_enabled:
            return None

        embedding_cfg = config.central.embedding
        if not embedding_cfg:
            # Default: same host as Qdrant, port 8200
            qdrant_cfg = config.central.qdrant
            if qdrant_cfg and qdrant_cfg.host:
                _client = EmbeddingClient(
                    host=qdrant_cfg.host,
                    port=8200,
                    timeout=60,
                )
            else:
                return None
        else:
            _client = EmbeddingClient(
                host=embedding_cfg.host,
                port=embedding_cfg.port,
                timeout=embedding_cfg.timeout_seconds,
            )

        if _client.is_available():
            return _client
        else:
            return None

    except Exception as e:
        log.error(f"Failed to initialize embedding client: {e}")
        return None


def reset_embedding_client():
    """Reset the global embedding client (for testing)."""
    global _client
    if _client:
        _client.close()
    _client = None
