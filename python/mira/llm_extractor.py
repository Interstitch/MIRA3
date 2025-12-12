"""
MIRA LLM Extractor Client

Client for the decision-extractor service that runs on the central server.
Triggers LLM-based extraction of decisions, preferences, and errors from sessions.
"""

import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Default extractor service URL (Tailscale IP)
DEFAULT_EXTRACTOR_URL = "http://100.107.224.88:8100"

# Lazy-loaded httpx
_httpx = None


def _get_httpx():
    """Lazy import httpx to avoid requiring it when not using LLM extraction."""
    global _httpx
    if _httpx is not None:
        return _httpx
    try:
        import httpx
        _httpx = httpx
        return _httpx
    except ImportError:
        log.warning("httpx not installed - LLM extraction unavailable")
        return None


class LLMExtractorClient:
    """
    Client for the decision-extractor service.

    The service runs on the central server and uses Ollama + llama3.2:8b
    to extract decisions, preferences, and error patterns from conversations.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0):
        """
        Initialize the extractor client.

        Args:
            base_url: URL of the decision-extractor service.
                      Defaults to MIRA_EXTRACTOR_URL env or the Tailscale IP.
            timeout: Request timeout in seconds. LLM extraction is slow (~10-30s).
        """
        self.base_url = base_url or os.getenv("MIRA_EXTRACTOR_URL", DEFAULT_EXTRACTOR_URL)
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Get or create httpx client."""
        httpx = _get_httpx()
        if httpx is None:
            return None
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def is_available(self) -> bool:
        """Check if the extractor service is available."""
        client = self._get_client()
        if client is None:
            return False
        try:
            response = client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            log.debug(f"Extractor service not available: {e}")
            return False

    def extract_from_text(
        self,
        conversation: str,
        session_id: Optional[str] = None,
        extract_types: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Extract insights from conversation text.

        Args:
            conversation: The conversation text to analyze
            session_id: Optional session ID to associate with extracted insights
            extract_types: Types to extract: ["decisions", "preferences", "errors"]
                          Defaults to all types.

        Returns:
            Dict with extracted decisions, preferences, and errors
        """
        client = self._get_client()
        if client is None:
            return {"error": "httpx not installed"}

        extract_types = extract_types or ["decisions", "preferences", "errors"]

        try:
            response = client.post(
                f"{self.base_url}/extract/all",
                json={
                    "conversation": conversation,
                    "session_id": session_id,
                    "extract_types": extract_types,
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"LLM extraction failed: {e}")
            return {"error": str(e)}

    def process_session(self, session_id: str) -> Dict[str, Any]:
        """
        Process a session from the database.

        The extractor service will:
        1. Fetch the conversation content from the archives table
        2. Run LLM extraction
        3. Store results back to Postgres
        4. Update llm_processed_at timestamp

        Args:
            session_id: The session ID to process

        Returns:
            Dict with processing results
        """
        client = self._get_client()
        if client is None:
            return {"error": "httpx not installed"}

        try:
            response = client.post(
                f"{self.base_url}/process/session",
                json={"session_id": session_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"Session processing failed: {e}")
            return {"error": str(e)}

    def process_batch(
        self,
        limit: int = 10,
        unprocessed_only: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple sessions in batch.

        Args:
            limit: Maximum number of sessions to process
            unprocessed_only: If True, only process sessions without llm_processed_at

        Returns:
            Dict with batch processing results
        """
        client = self._get_client()
        if client is None:
            return {"error": "httpx not installed"}

        try:
            response = client.post(
                f"{self.base_url}/process/batch",
                json={
                    "limit": limit,
                    "unprocessed_only": unprocessed_only,
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"Batch processing failed: {e}")
            return {"error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get extractor service statistics."""
        client = self._get_client()
        if client is None:
            return {"error": "httpx not installed"}

        try:
            response = client.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None


# Singleton client instance
_extractor_client: Optional[LLMExtractorClient] = None


def get_extractor_client() -> LLMExtractorClient:
    """Get the singleton extractor client."""
    global _extractor_client
    if _extractor_client is None:
        _extractor_client = LLMExtractorClient()
    return _extractor_client


def trigger_llm_extraction(session_id: str) -> bool:
    """
    Trigger LLM extraction for a session.

    This is called after session ingestion to queue LLM processing.
    Returns True if the request was successful, False otherwise.

    Note: This is fire-and-forget - the actual extraction happens async
    on the server and may take 10-30 seconds per session.
    """
    client = get_extractor_client()

    if not client.is_available():
        log.debug("LLM extractor service not available - skipping")
        return False

    result = client.process_session(session_id)
    if "error" in result:
        log.warning(f"LLM extraction trigger failed: {result['error']}")
        return False

    log.info(f"LLM extraction triggered for session {session_id[:8]}...")
    return True
