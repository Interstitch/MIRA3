"""
MIRA3 Embedding Module

Handles the sentence-transformers embedding model for Qdrant vector search.
Uses all-MiniLM-L6-v2 which produces 384-dimensional vectors.
"""

from typing import List, Union

from .utils import log, get_models_path, configure_model_cache
from .constants import EMBEDDING_MODEL_NAME

# Global embedding model instance
_embedding_model = None
_embedding_function = None


def get_embedding_model():
    """Get or initialize the sentence-transformers embedding model."""
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    # Ensure cache is configured
    configure_model_cache()

    try:
        from sentence_transformers import SentenceTransformer

        log(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            cache_folder=str(get_models_path())
        )
        log("Embedding model ready.")
        return _embedding_model
    except ImportError:
        log("ERROR: sentence-transformers not available")
        raise RuntimeError("sentence-transformers is required for MIRA")
    except Exception as e:
        log(f"ERROR: Failed to load embedding model: {e}")
        raise


def get_embedding_function():
    """Get the global embedding function (singleton)."""
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = MiraEmbeddingFunction()
    return _embedding_function


class MiraEmbeddingFunction:
    """
    Embedding function using all-MiniLM-L6-v2.

    Produces 384-dimensional vectors compatible with Qdrant.
    """

    def __init__(self):
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            self.model = get_embedding_model()

    def __call__(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Embed one or more texts.

        Args:
            texts: Single string or list of strings to embed

        Returns:
            List of embedding vectors (384-dimensional)
        """
        if isinstance(texts, str):
            texts = [texts]
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        result = self._embed([text])
        return result[0] if result else []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return self._embed(texts)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Internal embedding method."""
        self._ensure_model()
        if self.model is None:
            raise RuntimeError("Embedding model not available")

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]
