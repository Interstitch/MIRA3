"""
MIRA3 Embedding Module

Handles the sentence-transformers embedding model for ChromaDB.
"""

from .utils import log, get_models_path, configure_model_cache
from .constants import EMBEDDING_MODEL_NAME

# Global embedding model instance
_embedding_model = None


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
        log("WARNING: sentence-transformers not available, using ChromaDB defaults")
        return None
    except Exception as e:
        log(f"WARNING: Failed to load embedding model: {e}")
        return None


class MiraEmbeddingFunction:
    """
    Custom ChromaDB embedding function using all-MiniLM-L6-v2.

    Implements the ChromaDB EmbeddingFunction protocol with all required methods.
    """

    def __init__(self):
        self.model = None

    def name(self) -> str:
        """Return embedding function name (required by ChromaDB)."""
        return "mira_minilm"

    def _ensure_model(self):
        if self.model is None:
            self.model = get_embedding_model()

    def __call__(self, input: list) -> list:
        """Embed a list of documents (for adding to collection).

        Note: ChromaDB 0.4.16+ requires parameter name 'input' instead of 'texts'.
        """
        return self._embed(input)

    def embed_documents(self, texts: list) -> list:
        """Embed a list of documents (ChromaDB interface)."""
        return self._embed(texts)

    def embed_query(self, text: str) -> list:
        """Embed a single query text (ChromaDB interface)."""
        result = self._embed([text])
        return result[0] if result else []

    def _embed(self, texts: list) -> list:
        """Internal embedding method."""
        self._ensure_model()
        if self.model is None:
            raise RuntimeError("Embedding model not available")

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Return as list of lists (ChromaDB format)
        return [emb.tolist() for emb in embeddings]
