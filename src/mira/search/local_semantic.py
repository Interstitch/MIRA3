"""
MIRA Search - Local Semantic Search Module

Provides semantic search when remote storage is unavailable.
Uses fastembed (ONNX-based) + sqlite-vec for local vector search.

Lazy Loading Strategy:
- Model only downloads when remote is unavailable AND user searches
- First search after remote goes down: returns FTS5 results, starts background download
- Subsequent searches: use local semantic (model cached)

Architecture:
- fastembed: ~100MB ONNX model, no PyTorch needed
- sqlite-vec: SQLite extension for vector similarity
- local_vectors.db: Stores embeddings as BLOBs
"""

import struct
import threading
import time
from typing import List, Dict, Any, Optional

from mira.core import log
from mira.core.database import get_db_manager
from mira.core.constants import (
    DB_LOCAL_VECTORS,
    LOCAL_SEMANTIC_ENABLED,
    LOCAL_SEMANTIC_INDEX_INTERVAL,
    LOCAL_SEMANTIC_BATCH_SIZE,
    LOCAL_SEMANTIC_PROACTIVE,
    LOCAL_SEMANTIC_STARTUP_DELAY,
)

# Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, same as remote
EMBEDDING_DIM = 384
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 500
MAX_CHUNKS = 50

# Schema for local vector storage
LOCAL_VECTORS_SCHEMA = """
-- Session vectors (chunked like remote)
CREATE TABLE IF NOT EXISTS session_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    chunk_text TEXT,
    embedding BLOB NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_sv_session ON session_vectors(session_id);

-- Track which sessions have been indexed
CREATE TABLE IF NOT EXISTS indexed_sessions (
    session_id TEXT PRIMARY KEY,
    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    chunk_count INTEGER DEFAULT 0
);

-- Model status tracking
CREATE TABLE IF NOT EXISTS model_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    model_name TEXT,
    model_ready INTEGER DEFAULT 0,
    download_started_at TEXT,
    download_completed_at TEXT,
    last_error TEXT
);

INSERT OR IGNORE INTO model_status (id, model_name, model_ready) VALUES (1, '', 0);

-- Queue for sessions pending local vector indexing
CREATE TABLE IF NOT EXISTS indexing_queue (
    session_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT,
    queued_at TEXT DEFAULT CURRENT_TIMESTAMP,
    attempts INTEGER DEFAULT 0,
    last_error TEXT
);
"""


class LocalSemanticSearch:
    """
    Local semantic search using fastembed + sqlite-vec.

    Lazy loads the embedding model only when:
    1. Remote storage is unavailable AND
    2. User actually searches
    """

    _instance = None
    _lock = threading.Lock()
    _model = None
    _model_loading = False
    _sqlite_vec_available = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._ensure_schema()
        self._initialized = True

    def _ensure_schema(self):
        """Initialize the local vectors database."""
        try:
            db = get_db_manager()
            db.init_schema(DB_LOCAL_VECTORS, LOCAL_VECTORS_SCHEMA)
        except Exception as e:
            log(f"Failed to initialize local vectors schema: {e}")

    @property
    def sqlite_vec_available(self) -> bool:
        """Check if sqlite-vec extension can be loaded."""
        if self._sqlite_vec_available is not None:
            return self._sqlite_vec_available

        try:
            import sqlite3
            import sqlite_vec

            # Test if we can load the extension
            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.close()

            self._sqlite_vec_available = True
            log("sqlite-vec extension available")
        except Exception as e:
            self._sqlite_vec_available = False
            log(f"sqlite-vec not available: {e}")

        return self._sqlite_vec_available

    def is_model_ready(self) -> bool:
        """Check if the embedding model is cached and ready."""
        if self._model is not None:
            return True

        # Check database status
        db = get_db_manager()
        try:
            row = db.execute_read_one(
                DB_LOCAL_VECTORS,
                "SELECT model_ready FROM model_status WHERE id = 1",
                ()
            )
            return bool(row and row.get('model_ready'))
        except Exception:
            return False

    def is_download_in_progress(self) -> bool:
        """Check if model download is currently in progress."""
        return self._model_loading

    def get_status(self) -> Dict[str, Any]:
        """Get local semantic search status for mira_status."""
        db = get_db_manager()

        status = {
            "available": False,
            "sqlite_vec": self.sqlite_vec_available,
            "model_ready": self.is_model_ready(),
            "download_in_progress": self._model_loading,
            "indexed_sessions": 0,
            "total_vectors": 0,
        }

        if not self.sqlite_vec_available:
            status["limitation"] = "sqlite-vec extension not available"
            return status

        try:
            # Count indexed sessions
            row = db.execute_read_one(
                DB_LOCAL_VECTORS,
                "SELECT COUNT(*) as count FROM indexed_sessions",
                ()
            )
            status["indexed_sessions"] = row.get('count', 0) if row else 0

            # Count vectors
            row = db.execute_read_one(
                DB_LOCAL_VECTORS,
                "SELECT COUNT(*) as count FROM session_vectors",
                ()
            )
            status["total_vectors"] = row.get('count', 0) if row else 0

            # Get model status
            row = db.execute_read_one(
                DB_LOCAL_VECTORS,
                "SELECT model_name, download_started_at, download_completed_at, last_error FROM model_status WHERE id = 1",
                ()
            )
            if row:
                status["model_name"] = row.get('model_name', '')
                if row.get('last_error'):
                    status["last_error"] = row.get('last_error')

            status["available"] = status["sqlite_vec"] and status["model_ready"]

        except Exception as e:
            status["error"] = str(e)

        return status

    def _load_model(self) -> bool:
        """
        Load the fastembed model.

        This triggers the ~100MB download on first use.
        Should be called from background thread to avoid blocking.
        """
        if self._model is not None:
            return True

        if self._model_loading:
            return False  # Already loading

        with self._lock:
            if self._model is not None:
                return True

            self._model_loading = True
            db = get_db_manager()

            try:
                # Mark download started
                db.execute_write(
                    DB_LOCAL_VECTORS,
                    """UPDATE model_status SET
                       model_name = ?,
                       download_started_at = CURRENT_TIMESTAMP,
                       last_error = NULL
                       WHERE id = 1""",
                    (EMBEDDING_MODEL,)
                )

                log(f"Loading fastembed model: {EMBEDDING_MODEL} (~100MB download on first use)")

                from fastembed import TextEmbedding
                self._model = TextEmbedding(EMBEDDING_MODEL)

                # Mark download complete
                db.execute_write(
                    DB_LOCAL_VECTORS,
                    """UPDATE model_status SET
                       model_ready = 1,
                       download_completed_at = CURRENT_TIMESTAMP
                       WHERE id = 1""",
                    ()
                )

                log(f"fastembed model loaded successfully")
                return True

            except Exception as e:
                log(f"Failed to load fastembed model: {e}")
                db.execute_write(
                    DB_LOCAL_VECTORS,
                    "UPDATE model_status SET last_error = ?, model_ready = 0 WHERE id = 1",
                    (str(e),)
                )
                return False
            finally:
                self._model_loading = False

    def start_background_download(self):
        """Start model download in background thread."""
        if self._model is not None or self._model_loading:
            return  # Already ready or loading

        def download():
            log("Starting background download of fastembed model...")
            self._load_model()

        thread = threading.Thread(target=download, daemon=True)
        thread.start()

    def embed(self, texts: List[str]) -> List[bytes]:
        """
        Embed texts and return as BLOBs for sqlite-vec.

        Raises RuntimeError if model not ready.
        """
        if self._model is None:
            if not self._load_model():
                raise RuntimeError("Embedding model not ready")

        embeddings = list(self._model.embed(texts))
        return [struct.pack(f'{EMBEDDING_DIM}f', *emb) for emb in embeddings]

    def embed_query(self, query: str) -> bytes:
        """Embed a single query."""
        return self.embed([query])[0]

    def search(
        self,
        query: str,
        project_path: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using local vectors.

        Raises RuntimeError if not ready (model not cached, sqlite-vec unavailable).
        """
        if not self.sqlite_vec_available:
            raise RuntimeError("sqlite-vec not available")

        if not self.is_model_ready():
            raise RuntimeError("Embedding model not ready")

        # Embed query
        query_blob = self.embed_query(query)

        # Search vectors using sqlite-vec
        db = get_db_manager()

        # For now, use a brute-force approach with Python
        # TODO: Replace with proper sqlite-vec virtual table once integrated
        rows = db.execute_read(
            DB_LOCAL_VECTORS,
            """
            SELECT sv.session_id, sv.chunk_index, sv.chunk_text, sv.embedding,
                   i.indexed_at
            FROM session_vectors sv
            JOIN indexed_sessions i ON sv.session_id = i.session_id
            ORDER BY sv.id DESC
            LIMIT 1000
            """,
            ()
        )

        if not rows:
            return []

        # Compute similarities
        query_vec = struct.unpack(f'{EMBEDDING_DIM}f', query_blob)
        scored = []

        for row in rows:
            try:
                emb_blob = row['embedding']
                emb_vec = struct.unpack(f'{EMBEDDING_DIM}f', emb_blob)

                # Cosine similarity
                dot = sum(a * b for a, b in zip(query_vec, emb_vec))
                norm_q = sum(a * a for a in query_vec) ** 0.5
                norm_e = sum(a * a for a in emb_vec) ** 0.5
                similarity = dot / (norm_q * norm_e) if norm_q and norm_e else 0

                scored.append({
                    'session_id': row['session_id'],
                    'chunk_index': row['chunk_index'],
                    'chunk_text': row['chunk_text'],
                    'score': similarity,
                })
            except Exception as e:
                log(f"Error computing similarity: {e}")

        # Sort by score, deduplicate by session_id
        scored.sort(key=lambda x: x['score'], reverse=True)

        seen = set()
        results = []
        for item in scored:
            if item['session_id'] not in seen:
                seen.add(item['session_id'])
                results.append({
                    'session_id': item['session_id'],
                    'score': item['score'],
                    'chunk_preview': item['chunk_text'][:200] if item['chunk_text'] else '',
                    'search_source': 'local_semantic',
                })
                if len(results) >= limit:
                    break

        return results

    def index_session(
        self,
        session_id: str,
        content: str,
        summary: str = ""
    ) -> bool:
        """
        Index a session's content to local vectors.

        Returns False if model not ready.
        """
        if not self.is_model_ready():
            return False

        if not content:
            return False

        # Chunk content
        chunks = self._chunk_content(content)
        if not chunks:
            return False

        db = get_db_manager()

        try:
            # Delete existing vectors for this session
            db.execute_write(
                DB_LOCAL_VECTORS,
                "DELETE FROM session_vectors WHERE session_id = ?",
                (session_id,)
            )

            # Generate embeddings
            texts_to_embed = []
            for i, chunk in enumerate(chunks):
                # Prepend summary to first chunk
                if i == 0 and summary:
                    texts_to_embed.append(f"{summary}\n\n{chunk}")
                else:
                    texts_to_embed.append(chunk)

            embeddings = self.embed(texts_to_embed)

            # Insert vectors
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                db.execute_write(
                    DB_LOCAL_VECTORS,
                    """INSERT INTO session_vectors
                       (session_id, chunk_index, chunk_text, embedding)
                       VALUES (?, ?, ?, ?)""",
                    (session_id, i, chunk[:500], emb)
                )

            # Mark as indexed
            db.execute_write(
                DB_LOCAL_VECTORS,
                """INSERT OR REPLACE INTO indexed_sessions
                   (session_id, indexed_at, chunk_count)
                   VALUES (?, CURRENT_TIMESTAMP, ?)""",
                (session_id, len(chunks))
            )

            log(f"Indexed session {session_id[:8]} locally: {len(chunks)} chunks")
            return True

        except Exception as e:
            log(f"Failed to index session locally: {e}")
            return False

    def _chunk_content(self, content: str) -> List[str]:
        """Split content into chunks for embedding."""
        if len(content) <= CHUNK_SIZE:
            return [content]

        effective_chunk = CHUNK_SIZE - CHUNK_OVERLAP
        total_chunks_needed = (len(content) - CHUNK_OVERLAP) // effective_chunk + 1

        if total_chunks_needed <= MAX_CHUNKS:
            # Sequential chunking
            chunks = []
            start = 0
            while start < len(content) and len(chunks) < MAX_CHUNKS:
                end = min(start + CHUNK_SIZE, len(content))
                chunk = content[start:end]

                # Try to break at newline
                if end < len(content):
                    break_zone = chunk[-200:] if len(chunk) > 200 else chunk
                    newline_pos = break_zone.rfind('\n')
                    if newline_pos > 0:
                        adjust = len(chunk) - len(break_zone) + newline_pos + 1
                        chunk = content[start:start + adjust]

                chunks.append(chunk)
                start += effective_chunk

            return chunks

        # Large doc - sample evenly
        chunks = [content[:CHUNK_SIZE]]  # First

        middle_chunks = MAX_CHUNKS - 2
        if middle_chunks > 0:
            step = (len(content) - CHUNK_SIZE) / (middle_chunks + 1)
            for i in range(1, middle_chunks + 1):
                start = int(step * i)
                chunks.append(content[start:start + CHUNK_SIZE])

        chunks.append(content[-CHUNK_SIZE:])  # Last
        return chunks


# Global instance
_local_semantic: Optional[LocalSemanticSearch] = None


def get_local_semantic() -> LocalSemanticSearch:
    """Get the global LocalSemanticSearch instance."""
    global _local_semantic
    if _local_semantic is None:
        _local_semantic = LocalSemanticSearch()
    return _local_semantic


def is_local_semantic_available() -> bool:
    """
    Check if local semantic search is available.

    Returns True only if:
    - sqlite-vec extension can be loaded
    - fastembed model is cached (not downloading)
    """
    ls = get_local_semantic()
    return ls.sqlite_vec_available and ls.is_model_ready()


def trigger_local_semantic_download() -> Dict[str, str]:
    """
    Trigger background download of fastembed model.

    Called when remote is unavailable and user searches.
    Returns a message to include in search response.
    """
    ls = get_local_semantic()

    if not ls.sqlite_vec_available:
        return {
            "notice": "Local semantic search unavailable (sqlite-vec extension not supported)"
        }

    if ls.is_model_ready():
        return {}  # Already ready

    if ls.is_download_in_progress():
        return {
            "notice": "Local semantic search model is downloading (~100MB). Next search will use semantic matching."
        }

    # Start download
    ls.start_background_download()
    return {
        "notice": "Enabling local semantic search (downloading ~100MB model). This search uses keyword matching; next search will use semantic matching."
    }


def queue_session_for_indexing(session_id: str, content: str, summary: str = "") -> bool:
    """
    Queue a session for local vector indexing.

    Called after session is ingested to local_store.db.
    The background indexer will process the queue.
    """
    if not content:
        return False

    try:
        db = get_db_manager()

        # Check if already indexed
        row = db.execute_read_one(
            DB_LOCAL_VECTORS,
            "SELECT session_id FROM indexed_sessions WHERE session_id = ?",
            (session_id,)
        )
        if row:
            return True  # Already indexed

        # Add to queue (or update if already queued)
        db.execute_write(
            DB_LOCAL_VECTORS,
            """INSERT OR REPLACE INTO indexing_queue
               (session_id, content, summary, queued_at, attempts)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP, 0)""",
            (session_id, content, summary or "")
        )
        return True

    except Exception as e:
        log(f"Failed to queue session for local indexing: {e}")
        return False


def get_pending_indexing_count() -> int:
    """Get count of sessions pending local vector indexing."""
    try:
        db = get_db_manager()
        row = db.execute_read_one(
            DB_LOCAL_VECTORS,
            "SELECT COUNT(*) as count FROM indexing_queue",
            ()
        )
        return row.get('count', 0) if row else 0
    except Exception:
        return 0


class LocalSemanticIndexer:
    """
    Background worker that processes the local vector indexing queue.

    Similar to SyncWorker, runs as a daemon thread.

    Proactive Mode (LOCAL_SEMANTIC_PROACTIVE=True):
    - Downloads embedding model during startup (after delay)
    - Indexes all sessions to local vectors regardless of remote availability
    - Ensures offline semantic search is always ready

    Lazy Mode (LOCAL_SEMANTIC_PROACTIVE=False):
    - Only downloads model when remote is unavailable AND user searches
    - Original behavior for bandwidth-constrained environments
    """

    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0
        self._shutdown_event = threading.Event()

    def start(self):
        """Start the indexer thread."""
        if self.running:
            return

        if not LOCAL_SEMANTIC_ENABLED:
            log("Local semantic indexing disabled via config")
            return

        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        log("Local semantic indexer started")

    def stop(self):
        """Stop the indexer thread gracefully."""
        self.running = False
        self._shutdown_event.set()  # Signal worker to stop waiting
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        log("Local semantic indexer stopped")

    def _worker_loop(self):
        """Main worker loop - periodically processes indexing queue."""
        ls = get_local_semantic()

        # Check if sqlite-vec is available at all
        if not ls.sqlite_vec_available:
            log("Local semantic indexer: sqlite-vec not available, disabling")
            return

        # Proactive mode: wait startup delay then download model
        # Lazy mode: only download when triggered by failed remote search
        if LOCAL_SEMANTIC_PROACTIVE:
            log(f"Local semantic indexer: proactive mode, waiting {LOCAL_SEMANTIC_STARTUP_DELAY}s before model download")
            # Use shutdown event for interruptible sleep
            if self._shutdown_event.wait(timeout=LOCAL_SEMANTIC_STARTUP_DELAY):
                return  # Shutdown requested during startup delay

            if not ls.is_model_ready() and not ls.is_download_in_progress():
                log("Local semantic indexer: starting proactive model download (~100MB)")
                ls.start_background_download()
        else:
            # Lazy mode - just wait a bit for potential download triggered elsewhere
            if self._shutdown_event.wait(timeout=10):
                return

        while self.running and not self._shutdown_event.is_set():
            try:
                # Check if model is ready
                if not ls.is_model_ready():
                    if ls.is_download_in_progress():
                        # Wait for download to complete
                        self._shutdown_event.wait(timeout=LOCAL_SEMANTIC_INDEX_INTERVAL)
                        continue

                    if LOCAL_SEMANTIC_PROACTIVE:
                        # Proactive: trigger download if not started
                        ls.start_background_download()

                    self._shutdown_event.wait(timeout=LOCAL_SEMANTIC_INDEX_INTERVAL)
                    continue

                # Model ready - process queue
                processed = self._process_batch(LOCAL_SEMANTIC_BATCH_SIZE)
                if processed > 0:
                    self._consecutive_failures = 0
                    log(f"Local indexer: indexed {processed} sessions")

            except Exception as e:
                self._consecutive_failures += 1
                log(f"Local indexer error: {e}")
                # Back off on repeated failures
                if self._consecutive_failures > 5:
                    self._shutdown_event.wait(timeout=LOCAL_SEMANTIC_INDEX_INTERVAL * 5)

            self._shutdown_event.wait(timeout=LOCAL_SEMANTIC_INDEX_INTERVAL)

    def _process_batch(self, batch_size: int) -> int:
        """Process a batch of sessions from the queue."""
        db = get_db_manager()
        ls = get_local_semantic()

        # Get pending sessions (oldest first, max attempts < 3)
        rows = db.execute_read(
            DB_LOCAL_VECTORS,
            """SELECT session_id, content, summary, attempts
               FROM indexing_queue
               WHERE attempts < 3
               ORDER BY queued_at ASC
               LIMIT ?""",
            (batch_size,)
        )

        if not rows:
            return 0

        processed = 0
        for row in rows:
            session_id = row['session_id']
            content = row['content']
            summary = row.get('summary', '')

            try:
                success = ls.index_session(session_id, content, summary)

                if success:
                    # Remove from queue
                    db.execute_write(
                        DB_LOCAL_VECTORS,
                        "DELETE FROM indexing_queue WHERE session_id = ?",
                        (session_id,)
                    )
                    processed += 1
                else:
                    # Increment attempts
                    db.execute_write(
                        DB_LOCAL_VECTORS,
                        """UPDATE indexing_queue
                           SET attempts = attempts + 1,
                               last_error = 'Index returned False'
                           WHERE session_id = ?""",
                        (session_id,)
                    )

            except Exception as e:
                # Record error and increment attempts
                db.execute_write(
                    DB_LOCAL_VECTORS,
                    """UPDATE indexing_queue
                       SET attempts = attempts + 1,
                           last_error = ?
                       WHERE session_id = ?""",
                    (str(e)[:500], session_id)
                )
                log(f"Failed to index session {session_id[:8]}: {e}")

        return processed


# Global indexer instance
_indexer: Optional[LocalSemanticIndexer] = None


def start_local_indexer() -> Optional[LocalSemanticIndexer]:
    """Start the background local semantic indexer."""
    global _indexer
    if _indexer is None:
        _indexer = LocalSemanticIndexer()
    _indexer.start()
    return _indexer


def stop_local_indexer():
    """Stop the background local semantic indexer."""
    global _indexer
    if _indexer:
        _indexer.stop()
        _indexer = None
