# ULTRATHINK: Local Semantic Search Implementation Plan

**Goal:** Enable semantic search locally using fastembed + sqlite-vec when remote storage isn't available.

**Constraint:** ~100MB additional storage is acceptable for the embedding model.

---

## Architecture Overview

### Three-Tier Search System

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   Search Request                     │
                    └───────────────────────┬─────────────────────────────┘
                                            │
                    ┌───────────────────────▼─────────────────────────────┐
                    │                 Tier 1: Remote Semantic              │
                    │         (Qdrant + embedding-service)                 │
                    │         - Best quality, cross-machine                │
                    │         - Requires VPN/network                       │
                    └───────────────────────┬─────────────────────────────┘
                                            │ (unavailable)
                    ┌───────────────────────▼─────────────────────────────┐
                    │                 Tier 2: Local Semantic               │
                    │         (sqlite-vec + fastembed)                     │
                    │         - ~100MB model download on first use         │
                    │         - Same-quality semantic search               │
                    │         - Single-machine only                        │
                    └───────────────────────┬─────────────────────────────┘
                                            │ (model not ready)
                    ┌───────────────────────▼─────────────────────────────┐
                    │                 Tier 3: FTS5 Keyword                 │
                    │         (SQLite FTS5 - always available)             │
                    │         - Keyword matching only                      │
                    │         - Instant, no model needed                   │
                    └─────────────────────────────────────────────────────┘
```

### Decision Matrix

| Scenario | Search Tier | Latency | Quality |
|----------|-------------|---------|---------|
| Remote storage connected | Tier 1 (Remote) | ~200ms | Best (cross-machine) |
| Offline, model cached | Tier 2 (Local) | ~50ms | Good (semantic) |
| Offline, first run | Tier 3 (FTS5) | ~5ms | Fair (keyword) |
| Model downloading | Tier 3 (FTS5) | ~5ms | Fair (keyword) |

### Lazy Loading Trigger (Key Design)

The ~100MB fastembed model is **only downloaded when ALL of these conditions are met:**

1. ✅ Remote storage is **unavailable** (network down, not configured, etc.)
2. ✅ User performs a **search** (not just session start or status check)
3. ✅ Model is **not already cached**

**What happens on first offline search:**
```
User: mira_search("authentication bug")
       │
       ├── Check remote → UNAVAILABLE
       ├── Check local semantic → MODEL NOT CACHED
       ├── Start background download (non-blocking)
       ├── Return FTS5 results immediately
       └── Include notice: "Enabling local semantic search..."

Next search: Uses local semantic (model now cached)
```

**This means:**
- Remote users NEVER download the model (remote handles semantic search)
- Local-only users only download when they actually search
- First offline search is fast (FTS5), not blocked by download

---

## ULTRATHINK TODO Tasks

### Phase 1: Foundation (Day 1-2)

#### ULTRATHINK-001: Add fastembed to bootstrap dependencies
**File:** `python/mira/bootstrap.py`
**Changes:**
1. Add `fastembed` to REQUIREMENTS list
2. Create separate "heavy" requirements tier (fastembed ~100MB vs watchdog ~1MB)
3. Implement lazy download - only pull fastembed when semantic search first requested
4. Add progress logging during model download

```python
# bootstrap.py additions
REQUIREMENTS_CORE = ["watchdog", "psycopg2-binary", "httpx"]
REQUIREMENTS_SEMANTIC = ["fastembed", "sqlite-vec"]  # ~100MB, lazy-loaded

def ensure_semantic_deps():
    """Download semantic search dependencies on first use."""
    # Check if already installed
    # If not, pip install fastembed sqlite-vec
    # Log progress
```

**Considerations:**
- First-run model download from HuggingFace could take 30-60 seconds
- Need to handle network failures gracefully
- Cache model in `.mira/.venv/` alongside other deps

#### ULTRATHINK-002: Add sqlite-vec extension loading
**File:** `python/mira/db_manager.py` (new functionality)
**Changes:**
1. Detect if Python has loadable extension support
2. Load sqlite-vec extension when available
3. Fall back gracefully if extension loading fails

```python
def load_sqlite_vec():
    """Load sqlite-vec extension if available."""
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except Exception as e:
        log(f"sqlite-vec not available: {e}")
        return False
```

**Considerations:**
- pyenv Python often lacks `--enable-loadable-sqlite-extensions`
- System Python usually works fine
- Document this limitation in README

---

### Phase 2: Vector Storage Schema (Day 2-3)

#### ULTRATHINK-003: Create local_vectors.db schema
**File:** `python/mira/local_vectors.py` (new file)
**Schema:**
```sql
-- Session vectors (chunked like remote)
CREATE TABLE IF NOT EXISTS session_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    chunk_text TEXT,  -- For debugging/retrieval
    embedding BLOB NOT NULL,  -- 384-dim float32 via struct.pack
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, chunk_index)
);

CREATE INDEX idx_session_vectors_session ON session_vectors(session_id);

-- Virtual table for vector search (sqlite-vec)
CREATE VIRTUAL TABLE IF NOT EXISTS session_vec_index USING vec0(
    embedding float[384]
);
```

**Considerations:**
- Match remote embedding dimensions (384)
- Store chunk_text for result enrichment
- Keep vectors in separate DB to avoid bloating local_store.db

#### ULTRATHINK-004: Design vector ID mapping
**Problem:** sqlite-vec uses rowid, but we need session_id + chunk_index
**Solution:** Maintain ID mapping table or use deterministic ID generation

```python
def get_vector_id(session_id: str, chunk_index: int) -> int:
    """Generate stable integer ID from session_id and chunk_index."""
    # Same algorithm as remote service for consistency
    return hash(f"{session_id}:{chunk_index}") & 0x7FFFFFFFFFFFFFFF
```

---

### Phase 3: Embedding Pipeline (Day 3-4)

#### ULTRATHINK-005: Create local embedding client
**File:** `python/mira/local_embeddings.py` (new file)
**Responsibilities:**
1. Lazy-load fastembed model
2. Generate embeddings for text chunks
3. Match chunking strategy with remote (CHUNK_SIZE=4000, OVERLAP=500)

```python
from typing import List, Optional
import struct

class LocalEmbedder:
    """Local embedding generation using fastembed."""

    _model = None  # Lazy singleton

    @classmethod
    def get_model(cls):
        if cls._model is None:
            from fastembed import TextEmbedding
            # Use same model as remote OR compatible model
            cls._model = TextEmbedding("BAAI/bge-small-en-v1.5")
        return cls._model

    @classmethod
    def embed(cls, texts: List[str]) -> List[bytes]:
        """Embed texts, return as BLOBs for sqlite-vec."""
        model = cls.get_model()
        embeddings = list(model.embed(texts))
        return [struct.pack(f'{len(e)}f', *e) for e in embeddings]

    @classmethod
    def embed_query(cls, query: str) -> bytes:
        """Embed a single query."""
        return cls.embed([query])[0]
```

**Considerations:**
- fastembed uses ONNX, no PyTorch needed
- First call triggers ~100MB model download
- Subsequent calls use cached model (~50ms per embedding)

#### ULTRATHINK-006: Implement local indexing pipeline
**File:** `python/mira/local_vectors.py`
**Trigger:** After session ingested to local_store.db

```python
def index_session_locally(session_id: str, content: str, summary: str):
    """Index session content to local vector store."""
    # 1. Chunk content (match remote strategy)
    chunks = chunk_content(content)

    # 2. Generate embeddings
    embedder = LocalEmbedder()

    # 3. Delete existing vectors for session
    delete_session_vectors(session_id)

    # 4. Insert new vectors
    for i, chunk in enumerate(chunks):
        embed_text = f"{summary}\n\n{chunk}" if i == 0 else chunk
        embedding = embedder.embed([embed_text])[0]

        insert_vector(
            session_id=session_id,
            chunk_index=i,
            chunk_text=chunk[:500],  # Preview for results
            embedding=embedding
        )
```

---

### Phase 4: Search Integration (Day 4-5)

#### ULTRATHINK-007: Implement local semantic search
**File:** `python/mira/search.py` (modify existing)
**Changes:**
1. Add `search_local_vectors()` function
2. Integrate into search fallback chain

```python
def search_local_vectors(
    query: str,
    project_path: Optional[str] = None,
    limit: int = 10
) -> List[Dict]:
    """Semantic search using local sqlite-vec."""
    from .local_embeddings import LocalEmbedder
    from .local_vectors import query_vectors

    # Embed query
    query_embedding = LocalEmbedder.embed_query(query)

    # Search vectors
    results = query_vectors(
        embedding=query_embedding,
        limit=limit * 2,  # Get extra, filter later
    )

    # Filter by project if specified
    if project_path:
        results = [r for r in results if r['project_path'] == project_path]

    # Deduplicate by session_id (multiple chunks per session)
    seen = set()
    unique_results = []
    for r in results:
        if r['session_id'] not in seen:
            seen.add(r['session_id'])
            unique_results.append(r)
            if len(unique_results) >= limit:
                break

    return unique_results
```

#### ULTRATHINK-008: Implement search tier fallback
**File:** `python/mira/search.py`
**Logic:**

```python
def search(query: str, **kwargs) -> Dict:
    """Unified search with automatic tier selection."""

    # Tier 1: Try remote semantic search
    if is_central_available():
        try:
            return search_remote_semantic(query, **kwargs)
        except Exception as e:
            log(f"Remote search failed: {e}, falling back to local")

    # Tier 2: Try local semantic search
    if is_local_vectors_available():
        try:
            return search_local_vectors(query, **kwargs)
        except Exception as e:
            log(f"Local semantic search failed: {e}, falling back to FTS5")

    # Tier 3: FTS5 keyword search (always available)
    return search_fts5(query, **kwargs)

def is_local_vectors_available() -> bool:
    """Check if local vector search is ready."""
    # 1. sqlite-vec loaded?
    # 2. fastembed model cached?
    # 3. vectors indexed?
    pass
```

---

### Phase 5: Background Indexing (Day 5-6)

#### ULTRATHINK-009: Add local vector indexing to ingestion pipeline
**File:** `python/mira/ingestion.py`
**Changes:**
1. After storing session metadata, queue for local vector indexing
2. Index in background to avoid blocking ingestion

```python
def ingest_session(session_file: Path):
    # ... existing ingestion logic ...

    # Queue for local vector indexing (async)
    if local_vectors_enabled():
        queue_local_indexing(session_id, content, summary)
```

#### ULTRATHINK-010: Create local indexing worker
**File:** `python/mira/local_vectors.py`
**Behavior:**
1. Process queue of sessions to index
2. Batch embeddings for efficiency (fastembed supports batching)
3. Skip already-indexed sessions
4. Run in background thread

```python
def local_indexer_worker():
    """Background worker for local vector indexing."""
    while running:
        # Get unindexed sessions
        sessions = get_unindexed_sessions(limit=10)

        if not sessions:
            time.sleep(30)
            continue

        for session in sessions:
            try:
                index_session_locally(
                    session['session_id'],
                    session['content'],
                    session['summary']
                )
                mark_locally_indexed(session['session_id'])
            except Exception as e:
                log(f"Local indexing failed: {e}")
```

---

### Phase 6: Testing (Day 6-7)

#### ULTRATHINK-011: Create unit tests for local embeddings
**File:** `test/test_local_embeddings.py`
**Tests:**
1. Model loading (lazy initialization)
2. Embedding generation (correct dimensions)
3. BLOB encoding/decoding
4. Query embedding matches batch embedding

#### ULTRATHINK-012: Create unit tests for local vector search
**File:** `test/test_local_vectors.py`
**Tests:**
1. Vector insertion
2. Similarity search (known similar texts score high)
3. Session deduplication
4. Project filtering

#### ULTRATHINK-013: Create integration tests for search tiers
**File:** `test/test_search_tiers.py`
**Tests:**
1. Remote available → uses remote
2. Remote unavailable, local ready → uses local semantic
3. Both unavailable → uses FTS5
4. Fallback chain works correctly

---

### Phase 7: Documentation & Polish (Day 7)

#### ULTRATHINK-014: Update README with local semantic search
**Changes:**
1. Document three-tier search system
2. Explain ~100MB model download on first use
3. Note Python extension loading requirement for sqlite-vec

#### ULTRATHINK-015: Add mira_status indicators
**Changes:**
1. Show local vector index status
2. Show embedding model status (downloaded/not downloaded)
3. Show current search tier being used

---

## Technical Decisions

### Model Choice: BAAI/bge-small-en-v1.5
- **Dimensions:** 384 (matches our remote all-MiniLM-L6-v2)
- **Size:** ~120MB download, ~90MB on disk
- **Speed:** ~50ms per embedding on CPU
- **Quality:** Better than OpenAI ada-002 on benchmarks

### Why fastembed over sentence-transformers?
1. **Lighter:** No PyTorch dependency (~2GB → ~100MB)
2. **Faster cold start:** ONNX loads faster than PyTorch
3. **Same quality:** Both use transformer models
4. **Qdrant blessed:** Made by Qdrant team, well-maintained

### Why sqlite-vec over FAISS?
1. **Simpler:** Pure C, no complex dependencies
2. **Integrated:** Works with existing SQLite infrastructure
3. **Portable:** Runs anywhere SQLite runs
4. **Sufficient:** For local search (<100K vectors), performance is fine

### Chunking Strategy (match remote)
```python
CHUNK_SIZE = 4000  # Characters per chunk
CHUNK_OVERLAP = 500  # Overlap for context continuity
MAX_CHUNKS = 50  # Limit per session
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| First-run model download slow | User waits 30-60s | Show progress, fall back to FTS5 |
| sqlite-vec extension loading fails | No local semantic | Graceful FTS5 fallback, document fix |
| Model size too large | Disk space concerns | Lazy download, can skip if not needed |
| Embedding quality differs from remote | Inconsistent results | Use same-dimension model, test parity |

---

## Success Criteria

1. **Semantic search works offline** - User can search by meaning without network
2. **Transparent fallback** - User doesn't notice tier switches
3. **Reasonable first-run experience** - Clear progress, not blocking
4. **No breaking changes** - Existing remote users unaffected
5. **Same result quality** - Local semantic ≈ remote semantic (within 10%)

---

## Estimated Effort

| Phase | Days | Notes |
|-------|------|-------|
| Phase 1: Foundation | 2 | Bootstrap, extension loading |
| Phase 2: Schema | 1 | Vector storage design |
| Phase 3: Pipeline | 2 | Embedding generation |
| Phase 4: Search | 2 | Integration, fallbacks |
| Phase 5: Background | 1 | Async indexing |
| Phase 6: Testing | 1 | Unit + integration tests |
| Phase 7: Polish | 1 | Docs, status updates |
| **Total** | **10 days** | |

---

## Sources

- [FastEmbed GitHub](https://github.com/qdrant/fastembed) - Qdrant's ONNX-based embedding library
- [FastEmbed Documentation](https://qdrant.github.io/fastembed/) - Getting started guide
- [sqlite-vec GitHub](https://github.com/asg017/sqlite-vec) - SQLite vector search extension
- [Hybrid Search with SQLite](https://simonwillison.net/2024/Oct/4/hybrid-full-text-search-and-vector-search-with-sqlite/) - Simon Willison's exploration
