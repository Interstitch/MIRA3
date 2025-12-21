# MIRA Architecture

## Overview

MIRA (Memory Information Retriever and Archiver) is a pure Python MCP server that provides persistent memory for Claude Code sessions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code                               │
│                                                                  │
│  User ←→ Claude ←→ MCP Protocol ←→ MIRA Server                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MIRA MCP Server                             │
│                     (src/mira/server.py)                         │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ mira_init   │  │ mira_search │  │ mira_recent │  ...         │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Tools Layer                            │   │
│  │                  (src/mira/tools/)                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐       │
│  │  Storage   │       │   Search   │       │ Custodian  │       │
│  │            │       │            │       │            │       │
│  │ SQLite FTS │       │ Semantic   │       │ Learning   │       │
│  │ Postgres   │       │ Fuzzy      │       │ Profile    │       │
│  │ Qdrant     │       │ FTS5       │       │ Rules      │       │
│  └────────────┘       └────────────┘       └────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Background Workers                           │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ File Watcher│  │ Sync Worker │  │Local Indexer│              │
│  │ (watchdog)  │  │ (central)   │  │ (fastembed) │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## Package Structure

```
src/mira/
├── __init__.py          # Package root, exports VERSION
├── __main__.py          # CLI entry point
├── server.py            # MCP server using Python MCP SDK
│
├── core/                # Fundamental infrastructure
│   ├── constants.py     # VERSION, paths, DB names
│   ├── config.py        # Configuration classes
│   ├── database.py      # Thread-safe DatabaseManager
│   ├── bootstrap.py     # Venv and dependency management
│   ├── utils.py         # Logging, path utilities
│   └── parsing.py       # Conversation parsing
│
├── tools/               # MCP tool handlers (one file per tool)
│   ├── init.py          # mira_init - session context
│   ├── search.py        # mira_search - semantic search
│   ├── recent.py        # mira_recent - recent sessions
│   ├── errors.py        # mira_error_lookup - error patterns
│   ├── decisions.py     # mira_decisions - decision journal
│   ├── code_history.py  # mira_code_history - file changes
│   └── status.py        # mira_status - system health
│
├── storage/             # Storage backends
│   ├── storage.py       # Storage facade (local/central)
│   ├── local_store.py   # SQLite FTS5 backend
│   ├── migrations.py    # Schema version management
│   └── sync/            # Central storage sync
│
├── extraction/          # Content extraction from conversations
│   ├── metadata.py      # Summary, keywords, facts
│   ├── artifacts.py     # Code blocks, lists, tables
│   ├── insights.py      # Coordinator for errors/decisions
│   ├── errors.py        # Error pattern detection
│   ├── decisions.py     # Decision extraction
│   ├── concepts.py      # Codebase concept learning
│   └── code_history.py  # File change tracking
│
├── search/              # Multi-tier search
│   ├── core.py          # Search orchestration
│   ├── fuzzy.py         # Typo correction (Damerau-Levenshtein)
│   └── local_semantic.py # sqlite-vec + fastembed
│
├── custodian/           # User learning system
│   ├── learning.py      # Pattern extraction
│   ├── profile.py       # Profile building
│   ├── rules.py         # Always/never patterns
│   └── prerequisites.py # Environment prerequisites
│
└── ingestion/           # Conversation processing pipeline
    ├── core.py          # Main ingestion logic
    ├── batch.py         # Parallel batch processing
    └── watcher.py       # File system monitoring
```

## Data Flow

### Session Initialization (mira_init)

```
1. Claude Code starts session
2. SessionStart hook invokes: python -m mira --init
3. MIRA loads:
   - User profile (custodian)
   - Recent work context
   - Applicable prerequisites
   - Alerts and danger zones
4. Returns formatted context to Claude
```

### Conversation Ingestion

```
1. File watcher detects new/modified .jsonl
2. Debounce period (5 seconds)
3. Parse conversation messages
4. Extract metadata (summary, keywords, facts)
5. Extract artifacts (code blocks, lists)
6. Extract insights (errors, decisions)
7. Learn custodian patterns
8. Store in local SQLite
9. Optionally sync to central storage
```

### Search Query

```
1. Claude calls mira_search("error handling")
2. Three-tier search:
   a. Remote semantic (Qdrant) - if available
   b. Local semantic (sqlite-vec) - if available
   c. FTS5 keyword (always available)
3. Merge and deduplicate results
4. Apply time decay weighting
5. Return ranked results
```

## Storage Tiers

### Local (Default)
- SQLite with FTS5 for full-text search
- Optional: sqlite-vec + fastembed for local semantic search
- Works offline, single machine

### Central (Optional)
- Postgres for metadata storage
- Qdrant for vector search
- Embedding service for vector generation
- Enables cross-machine search

## Key Design Decisions

1. **Pure Python**: Removed TypeScript/Node.js layer for simplicity
2. **Thread-safe DB**: Write queue prevents SQLite locking issues
3. **Lazy initialization**: Components initialize on first use
4. **Graceful degradation**: Falls back to simpler search tiers
5. **Bootstrap pattern**: Auto-creates venv and installs deps
