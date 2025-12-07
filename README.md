
# Memory Information Retriever and Archiver - MIRA3

An MCP server that monitors, archives, and provides semantic search over Claude Code conversation history with vector embeddings and intelligent metadata extraction.

## Overview

MIRA3 watches your Claude Code chat directory, ingests conversations into a local vector database, and exposes MCP tools for intelligent retrieval. Unlike simple history viewers, MIRA3 extracts structured metadata and enables semantic search across your entire conversation history.

**Source Directory:** `~/.claude/projects/<project-path>/`
**Storage Directory:** `<workspace>/.mira/`

## Installation

```bash
npm install -g claude-mira3
```

The installation process automatically handles all prerequisites and dependencies including ChromaDB.

### Add to Claude Code

```bash
claude mcp add mira3 -- npx claude-mira3
```

Or manually add to your MCP configuration:

```json
{
  "mcpServers": {
    "mira3": {
      "command": "npx",
      "args": ["claude-mira3"],
      "env": {}
    }
  }
}
```

## Data Locations

### Source (Read-Only)

MIRA3 reads from Claude Code's native conversation storage:

```
~/.claude/
├── projects/
│   └── <project-path>/          # Path-encoded project directories
│       └── *.jsonl              # Conversation files (one per session)
└── history.jsonl                # Index of all conversations
```

Conversation files are JSONL format with one JSON object per line, containing messages, tool calls, and metadata.

### MIRA Storage (Read-Write)

MIRA3 maintains its own storage in the workspace:

```
<workspace>/.mira/
├── config.json                  # MIRA configuration and custodian info
├── state.json                   # Ingestion state (file hashes, timestamps)
├── .venv/                       # Python virtualenv (auto-created on first run)
│   └── ...                      # chromadb, sentence-transformers, watchdog
├── archives/
│   └── <session-id>.jsonl       # Full conversation copies
├── metadata/
│   └── <session-id>.json        # Extracted metadata per conversation
└── chroma/                      # ChromaDB persistent storage
    └── ...                      # Vector embeddings
```

### Storage Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Python venv** | Isolated environment with dependencies | `.mira/.venv/` |
| **Embedding Model** | all-MiniLM-L6-v2 for semantic embeddings | `.mira/models/` |
| **ChromaDB** | Vector embeddings for semantic search | `.mira/chroma/` |
| **Custodian DB** | Learned user preferences and patterns | `.mira/custodian.db` |
| **Insights DB** | Error patterns and architectural decisions | `.mira/insights.db` |
| **Archives** | Full conversation copies | `.mira/archives/` |
| **Metadata** | Pre-extracted summaries, keywords, facts | `.mira/metadata/` |

## Embedding Model

### What Are Embeddings?

An **embedding** is a list of numbers (a vector) that represents the meaning of text. Similar texts produce similar vectors, enabling semantic search - finding conversations by meaning rather than exact keywords.

```
"How do I fix a bug?"  → [0.12, -0.34, 0.56, ..., 0.78]  (384 numbers)
"Debug this error"     → [0.11, -0.32, 0.54, ..., 0.76]  (similar vectors!)
"I like pizza"         → [-0.45, 0.89, -0.12, ..., 0.23] (very different)
```

This means searching for "authentication" will find conversations about "login", "OAuth", or "user sessions" - even if they never use the word "authentication".

### Why all-MiniLM-L6-v2?

MIRA3 uses **all-MiniLM-L6-v2** from sentence-transformers:

| Property | Value |
|----------|-------|
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Dimensions** | 384 |
| **Size** | ~80MB |
| **Location** | `.mira/models/` |

This model was chosen because:
- **Small**: ~80MB vs 500MB+ for larger models
- **Fast**: Runs locally on CPU, no GPU required
- **Purpose-built**: Designed specifically for semantic similarity, not text generation
- **Offline**: No API calls after initial download

**Note:** This is an encoder-only model - it converts text to vectors but cannot generate text or be used as a chatbot. It's like a microphone that captures meaning, not a speaker that produces speech.

### How It Works

- ChromaDB uses all-MiniLM-L6-v2 for embeddings (384 dimensions)
- The model is automatically downloaded on first run (~80MB download)
- After initial download, MIRA works fully offline
- Embeddings are computed locally - no API calls or cloud dependencies

### Important: Token Limits

**all-MiniLM-L6-v2 has a 256 token limit (~1000 characters).** Text beyond this is truncated and lost - ChromaDB does NOT automatically chunk text.

MIRA3 handles this by creating dense, high-signal documents that fit within the limit:
1. Summary and task description (highest priority)
2. Keywords (direct search targets)
3. Key facts and TODO topics
4. Condensed message samples (if space remains)

This means searches match on the *essence* of a conversation, not raw message text.

### First Run

On first invocation, MIRA will:
1. Create a Python virtualenv in `.mira/.venv/`
2. Install CPU-only PyTorch (~600MB) - MIRA explicitly avoids the GPU version which adds 6GB of NVIDIA CUDA libraries
3. Install dependencies: chromadb, sentence-transformers, watchdog
4. Download the all-MiniLM-L6-v2 model (~80MB)
5. Begin ingesting conversations automatically

This may take 2-5 minutes depending on network speed. Subsequent runs start instantly.

**Total venv size:** ~1.5GB (vs ~7.5GB with GPU libraries). The major components are:
- PyTorch CPU (~600MB) - Neural network runtime
- sentence-transformers (~115MB) - Embedding model framework
- ChromaDB (~100MB) - Vector database
- Supporting libraries (~600MB) - numpy, scipy, transformers

## Architecture

### Process Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code                              │
│                        │                                    │
│                        ▼ spawns                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Node.js MCP Server                      │   │
│  │              (npx claude-mira3)                      │   │
│  │                        │                             │   │
│  │                        ▼ spawns                      │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │           Python Backend Daemon              │    │   │
│  │  │                                              │    │   │
│  │  │  First run:                                  │    │   │
│  │  │  1. Creates .mira/.venv                      │    │   │
│  │  │  2. pip install chromadb sentence-transformers│    │   │
│  │  │     watchdog                                 │    │   │
│  │  │  3. Downloads all-MiniLM-L6-v2 model        │    │   │
│  │  │  4. Continues with normal operation          │    │   │
│  │  │                                              │    │   │
│  │  │  Normal operation:                           │    │   │
│  │  │  - File watcher (polls ~/.claude/)           │    │   │
│  │  │  - Ingestion pipeline                        │    │   │
│  │  │  - ChromaDB semantic search                  │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │                        ▲                             │   │
│  │                        │ JSON-RPC over stdio         │   │
│  │                        │                             │   │
│  │         MCP tool calls route to Python backend       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### First Run (Automatic)

When Claude Code first invokes any MIRA3 tool:
1. Node.js MCP server starts
2. Spawns Python backend script (bundled with npm package)
3. Python backend detects missing `.mira/.venv`
4. Creates virtualenv, installs `chromadb`, `sentence-transformers`, `watchdog` via pip
5. Stores completion flag in `.mira/config.json`
6. Proceeds with tool invocation

Subsequent runs skip installation (venv already exists).

### Component Responsibilities

| Layer | Language | Responsibility |
|-------|----------|----------------|
| **MCP Server** | Node.js | MCP protocol, stdio transport, spawns backend |
| **Python Backend** | Python | Self-install deps, ChromaDB, file watching, metadata extraction |

### Search Strategy

MIRA3 uses ChromaDB with all-MiniLM-L6-v2 embeddings (384 dimensions) for semantic search.

**ChromaDB Configuration:**
- `hnsw:space: cosine` - Cosine similarity for text (0 = identical, 2 = opposite)
- `hnsw:M: 16` - Neighbors per node (good for 384-dim vectors)
- `hnsw:search_ef: 50` - Search thoroughness (higher = better recall)

**Key features:**
- **Query preprocessing** - Queries are cleaned and truncated to ~200 chars for optimal embedding
- **Project filtering** - Optionally restrict searches to a specific project
- **Intelligent message sampling** - Long conversations are intelligently sampled to capture topic shifts
- **Rich metadata** - Results include slug, git branch, task description, and more

## Conversation Indexing

MIRA3 parses Claude Code JSONL conversation files and extracts structured metadata for each session. Only conversations with actual user/assistant messages are indexed (file-history snapshots and agent sub-conversations are skipped).

### Long Conversation Handling

For users who maintain long-running sessions (using `claude --continue` over days/weeks), MIRA3 uses intelligent message sampling instead of truncating at a fixed limit:

1. **Always include first 5 messages** - Establishes initial context
2. **Always include last 10 messages** - Captures recent state
3. **Detect topic shifts** via multiple signals:
   - **Time gaps** - 2+ hour gaps between messages indicate session breaks
   - **Content patterns** - Phrases like "let's switch to", "new task", "good morning"
   - **TODO lists** - New task lists indicate new work planning
4. **Sample from middle** - Evenly samples remaining messages, prioritizing user messages

This ensures that even a 1000+ message conversation captures all major topics and shifts, not just the beginning.

### What Gets Indexed

For each conversation, MIRA3 extracts and stores:

| Field | Description | Storage |
|-------|-------------|---------|
| **Session ID** | Unique conversation identifier (UUID) | Metadata + ChromaDB |
| **Slug** | Human-readable session name (e.g., "velvet-hugging-reef") | Metadata |
| **Project Path** | Workspace path (cwd from messages) | Metadata |
| **Git Branch** | Active git branch during session | Metadata |
| **Summary** | Human-readable conversation summary | Metadata + ChromaDB |
| **Keywords** | Weighted technical terms (up to 50) | Metadata |
| **Task Description** | Clean first user request | Metadata |
| **Key Facts** | Rules, decisions, constraints (up to 15) | Metadata |
| **TODO Topics** | Task descriptions from TODO lists | Metadata |
| **Message Count** | Total user + assistant messages | Metadata |
| **Models Used** | AI models used (e.g., claude-opus-4-5) | Metadata |
| **Tools Used** | Tool usage counts (Bash, Edit, Read, etc.) | Metadata |
| **Files Touched** | File paths read/edited/written | Metadata |
| **Embeddings** | 384-dim semantic vectors | ChromaDB |

### Summary Generation

Summaries are built using this priority:

1. **Claude Code Summary** - If the conversation file contains a `summary` entry from Claude Code's own summarization, use it directly
2. **Task + Outcome** - Combine the first user request with completion indicators from the final assistant messages
3. **First Message Fallback** - Truncated first user message if no other summary available

Outcome detection looks for phrases like: "complete", "finished", "done", "implemented", "fixed", "created", "successfully", "working", "ready", "deployed"

### Keyword Extraction

Keywords are extracted using weighted pattern matching:

| Pattern | Weight | Examples |
|---------|--------|----------|
| **Known tech terms** | 5 | chromadb, typescript, react, docker, pytorch |
| **File names** | 3 | mira_backend, spawner, server |
| **Package imports** | 3 | sentence-transformers, chromadb |
| **Technical identifiers** | 2 | camelCase, snake_case, PascalCase names |
| **Function/class names** | 2 | extract_keywords, MiraEmbeddingFunction |
| **Error types** | 2 | TypeError, ImportError |
| **Frequent words** | 1 | Words appearing 3+ times (4+ chars) |

A comprehensive stopword list filters out common English words. The top 50 keywords by weight are stored.

### Key Facts Extraction

Key facts capture important rules, decisions, and constraints from assistant messages. Extraction uses priority scoring to surface the most important facts first:

| Priority | Pattern | Examples |
|----------|---------|----------|
| **Critical (5)** | Warnings, errors, security issues | "This will break if...", "Security vulnerability..." |
| **High (4)** | Never/always rules | "Never commit secrets", "Always validate input" |
| **Medium (3-4)** | Requirements, constraints | "Requires Python 3.8+", "Depends on chromadb" |
| **Normal (2)** | Decisions, explicit tips | "I recommend using...", "Note: ..." |
| **Low (1)** | Should/could suggestions | "You should consider...", "Could be improved by..." |

Facts are filtered to 30-300 characters, must be 70%+ alphabetic (no code), and are deduplicated. Up to 15 facts per conversation, sorted by priority then length.

### Task Description Cleaning

The first user message is cleaned to create a task description:

1. **Remove greetings** - "Hi Claude", "Hello", "Good morning", etc.
2. **Skip greeting sentences** - Multi-sentence messages skip short greetings to find the task
3. **Remove polite prefixes** - Extensive list including "please", "can you", "could you", "i need", "i want", "help me", "i was wondering if you could", etc.
4. **Capitalize** first letter
5. **Truncate** to 200 characters at word boundary

Examples:
- "Hi Claude! Please help me add authentication to the app" → "Add authentication to the app"
- "Good morning. I was wondering if you could fix the login bug" → "Fix the login bug"

### What Gets Skipped

MIRA3 only indexes conversations with actual messages. These are skipped:

- **File history snapshots** - `{"type": "file-history-snapshot", ...}` entries
- **Agent sub-conversations** - Files named `agent-*.jsonl` (these are subagent task logs)
- **Empty conversations** - Files with no user/assistant messages
- **Already indexed** - Conversations where the source file hasn't been modified since last ingestion

## Custodian Learning

MIRA3 learns about you (the "custodian") from your conversation patterns and provides this context to future Claude sessions automatically.

### What MIRA Learns

| Category | Examples | How It Helps |
|----------|----------|--------------|
| **Identity** | "My name is Max" | Claude greets you by name |
| **Preferences** | "I prefer pnpm", "No emojis" | Claude respects your choices |
| **Rules** | "Never commit to main", "Always run tests" | Claude follows your rules |
| **Danger Zones** | Files that caused repeated issues | Claude warns before touching them |
| **Work Patterns** | Test-first, planning before coding | Claude adapts to your workflow |

### How It Works

1. **Learning**: During ingestion, MIRA extracts patterns from your messages
2. **Storage**: Learned data stored in `custodian.db` with frequency tracking
3. **Context**: `mira_init` returns your profile to Claude at session start
4. **Evolution**: Confidence increases as patterns repeat across sessions

### Example Profile Output

```json
{
  "name": "Max",
  "preferences": {
    "tools": ["pnpm", "vitest"],
    "communication": ["concise", "no emojis"]
  },
  "rules": {
    "never": ["commit directly to main"],
    "always": ["run tests before pushing"]
  },
  "danger_zones": [
    {"path": "legacy-api.js", "issue_count": 3}
  ],
  "summary": "Custodian: Max | Preferred tools: pnpm, vitest | Never: commit directly to main"
}
```

## Error Pattern Recognition

MIRA3 extracts error patterns from conversations and links them to their solutions. This builds a searchable knowledge base of past problems and fixes.

### What Gets Captured

| Field | Description | Example |
|-------|-------------|---------|
| **Error message** | The actual error text | "TypeError: Cannot read property 'map' of undefined" |
| **Error type** | Categorized error type | TypeError, ImportError, SyntaxError |
| **File path** | Where the error occurred | src/components/List.tsx |
| **Solution** | How it was fixed | "Added null check before mapping" |
| **Context** | Surrounding discussion | The conversation explaining the fix |

### How It Works

1. **Detection**: During ingestion, assistant messages are scanned for error patterns (stack traces, error keywords)
2. **Normalization**: Error messages are cleaned (paths/IDs removed) for better matching
3. **Signature**: Each error gets a hash-based fingerprint for deduplication
4. **Solution linking**: When an error is followed by a fix, the solution is linked
5. **FTS indexing**: All errors are indexed for full-text search

### Use Cases

- **"I've seen this error before"**: Search past solutions instantly
- **"How did I fix the auth bug?"**: Find specific error resolutions
- **New team member**: Build understanding of common project issues

## Decision Journal

MIRA3 automatically extracts architectural and design decisions from conversations, preserving the reasoning behind choices.

### Decision Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **architecture** | System design, component structure | "Use microservices for auth" |
| **technology** | Library/framework choices | "Chose PostgreSQL over MongoDB" |
| **implementation** | Code patterns, algorithms | "Use factory pattern for plugins" |
| **testing** | Test strategies, coverage | "Integration tests for API" |
| **security** | Auth, validation, data protection | "JWT with refresh tokens" |
| **performance** | Optimization choices | "Add Redis caching layer" |
| **workflow** | Process and tooling | "Use conventional commits" |

### What Gets Captured

| Field | Description |
|-------|-------------|
| **Decision** | The actual choice made |
| **Reasoning** | Why this approach was chosen |
| **Alternatives** | Other options that were considered |
| **Category** | Which category this falls into |
| **Source session** | Which conversation made this decision |

### How It Helps

- **Consistency**: Remember why you chose specific approaches
- **Onboarding**: New sessions understand past decisions
- **Documentation**: Auto-generated decision log
- **Review**: Revisit decisions when requirements change

## Artifact Detection & Storage

MIRA3 automatically detects and stores structured content ("artifacts") from conversations in a dedicated SQLite database. This enables precise retrieval of important content that would otherwise be lost due to embedding truncation.

### Artifact Types

| Type | Description | Examples |
|------|-------------|----------|
| **code_block** | Code snippets with language annotation | Python functions, TypeScript classes, SQL queries |
| **numbered_list** | Ordered lists with 3+ items | Step-by-step instructions, procedures |
| **bullet_list** | Bullet point lists with 3+ items | Feature lists, requirements, options |
| **table** | Markdown tables | Data comparisons, specifications |
| **config** | Configuration blocks | JSON configs, package.json snippets |
| **error** | Error messages and stack traces | Tracebacks, TypeErrors, panics |
| **url** | URLs and links | GitHub repos, documentation links |
| **command** | Shell commands | npm install, git commands, docker |
| **document** | Large structured documents | Specifications with multiple sections |

### How It Works

1. **Detection**: During ingestion, MIRA3 scans each message for structured content patterns
2. **Extraction**: Matching content is extracted with context (title, language, line count)
3. **Storage**: Artifacts are stored in SQLite (`artifacts.db`) with full-text search indexing
4. **Deduplication**: Content is hashed to prevent duplicate storage across re-ingestion

### Artifact Storage Schema

```
.mira/
├── artifacts.db           # SQLite database
│   ├── artifacts          # Main table (id, session_id, type, content, metadata)
│   └── artifacts_fts      # FTS5 full-text search index
```

### Search Integration

Artifact search is automatically included in `mira_search` results. When you search for specific code, configurations, or error messages, matching artifacts are returned alongside conversation excerpts:

```json
{
  "results": [...],
  "artifacts": [
    {
      "session_id": "abc-123",
      "artifact_type": "code_block",
      "title": "python code",
      "language": "python",
      "line_count": 15,
      "excerpt": "def authenticate(user, password):..."
    }
  ]
}
```

## MCP Tools

### `mira_search`

**Hybrid search** combining semantic embeddings with full-text archive search:

**Path A: Semantic match found**
1. ChromaDB finds relevant conversations via embeddings
2. For each match, extracts excerpts from that conversation's archive

**Path B: No semantic match**
1. Falls back to full-text search across all archives
2. Returns conversations containing the query terms

```
mira_search(query: string, limit?: number, project_path?: string) -> ConversationResult[]
```

Parameters:
- `query` - Search query (semantic + fulltext matching)
- `limit` - Maximum results (default: 10)
- `project_path` - Optional: filter to specific project

Returns:
```json
{
  "session_id": "abc-123",
  "slug": "velvet-hugging-reef",
  "summary": "Implemented authentication system",
  "task_description": "Add OAuth login",
  "relevance": 0.85,
  "excerpts": [
    {
      "role": "assistant",
      "excerpt": "...the OAuth flow requires a redirect URI...",
      "matched_terms": ["oauth", "redirect"],
      "timestamp": "2025-12-07T05:04:24.159Z"
    }
  ],
  "has_archive_matches": true
}
```

This solves the 256-token embedding limit: even if specific code or error messages weren't indexed, they can still be found in the full archives.

### `mira_recent`

Display recent projects and tasks with a broader view of activity.

```
mira_recent(limit?: number) -> RecentActivity[]
```

### `mira_init`

Comprehensive initialization context for session start. Returns storage statistics, custodian profile, and current work context.

```
mira_init(project_path?: string) -> {
  indexed_conversations: number,
  recent_sessions: Session[],
  custodian: {
    name: string,
    tech_stack: string[],           // Top technologies from history
    active_projects: Project[],     // Most worked-on projects
    common_tools: Tool[],           // Most used Claude tools
    total_sessions: number,
    total_messages: number
  },
  storage: {
    total_mira: string,             // "125.3 MB"
    codebase: string,               // "45.2 MB"
    ratio_percent: number,          // 277.4
    components: {                   // Size breakdown
      venv, chroma, archives, metadata, models, artifacts_db
    },
    note: string                    // "MIRA storage is 277.4% of codebase size"
  },
  artifacts: {
    total: number,
    by_type: { code_block: n, table: n, ... },
    by_language: { python: n, typescript: n, ... }
  },
  current_work: {
    recent_tasks: string[],         // Recent task descriptions
    active_topics: string[],        // From TODO lists
    recent_decisions: string[]      // Key facts/rules
  }
}
```

### `mira_status`

Display ingestion statistics, artifact counts, and system health.

```
mira_status() -> {
  total_files: number,
  ingested: number,
  pending: number,
  last_sync: timestamp,
  artifacts: {
    total: number,
    by_type: { code_block: number, table: number, ... },
    by_language: { python: number, typescript: number, ... }
  },
  errors: { total: number, with_solutions: number },
  decisions: { total: number, by_category: { ... } }
}
```

### `mira_error_lookup`

Search for past error solutions. Find how similar errors were resolved in previous sessions.

```
mira_error_lookup(query: string, limit?: number) -> {
  results: [
    {
      error_message: string,         // The original error
      error_type: string,            // TypeError, ImportError, etc.
      file_path: string,             // Where it occurred
      solution: string,              // How it was fixed
      session_id: string,            // Source conversation
      timestamp: string              // When it was resolved
    }
  ],
  total: number,
  query: string
}
```

**Use Cases:**
- "TypeError async function" - Find how async type errors were fixed
- "ImportError chromadb" - See past ChromaDB import resolutions
- "ENOENT file not found" - Find file access error solutions

### `mira_decisions`

Search for past architectural and design decisions with their reasoning.

```
mira_decisions(query: string, category?: string, limit?: number) -> {
  results: [
    {
      decision: string,              // What was decided
      category: string,              // architecture, technology, etc.
      reasoning: string,             // Why this choice was made
      alternatives: string[],        // Other options considered
      session_id: string,            // Source conversation
      timestamp: string
    }
  ],
  total: number,
  query: string,
  category: string | null
}
```

**Categories:** `architecture`, `technology`, `implementation`, `testing`, `security`, `performance`, `workflow`

**Use Cases:**
- "database" - Find all database-related decisions
- "authentication" with category "security" - Security decisions about auth
- "" (empty query) - Get recent decisions across all categories

## Requirements

- Node.js >= 20.0.0
- Python >= 3.8 (for backend - auto-detected, prompts if missing)
- Claude Code installation

Python dependencies (ChromaDB, sentence-transformers, watchdog) are automatically installed into `.mira/.venv/` on first run.

## How It Differs from Claude Historian

| Feature | MIRA3 | Claude Historian |
|---------|-------|------------------|
| Storage | ChromaDB vectors | In-memory algorithms |
| Indexing | Persistent | On-demand |
| Metadata | Structured extraction | Query-time parsing |
| Background Processing | Daemon | None |
| Search | Semantic (MiniLM) | TF-IDF + fuzzy |

## License

MIT

---

## Development Status

**MVP Complete** - All core functionality implemented and tested.

### Completed Features

| Feature | Status | Notes |
|---------|--------|-------|
| Project scaffolding | Done | TypeScript + ES modules |
| Python self-installing backend | Done | Creates venv, installs ChromaDB, sentence-transformers on first run |
| Node.js → Python JSON-RPC | Done | Bidirectional stdio communication |
| JSONL parsing | Done | Claude Code conversation format |
| Conversation discovery | Done | Scans ~/.claude/projects/ |
| Metadata extraction | Done | Summary, keywords, key facts, TODO topics |
| Archive writer | Done | Copies to .mira/archives/ |
| ChromaDB integration | Done | Semantic vector search with all-MiniLM-L6-v2 |
| Artifact detection | Done | Code, lists, tables, configs, errors, URLs stored in SQLite |
| MCP tools (6) | Done | mira_search, mira_recent, mira_init, mira_status, mira_error_lookup, mira_decisions |
| Error pattern recognition | Done | Extracts errors with solutions, FTS5 searchable |
| Decision journal | Done | Captures architectural decisions with reasoning |
| File watcher | Done | watchdog-based, auto-ingests new conversations |
| CLI entry point | Done | npx claude-mira3 |

### Test Results

**Integration Tests:**
```
=== MIRA3 Integration Tests ===
✓ Backend spawned successfully
✓ Status response: indexed: 1
✓ Search response: found conversation with relevance score
✓ Recent response: 1 session found
✓ Init response: custodian discovered
=== All tests passed! ===
```

**Unit Tests (78 passing):**
- TestArtifactDetection (11 tests)
- TestConversationParsing (1 test)
- TestSearch (3 tests)
- TestIngestion (2 tests)
- TestEmbedding (1 test)
- TestWatcher (1 test)
- TestHandlers (6 tests)
- TestCustodian (8 tests)
- TestInsights (7 tests)
- TestBootstrap (1 test)
- TestUserExperienceScenarios (14 tests)

### Future Enhancements

- **Conversation statistics and analytics** - Usage patterns, topic frequency, productivity metrics
- **Export functionality** - Export conversation history, decisions, and learnings to markdown/JSON
- **Proactive insights** - Claude automatically surfaces relevant past context during sessions
- **Custodian rule enforcement** - Automatic warnings when actions conflict with learned rules