# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Session Initialization

**At the start of every session, run `mira_init` to get personalized context.**

```
mira_init with project_path set to the current working directory
```

The `mira_init` tool returns:
- **`guidance.actions`**: Specific instructions on how to work with this user - FOLLOW THESE
- **`alerts`**: Issues requiring immediate attention (uncommitted changes, danger zones)
- **`core.custodian`**: User profile including name, preferred development workflow, and interaction style
- **`core.current_work`**: Recent tasks and active topics for continuity

**Why this matters:** MIRA learns from past sessions. The guidance tells you:
- The user's name (address them naturally)
- Their development cycle (e.g., "Plan → Write Tests → Implement → Commit")
- Work style preferences (incremental vs big changes, concise vs detailed responses)
- Files that have caused issues before (proceed carefully)
- Current work context (connect to what they were doing)

**Run `mira_init` again** if the session runs long or context seems stale.

---

## Project Overview

MIRA3 (Memory Information Retriever and Archiver) is an MCP server that monitors, archives, and provides semantic search over Claude Code conversation history. It uses ChromaDB for vector embeddings with the all-MiniLM-L6-v2 model.

## Build and Development Commands

```bash
npm run build        # Compile TypeScript to dist/
npm run dev          # Watch mode for development
npm run test         # Run tests with vitest
npm run lint         # ESLint on src/
npm run typecheck    # TypeScript type checking without emit
```

### Running the Integration Test

```bash
npm run build && node dist/../test/integration.js
```

Or directly:
```bash
npx tsx test/integration.ts
```

## Architecture

### Two-Layer Process Model

1. **Node.js MCP Server** (`src/`) - Thin MCP protocol layer that:
   - Handles stdio transport with Claude Code
   - Registers 6 MCP tools: `mira_search`, `mira_recent`, `mira_init`, `mira_status`, `mira_error_lookup`, `mira_decisions`
   - Spawns and communicates with Python backend via JSON-RPC over stdio

2. **Python Backend Daemon** (`python/mira/`) - Self-installing modular backend:
   - On first run: Creates `.mira/.venv/`, installs chromadb, sentence-transformers, watchdog
   - Runs file watcher on `~/.claude/projects/` for new conversations
   - Manages ChromaDB for semantic search
   - Uses all-MiniLM-L6-v2 (384-dim) for embeddings

### Key Files

| File | Purpose |
|------|---------|
| `src/cli.ts` | CLI entry point, parses --help/--version |
| `src/mcp/server.ts` | MCP server setup and tool registration |
| `src/backend/spawner.ts` | Spawns Python backend, handles JSON-RPC communication |
| `python/mira/main.py` | Entry point, JSON-RPC loop, initializes components |
| `python/mira/bootstrap.py` | Venv creation and dependency installation |
| `python/mira/handlers.py` | RPC request handlers for MCP tools |
| `python/mira/search.py` | Semantic and fulltext search logic |
| `python/mira/ingestion.py` | Conversation ingestion and indexing |
| `python/mira/metadata.py` | Summary, keyword, and fact extraction |
| `python/mira/artifacts.py` | SQLite FTS5 artifact storage |
| `python/mira/watcher.py` | File watcher with debouncing |
| `python/mira/embedding.py` | MiraEmbeddingFunction for ChromaDB |
| `python/mira/custodian.py` | Custodian learning and profile management |
| `python/mira/insights.py` | Error pattern recognition and decision journal |
| `python/mira/concepts.py` | Codebase concept extraction and tracking |

### Communication Flow

```
Claude Code → stdio → Node.js MCP Server → JSON-RPC over stdio → Python Backend
```

The Node.js layer is intentionally thin. All storage, search, and ingestion logic lives in Python.

### Storage Layout

```
.mira/
├── .venv/           # Auto-created Python virtualenv
├── config.json      # Installation state
├── chroma/          # ChromaDB persistent storage
├── artifacts.db     # SQLite database for structured content (code, lists, tables, etc.)
├── custodian.db     # SQLite database for learned user preferences and patterns
├── insights.db      # SQLite database for error patterns and architectural decisions
├── concepts.db      # SQLite database for learned codebase concepts
├── archives/        # Conversation copies
├── metadata/        # Extracted session metadata (JSON)
└── models/          # Cached sentence-transformers model
```

## Key Design Decisions

- The Python backend re-executes itself inside `.mira/.venv/` after bootstrapping dependencies
- Backend sends a `{"method": "ready"}` notification when startup completes
- ChromaDB collection uses custom `MiraEmbeddingFunction` wrapping sentence-transformers
- ChromaDB uses all-MiniLM-L6-v2 (384-dim) for semantic embeddings
- File watcher has 5-second debounce to avoid duplicate ingestion
- First-run timeout is 10 minutes to allow for dependency installation and model download

## Indexing Behavior

- Only indexes conversations with actual user/assistant messages
- Skips agent-*.jsonl files (subagent task logs)
- Extracts: summary, keywords (weighted), key facts, task description, TODO topics
- Session metadata: slug, git branch, models used, tools used, files touched
- Summary priority: Claude's own summary → task+outcome → first message
- Long conversation handling: time-gap detection (2hr+), TODO list tracking, content-based topic shifts

## Artifact Detection

Structured content is detected and stored in SQLite (`artifacts.db`) for precise retrieval:
- Code blocks (with language detection)
- Numbered and bullet lists (3+ items)
- Markdown tables
- Configuration blocks (JSON, YAML)
- Error messages and stack traces
- URLs and shell commands
- Large documents with multiple sections

Artifacts are searchable via FTS5 full-text search and integrated into `mira_search` results.

## Custodian Learning

MIRA learns about the user (custodian) from conversation patterns and provides this context to future Claude sessions via `mira_init`:

**What MIRA Learns:**
- **Identity**: User's name from self-introductions
- **Development lifecycle**: The user's preferred workflow sequence (e.g., "Plan → Write Tests → Implement → Commit")
- **Preferences**: Coding style, tools (pnpm vs npm), frameworks, communication style
- **Rules**: Explicit always/never/avoid patterns from conversations
- **Danger zones**: Files or modules that have caused repeated issues
- **Work patterns**: Iterative vs big-bang changes, planning preference

**Development Lifecycle Detection:**
- Analyzes the order in which users mention planning, testing, implementing, and committing
- Tracks confidence based on consistency across sessions
- Recent sessions weighted more heavily (habits can change)
- Outputs like "Plan → Write Tests → Implement (85% confidence)"

**Storage:**
- Learned data stored in `custodian.db` (SQLite)
- Frequency tracking increases confidence over time
- Recency weighting: last 7 days = 2x, last 30 days = 1.5x
- Source sessions tracked for provenance

**How It Helps:**
- Claude knows your name without re-introduction
- Claude follows your preferred development workflow
- Claude respects your stated preferences
- Claude warns when touching files that caused past issues
- Claude adapts as your workflow evolves over time

## Error Pattern Recognition

MIRA extracts and indexes error patterns from conversations, linking them to their solutions via `mira_error_lookup`:

**What MIRA Captures:**
- **Error messages**: Stack traces, compiler errors, runtime exceptions
- **Solutions**: The fix that resolved each error (from subsequent assistant messages)
- **Context**: File paths, error types, and surrounding discussion
- **Normalized signatures**: Hash-based error fingerprinting for deduplication

**How It Helps:**
- Search past errors: "TypeError in authentication"
- Find how similar errors were solved before
- Build institutional knowledge of common issues
- FTS5 search for exact error text matching

## Decision Journal

MIRA extracts architectural and design decisions from conversations via `mira_decisions`:

**Decision Categories:**
- `architecture`: System design, component structure
- `technology`: Library/framework choices
- `implementation`: Code patterns, algorithms
- `testing`: Test strategies, coverage approaches
- `security`: Auth, validation, data protection
- `performance`: Optimization choices
- `workflow`: Process and tooling decisions

**What MIRA Captures:**
- **Decision text**: The actual choice made
- **Reasoning**: Why this approach was chosen (from discussion context)
- **Alternatives**: Other options that were considered
- **Source session**: Which conversation made this decision

**How It Helps:**
- Search decisions by topic or category
- Understand why past choices were made
- Maintain consistency across sessions
- Onboard to project decisions quickly

## Codebase Concept Tracking

MIRA extracts and tracks key concepts about the codebase from conversation analysis, providing this context via `mira_init`:

**What MIRA Captures:**
- **Components**: Major architectural pieces (e.g., "Python backend", "MCP server")
- **Module purposes**: What each file does (learned from discussion)
- **Technology roles**: How technologies are used (e.g., "ChromaDB for vector search")
- **Integration patterns**: How components communicate (e.g., "JSON-RPC over stdio")
- **Design patterns**: Architectural approaches (e.g., "two-layer architecture")
- **User-provided facts**: Explicit statements about the codebase
- **User-provided rules**: Conventions and requirements

**Extraction Approach:**
- Pattern-based extraction from conversation content
- Higher confidence for assistant explanations
- Known technology detection with boosted confidence
- Frequency tracking for repeated mentions
- Case-normalized deduplication

**Storage:**
- Concepts stored in `concepts.db` (SQLite)
- Scoped by project path
- Confidence scores based on frequency and corroboration

**How It Helps:**
- New Claude sessions immediately understand codebase architecture
- Know which files are central/frequently discussed
- Understand how components relate without re-exploration
- Respect user-stated conventions and rules