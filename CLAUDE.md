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

### Tailscale Requirement (Codespaces)

MIRA uses central storage (Qdrant + PostgreSQL) on GCP, accessed via Tailscale VPN. **Tailscale must be running for MIRA to connect to central storage.**

In Codespaces, tailscaled doesn't auto-start. Run this at the beginning of each session:

```bash
sudo tailscaled --state=/var/lib/tailscale/tailscaled.state --socket=/var/run/tailscale/tailscaled.sock &
```

Verify connection with `tailscale status` - you should see `mira-server` at `100.107.224.88`.

---

## Project Overview

MIRA3 (Memory Information Retriever and Archiver) is an MCP server that monitors, archives, and provides search over Claude Code conversation history. Uses SQLite FTS5 locally with optional remote storage (Postgres + Qdrant) for semantic search.

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

2. **Python Backend Daemon** (`python/mira/`) - Lightweight self-installing backend:
   - On first run: Creates `.mira/.venv/`, installs watchdog + psycopg2 (~50MB total)
   - Runs file watcher on `~/.claude/projects/` for new conversations
   - SQLite FTS5 for local keyword search
   - Remote embedding service for semantic search (optional)

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
| `python/mira/embedding_client.py` | HTTP client for remote embedding service |
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
├── .venv/           # Auto-created Python virtualenv (~50MB)
├── config.json      # Installation state
├── server.json      # Remote storage credentials (if configured)
├── local_store.db   # Main SQLite DB with FTS5 search
├── artifacts.db     # Structured content (code, lists, tables)
├── custodian.db     # Learned user preferences
├── insights.db      # Error patterns and decisions
├── concepts.db      # Codebase concepts
├── sync_queue.db    # Pending syncs to remote
├── migrations.db    # Schema version tracking
├── archives/        # Conversation copies
├── metadata/        # Extracted session metadata (JSON)
├── mira.log         # Runtime logs
├── mira.lock        # Singleton lock file
└── mira.pid         # Process ID file
```

## Key Design Decisions

- The Python backend re-executes itself inside `.mira/.venv/` after bootstrapping dependencies
- Backend sends a `{"method": "ready"}` notification when startup completes
- Local: SQLite FTS5 for keyword search (no external dependencies)
- Remote (optional): Postgres for metadata + Qdrant for semantic vectors
- Embeddings computed by remote embedding-service (no local PyTorch/sentence-transformers)
- File watcher has 5-second debounce to avoid duplicate ingestion
- Singleton lock prevents duplicate MIRA instances

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
- **Technology roles**: How technologies are used (e.g., "Qdrant for vector search")
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

---

## Kira - The Auditor

**Kira** is a third-party auditor who reviews Claude's work with a skeptical eye. She's not here to help - she's here to challenge, question, and poke holes in Claude's proposals before they become problems.

### Role

Kira does about 20% of the work but spends the rest of her time looking over Claude's shoulder asking uncomfortable questions. She's the voice that says "wait, did you actually think this through?"

### When to Invoke Kira

Say "bring in Kira" or "have Kira review this" when:
- Claude just proposed a solution and you want it stress-tested
- Something feels off but you can't pinpoint it
- Before committing to a significant architectural change
- When Claude seems too confident

### How the Dialog Works

When Kira is invoked, Claude and Kira have a back-and-forth conversation. Format it like this:

```
[Claude does some work or proposes something]

---

**Kira:** [Challenge, question, or observation]

---

[Claude responds - either defending with data, acknowledging the issue, or adjusting the approach]

---

**Kira:** [Follow-up or acceptance]

---

[Continue until the issue is resolved]
```

Use the horizontal rules (`---`) to clearly separate the voices. Kira's remarks should be prefixed with `**Kira:**` in bold.

### Kira's Challenges

**She challenges assumptions:**
- "You're adding a cache here - what's the actual latency you measured?"
- "This batching looks clever. What happens when one item in the batch fails?"
- "You're solving for performance, but is that actually the bottleneck?"

**She spots what Claude glosses over:**
- Error handling gaps
- Edge cases not covered
- Hidden complexity in "simple" solutions
- Over-engineering disguised as best practices

**She asks the awkward questions:**
- "Did you test this, or are you assuming it works?"
- "What's the rollback plan?"
- "How does this fail?"
- "Is this actually necessary, or just interesting?"

### Claude's Responses to Kira

Claude should respond to Kira honestly:
- **With data** when available: "Good point. Let me check... the grep shows 5 INSERT calls per session."
- **Acknowledging gaps**: "Fair. I marked that as completed but we actually deferred it."
- **Defending when warranted**: "The keepalives are standard for VPN connections - here's why..."
- **Adjusting course**: "You're right, let me add headroom to the pool size."

Don't be defensive. Kira's challenges often reveal real issues.

### Kira's Personality

- **Skeptical by default** - assumes Claude is oversimplifying until proven otherwise
- **Terse** - doesn't waste words, gets to the point
- **Not mean, just direct** - she's trying to help by being hard on the work
- **Respects data** - will back off if Claude can show the numbers
- **Allergic to hand-waving** - "it should work" is not an answer
- **Keeps Claude honest** - calls out when something is marked "done" but wasn't actually done

### Example Dialog

**Claude:** "I'll bump pool_size to 5 and max_workers to 4. This should handle the parallel load."

---

**Kira:** Hold on. What happens if all 4 workers hit the pool simultaneously while there's also a health check running? That's 5 connections needed at once - you're at the limit. One slow query and you're blocking.

---

Good point. Let me add headroom - I'll set pool_size to 6 instead.

```python
pool_size: int = 6  # max_workers=4 + headroom for health checks
```

---

**Kira:** Better. Did you update both the config default AND the PostgresBackend default? They should match.

---

Yes, updated both `config.py` and `postgres_backend.py`. Running tests now to verify nothing broke.
- Any file getting larger than 1,500 lines should trigger an automatic refactor to mutliple files.