# MIRA3 - Memory Information Retriever and Archiver

[![GitHub](https://img.shields.io/badge/GitHub-Interstitch%2FMIRA3-blue?logo=github)](https://github.com/Interstitch/MIRA3)

An MCP server that gives Claude Code persistent memory across sessions, machines, and projects.

## The Problem

Every Claude Code session starts fresh. Claude doesn't remember:
- That you fixed this exact error last Tuesday
- That `config.py` breaks every time someone touches it
- That you prefer tests before implementation
- What you were even working on yesterday

You end up re-explaining context, re-debugging solved problems, and watching Claude confidently break files that have burned you before.

## The Solution

**MIRA gives Claude a memory.** Ask things like:

> *"What were we working on last week?"*
> *"How did we fix that authentication bug?"*
> *"Show me the code we used for the caching layer in the other project"*

And Claude just *knows* - no re-explanation needed.

## What This Actually Looks Like

**Scenario 1: You hit an error you've seen before**
```
You: Getting "connection refused" on the database
Claude: [Searches MIRA first] Last time this happened (Dec 5), it was because
        the Docker container wasn't running. Want me to check `docker ps`?
```

**Scenario 2: You're about to touch a cursed file**
```
You: Let's refactor ingestion.py
Claude: ⚠️ Heads up - ingestion.py has caused 51 recorded issues in past sessions.
        Want me to search what went wrong before to avoid repeating mistakes?
```

**Scenario 3: You forgot where you left off**
```
You: What was I working on Friday?
Claude: [Calls mira_recent(days=3)] You had 2 sessions Friday:
        1. Implementing sync worker retry logic - got it working
        2. Debugging the connection pool - noted you wanted exponential backoff
```

**Scenario 4: Reusing code from another project**
```
You: How did we implement rate limiting in the API project?
Claude: [Searches across projects] Found it in api-gateway (Dec 2). You used a
        token bucket algorithm with Redis. Want me to show the implementation?
```

## Key Features

- **Zero-configuration context** - Claude knows who you are the moment a session starts
- **Semantic search** - Find "that auth conversation" even if you never used that word
- **Cross-machine sync** - Your history follows you across laptop, desktop, Codespaces
- **Error pattern database** - Indexed errors linked to solutions, not just ad-hoc search
- **Danger zone warnings** - Claude warns before touching files that caused past issues
- **Workflow enforcement** - MIRA detects your dev pattern (test first? plan first?) and Claude enforces it
- **Decision journal** - Track architectural decisions with reasoning for future reference
- **Accomplishment tracking** - Know what you shipped: commits, fixes, releases

## Why MIRA?

Other conversation history tools require you to manually search. **MIRA is proactive:**

| Capability | MIRA | Others |
|------------|------|--------|
| Claude knows your name at session start | ✅ Automatic | ❌ You must ask |
| Learns your workflow preferences | ✅ Detects patterns | ❌ Stateless |
| Warns about problematic files | ✅ Danger zones | ❌ No tracking |
| Cross-machine history sync | ✅ Central storage | ❌ Local only |
| Meaning-based search | ✅ Semantic vectors | ❌ Keywords only |
| Error → solution linking | ✅ Indexed database | ❌ Ad-hoc search |
| Decision tracking | ✅ Searchable journal | ❌ Not captured |

**The difference:** Other tools are passive archives you query. MIRA actively shapes Claude's behavior - enforcing your workflow, warning about past mistakes, and never letting Claude say "I don't know" without checking your history first.

## Claude Knows You From the Start

**This is where the magic happens.** The SessionStart hook automatically injects your profile before you type anything:

```
=== MIRA Session Context ===

## User Profile
Name: Alex
Summary: Alex is the lead developer (123 sessions). Prefers planning before implementation.
Development Lifecycle: Plan → Test → Implement → Commit (85% confidence)
Interaction Tips:
  - Work pattern: Iterative development with frequent edits
  - Be careful with: ingestion.py (51 recorded issues)

## When to Consult MIRA
- [CRITICAL] Encountering an error → call mira_error_lookup first
- [CRITICAL] About to say "I don't know" → search MIRA before admitting ignorance
- [CRITICAL] User mentions past discussions → search MIRA to recall context
- [CRITICAL] Making architectural decisions → call mira_decisions for precedents

## Alerts
- [HIGH] In Codespaces, start tailscaled first
```

**What this means:**
- Claude addresses you by name without asking
- Claude follows your preferred workflow (test first? plan first?)
- Claude warns before touching files that caused past issues
- Claude searches your history before saying "I don't know"
- Claude recalls past conversations when you reference them
- Claude reminds you of environment-specific prerequisites

No manual prompting. No "remember that I prefer..." every session. MIRA learns from your conversations and Claude just *knows*.

**How it works:** MIRA installs a [SessionStart hook](https://docs.anthropic.com/en/docs/claude-code/hooks) that runs `mira_init` before every conversation. The hook returns your profile, danger zones, critical reminders, and environment-specific alerts - all injected into Claude's context automatically.

**Storage modes:**
- **Local** - Semantic search works out of the box with fastembed + sqlite-vec (~100MB, auto-downloads on first search)
- **Remote (optional)** - Add Postgres + Qdrant for cross-machine sync and shared history across devices

## Installation

```bash
claude mcp add claude-mira3 -- npx claude-mira3
```

The SessionStart hook auto-configures on install, injecting MIRA context at the start of every Claude Code session. Semantic search works immediately - on your first search, MIRA downloads a ~100MB embedding model in the background.

**Optional:** [Set up remote storage](#remote-storage-optional) for cross-machine sync if you work across multiple devices.

### Manual MCP Configuration (Alternative)

Add to your MCP settings (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "claude-mira3": {
      "command": "npx",
      "args": ["claude-mira3"]
    }
  }
}
```

## Remote Storage (Optional)

Remote storage enables **cross-machine sync** - your conversation history follows you everywhere:

- **Work from anywhere** - Seamless history across laptop, desktop, and Codespaces
- **Persistent memory** - Rebuild a Codespace and your full history is already there
- **Cross-project search** - Search all your projects from any machine
- **Team sharing** - Multiple developers can share the same memory pool

> **Note:** Semantic search works locally without remote storage. Remote is only needed if you want history to sync across multiple machines.

### Option A: Quick Setup Script

**On your server** (any Linux machine with Docker):

```bash
curl -sL https://raw.githubusercontent.com/Interstitch/MIRA3/master/server/install.sh | bash
```

The script prompts for server IP, PostgreSQL password, and Qdrant API key, then starts everything via Docker Compose.

### Option B: Manual Docker Compose

Create `docker-compose.yml` on your server:

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: mira
      POSTGRES_USER: mira
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  qdrant:
    image: qdrant/qdrant:latest
    environment:
      QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY}
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"

  embedding:
    image: ghcr.io/interstitch/mira-embedding:latest
    ports:
      - "8100:8100"

volumes:
  postgres_data:
  qdrant_data:
```

```bash
export POSTGRES_PASSWORD=your-secure-password
export QDRANT_API_KEY=your-api-key
docker-compose up -d
```

### Connecting Your Machines

**On each Claude Code machine**, create `.mira/server.json`:

```json
{
  "version": 1,
  "central": {
    "enabled": true,
    "qdrant": { "host": "YOUR_SERVER_IP", "port": 6333, "api_key": "YOUR_QDRANT_KEY" },
    "postgres": { "host": "YOUR_SERVER_IP", "password": "YOUR_PG_PASSWORD" }
  }
}
```

```bash
chmod 600 ~/.mira/server.json  # Protect credentials
```

**Full guide:** [SERVER_SETUP.md](SERVER_SETUP.md) covers firewall ports, verification, and troubleshooting.

### Local-Only Mode

MIRA's default mode - **full semantic search without any server setup:**

- **Meaning-based search** - Find "that auth conversation" even if you never used that word
- **Zero configuration** - Works immediately after installation
- **Lightweight** - ~100MB embedding model (fastembed) + sqlite-vec for vector storage
- **Lazy loading** - Model only downloads when you first search, not on install

**How it works:** When you search without remote storage configured, MIRA downloads a compact ONNX-based embedding model (~100MB) in the background on first use. Vectors are stored in SQLite with the sqlite-vec extension for fast similarity search. All processing happens locally - no API calls, no external services.

**When to add remote storage:** Only if you work across multiple machines (laptop + desktop + Codespaces) and want your history to sync automatically.

### Configuration

MIRA configuration is stored in `.mira/config.json`:

```json
{
  "project_path": "/workspaces/MIRA3"
}
```

| Option | Description |
|--------|-------------|
| `project_path` | Restrict MIRA to only index conversations from this project. Use the full filesystem path (e.g., `/workspaces/MIRA3`). When set, MIRA ignores conversations from other projects. |

**When to use `project_path`:** If you work on many projects but only want MIRA memory for a specific one, set this to avoid indexing unrelated conversations. This keeps search results focused and reduces storage usage.

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

MIRA maintains its own storage in `<workspace>/.mira/`:

| File/Directory | Purpose |
|----------------|---------|
| `config.json` | MIRA configuration (project_path filter, etc.) |
| `server.json` | Remote storage credentials (if configured) |
| `.venv/` | Python virtualenv with dependencies (~50MB) |
| `archives/*.jsonl` | Full conversation copies for offline search |
| `metadata/*.json` | Pre-extracted summaries, keywords, key facts |
| `local_store.db` | Main SQLite DB with FTS5 full-text search |
| `local_vectors.db` | Semantic search vectors (fastembed + sqlite-vec) |
| `custodian.db` | Learned user profile, preferences, danger zones |
| `insights.db` | Error patterns and architectural decisions |
| `concepts.db` | Codebase concepts extracted from conversations |
| `artifacts.db` | Indexed code blocks, commands, configs, tables |
| `sync_queue.db` | Pending syncs to remote (if configured) |
| `mira.log` | Runtime logs for debugging |

## Search

MIRA provides semantic search that finds conversations by meaning, not just keywords. Search for "authentication bug" and find discussions about "login issues" or "JWT token problems."

Semantic search works both with remote storage (cross-machine) and locally (single-machine). MIRA automatically uses the best available method.

### Time Decay (Recency Bias)

By default, search results are ranked with **exponential time decay** - recent conversations are weighted higher than older ones:

| Age | Score Multiplier |
|-----|------------------|
| Today | 1.00 |
| 30 days | 0.79 |
| 90 days | 0.50 (half-life) |
| 180 days | 0.25 |
| 1 year | 0.10 (floor) |

This ensures recent context surfaces first while still including old but highly relevant results.

**Disable time decay** with `recency_bias: false` for historical searches:
```
mira_search(query='original architecture decision', recency_bias=false)
mira_search(query='how we first implemented auth', recency_bias=false)
```

Use `recency_bias: false` when:
- Looking for **original** decisions or implementations
- Searching for the **first time** something was discussed
- Wanting **comprehensive** results regardless of age

**Optimized responses:** Search results use a compact format by default, reducing token usage by ~79%. Each result includes:
- Short session ID (8 chars)
- Consolidated summary (max 100 chars)
- Date (YYYY-MM-DD)
- Top 5 topics (stopwords filtered)
- Best matching excerpt (full length preserved)

Pass `compact: false` to get verbose format for debugging.

## Architecture

```
Claude Code → Node.js MCP Server → Python Backend
                                        ↓
                    ┌───────────────────┴───────────────────┐
                    ↓                                       ↓
              Local Storage                          Remote Storage
           (SQLite FTS5 search)            (Postgres + Qdrant semantic search)
```

- **Node.js layer**: MCP protocol, spawns Python backend
- **Python layer**: File watching, ingestion, search, metadata extraction
- **Local storage**: SQLite databases with FTS5 full-text search
- **Remote storage**: Postgres for metadata, Qdrant for vector embeddings, embedding service for semantic search

## What Gets Indexed

When MIRA detects a new or modified conversation file, it runs an **ingestion pipeline** that extracts structured data from the raw JSONL:

| Extraction | What It Captures | How It's Used |
|------------|------------------|---------------|
| **Summary** | Claude's own summarization, or synthesized from first exchange | Quick context in search results |
| **Keywords** | Technical terms, file names, imports, libraries mentioned | FTS5 search ranking |
| **Key Facts** | Rules, decisions, constraints stated during conversation | Custodian learning, decision journal |
| **Task Description** | Cleaned first user request | Understanding session purpose |
| **Error Patterns** | Errors encountered and their solutions | `mira_error_lookup` database |
| **Artifacts** | Code blocks, commands, configs, tables, URLs | Precise retrieval of structured content |
| **Vector Embeddings** | Semantic representation of conversation content | Meaning-based search |
| **Metadata** | Git branch, models used, tools invoked, files touched | Filtering and context |

**Automatic filtering:** MIRA filters out noise to index only meaningful content:
- **Agent sub-conversations** (`agent-*.jsonl`) - Subagent task logs are skipped entirely
- **Empty conversations** - Sessions with no user/assistant messages are skipped
- **Non-message content** - Only `user`, `assistant`, and `summary` message types are extracted; other types like `file-history-snapshot` and `tool-result` are ignored

**Incremental updates:** When a conversation is modified (you continue a session), MIRA re-ingests only the changed content, updating summaries and adding new artifacts.

## Custodian Learning

MIRA builds your profile from conversations - this is what powers the [automatic context injection](#claude-knows-you-from-the-start):

- **Identity** - Your name (from "My name is John" or "I'm Sarah")
- **Preferences** - Tool preferences ("I prefer pnpm"), coding style, frameworks
- **Rules** - Explicit constraints ("never commit to main", "always run tests first")
- **Danger zones** - Files that have caused repeated issues (tracked automatically)
- **Development lifecycle** - Your workflow pattern, detected from how you work across sessions
- **Prerequisites** - Environment-specific setup (see below)

The more you use Claude Code with MIRA, the more it learns. Everything feeds into the session start context.

## Learned Prerequisites

MIRA learns environment-specific prerequisites from your conversations. State them naturally:

```
"In Codespaces, I need to start tailscaled first"
"On my home workstation, run docker-compose up before tests"
"When SSHed into the server, source the env file first"
```

MIRA extracts the environment, action, command, and reason - then reminds you in future sessions when that environment is detected.

**Environment detection:** Codespaces, Gitpod, WSL, SSH, Docker, hostname, OS, and more. Set explicitly with `export MIRA_ENVIRONMENT=my-workstation`.

## Error Pattern Recognition

MIRA indexes errors and their solutions from conversations. Search past errors with `mira_error_lookup` to find how similar issues were resolved before.

## Decision Journal

MIRA extracts architectural decisions with reasoning. Record decisions explicitly for high confidence:

```
"Decision: use PostgreSQL for the database"
"ADR: all API responses include meta field"
"Policy: configs in YAML format"
"Going forward, use pnpm instead of npm"
```

Search with `mira_decisions` to understand past choices and maintain consistency.

## Artifacts

MIRA detects and indexes structured content: code blocks, commands, configs, tables, error messages, URLs. Included in search results automatically.

## MCP Tools

| Tool | Purpose |
|------|---------|
| `mira_search` | Search conversations. Params: `query`, `limit`, `project_path`, `days`, `recency_bias`, `compact` |
| `mira_recent` | Recent sessions. Params: `limit`, `days` (e.g., `days: 7` for last week) |
| `mira_init` | Session initialization - user profile, prerequisites, danger zones |
| `mira_status` | Ingestion stats and system health |
| `mira_error_lookup` | Search past errors and their solutions. Params: `query`, `limit` |
| `mira_decisions` | Search architectural decisions. Params: `query`, `category`, `limit` |

### Search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query |
| `limit` | number | 10 | Maximum results |
| `project_path` | string | - | Filter to specific project |
| `days` | number | - | Filter to last N days (hard cutoff) |
| `recency_bias` | boolean | true | Apply time decay to boost recent results. Set `false` for historical searches |
| `compact` | boolean | true | Compact format (~79% smaller) |

**Note:** `mira_init` is called automatically via the SessionStart hook. You don't need to call it manually unless context seems stale.

## FAQ

**How long until MIRA learns my name?**

Usually 1-2 sessions. Just mention it naturally: "I'm Sarah" or sign off with your name. MIRA extracts it automatically - no configuration needed.

**Does MIRA read my code?**

No. MIRA only indexes Claude Code conversation history (the `.jsonl` files Claude creates). It never reads your source code directly. The insights come from what you and Claude discussed, not from scanning your codebase.

**What if I use multiple machines?**

With remote storage configured, your full history syncs automatically. Start a Codespace, and Claude already knows your name, workflow preferences, and past errors - even if you've never used that specific machine before.

**How do I teach MIRA my workflow preference?**

Just work normally. MIRA detects patterns: if you consistently write tests before implementing, it learns "Test → Implement" as your workflow and tells Claude to enforce that sequence.

## Requirements

- Node.js >= 20.0.0
- Python >= 3.8
- Claude Code

**Note:** MIRA has only been tested with Claude Code on Linux (Ubuntu, Debian, Codespaces). macOS and Windows support is untested - contributions welcome.

## Known Limitations

**Fresh Install Testing:** Most development and testing has been done on systems with existing MIRA data. Fresh installs (first-time setup with no prior conversation history) have received limited testing. If you encounter issues during initial setup, please [open an issue](https://github.com/Interstitch/MIRA3/issues).

## License

MIT