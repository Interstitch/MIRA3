# MIRA3 - Memory Information Retriever and Archiver

An MCP server that gives Claude Code persistent memory across sessions, machines, and projects.

## Overview

MIRA watches your Claude Code conversations, learns your preferences, indexes errors and solutions, and makes everything searchable. With remote storage, you get semantic search ("that auth conversation") and seamless history across all your machines.

**Key Features:**
- **Semantic search** - Find conversations by meaning, not just keywords (requires remote storage)
- **Cross-machine sync** - Your full history follows you across laptop, desktop, Codespaces
- **Automatic context injection** - Claude knows your name, workflow, and preferences from session start
- **Error pattern recognition** - Search 88+ resolved errors before debugging from scratch
- **Learned prerequisites** - MIRA learns environment-specific setup (e.g., "start tailscaled in Codespaces")

**Storage modes:**
- **Remote (recommended)** - Postgres + Qdrant for semantic search and cross-machine sync
- **Local-only** - SQLite FTS5 for offline/air-gapped environments

## Installation

```bash
claude mcp add claude-mira3 -- npx claude-mira3
```

The SessionStart hook auto-configures on install, injecting MIRA context at the start of every Claude Code session.

**Next step:** [Set up remote storage](#remote-storage) to unlock semantic search and cross-machine sync.

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

## Remote Storage

Remote storage unlocks MIRA's full potential:

- **Semantic search** - Find "that conversation about authentication" even if you never used that word
- **Cross-project search** - Search all your projects at once
- **Cross-machine sync** - Seamless history across laptop, desktop, and Codespaces
- **Persistent memory** - Rebuild a Codespace and your full history is already there

### Quick Setup

**On your server** (any Linux machine with Docker):

```bash
curl -sL https://raw.githubusercontent.com/Interstitch/MIRA3/master/server/install.sh | bash
```

The script prompts for:
- Server IP address
- PostgreSQL password
- Qdrant API key

Then starts Postgres, Qdrant, and an embedding service via Docker Compose.

**On each dev machine**, create `.mira/server.json` with the credentials you set:

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

Without remote storage, MIRA works locally with SQLite FTS5 keyword search. This is useful for:
- Air-gapped environments
- Quick evaluation before setting up a server
- Offline work (syncs when reconnected if remote is configured)

Local mode creates `.mira/.venv/` and SQLite databases (~50MB) on first use.

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
├── config.json                  # MIRA configuration
├── server.json                  # Remote storage credentials (if configured)
├── .venv/                       # Python virtualenv (auto-created)
├── archives/
│   └── <session-id>.jsonl       # Full conversation copies
├── metadata/
│   └── <session-id>.json        # Extracted metadata per conversation
├── local_store.db               # Main SQLite DB with FTS5 search
├── custodian.db                 # Learned user preferences
├── insights.db                  # Error patterns and decisions
├── concepts.db                  # Codebase concepts
├── artifacts.db                 # Structured content (code, lists, tables)
├── sync_queue.db                # Pending syncs to remote (if configured)
├── migrations.db                # Schema version tracking
├── mira.log                     # Runtime logs
├── mira.lock                    # Singleton lock file
└── mira.pid                     # Process ID file
```

### Storage Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Python venv** | Isolated environment with dependencies | `.mira/.venv/` |
| **Local Store** | Sessions, messages, FTS5 search index | `.mira/local_store.db` |
| **Custodian DB** | Learned user preferences and patterns | `.mira/custodian.db` |
| **Insights DB** | Error patterns and architectural decisions | `.mira/insights.db` |
| **Concepts DB** | Codebase concepts and patterns | `.mira/concepts.db` |
| **Artifacts DB** | Structured content (code, lists, tables) | `.mira/artifacts.db` |
| **Sync Queue** | Pending operations for remote sync | `.mira/sync_queue.db` |
| **Server Config** | Remote storage connection credentials | `.mira/server.json` |
| **Archives** | Full conversation copies | `.mira/archives/` |
| **Metadata** | Pre-extracted summaries, keywords, facts | `.mira/metadata/` |

## Search

MIRA provides two search modes:

| Mode | Storage | Capability |
|------|---------|------------|
| **Semantic** | Remote | Find by meaning - "auth conversation" matches "login flow discussion" |
| **Keyword** | Local/Remote | FTS5 full-text search on exact terms |

With remote storage, searches use semantic similarity first, then keyword fallback. Local-only mode uses keyword search.

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

For each conversation:
- **Summary** - From Claude's summarization or first user message
- **Keywords** - Technical terms, file names, imports
- **Key facts** - Rules, decisions, constraints mentioned
- **Task description** - Cleaned first user request
- **Full text** - All message content (FTS5 indexed)
- **Metadata** - Git branch, models used, tools used, files touched

Skipped: file-history snapshots, agent sub-conversations, empty conversations.

## Custodian Learning

MIRA learns your preferences from conversations:
- **Identity** - Your name (from statements like "My name is John")
- **Preferences** - Tool preferences ("I prefer pnpm"), coding style
- **Rules** - Explicit constraints ("never commit to main")
- **Danger zones** - Files that have caused repeated issues
- **Development lifecycle** - Your workflow pattern (e.g., "Plan → Test → Implement")
- **Prerequisites** - Environment-specific setup requirements (see below)

This context is automatically injected at session start via the SessionStart hook.

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
| `mira_search` | Search conversations. Params: `query`, `limit`, `project_path`, `days`, `compact` |
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
| `days` | number | - | Filter to last N days |
| `compact` | boolean | true | Compact format (~79% smaller) |

**Note:** `mira_init` is called automatically via the SessionStart hook. You don't need to call it manually unless context seems stale.

## Requirements

- Node.js >= 20.0.0
- Python >= 3.8
- Claude Code

## License

MIT