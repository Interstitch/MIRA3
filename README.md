# MIRA3 - Memory Information Retriever and Archiver

[![GitHub](https://img.shields.io/badge/GitHub-Interstitch%2FMIRA3-blue?logo=github)](https://github.com/Interstitch/MIRA3)

An MCP server that gives Claude Code persistent memory across sessions.

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
> *"Show me the code we used for the caching layer"*

And Claude just *knows* - no re-explanation needed.

## What This Actually Looks Like

**Scenario 1: You forgot where you left off**
```
You: What was I working on Friday?
Claude: [Calls mira_recent(days=3)] You had 2 sessions Friday:
        1. Implementing sync worker retry logic - got it working
        2. Debugging the connection pool - noted you wanted exponential backoff
```

**Scenario 2: You want to reuse a solution from another project**
```
You: How did we implement rate limiting in the API project?
Claude: [Searches across projects] Found it in api-gateway (Nov 15). You used
        a token bucket algorithm with Redis. Want me to show the implementation?
```

**Scenario 3: You're about to touch a cursed file**
```
You: Let's refactor ingestion.py
Claude: ⚠️ Heads up - ingestion.py has caused 51 recorded issues in past sessions.
        Want me to search what went wrong before to avoid repeating mistakes?
```

## Key Features

- **Zero-configuration context** - Claude knows who you are the moment a session starts
- **Semantic search** - Find "that auth conversation" even if you never used that word
- **Error pattern database** - Indexed errors linked to solutions
- **Danger zone warnings** - Claude warns before touching files that caused past issues
- **Workflow enforcement** - MIRA detects your dev pattern (test first? plan first?) and Claude enforces it
- **Decision journal** - Track architectural decisions with reasoning

## Installation

```bash
claude mcp add claude-mira3 -- npx claude-mira3
```

That's it. The SessionStart hook auto-configures on install, injecting MIRA context at the start of every Claude Code session.

**First search:** When you first search, MIRA downloads a ~100MB embedding model in the background for local semantic search. This only happens once. (If you [set up remote storage](#remote-storage-optional) first, the server handles embeddings and this download is skipped.)

### Configuration

MIRA configuration is stored in `.mira/config.json`:

```json
{
  "project_path": "/workspaces/MIRA3"
}
```

| Option | Description |
|--------|-------------|
| `project_path` | Restrict MIRA to only index this project's conversations |

## How It Works

**The magic is in the SessionStart hook.** Before you type anything, MIRA injects your profile:

```
=== MIRA Session Context ===

## User Profile
Name: Alex
Summary: Alex is the lead developer (123 sessions). Prefers planning before implementation.
Development Lifecycle: Plan → Test → Implement → Commit (85% confidence)
Interaction Tips:
  - Be careful with: ingestion.py (51 recorded issues)

## When to Consult MIRA
- [CRITICAL] Encountering an error → call mira_error_lookup first
- [CRITICAL] About to say "I don't know" → search MIRA before admitting ignorance
- [CRITICAL] User mentions past discussions → search MIRA to recall context
```

**What this means:**
- Claude addresses you by name without asking
- Claude follows your preferred workflow (test first? plan first?)
- Claude warns before touching files that caused past issues
- Claude searches your history before saying "I don't know"

No manual prompting. No "remember that I prefer..." every session. MIRA learns from your conversations and Claude just *knows*.

## MCP Tools

| Tool | Purpose |
|------|---------|
| `mira_init` | Session initialization (called automatically via hook) |
| `mira_search` | Search conversations by meaning |
| `mira_recent` | Recent sessions. Use `days: 7` for last week |
| `mira_error_lookup` | Search past errors and their solutions |
| `mira_decisions` | Search architectural decisions |
| `mira_code_history` | Track file/function changes across sessions |
| `mira_status` | System health and ingestion stats |

### Search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query |
| `limit` | number | 10 | Maximum results |
| `days` | number | - | Filter to last N days |
| `recency_bias` | boolean | true | Boost recent results. Set `false` for historical searches |
| `compact` | boolean | true | Compact format (~79% smaller) |

## Search Features

### Semantic Search

MIRA finds conversations by meaning, not just keywords. Search for "authentication bug" and find discussions about "login issues" or "JWT token problems."

**Local mode:** On first search, MIRA downloads `BAAI/bge-small-en-v1.5` (~100MB ONNX model via fastembed). Vectors are stored in SQLite with sqlite-vec. All processing happens locally.

**Remote mode:** If you configure [remote storage](#remote-storage-optional), the embedding-service on your server handles vector generation using `all-MiniLM-L6-v2` (sentence-transformers) with Qdrant for storage and search. No model download needed on client machines.

### Time Decay

Results are ranked with exponential time decay - recent conversations surface first:

| Age | Score Multiplier |
|-----|------------------|
| Today | 1.00 |
| 30 days | 0.79 |
| 90 days | 0.50 |
| 1 year | 0.10 (floor) |

Disable with `recency_bias: false` for historical searches like "original architecture decision".

### Fuzzy Matching

Typos are auto-corrected:
```
Query: "authentcation implementaton"
       ↓ (auto-corrected)
Search: "authentication implementation"
```

## Data & Indexing

MIRA stores data in `<workspace>/.mira/`. When a new conversation is detected, it extracts and indexes:

| What's Extracted | Stored In | Purpose |
|------------------|-----------|---------|
| Summary, keywords, metadata | `local_store.db` | FTS5 full-text search |
| Semantic vectors | `local_vectors.db` | Meaning-based search |
| Error patterns & solutions | `insights.db` | `mira_error_lookup` |
| Architectural decisions | `insights.db` | `mira_decisions` |
| Code blocks, commands, configs | `artifacts.db` | Structured content retrieval |
| User profile, preferences | `custodian.db` | Session context injection |
| Conversation copies | `archives/` | Offline search, excerpts |
| Python dependencies | `.venv/` | ~50MB runtime |

**Filtering:** Agent sub-conversations and empty sessions are skipped automatically.

## Learning Features

### Custodian Profile

MIRA builds your profile from conversations:

- **Identity** - Your name (from "I'm Sarah" or similar)
- **Preferences** - Tool preferences, coding style
- **Rules** - "never commit to main", "always run tests first"
- **Danger zones** - Files that have caused repeated issues
- **Workflow** - Your development pattern (test first? plan first?)

### Environment Prerequisites

State prerequisites naturally and MIRA remembers:

```
"In Codespaces, I need to start tailscaled first"
"On my workstation, run docker-compose up before tests"
```

MIRA detects your environment and reminds you in future sessions.

### Decision Journal

Record decisions explicitly for high confidence:

```
"Decision: use PostgreSQL for the database"
"ADR: all API responses include meta field"
"Going forward, use pnpm instead of npm"
```

Search with `mira_decisions` to understand past choices.

## Architecture

```
Claude Code → Node.js MCP Server → Python Backend → SQLite (FTS5 + vectors)
```

- **Node.js layer**: MCP protocol handling
- **Python layer**: File watching, ingestion, search, extraction
- **SQLite**: FTS5 for text search, sqlite-vec for semantic search

## FAQ

**How long until MIRA learns my name?**
Usually 1-2 sessions. Just mention it naturally: "I'm Sarah" or sign off with your name.

**Does MIRA read my code?**
No. MIRA only indexes Claude Code conversation history. It never reads your source code files directly.

**Does MIRA store code from my conversations?**
Yes. Code snippets, commands, and configs discussed in conversations are stored in MIRA's local SQLite databases (in `.mira/`). This data stays on your machine and is only exchanged with Claude Code during your sessions. If you configure remote storage, conversation data syncs to your server - but never to any third-party service.

**How do I teach MIRA my workflow?**
Just work normally. MIRA detects patterns: if you consistently write tests before implementing, it learns that as your workflow.

## Requirements

- Node.js >= 20.0.0
- Python >= 3.8
- Claude Code

**Note:** MIRA has only been tested on Linux (Ubuntu, Debian, Codespaces). macOS and Windows support is untested.

## Known Limitations

**Fresh Install Testing:** Most development has been done on systems with existing MIRA data. Fresh installs have received limited testing. If you encounter issues during initial setup, please [open an issue](https://github.com/Interstitch/MIRA3/issues).

---

## Remote Storage (Optional)

> **Most users don't need this.** Local semantic search works out of the box. Remote storage is only for users who want history to sync across multiple machines or share between team members.

Remote storage enables:
- **Cross-machine sync** - History follows you across laptop, desktop, Codespaces
- **Persistent memory** - Rebuild a Codespace and your history is already there
- **Team sharing** - Multiple developers share the same memory pool

### Quick Setup

**On your server** (any Linux machine with Docker):

```bash
curl -sL https://raw.githubusercontent.com/Interstitch/MIRA3/master/server/install.sh | bash
```

The script prompts for server IP, PostgreSQL password, and Qdrant API key.

### Manual Docker Compose

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
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      QDRANT_HOST: qdrant
      QDRANT_API_KEY: ${QDRANT_API_KEY}
    ports:
      - "8100:8100"
    depends_on:
      - postgres
      - qdrant

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

Create `~/.mira/server.json` on each machine:

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

See [SERVER_SETUP.md](SERVER_SETUP.md) for firewall ports and troubleshooting.

---

## License

MIT
