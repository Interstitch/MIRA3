
# MIRA3 - Memory Information Retriever and Archiver

An MCP server that archives and searches your Claude Code conversation history.

## Overview

MIRA watches `~/.claude/projects/`, archives conversations, extracts metadata (summaries, keywords, decisions), and provides full-text search. Works offline with SQLite. Optional remote storage adds semantic search.

**Source Directory:** `~/.claude/projects/<project-path>/`
**Storage Directory:** `<workspace>/.mira/`

## Installation

```bash
claude mcp add claude-mira3 -- npx claude-mira3
```

That's it. On first use, MIRA3 creates a lightweight Python environment with minimal dependencies.

### Alternative: Manual Configuration

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

Uses SQLite FTS5 for keyword search. Searches conversation content, summaries, and artifacts.

**First run:** Creates `.mira/.venv/`, installs `watchdog`, creates SQLite databases (~50MB total). Takes under a minute.

**Want semantic search?** Set up [remote storage](#remote-storage-optional).

## Architecture

```
Claude Code → Node.js MCP Server → Python Backend (SQLite/FTS5, file watcher)
```

- **Node.js layer**: MCP protocol, spawns Python backend
- **Python layer**: File watching, ingestion, search, metadata extraction

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

MIRA learns your preferences from conversations: your name (from statements like "My name is John"), tool preferences ("I prefer pnpm"), rules ("never commit to main"), and files that cause issues. This context is provided to Claude via `mira_init`.

## Error Pattern Recognition (Under Development)

MIRA indexes errors and their solutions from conversations. Search past errors with `mira_error_lookup`.

## Decision Journal (Under Development)

MIRA extracts architectural decisions (technology choices, patterns, etc.) with reasoning. Search with `mira_decisions`.

## Artifacts

MIRA detects and indexes structured content: code blocks, commands, configs, tables, error messages, URLs. Included in search results automatically.

## MCP Tools

| Tool | Purpose |
|------|---------|
| `mira_search` | Full-text search across conversations |
| `mira_recent` | Recent activity across projects |
| `mira_init` | Session initialization with custodian profile |
| `mira_status` | Ingestion stats and system health |
| `mira_error_lookup` | Search past errors and solutions |
| `mira_decisions` | Search architectural decisions |

## Requirements

- Node.js >= 20.0.0
- Python >= 3.8
- Claude Code

## Remote Storage (Optional)

Run your own server to unlock:

- **Semantic search** - Find "that conversation about authentication" even if you never used that word
- **Cross-project search** - Search all your projects at once
- **Cross-machine sync** - Seamless history across laptop, desktop, and Codespaces
- **Persistent memory** - Rebuild a Codespace and your full history is already there

```bash
curl -sL https://raw.githubusercontent.com/Interstitch/MIRA3/master/server/install.sh | bash
```

Then create `~/.mira/server.json` on your dev machines pointing to the server.

**Full guide:** [SERVER_SETUP.md](SERVER_SETUP.md)

## License

MIT