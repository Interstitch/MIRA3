# MIRA MCP Tools

MIRA provides 7 MCP tools for Claude Code session memory.

## Tool Reference

### mira_init

Initialize MIRA context for a new session.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_path | string | No | Current project path |

**Returns:**
- User profile (name, preferences, workflow)
- Recent work context
- Usage guidance (when to consult MIRA)
- Alerts and danger zones
- Storage status

**Example:**
```json
{
  "name": "mira_init",
  "arguments": {
    "project_path": "/home/user/myproject"
  }
}
```

---

### mira_search

Search past Claude Code conversations.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| query | string | Yes | Search query |
| limit | integer | No | Maximum results (default: 10) |

**Returns:**
- Matching sessions with summaries
- Relevant artifacts (code blocks, etc.)
- Excerpts highlighting matches

**Example:**
```json
{
  "name": "mira_search",
  "arguments": {
    "query": "authentication flow",
    "limit": 5
  }
}
```

---

### mira_recent

Get summaries of recent conversation sessions.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| limit | integer | No | Maximum sessions (default: 5) |
| project_only | boolean | No | Filter to current project (default: false) |

**Returns:**
- Recent sessions grouped by project
- Session summaries and dates
- Message counts

**Example:**
```json
{
  "name": "mira_recent",
  "arguments": {
    "limit": 10,
    "project_only": true
  }
}
```

---

### mira_error_lookup

Search past error patterns and their solutions.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| query | string | Yes | Error message or pattern |
| limit | integer | No | Maximum results (default: 5) |

**Returns:**
- Matching error patterns
- Solutions that resolved each error
- Occurrence counts

**Example:**
```json
{
  "name": "mira_error_lookup",
  "arguments": {
    "query": "TypeError: Cannot read property",
    "limit": 3
  }
}
```

---

### mira_decisions

Search architectural and design decisions.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| query | string | No | Decision topic |
| category | string | No | Filter by category |
| limit | integer | No | Maximum results (default: 10) |

**Categories:**
- `architecture` - System design choices
- `technology` - Library/framework choices
- `implementation` - Code pattern choices
- `testing` - Test strategy decisions
- `security` - Security-related decisions
- `performance` - Optimization decisions
- `workflow` - Process decisions

**Example:**
```json
{
  "name": "mira_decisions",
  "arguments": {
    "query": "database",
    "category": "technology"
  }
}
```

---

### mira_code_history

Get the history of changes to a file across sessions.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file_path | string | Yes | Path to the file |
| limit | integer | No | Maximum operations (default: 50) |

**Returns:**
- Timeline of file modifications
- Operation types (create, edit, delete)
- Session context for each change

**Example:**
```json
{
  "name": "mira_code_history",
  "arguments": {
    "file_path": "src/auth/login.py",
    "limit": 20
  }
}
```

---

### mira_status

Get MIRA system status, health, and statistics.

**Parameters:**
None required.

**Returns:**
- Storage mode (local/central)
- Connection health
- Session counts
- Indexing progress
- Storage statistics

**Example:**
```json
{
  "name": "mira_status",
  "arguments": {}
}
```

## When to Use Each Tool

| Situation | Tool |
|-----------|------|
| Starting a new session | `mira_init` |
| Looking for past discussions | `mira_search` |
| Seeing recent work | `mira_recent` |
| Debugging an error | `mira_error_lookup` |
| Making an architectural choice | `mira_decisions` |
| Understanding file history | `mira_code_history` |
| Checking MIRA health | `mira_status` |

## Recording Decisions

To record decisions explicitly (high confidence):

```
"Decision: use PostgreSQL for the primary database"
"ADR: all API responses include pagination metadata"
"For the record, we're using React Query for data fetching"
"Policy: all configs must be in YAML format"
"Going forward, use pnpm instead of npm"
```

These patterns are detected and stored with 0.90-0.95 confidence.
