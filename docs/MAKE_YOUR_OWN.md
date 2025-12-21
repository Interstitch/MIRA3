# MAKE YOUR OWN: AI Prompt to Recreate MIRA

This document contains a comprehensive AI prompt that captures the complete architecture, algorithms, and implementation details of MIRA3. Use this prompt with an AI assistant to recreate a similar persistent memory system for Claude Code or any AI coding assistant.

---

## The Prompt

```markdown
# Build a Persistent Memory System for AI Coding Assistants

## Goal

Create an MCP (Model Context Protocol) server that gives AI coding assistants persistent memory across sessions. The system should:

1. **Monitor and index** AI conversation history automatically
2. **Learn about the user** (name, preferences, workflow patterns, danger zones)
3. **Provide semantic search** across all past conversations
4. **Track error patterns** and link them to solutions
5. **Record architectural decisions** with reasoning
6. **Inject context** at session start so the AI "remembers" the user

## Problem Statement

Every AI coding session starts fresh. The AI doesn't remember:
- That you fixed this exact error last Tuesday
- That `config.py` breaks every time someone touches it
- That you prefer tests before implementation
- What you were working on yesterday

The solution gives the AI memory by indexing conversations, learning user patterns, and injecting context automatically.

---

## Architecture Overview

### System Design

```
AI Coding Tool (Claude Code, Cursor, etc.)
          │
          ▼ (MCP over stdio)
    ┌─────────────────┐
    │   MCP Server    │ ← Pure Python, runs as subprocess
    │   (Python)      │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  SQLite Storage │ ← Multiple specialized databases
    │  (FTS5 + vec)   │
    └────────┬────────┘
             │ (optional)
    ┌────────▼────────┐
    │ Central Storage │ ← PostgreSQL + Qdrant (cross-machine sync)
    └─────────────────┘
```

### Data Flow

1. **File Watcher** monitors conversation directory for changes
2. **Ingestion Pipeline** parses and extracts structured data
3. **Multiple DBs** store specialized data (sessions, artifacts, errors, decisions, learnings)
4. **Search System** provides semantic + keyword search with time decay
5. **MCP Tools** expose capabilities to the AI

---

## Database Schemas

Implement these SQLite databases:

### 1. local_store.db (Sessions & Projects)

```sql
-- Projects table
CREATE TABLE projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    slug TEXT,
    git_remote TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id),
    session_id TEXT NOT NULL,
    summary TEXT,
    keywords TEXT,                    -- JSON array of top 50 terms
    facts TEXT,                       -- JSON array of key assertions
    task_description TEXT,
    git_branch TEXT,
    models_used TEXT,                 -- JSON array
    tools_used TEXT,                  -- JSON array
    files_touched TEXT,               -- JSON array
    message_count INTEGER DEFAULT 0,
    started_at TEXT,
    ended_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, session_id)
);

-- FTS5 for keyword search
CREATE VIRTUAL TABLE sessions_fts USING fts5(
    summary, task_description, keywords, facts,
    content='sessions', content_rowid='id'
);

-- Archives (full conversation content)
CREATE TABLE archives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES sessions(id),
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id)
);

-- Vocabulary for fuzzy matching
CREATE TABLE vocabulary (
    term TEXT PRIMARY KEY,
    frequency INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 2. artifacts.db (Structured Content)

```sql
-- File operations tracking
CREATE TABLE file_operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,     -- 'write' or 'edit'
    file_path TEXT NOT NULL,
    content TEXT,
    old_string TEXT,
    new_string TEXT,
    sequence_num INTEGER,
    timestamp TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Detected artifacts
CREATE TABLE artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,      -- code, list, table, config, error, url, command
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    language TEXT,
    title TEXT,
    line_count INTEGER,
    role TEXT,                        -- 'user' or 'assistant'
    message_index INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, content_hash)
);

CREATE VIRTUAL TABLE artifacts_fts USING fts5(
    content, title,
    content='artifacts', content_rowid='id'
);
```

### 3. custodian.db (User Learning)

```sql
-- User identity
CREATE TABLE identity (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    source_session TEXT,
    learned_at TEXT
);

-- User preferences
CREATE TABLE preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    preference TEXT NOT NULL,
    value TEXT,
    frequency INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.5,
    first_seen TEXT,
    last_seen TEXT,
    source_sessions TEXT              -- JSON array
);

-- Always/never rules
CREATE TABLE rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type TEXT NOT NULL,          -- 'always', 'never', 'when'
    rule_text TEXT NOT NULL,
    scope TEXT,                       -- 'global', 'project', 'file'
    frequency INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.8,
    first_seen TEXT,
    last_seen TEXT,
    source_sessions TEXT
);

-- Danger zones (problem files)
CREATE TABLE danger_zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path_pattern TEXT NOT NULL,
    issue_description TEXT,
    issue_count INTEGER DEFAULT 1,
    last_issue TEXT,
    source_sessions TEXT
);

-- Work patterns
CREATE TABLE work_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.5,
    first_seen TEXT,
    last_seen TEXT
);

-- Environment prerequisites
CREATE TABLE prerequisites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    environment TEXT NOT NULL,        -- e.g., "codespaces"
    action TEXT NOT NULL,
    command TEXT,
    reason TEXT,
    confidence REAL DEFAULT 0.5,
    learned_at TEXT
);

-- Name candidates
CREATE TABLE name_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    confidence REAL NOT NULL,
    pattern_type TEXT NOT NULL,
    source_session TEXT NOT NULL,
    context TEXT,
    extracted_at TEXT NOT NULL,
    UNIQUE(name, source_session)
);
```

### 4. insights.db (Errors & Decisions)

```sql
-- Error patterns
CREATE TABLE error_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_signature TEXT NOT NULL,    -- SHA256[:16]
    error_type TEXT,
    error_message TEXT NOT NULL,
    normalized_message TEXT,
    solution_summary TEXT,
    file_context TEXT,
    occurrence_count INTEGER DEFAULT 1,
    first_seen TEXT,
    last_seen TEXT,
    source_sessions TEXT
);

-- Decision journal
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_hash TEXT UNIQUE,
    decision_summary TEXT NOT NULL,
    reasoning TEXT,
    category TEXT,                    -- architecture, technology, testing, etc.
    session_id TEXT,
    timestamp TEXT,
    confidence REAL DEFAULT 0.5
);

CREATE VIRTUAL TABLE errors_fts USING fts5(
    error_message, solution_summary, file_context,
    content='error_patterns', content_rowid='id'
);

CREATE VIRTUAL TABLE decisions_fts USING fts5(
    decision_summary, reasoning,
    content='decisions', content_rowid='id'
);
```

### 5. concepts.db (Codebase Knowledge)

```sql
CREATE TABLE codebase_concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_path TEXT NOT NULL,
    concept_type TEXT NOT NULL,       -- component, module, technology, pattern, fact, rule
    name TEXT NOT NULL,
    description TEXT,
    related_files TEXT,
    frequency INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.5,
    source_sessions TEXT,
    first_seen TEXT,
    last_updated TEXT,
    UNIQUE(project_path, concept_type, name)
);

CREATE VIRTUAL TABLE concepts_fts USING fts5(
    name, description,
    content='codebase_concepts', content_rowid='id'
);
```

### 6. local_vectors.db (Semantic Search)

```sql
-- Session embeddings
CREATE TABLE session_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    chunk_text TEXT,
    embedding BLOB NOT NULL,          -- 384-dim float32
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, chunk_index)
);

-- Track indexed sessions
CREATE TABLE indexed_sessions (
    session_id TEXT PRIMARY KEY,
    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    chunk_count INTEGER DEFAULT 0
);

-- Model status
CREATE TABLE model_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    model_name TEXT,
    model_ready INTEGER DEFAULT 0,
    download_started_at TEXT,
    download_completed_at TEXT
);

-- Indexing queue
CREATE TABLE indexing_queue (
    session_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT,
    queued_at TEXT DEFAULT CURRENT_TIMESTAMP,
    attempts INTEGER DEFAULT 0
);
```

---

## MCP Tools to Implement

### Tool 1: mira_init (Session Initialization)

**Purpose:** Inject user context at session start

**Returns:**
- User profile (name, preferences, workflow)
- Alerts (danger zones, prerequisites)
- Current work context
- Tool quick reference

**Example output:**
```json
{
  "core": {
    "custodian": {
      "name": "Alex",
      "summary": "Alex is the lead developer (123 sessions). Prefers test-first.",
      "development_lifecycle": "Plan → Test → Implement (85%)",
      "danger_zones": ["ingestion.py (51 issues)"]
    }
  },
  "alerts": [
    {"priority": "warn", "message": "File ingestion.py has caused 51 issues"}
  ]
}
```

### Tool 2: mira_search (Semantic Search)

**Parameters:**
- `query` (string): Search query
- `limit` (int): Max results (default 10)
- `project_path` (string, optional): Filter by project

**Returns:**
- Results with session ID, summary, date, topics, excerpt
- Search source (semantic vs keyword)
- Any fuzzy corrections applied

### Tool 3: mira_recent (Recent Sessions)

**Parameters:**
- `limit` (int): Max results
- `days` (int, optional): Filter to last N days

**Returns:** List of recent sessions with summaries and metadata

### Tool 4: mira_error_lookup (Error Search)

**Parameters:**
- `query` (string): Error message or type

**Returns:** Matching errors with solutions, occurrence counts, confidence

### Tool 5: mira_decisions (Decision Journal)

**Parameters:**
- `query` (string, optional): Search decisions
- `category` (string, optional): Filter by category

**Returns:** Decisions with reasoning and confidence

### Tool 6: mira_code_history (File Timeline)

**Parameters:**
- `file_path` (string, optional): Specific file
- `symbol` (string, optional): Function/class name

**Returns:** Timeline of file operations across sessions

### Tool 7: mira_status (System Status)

**Returns:** Version, storage mode, indexing status, semantic search availability

---

## Key Algorithms

### 1. Time Decay Scoring

Apply exponential decay with 90-day half-life:

```python
import math
from datetime import datetime

def calculate_time_decay(timestamp_str):
    now = datetime.now()
    timestamp = datetime.fromisoformat(timestamp_str)
    age_days = (now - timestamp).total_seconds() / 86400

    half_life = 90  # days
    decay_rate = math.log(2) / half_life
    decay = math.exp(-decay_rate * age_days)

    return max(decay, 0.1)  # Floor at 0.1
```

### 2. Text Chunking for Embeddings

```python
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 500
MAX_CHUNKS = 50

def chunk_content(content):
    if len(content) <= CHUNK_SIZE:
        return [content]

    chunks = []
    start = 0
    while start < len(content) and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, len(content))

        # Try to break at newline
        if end < len(content):
            newline_pos = content[start:end].rfind('\n')
            if newline_pos > CHUNK_SIZE - 200:
                end = start + newline_pos + 1

        chunks.append(content[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
```

### 3. Error Signature Generation

```python
import hashlib
import re

def normalize_error_message(msg):
    # Replace variable parts with placeholders
    msg = re.sub(r'/[\w/.-]+\.\w+', '<FILE>', msg)
    msg = re.sub(r'line \d+', 'line <N>', msg)
    msg = re.sub(r'0x[0-9a-fA-F]+', '<ADDR>', msg)
    msg = re.sub(r'\d{4}-\d{2}-\d{2}', '<DATE>', msg)
    return msg

def generate_error_signature(error_msg, error_type=None):
    normalized = normalize_error_message(error_msg)
    sig_input = f"{error_type}:{normalized[:200]}" if error_type else normalized[:200]
    return hashlib.sha256(sig_input.encode()).hexdigest()[:16]
```

### 4. Search Tier Fallback

```python
def search(query, limit=10):
    # Tier 1: Central semantic (if configured)
    if storage.using_central:
        results = central_semantic_search(query, limit)
        if results:
            return apply_time_decay(results), "central_semantic"

    # Tier 2: Local semantic (if model ready)
    if local_semantic_available():
        results = local_semantic_search(query, limit)
        if results:
            return apply_time_decay(results), "local_semantic"

    # Tier 3: FTS5 keyword (always available)
    results = fts_search(query, limit)
    return apply_time_decay(results), "local_fts"
```

---

## Extraction Patterns

### Artifact Detection

```python
import re

# Code blocks
RE_CODE_BLOCK = re.compile(r'```(\w*)\n([\s\S]*?)```')

# Lists (3+ items)
RE_NUMBERED_LIST = re.compile(r'(?:^\d+[.)]\s+.+\n?){3,}', re.MULTILINE)
RE_BULLET_LIST = re.compile(r'(?:^[\-\*\+]\s+.+\n?){3,}', re.MULTILINE)

# Tables
RE_TABLE = re.compile(r'(?:^\|.+\|.*\n?)+', re.MULTILINE)

# Error patterns
ERROR_PATTERNS = [
    r'Error:',
    r'Exception:',
    r'Traceback \(most recent call last\):',
    r'TypeError:',
    r'ValueError:',
    r'ModuleNotFoundError:',
]

# Commands
RE_SHELL_COMMAND = re.compile(r'^(?:\$|>|#)\s+.+', re.MULTILINE)
```

### Language Detection

```python
LANGUAGE_PATTERNS = {
    'python': [r'\bdef\s+\w+\s*\(', r'\bimport\s+\w+', r'\bclass\s+\w+:'],
    'javascript': [r'\bfunction\s+\w+', r'\bconst\s+\w+\s*=', r'=>'],
    'typescript': [r':\s*(string|number|boolean)\b', r'\binterface\s+\w+'],
    'bash': [r'^#!/bin/bash', r'\$\{\w+\}'],
    'sql': [r'\bSELECT\b', r'\bCREATE\s+TABLE\b', r'\bINSERT\b'],
}

def detect_language(code):
    scores = {}
    for lang, patterns in LANGUAGE_PATTERNS.items():
        scores[lang] = sum(1 for p in patterns if re.search(p, code, re.IGNORECASE))
    return max(scores, key=scores.get) if max(scores.values()) > 0 else None
```

### Name Detection

```python
NAME_PATTERNS = [
    (r"my name is\s+([A-Z][a-z]{2,15})", 0.95),
    (r"(?:hi,?\s+)?i'?m\s+([A-Z][a-z]{2,15})", 0.9),
    (r"call me\s+([A-Z][a-z]{2,15})", 0.85),
]

# Blocklist: common tech terms that aren't names
BLOCKLIST = {'Python', 'React', 'Docker', 'Claude', 'GitHub', ...}

def extract_name(text):
    for pattern, confidence in NAME_PATTERNS:
        match = re.search(pattern, text)
        if match:
            name = match.group(1)
            if name not in BLOCKLIST and 3 <= len(name) <= 15:
                return name, confidence
    return None, 0
```

### Rule Extraction

```python
RULE_PATTERNS = {
    'always': [
        r'always\s+(.+?)(?:\.|$)',
        r'must\s+(.+?)(?:\.|$)',
    ],
    'never': [
        r'never\s+(.+?)(?:\.|$)',
        r'must not\s+(.+?)(?:\.|$)',
        r"don't\s+(.+?)(?:\.|$)",
    ],
}

def extract_rules(text):
    rules = []
    for rule_type, patterns in RULE_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                rules.append({
                    'type': rule_type,
                    'text': match.group(1).strip(),
                    'confidence': 0.8
                })
    return rules
```

### Decision Detection

```python
# Explicit patterns (high confidence)
EXPLICIT_DECISION_PATTERNS = [
    (r"^Decision:\s*(.+?)(?:\n|$)", 0.95),
    (r"^ADR:\s*(.+?)(?:\n|$)", 0.95),
    (r"^For the record[,:]?\s*(.+?)(?:\n|$)", 0.90),
    (r"^Going forward[,:]?\s*(.+?)(?:\n|$)", 0.85),
]

# Implicit patterns (lower confidence)
IMPLICIT_DECISION_PATTERNS = [
    (r"I decided to\s+(.+?)(?:\.|because)", 0.75),
    (r"I recommend using\s+(.+?)(?:\.|$)", 0.65),
]

DECISION_CATEGORIES = {
    'architecture': ['architecture', 'design', 'structure', 'pattern'],
    'technology': ['use', 'library', 'framework', 'database'],
    'testing': ['test', 'coverage', 'mock', 'fixture'],
    'security': ['auth', 'security', 'credential', 'encrypt'],
}

def categorize_decision(text):
    text_lower = text.lower()
    for category, keywords in DECISION_CATEGORIES.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return 'general'
```

### Development Lifecycle Detection

```python
LIFECYCLE_SIGNALS = {
    'plan': ['plan', 'design', 'architect', 'approach'],
    'test_first': ['write test', 'test first', 'tdd', 'failing test'],
    'implement': ['implement', 'build', 'code', 'develop'],
    'test_after': ['run test', 'verify', 'check', 'validate'],
    'commit': ['commit', 'push', 'pr', 'merge'],
}

def detect_lifecycle(messages):
    phase_counts = {phase: 0 for phase in LIFECYCLE_SIGNALS}

    for msg in messages:
        text = msg['content'].lower()
        for phase, signals in LIFECYCLE_SIGNALS.items():
            if any(sig in text for sig in signals):
                phase_counts[phase] += 1

    # Sort by frequency
    sorted_phases = sorted(phase_counts.items(), key=lambda x: -x[1])
    active_phases = [(p, c) for p, c in sorted_phases if c > 0]

    if not active_phases:
        return None, 0

    total = sum(c for _, c in active_phases)
    confidence = active_phases[0][1] / total

    return " → ".join(p.replace('_', ' ').title() for p, _ in active_phases[:4]), confidence
```

---

## User Learning System

### What to Learn

| Category | Examples | Storage |
|----------|----------|---------|
| Identity | Name, pronouns | identity table |
| Preferences | "prefers pnpm", "uses pytest" | preferences table |
| Rules | "never commit to main" | rules table |
| Danger Zones | Files with repeated issues | danger_zones table |
| Work Patterns | test-first, plan-first | work_patterns table |
| Prerequisites | "start tailscaled in Codespaces" | prerequisites table |

### Confidence Scoring

```python
def update_confidence(existing_confidence, is_repeated, recency_days):
    # Base increase for repetition
    if is_repeated:
        new_conf = min(existing_confidence + 0.05, 0.95)
    else:
        new_conf = existing_confidence

    # Recency weighting
    if recency_days <= 7:
        new_conf *= 1.1
    elif recency_days <= 30:
        new_conf *= 1.05

    return min(new_conf, 0.95)
```

### Profile Assembly

```python
def get_custodian_profile():
    return {
        "name": get_most_confident_name(),
        "summary": build_summary(),
        "development_lifecycle": detect_lifecycle(),
        "interaction_tips": [
            rule['text'] for rule in get_high_confidence_rules()
        ],
        "danger_zones": [
            {"path": dz['path'], "issues": dz['count']}
            for dz in get_danger_zones()
        ],
        "prerequisites": get_environment_prerequisites(),
        "total_sessions": count_sessions()
    }
```

---

## File Structure

```
your_mcp_server/
├── __init__.py
├── __main__.py              # Entry point
├── server.py                # MCP server loop
│
├── core/
│   ├── config.py            # Configuration loading
│   ├── constants.py         # Magic numbers, paths
│   ├── database.py          # Thread-safe SQLite
│   ├── parsing.py           # Conversation parsing
│   └── utils.py             # Logging, helpers
│
├── storage/
│   ├── local_store.py       # Session storage
│   ├── storage.py           # Storage abstraction
│   └── migrations.py        # Schema versioning
│
├── extraction/
│   ├── metadata.py          # Summary, keywords, facts
│   ├── artifacts.py         # Code, lists, tables
│   ├── errors.py            # Error patterns
│   ├── decisions.py         # Decision journal
│   └── concepts.py          # Codebase knowledge
│
├── search/
│   ├── core.py              # Search logic
│   ├── local_semantic.py    # Embedding search
│   └── fuzzy.py             # Typo correction
│
├── custodian/
│   ├── learning.py          # User learning
│   ├── profile.py           # Profile assembly
│   └── rules.py             # Rule extraction
│
├── ingestion/
│   ├── core.py              # Ingestion pipeline
│   └── watcher.py           # File watching
│
└── tools/
    ├── init.py              # mira_init
    ├── search.py            # mira_search
    ├── recent.py            # mira_recent
    ├── errors.py            # mira_error_lookup
    ├── decisions.py         # mira_decisions
    ├── code_history.py      # mira_code_history
    └── status.py            # mira_status
```

---

## Key Design Principles

1. **Local-First**: All data writes to local SQLite first. Central sync is optional and async.

2. **Lazy Loading**: Embedding model only downloads when needed (on first offline search).

3. **Thread-Safe Database**: Single writer thread with queue prevents lock conflicts.

4. **Graceful Degradation**: Each feature has fallbacks (semantic → keyword search).

5. **Confidence Scoring**: All learned data has 0.0-1.0 confidence. Repeated patterns increase confidence.

6. **Time Decay**: Recent results rank higher via exponential decay (90-day half-life).

7. **Incremental Processing**: Only process new messages on re-ingestion.

8. **Content Deduplication**: SHA256 hashes prevent duplicate artifacts.

---

## Implementation Order

1. **Database Manager** - Thread-safe SQLite with write queue
2. **Schema Migrations** - Create all tables with proper indexes
3. **Conversation Parser** - Read JSONL conversation files
4. **Ingestion Pipeline** - Extract and store metadata
5. **FTS5 Search** - Basic keyword search
6. **MCP Server** - Tool registration and request handling
7. **User Learning** - Name, preferences, rules extraction
8. **Error Tracking** - Pattern recognition and solution linking
9. **Decision Journal** - Explicit and implicit decision capture
10. **Semantic Search** - Embedding generation and vector search
11. **Session Context** - mira_init profile assembly

---

## Testing Recommendations

1. **Unit Tests**: Chunking, normalization, pattern extraction
2. **Integration Tests**: Full ingestion pipeline, search tiers
3. **E2E Tests**: MCP tool responses, CLI functionality

Use pytest with temp directories for database isolation.

---

## Dependencies

```
# Core
mcp>=1.25.0              # MCP protocol SDK

# Databases
# SQLite (built-in)      # FTS5 included in Python 3.8+

# Search
fastembed               # ONNX-based embeddings (~100MB model)
sqlite-vec              # Vector search extension (optional)

# Optional (for central storage)
psycopg2-binary         # PostgreSQL client
qdrant-client           # Vector database client
```

---

This prompt provides a complete specification for building a persistent memory system for AI coding assistants. Implement each section incrementally, testing as you go.
```

---

## Usage Notes

To use this prompt:

1. **Copy the entire prompt** (between the ``` markers)
2. **Paste into your AI assistant** (Claude, GPT-4, etc.)
3. **Request incremental implementation** - Ask for one section at a time
4. **Adapt to your use case** - Modify schemas and tools as needed

The prompt is designed to be comprehensive enough to recreate the core functionality while being flexible enough to adapt to different AI tools and environments.

---

## What This Captures

| Aspect | Included |
|--------|----------|
| Database Schemas | All 6 SQLite databases with full schemas |
| MCP Tools | All 7 tools with parameters and return types |
| Algorithms | Time decay, chunking, fuzzy matching, signatures |
| Extraction | Artifacts, errors, decisions, names, rules |
| Learning | Identity, preferences, danger zones, workflow |
| Search | Three-tier fallback chain |
| Architecture | File structure, data flow, design principles |

---

## Limitations

This prompt does not include:

- **Central storage implementation** (PostgreSQL + Qdrant sync logic)
- **Remote embedding service** (server-side embedding generation)
- **SessionStart hook configuration** (Claude Code specific)
- **Production deployment details** (Docker, systemd, etc.)

These are intentionally omitted as they are either optional features or tool-specific configurations.
