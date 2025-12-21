# Ingestion Module

Conversation processing pipeline.

## Files

| File | Purpose |
|------|---------|
| `core.py` | Main ingestion logic and discovery |
| `batch.py` | Parallel batch processing |
| `watcher.py` | File system monitoring (watchdog) |

## Ingestion Pipeline

```
1. Discover .jsonl files in ~/.claude/projects/
2. Parse conversation messages
3. Extract metadata (summary, keywords)
4. Extract artifacts (code blocks)
5. Extract insights (errors, decisions)
6. Learn custodian patterns
7. Archive and index
8. Optionally sync to central storage
```

## Key Exports

```python
from mira.ingestion import (
    ingest_conversation,
    run_full_ingestion,
    discover_conversations,
)
from mira.ingestion.watcher import run_file_watcher
```
