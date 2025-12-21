# Tools Module

MCP tool handlers for MIRA.

## Files

One file per MCP tool:

| File | Tool | Purpose |
|------|------|---------|
| `init.py` | mira_init | Session initialization and context |
| `search.py` | mira_search | Semantic and keyword search |
| `recent.py` | mira_recent | Recent session summaries |
| `errors.py` | mira_error_lookup | Error pattern lookup |
| `decisions.py` | mira_decisions | Decision journal search |
| `code_history.py` | mira_code_history | File change history |
| `status.py` | mira_status | System health and stats |

## Key Exports

```python
from mira.tools import (
    handle_init,
    handle_search,
    handle_recent,
    handle_error_lookup,
    handle_decisions,
    handle_code_history,
    handle_status,
)
```

All handlers follow the signature:
```python
def handle_*(params: dict, storage: Optional[Storage]) -> dict
```
