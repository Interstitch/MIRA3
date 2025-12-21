# Extraction Module

Content extraction from conversations.

## Files

| File | Purpose |
|------|---------|
| `metadata.py` | Summary, keywords, facts extraction |
| `artifacts.py` | Code blocks, lists, tables |
| `insights.py` | Coordinator for errors/decisions |
| `errors.py` | Error pattern detection |
| `decisions.py` | Decision journal extraction |
| `concepts.py` | Codebase concept learning |
| `code_history.py` | File change tracking |

## Key Exports

```python
from mira.extraction import (
    extract_metadata,
    extract_artifacts_from_messages,
    extract_insights_from_conversation,
    extract_concepts_from_conversation,
)
```
