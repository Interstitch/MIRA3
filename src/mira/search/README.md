# Search Module

Multi-tier search implementation.

## Files

| File | Purpose |
|------|---------|
| `core.py` | Search orchestration and result merging |
| `fuzzy.py` | Typo correction (Damerau-Levenshtein) |
| `local_semantic.py` | Local semantic with sqlite-vec + fastembed |

## Search Tiers

1. **Remote Semantic** (Qdrant) - if central storage available
2. **Local Semantic** (sqlite-vec) - if fastembed installed
3. **FTS5 Keyword** - always available

## Key Exports

```python
from mira.search import (
    handle_search,
    fulltext_search_archives,
    start_local_indexer,
    stop_local_indexer,
)
```
