# Storage Module

Storage backends and synchronization.

## Files

| File | Purpose |
|------|---------|
| `storage.py` | Storage facade (local/central switching) |
| `local_store.py` | SQLite FTS5 storage |
| `migrations.py` | Schema version management |
| `sync/` | Central storage synchronization |

## Storage Modes

### Local (Default)
- SQLite databases in `~/.mira/`
- FTS5 full-text search
- Works offline

### Central (Optional)
- PostgreSQL for metadata
- Qdrant for vectors
- Cross-machine search

## Key Exports

```python
from mira.storage import get_storage, Storage
```
