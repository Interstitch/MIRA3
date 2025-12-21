# Integration Tests

Tests that require database connections, file system access, or storage backends.

## Test Files

| File | Tests |
|------|-------|
| `test_storage.py` | Storage facade and backend switching |
| `test_backends.py` | Postgres and Qdrant backends |
| `test_migrations.py` | Database schema migrations |
| `test_sync.py` | Central storage synchronization |
| `test_ingestion.py` | Full ingestion pipeline |
| `test_watcher.py` | File system watcher |

## Prerequisites

These tests may require:
- Temporary directories (provided by fixtures)
- SQLite databases (created automatically)
- Optional: Postgres/Qdrant for backend tests (skipped if unavailable)

## Running

```bash
pytest tests/integration/ -v
```
