# Unit Tests

Fast, isolated tests for individual components. No external dependencies required.

## Test Files

| File | Tests |
|------|-------|
| `test_imports.py` | Verifies all module imports work correctly |
| `test_parsing.py` | Conversation parsing logic |
| `test_metadata.py` | Metadata extraction (summary, keywords) |
| `test_config.py` | Configuration loading and validation |
| `test_bootstrap.py` | Venv creation and dependency installation |
| `test_artifacts.py` | Code block and artifact detection |
| `test_custodian.py` | User learning and profile extraction |
| `test_insights.py` | Error patterns and decision extraction |
| `test_local_semantic.py` | Local semantic search (sqlite-vec) |
| `test_search.py` | Search engine logic |
| `test_handlers.py` | MCP tool handlers |
| `test_status.py` | Status reporting |
| `test_utils.py` | Utility functions |
| `test_embedding.py` | Embedding client |

## Running

```bash
pytest tests/unit/ -v
```
