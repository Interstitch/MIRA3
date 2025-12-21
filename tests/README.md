# Tests

MIRA test suite organized by scope.

## Structure

```
tests/
├── conftest.py          # Shared fixtures
├── unit/                # Isolated component tests
│   ├── core/            # Config, database, utilities
│   ├── extraction/      # Metadata, artifacts, concepts
│   ├── search/          # Search engine, fuzzy matching
│   ├── storage/         # Storage backends
│   └── custodian/       # User learning
├── integration/         # Cross-component tests
└── e2e/                 # Full MCP protocol tests
```

## Running Tests

```bash
# All tests
pytest

# Specific category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Specific file
pytest tests/unit/test_search.py

# With coverage
pytest --cov=mira

# Verbose
pytest -v
```

## Writing Tests

1. **Unit tests**: Test single functions/classes in isolation
2. **Integration tests**: Test component interactions
3. **E2E tests**: Test full MCP tool workflows

Use fixtures from `conftest.py` for common setup.
