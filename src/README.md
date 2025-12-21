# MIRA Source Code

The main MIRA Python package lives in `src/mira/`.

## Package Structure

```
src/
└── mira/                    # Main package
    ├── __init__.py          # Package root, exports VERSION
    ├── __main__.py          # CLI entry point (python -m mira)
    ├── server.py            # MCP server using Python MCP SDK
    │
    ├── core/                # Fundamental infrastructure
    │   ├── constants.py     # VERSION, paths, DB names
    │   ├── config.py        # Configuration classes
    │   ├── database.py      # Thread-safe DatabaseManager
    │   ├── bootstrap.py     # Venv and dependency management
    │   ├── utils.py         # Logging, path utilities
    │   └── parsing.py       # Conversation parsing
    │
    ├── tools/               # MCP tool handlers
    │   ├── init.py          # mira_init
    │   ├── search.py        # mira_search
    │   ├── recent.py        # mira_recent
    │   ├── errors.py        # mira_error_lookup
    │   ├── decisions.py     # mira_decisions
    │   ├── code_history.py  # mira_code_history
    │   └── status.py        # mira_status
    │
    ├── storage/             # Storage backends
    ├── extraction/          # Content extraction
    ├── search/              # Multi-tier search
    ├── custodian/           # User learning
    └── ingestion/           # Conversation processing
```

## Running

```bash
# Run as MCP server (for Claude Code)
python -m mira

# Run mira_init for hooks
python -m mira --init --project /path/to/project

# Check version
python -m mira --version
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type check
mypy src/mira
```

See `docs/ARCHITECTURE.md` for detailed architecture documentation.
