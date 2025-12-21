# Core Module

Fundamental infrastructure for MIRA.

## Files

| File | Purpose |
|------|---------|
| `constants.py` | VERSION, paths, database names |
| `config.py` | ServerConfig, CentralConfig classes |
| `database.py` | Thread-safe DatabaseManager with write queue |
| `bootstrap.py` | Venv creation and dependency installation |
| `utils.py` | Logging, path utilities, helpers |
| `parsing.py` | Conversation parsing and message extraction |

## Key Exports

```python
from mira.core import (
    VERSION,
    get_mira_path,
    get_config,
    get_db_manager,
    shutdown_db_manager,
    log,
    parse_conversation,
)
```
