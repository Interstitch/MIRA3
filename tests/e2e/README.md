# End-to-End Tests

Full scenario tests that simulate real Claude Code usage patterns.

## Test Files

| File | Tests |
|------|-------|
| `test_scenarios.py` | Multi-step workflow scenarios |
| `test_cli.py` | Command-line interface tests |

## What These Test

- Complete MCP tool workflows
- Session lifecycle (init → search → recent)
- Error handling across the stack
- Real-world usage patterns

## Running

```bash
pytest tests/e2e/ -v
```
