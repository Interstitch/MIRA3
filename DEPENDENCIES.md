# MIRA Dependencies

## Runtime

### mcp `>=1.25.0`
> MCP protocol communication with Claude Code

- **Repository**: [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)
- **Stars**: 20.7k | **License**: MIT | **Author**: Anthropic
- **Why trusted**: Official Anthropic SDK. Microsoft and GitHub sit on the MCP steering committee. Industry standard for LLM tool integration.

### watchdog `>=3.0.0`
> Monitors `~/.claude/projects/` for conversation changes

- **Repository**: [gorakhargosh/watchdog](https://github.com/gorakhargosh/watchdog)
- **Stars**: 7.2k | **License**: Apache 2.0 | **Author**: Yesudeep Mangalapilly
- **Why trusted**: Battle-tested since 2010. Cross-platform (Linux, macOS, Windows). Used by major projects including Flask's reloader.

### psycopg2-binary `>=2.9`
> PostgreSQL client for central storage sync

- **Repository**: [psycopg/psycopg2](https://github.com/psycopg/psycopg2)
- **Stars**: 3.6k | **License**: LGPL | **Author**: psycopg team
- **Why trusted**: The de-facto Python PostgreSQL adapter since 2001. C implementation for performance. Production-proven at scale.

---

## Local Semantic Search (Optional)

### fastembed `>=0.3.0`
> Generates text embeddings locally using ONNX runtime

- **Repository**: [qdrant/fastembed](https://github.com/qdrant/fastembed)
- **Stars**: 2.6k | **License**: Apache 2.0 | **Author**: Qdrant
- **Why trusted**: Built by Qdrant (vector DB company). ONNX-based: ~50MB footprint, no GPU required, 10x lighter than PyTorch alternatives.

### sqlite-vec `>=0.1.0`
> Vector similarity search directly in SQLite

- **Repository**: [asg017/sqlite-vec](https://github.com/asg017/sqlite-vec)
- **Stars**: 6.4k | **License**: MIT | **Author**: Alex Garcia
- **Why trusted**: Mozilla Builders project. Sponsored by Fly.io, Turso, SQLite Cloud. Pure C, zero dependencies. Runs anywhere SQLite runs.

---

## Remote Services (Optional)

### PostgreSQL `14+`
> Session metadata, custodian profiles, decision journal

- **Why trusted**: The world's most advanced open source database. 35+ years of development.

### Qdrant `latest`
> Vector similarity search for semantic queries

- **Repository**: [qdrant/qdrant](https://github.com/qdrant/qdrant)
- **Stars**: 19k+ | **License**: Apache 2.0 | **Author**: Qdrant
- **Why trusted**: Purpose-built vector database. Written in Rust. Used by major AI companies.

See [`server/README.md`](server/README.md) for deployment.

---

## Development

### pytest `>=7.0`
> Unit, integration, and e2e test framework

- **Repository**: [pytest-dev/pytest](https://github.com/pytest-dev/pytest)
- **Stars**: 13.4k | **License**: MIT | **Author**: pytest-dev
- **Why trusted**: The Python testing standard. 1300+ plugins. Used by nearly every major Python project.

### pytest-asyncio `>=0.21`
> Async test support for MCP handlers

- **Repository**: [pytest-dev/pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio)
- **Stars**: 1.6k | **License**: Apache 2.0 | **Author**: pytest-dev
- **Why trusted**: Official pytest async plugin. 5.4M weekly downloads.

### ruff `>=0.1.0`
> Linting and formatting

- **Repository**: [astral-sh/ruff](https://github.com/astral-sh/ruff)
- **Stars**: 44.4k | **License**: MIT | **Author**: Astral
- **Why trusted**: Written in Rust, 10-100x faster than flake8/black. From Astral (creators of `uv`). Explosive adoption curve.

### mypy `>=1.0`
> Static type checking

- **Repository**: [python/mypy](https://github.com/python/mypy)
- **Stars**: 20.1k | **License**: MIT | **Author**: Python org
- **Why trusted**: The original Python type checker. Contributions from Guido van Rossum. Reference implementation for PEP 484.

---

## Bootstrap Tooling

### uv `>=0.1.0`
> 10-100x faster dependency installation

- **Repository**: [astral-sh/uv](https://github.com/astral-sh/uv)
- **Stars**: 55k+ | **License**: MIT | **Author**: Astral
- **Why trusted**: From Astral (creators of `ruff`). Written in Rust. Drop-in replacement for pip. Explosive adoption in Python ecosystem.

MIRA installs `uv` as its first dependency (via pip), then uses `uv pip` for all subsequent installations. This provides 10-100x faster setup with no user configuration needed.

---

## System Requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Type hint syntax, match statements |
| SQLite | 3.35+ | FTS5 full-text search support |

## Security Summary

| Aspect | Status |
|--------|--------|
| Licenses | All permissive (MIT, Apache 2.0, BSD, LGPL) |
| Maintenance | All actively maintained |
| Network access | None required unless central storage configured |
| Supply chain | All from established orgs with verified identities |
