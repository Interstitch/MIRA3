# Changelog

All notable changes to MIRA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-12-21

### Changed
- **Breaking**: Renamed PyPI package from `claude-mira` to `claude-mira3` (the old name could be confused with an Anthropic project)
- **Breaking**: Hybrid storage model - venv and user databases now stored globally in `~/.mira/`
- **Breaking**: Windows compatibility fixes require venv reinstall

### Added
- Hybrid storage architecture:
  - Global (`~/.mira/`): Shared venv, user preferences (custodian.db), error patterns (insights.db)
  - Project (`<cwd>/.mira/`): Conversation index, artifacts, logs, archives
- Cross-platform path utilities: `get_global_mira_path()`, `get_project_mira_path()`, `get_db_path()`
- `mira_status` now shows both global and project storage paths with sizes
- Database routing: `GLOBAL_DATABASES` and `PROJECT_DATABASES` constants for clear separation

### Fixed
- Windows compatibility: Fixed hardcoded Unix paths (`bin/` → `Scripts/` on Windows)
- Windows compatibility: Fixed path encoding for backslashes in project paths
- Windows compatibility: Skip Unix-only `/usr/bin/python3` check on Windows
- Bootstrap: Fixed silent failure where `deps_installed: true` was written even when core dependencies failed to install
- MCP config now uses stable global venv path, works from any project directory

### Migration from claude-mira
If upgrading from `claude-mira`, run:
```bash
pip uninstall claude-mira
pip install claude-mira3
rm -rf .mira/.venv  # Old per-project venv (no longer used)
rm -rf ~/.mira      # Start fresh with global storage
mira-install
```

## [0.3.3] - 2024-12-20

### Changed
- **Breaking**: Complete restructure to pure Python (removed TypeScript/Node.js)
- **Breaking**: New package structure under `src/mira/`
- **Breaking**: Now distributed via PyPI as `claude-mira` instead of npm
- Entry point changed from `node dist/cli.js` to `python -m mira`

### Added
- Python MCP SDK integration (`mcp` package)
- Organized subpackages: core, tools, ingestion, extraction, storage, search, custodian
- Comprehensive README files in all directories
- `pyproject.toml` for modern Python packaging
- Scripts directory with dev, deploy, validate, and cli categories

### Removed
- TypeScript/Node.js layer (`src/`, `package.json`, `tsconfig.json`)
- `node_modules/` dependency

## [0.3.2] - 2024-12-19

### Fixed
- Central storage sync reliability
- Quiet mode for startup
- Custodian stats table name typo

## [0.3.1] - 2024-12-18

### Added
- Local semantic search with fastembed + sqlite-vec
- Background indexing worker for local vectors

### Fixed
- Bootstrap now prefers system Python for sqlite extension support

## [0.3.0] - 2024-12-15

### Added
- Initial public release
- MCP server with 7 tools
- Local SQLite FTS5 search
- Optional remote storage (Postgres + Qdrant)
- Custodian learning system
- Error pattern recognition
- Decision journal
