# Changelog

All notable changes to MIRA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
