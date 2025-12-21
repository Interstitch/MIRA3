# Scripts

Utility scripts for MIRA development, deployment, and operations.

## Directory Structure

```
scripts/
├── cli/        # Standalone CLI tools (run MCP tools outside Claude Code)
├── dev/        # Local development helpers
├── deploy/     # Deployment and installation scripts
├── validate/   # Health checks and validation
├── onetime/    # One-time migration or setup scripts
└── run.py      # Main MIRA entry point
```

## Directories

### cli/
Standalone CLI wrappers for MIRA MCP tools:
- `mira_search.py` - Search conversation history
- `mira_status.py` - Check MIRA status
- `mira_recent.py` - View recent sessions
- `mira_errors.py` - Search error patterns
- `mira_decisions.py` - Search decisions

### dev/
Local development helpers:
- `reset_db.py` - Wipe local databases for fresh start
- `find_unused_code.py` - Find dead code in codebase

### deploy/
Deployment scripts (see server/ for remote setup):
- Installation and upgrade scripts

### validate/
Health checks and validation:
- `health_check.py` - System health diagnostics

### onetime/
One-time scripts for migrations or data fixes.

## Usage

```bash
# Run MIRA MCP server
python scripts/run.py

# CLI tools
python scripts/cli/mira_search.py "error handling"
python scripts/cli/mira_status.py

# Validation
python scripts/validate/health_check.py

# Development
python scripts/dev/reset_db.py --force
```
