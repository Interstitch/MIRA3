# Custodian Module

User learning and profile management.

## Files

| File | Purpose |
|------|---------|
| `learning.py` | Pattern extraction from conversations |
| `profile.py` | Profile building and retrieval |
| `rules.py` | Always/never/prefer pattern detection |
| `prerequisites.py` | Environment-specific prerequisites |

## What MIRA Learns

- User's name from introductions
- Development lifecycle (Plan → Test → Implement)
- Tool preferences (pnpm vs npm)
- Coding style preferences
- Danger zones (files that cause issues)
- Environment prerequisites

## Key Exports

```python
from mira.custodian import (
    init_custodian_db,
    extract_custodian_learnings,
    get_full_custodian_profile,
)
```
