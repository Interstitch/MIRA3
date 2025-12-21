"""
MIRA Custodian Package

Handles learning about the user (custodian) from conversation patterns:
- Identity (name, etc.)
- Preferences (tools, frameworks, coding style)
- Rules (always/never/prefer/avoid patterns)
- Work patterns and development lifecycle
- Danger zones (files that cause issues)
- Prerequisites (environment-specific setup)
"""

from .learning import (
    # Constants
    PREF_CODING_STYLE,
    PREF_TOOLS,
    PREF_FRAMEWORKS,
    PREF_WORKFLOW,
    PREF_COMMUNICATION,
    PREF_TESTING,
    # Schema
    CUSTODIAN_SCHEMA,
    # Functions
    init_custodian_db,
    extract_custodian_learnings,
)

from .profile import (
    compute_best_name,
    get_all_name_candidates,
    sync_from_central,
    get_full_custodian_profile,
    get_danger_zones_for_files,
    get_custodian_stats,
)

from .rules import (
    # Constants
    RULE_TYPES,
    RULE_PATTERNS,
    CONDITIONAL_RULE_PATTERNS,
    RULE_REVOCATION_PATTERNS,
    RULE_FILTER_WORDS,
    # Functions
    normalize_rule_text,
    is_rule_false_positive,
    find_similar_rule,
    get_rules_with_decay,
    format_rule_for_display,
    extract_scope_from_content,
)

from .prerequisites import (
    # Constants
    PREREQ_STATEMENT_PATTERNS,
    PREREQ_COMMAND_PATTERNS,
    PREREQ_REASON_PATTERNS,
    PREREQ_CHECK_TEMPLATES,
    PREREQ_KEYWORDS,
    # Functions
    detect_environment,
    get_applicable_prerequisites,
    check_prerequisites_and_alert,
)

__all__ = [
    # Learning constants
    "PREF_CODING_STYLE",
    "PREF_TOOLS",
    "PREF_FRAMEWORKS",
    "PREF_WORKFLOW",
    "PREF_COMMUNICATION",
    "PREF_TESTING",
    "CUSTODIAN_SCHEMA",
    # Learning functions
    "init_custodian_db",
    "extract_custodian_learnings",
    # Profile functions
    "compute_best_name",
    "get_all_name_candidates",
    "sync_from_central",
    "get_full_custodian_profile",
    "get_danger_zones_for_files",
    "get_custodian_stats",
    # Rules constants
    "RULE_TYPES",
    "RULE_PATTERNS",
    "CONDITIONAL_RULE_PATTERNS",
    "RULE_REVOCATION_PATTERNS",
    "RULE_FILTER_WORDS",
    # Rules functions
    "normalize_rule_text",
    "is_rule_false_positive",
    "find_similar_rule",
    "get_rules_with_decay",
    "format_rule_for_display",
    "extract_scope_from_content",
    # Prerequisites constants
    "PREREQ_STATEMENT_PATTERNS",
    "PREREQ_COMMAND_PATTERNS",
    "PREREQ_REASON_PATTERNS",
    "PREREQ_CHECK_TEMPLATES",
    "PREREQ_KEYWORDS",
    # Prerequisites functions
    "detect_environment",
    "get_applicable_prerequisites",
    "check_prerequisites_and_alert",
]
