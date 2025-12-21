"""
MIRA Extraction Package

Extracts structured insights from conversation content:
- Error patterns and solutions
- Architectural decisions
- Codebase concepts
- Code history (file operations)
- Artifacts (code blocks, lists, etc.)
- Metadata (summary, keywords, facts)
"""

from .errors import (
    init_insights_db,
    normalize_error_message,
    extract_error_type,
    generate_error_signature,
    record_error_pattern,
    search_error_solutions,
    get_error_stats,
    extract_errors_from_conversation,
)

from .decisions import (
    categorize_decision,
    generate_decision_hash,
    record_decision,
    search_decisions,
    get_decision_stats,
    extract_decisions_from_text,
    extract_decisions_from_conversation,
)

from .insights import (
    extract_insights_from_conversation,
)

from .concepts import (
    init_concepts_db,
    ConceptExtractor,
    ConceptStore,
    get_codebase_knowledge,
    get_concepts_stats,
    extract_concepts_from_conversation,
)

from .code_history import (
    init_code_history_db,
    FileOperation,
    get_file_timeline,
    get_symbol_history,
    reconstruct_file_at_date,
    get_edits_between,
    get_file_snapshot_at_date,
    get_code_history_stats,
)

from .artifacts import (
    ARTIFACTS_SCHEMA,
    init_artifact_db,
    store_file_operation,
    get_file_operations,
    reconstruct_file,
    detect_language,
    store_artifact,
    collect_artifacts_from_content,
    extract_file_operations_from_messages,
    extract_artifacts_from_messages,
    get_artifact_stats,
    get_journey_stats,
    search_artifacts_for_query,
)

from .metadata import (
    STOPWORDS,
    build_summary,
    extract_keywords,
    extract_accomplishments,
    extract_key_facts,
    clean_task_description,
    extract_todo_topics,
    sample_messages_for_embedding,
    build_document_content,
    extract_metadata,
)

__all__ = [
    # Errors
    "init_insights_db",
    "normalize_error_message",
    "extract_error_type",
    "generate_error_signature",
    "record_error_pattern",
    "search_error_solutions",
    "get_error_stats",
    "extract_errors_from_conversation",
    # Decisions
    "categorize_decision",
    "generate_decision_hash",
    "record_decision",
    "search_decisions",
    "get_decision_stats",
    "extract_decisions_from_text",
    "extract_decisions_from_conversation",
    # Insights (coordinator)
    "extract_insights_from_conversation",
    # Concepts
    "init_concepts_db",
    "ConceptExtractor",
    "ConceptStore",
    "get_codebase_knowledge",
    "get_concepts_stats",
    "extract_concepts_from_conversation",
    # Code History
    "init_code_history_db",
    "FileOperation",
    "get_file_timeline",
    "get_symbol_history",
    "reconstruct_file_at_date",
    "get_edits_between",
    "get_file_snapshot_at_date",
    "get_code_history_stats",
    # Artifacts
    "ARTIFACTS_SCHEMA",
    "init_artifact_db",
    "store_file_operation",
    "get_file_operations",
    "reconstruct_file",
    "detect_language",
    "store_artifact",
    "collect_artifacts_from_content",
    "extract_file_operations_from_messages",
    "extract_artifacts_from_messages",
    "get_artifact_stats",
    "get_journey_stats",
    "search_artifacts_for_query",
    # Metadata
    "STOPWORDS",
    "build_summary",
    "extract_keywords",
    "extract_accomplishments",
    "extract_key_facts",
    "clean_task_description",
    "extract_todo_topics",
    "sample_messages_for_embedding",
    "build_document_content",
    "extract_metadata",
]
