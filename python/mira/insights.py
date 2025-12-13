"""
MIRA3 Insights Module

Provides advanced analysis features:
- Error pattern recognition and solution lookup
- Decision journal with reasoning
- Conversation similarity detection

Uses central Postgres for storage with local SQLite fallback.
"""

import json
import re
import hashlib
from datetime import datetime
from typing import Optional, List, Dict

from .utils import log, extract_query_terms
from .db_manager import get_db_manager


# Database name for insights
INSIGHTS_DB = "insights.db"

# Schema for insights database
INSIGHTS_SCHEMA = """
-- Error patterns table - tracks errors and their solutions
CREATE TABLE IF NOT EXISTS error_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_signature TEXT NOT NULL,
    error_type TEXT,
    error_message TEXT NOT NULL,
    normalized_message TEXT,
    solution_summary TEXT,
    solution_details TEXT,
    file_context TEXT,
    occurrence_count INTEGER DEFAULT 1,
    first_seen TEXT,
    last_seen TEXT,
    source_sessions TEXT,
    resolution_success INTEGER DEFAULT 0
);

-- Error solutions - links errors to specific solutions found
CREATE TABLE IF NOT EXISTS error_solutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_pattern_id INTEGER,
    solution_text TEXT NOT NULL,
    solution_type TEXT,
    confidence REAL DEFAULT 0.5,
    session_id TEXT,
    timestamp TEXT,
    FOREIGN KEY (error_pattern_id) REFERENCES error_patterns(id)
);

-- Decision journal - tracks architectural and design decisions
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_hash TEXT UNIQUE,
    decision_summary TEXT NOT NULL,
    decision_details TEXT,
    reasoning TEXT,
    alternatives_considered TEXT,
    context TEXT,
    category TEXT,
    outcome TEXT,
    session_id TEXT,
    timestamp TEXT,
    confidence REAL DEFAULT 0.5
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_error_sig ON error_patterns(error_signature);
CREATE INDEX IF NOT EXISTS idx_error_type ON error_patterns(error_type);
CREATE INDEX IF NOT EXISTS idx_decision_cat ON decisions(category);

-- Create FTS for error search
CREATE VIRTUAL TABLE IF NOT EXISTS errors_fts USING fts5(
    error_message,
    solution_summary,
    file_context,
    content='error_patterns',
    content_rowid='id'
);

-- Create FTS for decision search
CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
    decision_summary,
    reasoning,
    context,
    content='decisions',
    content_rowid='id'
);
"""


def init_insights_db():
    """Initialize the insights database for errors, decisions, etc."""
    db = get_db_manager()
    db.init_schema(INSIGHTS_DB, INSIGHTS_SCHEMA)
    log("Insights database initialized")


# =============================================================================
# ERROR PATTERN RECOGNITION
# =============================================================================

def normalize_error_message(error_msg: str) -> str:
    """
    Normalize an error message for comparison.

    Removes variable parts (line numbers, file paths, memory addresses, etc.)
    to allow matching similar errors across different contexts.
    """
    normalized = error_msg

    # Remove file paths (Unix and Windows) - FIRST to avoid path:line confusion
    normalized = re.sub(r'(?:/[^\s:,\)]+)+(?:\.[a-zA-Z0-9]+)?', '<FILE>', normalized)
    normalized = re.sub(r'[A-Z]:\\[^\s:,\)]+', '<FILE>', normalized)
    # Go-style paths: package/subpackage/file.go
    normalized = re.sub(r'\b[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)+\.go\b', '<FILE>', normalized)

    # Remove timestamps BEFORE line numbers (timestamps have :NN:NN that would match line patterns)
    normalized = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?', '<TIME>', normalized)
    normalized = re.sub(r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}', '<TIME>', normalized)
    normalized = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '[<TIME>]', normalized)

    # Remove port numbers BEFORE generic line numbers (ports are :NNNN at end of hostnames)
    normalized = re.sub(r'(localhost|[\w.-]+):(\d{4,5})\b', r'\1:<PORT>', normalized)

    # Remove IP addresses (before line numbers to avoid IP:port confusion)
    normalized = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', normalized)

    # Remove line/column numbers in various formats
    normalized = re.sub(r'line \d+', 'line <N>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r':\d+:\d+:\d+', ':<N>:<N>:<N>', normalized)  # file:line:col:pos
    normalized = re.sub(r':\d+:\d+', ':<N>:<N>', normalized)
    normalized = re.sub(r':\d+\b', ':<N>', normalized)
    normalized = re.sub(r'\bLine \d+\b', 'Line <N>', normalized)
    normalized = re.sub(r'\bline=\d+', 'line=<N>', normalized)
    normalized = re.sub(r'\brow \d+', 'row <N>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bcolumn \d+', 'column <N>', normalized, flags=re.IGNORECASE)

    # Remove memory addresses (various formats)
    normalized = re.sub(r'0x[0-9a-fA-F]+', '<ADDR>', normalized)
    normalized = re.sub(r'\bat [0-9a-fA-F]{8,16}\b', 'at <ADDR>', normalized)

    # Remove UUIDs
    normalized = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<UUID>', normalized, flags=re.IGNORECASE)

    # Remove process/thread IDs
    normalized = re.sub(r'\bpid[=: ]\d+', 'pid=<N>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\btid[=: ]\d+', 'tid=<N>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bthread[- ]?\d+', 'thread-<N>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bgoroutine \d+', 'goroutine <N>', normalized)

    # Remove error codes that are variable
    normalized = re.sub(r'\berror code[=: ]\d+', 'error code=<N>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\berrno[=: ]\d+', 'errno=<N>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bstatus[=: ]\d{3}\b', 'status=<N>', normalized, flags=re.IGNORECASE)

    # Remove specific variable values in quotes (long values only)
    normalized = re.sub(r"'[^']{20,}'", "'<VALUE>'", normalized)
    normalized = re.sub(r'"[^"]{20,}"', '"<VALUE>"', normalized)

    # Remove request/transaction IDs
    normalized = re.sub(r'\brequest[_-]?id[=: ][^\s,]+', 'request_id=<ID>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\btx[_-]?id[=: ][^\s,]+', 'tx_id=<ID>', normalized, flags=re.IGNORECASE)

    # Remove Java-style package paths but preserve class names
    normalized = re.sub(r'(?:at )?(?:[a-z][a-z0-9_]*\.)+([A-Z][a-zA-Z0-9_]*)', r'\1', normalized)

    return normalized.strip()


def extract_error_type(error_msg: str) -> Optional[str]:
    """
    Extract the error type from an error message.

    Supports many languages and formats:
    - Python: TypeError, ValueError, Exception
    - JavaScript: TypeError, ReferenceError, SyntaxError
    - Java: NullPointerException, IOException
    - Go: panic, fatal error
    - Rust: error[E####]
    - Ruby: NoMethodError, ArgumentError
    - And many more
    """
    # Ordered by specificity - most specific patterns first
    patterns = [
        # Python/Ruby/JS style - TypeNameError or TypeNameException
        r'^((?:[A-Z][a-z]+)+Error)(?:\s*:|\s*\()',
        r'^((?:[A-Z][a-z]+)+Exception)(?:\s*:|\s*\()',
        r'((?:[A-Z][a-z]+)+Error)\s*:',
        r'((?:[A-Z][a-z]+)+Exception)\s*:',

        # Go errors
        r'\b(panic):\s',
        r'^(fatal error):\s',
        r'\b(runtime error):\s',

        # Rust errors with codes
        r'(error\[E\d{4}\])',

        # Java/JVM specific
        r'(java\.lang\.\w+Exception)',
        r'(java\.io\.\w+Exception)',
        r'(java\.util\.\w+Exception)',
        r'(kotlin\.\w+Exception)',
        r'(scala\.\w+Error)',

        # C/C++ specific
        r'\b(segmentation fault)',
        r'\b(SIGSEGV|SIGABRT|SIGBUS|SIGFPE)',
        r'\b(undefined reference)',
        r'\b(core dumped)',

        # Database errors
        r'\b(SQLITE_\w+)',
        r'\b(ERROR \d{4,5})',  # MySQL error codes
        r'(MongoError|MongoServerError)',
        r'(PG::Error|PGError)',

        # Build tool errors
        r'(BUILD FAILURE)',
        r'(FAILURE: Build failed)',
        r'(CMake Error)',
        r'(error: linker)',

        # Package manager errors
        r'(npm ERR!)',
        r'(yarn error)',
        r'(ERR_PNPM_\w+)',
        r'(pip\._internal\.exceptions\.\w+)',

        # TypeScript errors
        r'(error TS\d{4})',

        # ESLint/linter errors
        r'(error\s+[a-z-]+/[a-z-]+)',  # eslint rule format

        # HTTP errors
        r'\b(HTTP [45]\d{2})',
        r'\b([45]\d{2} \w+)',  # "404 Not Found"

        # Cloud/infra errors
        r'(Error from server)',
        r'(AccessDenied|AccessDeniedException)',
        r'(ResourceNotFoundException)',

        # Generic fallbacks (less specific, checked last)
        r'^(Error):\s+[A-Z]',
        r'^(FATAL|Fatal):\s',
        r'^(FAILED|Failed):\s',
        r'\b(Error|Exception|Panic|Fatal|Failed)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, error_msg, re.IGNORECASE if 'IGNORECASE' in pattern else 0)
        if match:
            return match.group(1)

    return None


def compute_error_signature(error_msg: str) -> str:
    """Compute a signature for an error for deduplication."""
    normalized = normalize_error_message(error_msg)
    error_type = extract_error_type(error_msg) or "Unknown"

    msg_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]
    return f"{error_type}:{msg_hash}"


# =============================================================================
# ERROR PATTERN DEFINITIONS - Organized by confidence tier
# =============================================================================

# Tier 1: HIGH CONFIDENCE (0.95) - Very specific formats, almost no false positives
# These have unique syntax that doesn't appear in normal prose
ERROR_PATTERNS_TIER1 = [
    # Python tracebacks - unmistakable format
    (r'(Traceback \(most recent call last\):[\s\S]*?(?:\w+Error|\w+Exception):[^\n]+)', 'python_traceback'),

    # Rust errors with codes - error[E0001]: message
    (r'(error\[E\d{4}\]:[^\n]{10,200})', 'rust_error'),

    # TypeScript errors - error TS2345: message
    (r'(error TS\d{4}:[^\n]{10,200})', 'typescript_error'),

    # Go panic with goroutine dump
    (r'(panic:.*?goroutine \d+[^\n]*(?:\n[^\n]*){1,5})', 'go_panic'),

    # Java/JVM stack traces - "Exception in thread" or "at com.package.Class"
    (r'((?:Exception in thread "[^"]+"|Caused by:)\s+[\w.]+(?:Exception|Error):[^\n]*(?:\n\s+at [^\n]+){1,5})', 'java_stacktrace'),

    # MySQL error codes - ERROR 1045 (28000)
    (r'(ERROR \d{4} \(\d+\)[^\n]{10,150})', 'mysql_error'),

    # PostgreSQL errors - ERROR: message / FATAL: message
    (r'((?:ERROR|FATAL|PANIC):\s{2}[^\n]{10,200})', 'postgres_error'),

    # Docker daemon errors - unmistakable prefix
    (r'(Error response from daemon:[^\n]{10,200})', 'docker_error'),

    # Kubernetes errors
    (r'(Error from server \([^)]+\):[^\n]{10,200})', 'k8s_error'),

    # CMake errors with specific format
    (r'(CMake Error at [^\n]+:\d+[^\n]*(?:\n[^\n]{2,})*)', 'cmake_error'),

    # Make errors - make: *** [target] Error N
    (r'(make: \*\*\* \[[^\]]+\] Error \d+)', 'make_error'),

    # Linker errors - undefined reference to `symbol'
    (r"(undefined reference to [`'][^'`]+[`'][^\n]*)", 'linker_error'),

    # Segmentation fault with signal info
    (r'((?:Segmentation fault|SIGSEGV|SIGABRT|SIGBUS)(?:\s*\([^)]+\))?(?:[^\n]*core dumped)?)', 'segfault'),
]

# Tier 2: GOOD CONFIDENCE (0.85) - Specific prefixes with content requirements
ERROR_PATTERNS_TIER2 = [
    # Python-style errors - TypeNameError: message (requires content)
    (r'((?:[A-Z][a-z]+)+Error:\s+[^\n]{15,200})', 'python_error'),
    (r'((?:[A-Z][a-z]+)+Exception:\s+[^\n]{15,200})', 'python_exception'),

    # npm errors with content
    (r'(npm ERR! [A-Z][^\n]{15,150})', 'npm_error'),

    # yarn errors
    (r'(error An unexpected error occurred:[^\n]{10,150})', 'yarn_error'),

    # pnpm errors with specific prefix
    (r'(ERR_PNPM_[A-Z_]+[^\n]{10,150})', 'pnpm_error'),

    # pip errors
    (r'(ERROR: (?:Could not|Cannot|Failed to|No matching)[^\n]{10,150})', 'pip_error'),

    # Git fatal errors
    (r'(fatal: [^\n]{15,150})', 'git_fatal'),

    # Git errors (less specific than fatal)
    (r'(error: [a-z][^\n]{15,150})', 'git_error'),

    # Gradle build failures
    (r'(FAILURE: Build failed with an exception[^\n]*(?:\n[*>\s][^\n]*){1,5})', 'gradle_error'),

    # Maven build failures
    (r'(\[ERROR\] (?:Failed to execute|Could not|Cannot)[^\n]{10,150})', 'maven_error'),

    # Ruby errors
    (r'((?:NoMethodError|ArgumentError|NameError|RuntimeError|LoadError):[^\n]{10,150})', 'ruby_error'),

    # Elixir/Erlang errors
    (r'(\*\* \((?:\w+Error|\w+Exception)\)[^\n]{10,150})', 'elixir_error'),

    # PHP errors
    (r'((?:Fatal error|Parse error|Warning):\s+[^\n]{15,150}\s+in\s+[^\n]+)', 'php_error'),

    # C# / .NET errors
    (r'(System\.\w+Exception:[^\n]{10,150})', 'dotnet_error'),
    (r'(Unhandled exception\.[^\n]{10,150})', 'dotnet_unhandled'),

    # Swift/Objective-C
    (r'(Fatal error:[^\n]{15,150})', 'swift_fatal'),
    (r'(NSException[^\n]{10,150})', 'objc_exception'),

    # Terraform errors
    (r'(Error: (?:Invalid|Cannot|Failed|Missing|Error)[^\n]{15,150})', 'terraform_error'),

    # Ansible errors
    (r'(fatal: \[[^\]]+\]:[^\n]{10,150})', 'ansible_fatal'),

    # Shell/Bash errors
    (r'(bash: [^\n]{10,100}: (?:command not found|No such file|Permission denied))', 'bash_error'),
    (r'(zsh: [^\n]{10,100}: (?:command not found|no such file|permission denied))', 'zsh_error'),

    # SSL/TLS errors
    (r'(SSL[_: ](?:error|Error|ERROR)[^\n]{10,150})', 'ssl_error'),
    (r'(certificate verify failed[^\n]{0,100})', 'cert_error'),

    # CORS errors
    (r'((?:Access to|Cross-Origin|CORS)[^\n]*(?:blocked|denied|policy)[^\n]{10,100})', 'cors_error'),

    # MongoDB errors
    (r'(MongoError:[^\n]{10,150})', 'mongo_error'),
    (r'(MongoServerError:[^\n]{10,150})', 'mongo_server_error'),

    # Redis errors
    (r'((?:WRONGTYPE|ERR|NOSCRIPT)[^\n]{10,100})', 'redis_error'),

    # AWS errors
    (r'(An error occurred \([^)]+\)[^\n]{10,150})', 'aws_error'),
    (r'(AccessDenied(?:Exception)?:[^\n]{10,150})', 'aws_access_denied'),
]

# Tier 3: MODERATE CONFIDENCE (0.70) - More generic patterns, require additional validation
ERROR_PATTERNS_TIER3 = [
    # Generic Error: with capital letter start and minimum content
    (r'(Error: [A-Z][^\n]{20,150})', 'generic_error'),

    # Failed: patterns
    (r'((?:FAILED|Failed):\s+[^\n]{15,150})', 'failed_status'),

    # Build failures (generic)
    (r'(Build failed[^\n]{10,100})', 'build_failed'),

    # Connection errors
    (r'((?:Connection|Connect) (?:refused|failed|timed out|reset)[^\n]{0,100})', 'connection_error'),

    # Permission errors
    (r'(Permission denied[^\n]{0,100})', 'permission_error'),
    (r'(Access denied[^\n]{0,100})', 'access_denied'),

    # Not found errors
    (r'((?:File|Module|Package|Command|Resource) not found[^\n]{0,100})', 'not_found'),

    # Timeout errors
    (r'((?:Timeout|timed out)[^\n]{10,100})', 'timeout_error'),

    # Out of memory
    (r'((?:Out of memory|OOM|Cannot allocate memory)[^\n]{0,100})', 'oom_error'),

    # Assertion failures
    (r'((?:Assertion|Assert) failed[^\n]{10,150})', 'assertion_failed'),
    (r'(AssertionError:[^\n]{10,150})', 'assertion_error'),
]

# Combine all patterns with their confidence levels
ERROR_PATTERNS_ALL = (
    [(p, t, 0.95) for p, t in ERROR_PATTERNS_TIER1] +
    [(p, t, 0.85) for p, t in ERROR_PATTERNS_TIER2] +
    [(p, t, 0.70) for p, t in ERROR_PATTERNS_TIER3]
)


def _is_error_false_positive(error_text: str, full_content: str) -> bool:
    """
    Check if an error match is likely a false positive.

    Returns True if the match should be rejected.
    """
    error_lower = error_text.lower().strip()
    error_stripped = error_text.strip()
    content_lower = full_content.lower()

    # Too short to be a real error
    if len(error_stripped) < 10:
        return True

    # Markdown headers that contain error-like words
    if re.match(r'^#+\s', error_stripped):
        return True

    # Markdown list items about errors (not actual errors)
    if re.match(r'^[-*]\s*\*?\*?(?:errors?|error handling|error patterns)', error_lower):
        return True

    # Code comments explaining error handling
    if re.match(r'^\s*(//|#|/\*|\*|<!--)', error_stripped):
        return True

    # Code definitions - function/class/variable definitions containing "error"
    # Be careful not to reject actual error messages like "ERROR:" (Postgres) or "Error:" (generic)
    code_definition_patterns = [
        r'^(?:def|class|function|const|let|var|async)\s+\w*error',  # def handle_error(
        r'^[a-z_]+error\w*\s*[=(]',  # error_patterns( or error_handler = (lowercase = variable)
        r'^exception\s+as\s+\w+:',  # Exception as e:
        r'^\s*except\s+\w*(?:error|exception)',  # except ValueError:
        r'^\s*catch\s*\(',  # catch (
        r'^\s*try\s*:',  # try:
        r'^errors\s*[=\[]',  # errors = [] (but NOT "ERROR:" which is a real error prefix)
        r'^["\']errors?["\']\s*:',  # "errors": or 'error':
    ]
    for pattern in code_definition_patterns:
        if re.match(pattern, error_lower):
            return True

    # Documentation patterns
    doc_patterns = [
        r'^\s*\*\s',  # JSDoc
        r'^\s*:param',  # Python docstring
        r'^\s*@throws',  # Java doc
        r'^\s*@raises',  # Python doc
        r'^\s*raises:',  # Python docstring
        r'returns?\s+an?\s+error',  # "returns an error"
        r'throws?\s+an?\s+error',  # "throws an error"
        r'error\s+handling',  # discussion of error handling
        r'error\s+message',  # discussing error messages
        r'error\s+patterns?',  # discussing error patterns
        r'error\s+types?',  # discussing error types
    ]
    for pattern in doc_patterns:
        if re.search(pattern, error_lower):
            return True

    # Example/sample indicators
    example_patterns = [
        r'\bfor example\b',
        r'\bexample[,:]',
        r'\bsample\b',
        r'\bsuch as\b',
        r'\blike\b.{0,10}$',
        r'for instance',
        r'e\.g\.',
        r'you might see',
        r'you could get',
    ]
    error_pos = content_lower.find(error_lower[:30])
    if error_pos > 0:
        context_before = content_lower[max(0, error_pos - 50):error_pos]
        for pattern in example_patterns:
            if re.search(pattern, context_before):
                return True

    # Variable/function names that happen to contain "error"
    if re.match(r'^[a-z_]+error[a-z_]*\s*[=(]', error_stripped):  # snake_case variable
        return True
    if re.match(r'^[a-z]+Error[a-zA-Z]*\s*[=(]', error_stripped):  # camelCase function
        return True

    # Import statements
    if re.match(r'^\s*(import|from|require|use)\b', error_stripped):
        return True

    # Conditional/test patterns (checking for errors, not actual errors)
    check_patterns = [
        r'if.*error',
        r'when.*error',
        r'catch.*error',
        r'expect.*error',
        r'assert.*error',
        r'should.*error',
        r'test.*error',
    ]
    for pattern in check_patterns:
        if re.match(pattern, error_lower):
            return True

    # Code snippets with function calls - but NOT tracebacks
    # "Traceback (most recent call last):" is NOT a function call
    if not error_lower.startswith('traceback'):
        # Only reject if it looks like a function definition, not a traceback
        if re.match(r'^[a-z_]+\s*\([^)]*\)\s*:', error_lower):  # function(): pattern
            return True
    if error_lower.count('(') > 3:  # More than 3 parens = probably code
        return True

    # Markdown formatting - bold/italic section headers
    if re.match(r'^\*\*[^*]+\*\*:?\s*$', error_stripped):  # **Errors**:
        return True

    return False


def extract_errors_from_conversation(
    conversation: dict,
    session_id: str,
    project_path: str = None,
    storage=None
) -> int:
    """
    Extract error patterns and solutions from a conversation.

    Uses tiered pattern matching:
    - Tier 1 (0.95): Very specific formats (tracebacks, error codes)
    - Tier 2 (0.85): Language-specific patterns with content requirements
    - Tier 3 (0.70): Generic patterns with additional validation

    Applies false positive detection to filter out:
    - Markdown headers, code comments, documentation
    - Example/sample mentions
    - Error handling code (not actual errors)

    Uses central Postgres if available, falls back to local SQLite.
    """
    messages = conversation.get('messages', [])
    if len(messages) < 2:
        return 0

    # Get storage if not provided
    use_central = False
    project_id = None
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            pass

    if storage and storage.using_central and project_path:
        try:
            project_id = storage.postgres.get_or_create_project(project_path)
            use_central = True
        except Exception as e:
            log(f"Failed to get project_id for errors: {e}")

    db = get_db_manager()
    now = datetime.now().isoformat()

    errors_found = 0
    seen_signatures = set()  # Dedupe within this conversation

    try:
        for i, msg in enumerate(messages):
            if msg.get('role') != 'user':
                continue

            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join(
                    item.get('text', '') for item in content
                    if isinstance(item, dict) and item.get('type') == 'text'
                )

            # Skip very short messages
            if len(content) < 20:
                continue

            # Try each pattern tier, collecting matches with confidence
            for pattern, error_category, confidence in ERROR_PATTERNS_ALL:
                try:
                    # Use DOTALL for multiline patterns (tracebacks)
                    flags = re.DOTALL if r'[\s\S]' in pattern else 0
                    matches = re.findall(pattern, content, flags)

                    for error_match in matches:
                        # Handle tuple matches (from groups)
                        if isinstance(error_match, tuple):
                            error_msg = error_match[0].strip()
                        else:
                            error_msg = error_match.strip()

                        # Skip if too short
                        if len(error_msg) < 10:
                            continue

                        # Apply false positive detection
                        if _is_error_false_positive(error_msg, content):
                            continue

                        # Compute signature for deduplication
                        signature = compute_error_signature(error_msg)

                        # Skip if already seen in this conversation
                        if signature in seen_signatures:
                            continue
                        seen_signatures.add(signature)

                        error_type = extract_error_type(error_msg)

                        # Extract solution from next 2-3 assistant messages
                        # Real solutions often come after some investigation
                        solution_summary = None
                        for look_ahead in range(1, 4):  # Check next 3 messages
                            msg_idx = i + look_ahead
                            if msg_idx >= len(messages):
                                break
                            next_msg = messages[msg_idx]
                            if next_msg.get('role') != 'assistant':
                                continue

                            solution_details = next_msg.get('content', '')
                            if isinstance(solution_details, list):
                                solution_details = ' '.join(
                                    item.get('text', '') for item in solution_details
                                    if isinstance(item, dict) and item.get('type') == 'text'
                                )
                            candidate = _extract_solution_summary(solution_details)
                            if candidate:
                                solution_summary = candidate
                                break  # Found a solution, stop looking

                        # Store in central Postgres
                        if use_central and project_id:
                            try:
                                storage.postgres.upsert_error_pattern(
                                    project_id=project_id,
                                    signature=signature,
                                    error_type=error_type,
                                    error_text=error_msg[:500],
                                    solution=solution_summary,
                                )
                                errors_found += 1
                                continue
                            except Exception as e:
                                log(f"Central error storage failed: {e}")

                        # Fallback to local SQLite
                        normalized = normalize_error_message(error_msg)

                        def upsert_error(cursor):
                            cursor.execute("""
                                SELECT id, occurrence_count, source_sessions FROM error_patterns
                                WHERE error_signature = ?
                            """, (signature,))

                            row = cursor.fetchone()
                            if row:
                                pattern_id, count, sources = row
                                sources_list = json.loads(sources) if sources else []
                                if session_id not in sources_list:
                                    sources_list.append(session_id)

                                cursor.execute("""
                                    UPDATE error_patterns
                                    SET occurrence_count = ?, last_seen = ?, source_sessions = ?,
                                        solution_summary = COALESCE(?, solution_summary)
                                    WHERE id = ?
                                """, (count + 1, now, json.dumps(sources_list[-10:]),
                                      solution_summary, pattern_id))
                                return 0
                            else:
                                cursor.execute("""
                                    INSERT INTO error_patterns
                                    (error_signature, error_type, error_message, normalized_message,
                                     solution_summary, first_seen, last_seen, source_sessions)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (signature, error_type, error_msg[:500], normalized[:500],
                                      solution_summary, now, now, json.dumps([session_id])))

                                cursor.execute("""
                                    INSERT INTO errors_fts(rowid, error_message, solution_summary, file_context)
                                    VALUES (last_insert_rowid(), ?, ?, ?)
                                """, (error_msg[:500], solution_summary or '', ''))
                                return 1

                        errors_found += db.execute_write_func(INSIGHTS_DB, upsert_error)

                except Exception as pattern_error:
                    # Log but continue with other patterns
                    pass

    except Exception as e:
        log(f"Error extracting error patterns: {e}")

    return errors_found


def _extract_solution_summary(solution_text: str) -> Optional[str]:
    """Extract a brief summary of the solution from assistant response.

    Only extracts if the response actually contains solution-like content.
    Returns None if no clear solution is found.

    Pattern priority (IMPORTANT - order matters!):
    1. Command/code fixes (most concrete - "Run: pip install X")
    2. Action descriptions ("I changed...", "I added...")
    3. Direct fix statements ("The fix is...", "To fix this...")
    4. Root cause with action (only if contains action verb after explanation)

    NEVER extract pure problem descriptions without accompanying solutions.
    """
    if not solution_text or len(solution_text) < 30:
        return None

    # Expanded solution-indicating words
    solution_indicators = [
        'fix', 'solve', 'resolv', 'chang', 'updat', 'replac', 'add', 'remov',
        'correct', 'install', 'upgrad', 'downgrad', 'set', 'configur', 'modif',
        'adjust', 'run', 'execut', 'creat', 'delet', 'restart', 'reload',
        'clear', 'reset', 'rebuild', 'recompil', 'missing', 'need to', 'should',
    ]
    text_lower = solution_text.lower()
    if not any(ind in text_lower for ind in solution_indicators):
        return None

    # TIER 1: Command/code fixes (most concrete, highest priority)
    # These are actionable commands - extract them first
    command_patterns = [
        # Code blocks with shell commands
        r"```(?:bash|sh|shell|console|terminal)?\s*\n\s*([^\n`]{5,100})",
        # Inline code after action words
        r"(?:Run|Execute|Try|Install|Use)[:\s]+`([^`]{5,100})`",
        # Code block after action words
        r"(?:Run|Execute|Try|Install|Use)[:\s]*\n```[^\n]*\n\s*([^\n`]{5,100})",
    ]

    # TIER 2: Language-specific package manager commands (very concrete)
    package_manager_patterns = [
        # Full command capture (npm/yarn/pnpm)
        r"((?:npm|yarn|pnpm)\s+(?:install|add|update|remove|run|i)\s+[^\n]{3,80})",
        # pip commands
        r"(pip\s+(?:install|uninstall|upgrade)\s+[^\n]{3,80})",
        # apt/brew/yum
        r"((?:apt|apt-get|brew|yum|dnf)\s+(?:install|update|remove)\s+[^\n]{3,80})",
        # git commands
        r"(git\s+(?:checkout|reset|revert|pull|fetch|stash|clean)\s+[^\n]{3,80})",
        # docker commands
        r"(docker(?:-compose)?\s+(?:run|pull|build|up|down)\s+[^\n]{3,80})",
        # cargo commands
        r"(cargo\s+(?:add|install|update|build|run)\s+[^\n]{3,80})",
    ]

    # TIER 3: Explicit action descriptions (what the assistant DID)
    action_patterns = [
        # Past tense actions - very reliable
        r"I\s+(?:fixed|resolved|corrected|changed|updated|modified|added|removed|replaced|installed)\s+([^.!?\n]{10,150}[.!?]?)",
        # Present perfect - also reliable
        r"I(?:'ve| have)\s+(?:fixed|resolved|corrected|changed|updated|modified|added|removed)\s+([^.!?\n]{10,150}[.!?]?)",
        # Imperative sentence starters (Claude often uses these)
        r"(?:^|\n)\s*(?:Change|Update|Fix|Replace|Add|Remove|Modify|Delete|Create|Install|Run|Set)\s+([^.!?\n]{10,120}[.!?\n])",
    ]

    # TIER 4: Direct fix statements (explicit solution framing)
    fix_statement_patterns = [
        r"(?:The\s+(?:fix|solution)\s+(?:is|was))[:\s]+([^.!?\n]{15,180}[.!?])",
        r"(?:To\s+(?:fix|solve|resolve)\s+(?:this|the|it))[,:\s]+([^.!?\n]{15,180}[.!?])",
        r"(?:You\s+(?:need|should|must|can)\s+(?:to\s+)?(?:change|update|fix|add|remove|install|run))\s+([^.!?\n]{10,150}[.!?]?)",
    ]

    # TIER 5: Root cause WITH action (must have action verb AFTER the explanation)
    # These patterns require both explanation AND action in the same sentence
    root_cause_with_action_patterns = [
        # "The issue was X, so I did Y" - capture Y
        r"(?:The\s+(?:issue|problem|error)\s+(?:is|was)[^.]{10,80}),?\s+(?:so\s+)?I\s+([^.!?\n]{10,150}[.!?])",
        # "This happened because X. I fixed it by Y" - capture Y
        r"(?:because|due to)[^.]{10,80}\.\s*I\s+(?:fixed|resolved|changed|updated)\s+([^.!?\n]{10,150}[.!?])",
    ]

    # Combine in priority order
    all_patterns = (
        command_patterns +
        package_manager_patterns +
        action_patterns +
        fix_statement_patterns +
        root_cause_with_action_patterns
    )

    # Patterns that indicate NOT a solution (generic task continuation)
    non_solution_patterns = [
        r"^i'?ll\s+(?:continue|proceed|move|go|start|begin|check|look|examine|review|analyze|investigate)",
        r"^(?:continue|proceed|move|go|start|begin)\s+(?:with|to|on)",
        r"^(?:let me|let's)\s+(?:continue|proceed|move|check|look|examine)",
        r"^(?:now|next)\s+(?:i'?ll|let me|let's|we)",
        r"^testing\s+the",
        r"^verif(?:y|ying)\s+(?:the|that|all)",
        r"data validation",
        r"^checking\s+(?:if|the|that)",
    ]

    for pattern in all_patterns:
        match = re.search(pattern, solution_text, re.IGNORECASE | re.MULTILINE)
        if match:
            summary = match.group(1).strip()
            # Clean up: remove leading/trailing punctuation noise
            summary = re.sub(r'^[:\-\s]+', '', summary)
            summary = re.sub(r'[:\-\s]+$', '', summary)

            # Reject generic task continuation statements
            summary_lower = summary.lower()
            is_non_solution = any(re.match(p, summary_lower) for p in non_solution_patterns)
            if is_non_solution:
                continue  # Try next pattern

            # Lower minimum for commands (pip install X = 14 chars)
            min_len = 8 if any(pm in summary_lower for pm in ['npm', 'pip', 'apt', 'brew', 'git', 'cargo', 'docker']) else 12
            if min_len < len(summary) < 250:
                return summary

    # Don't fall back to arbitrary sentences - too noisy
    return None


def search_error_solutions(query: str, limit: int = 5, project_path: str = None, storage=None) -> List[Dict]:
    """
    Search for past error solutions matching the query.

    Uses central Postgres if available, falls back to local SQLite.

    Returns list of matching errors with their solutions.
    """
    # Try central Postgres first
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            pass

    if storage and storage.using_central:
        try:
            project_id = None
            if project_path:
                project_id = storage.postgres.get_or_create_project(project_path)
            results = storage.postgres.search_error_patterns(query, project_id=project_id, limit=limit)
            formatted = []
            for r in results:
                formatted.append({
                    'error_type': r.get('error_type'),
                    'error_message': r.get('error_text'),
                    'solution_summary': r.get('solution'),
                    'occurrence_count': r.get('occurrences', 1),
                    'last_seen': str(r.get('last_seen', '')),
                    'project_path': r.get('project_path', ''),
                })
            return formatted
        except Exception as e:
            log(f"Central error search failed: {e}")

    # Fallback to local SQLite
    return _search_error_solutions_local(query, limit)


def _search_error_solutions_local(query: str, limit: int = 5) -> List[Dict]:
    """Search local SQLite for error solutions."""
    db = get_db_manager()
    results = []

    def _extract_session_id(source_sessions_json: str) -> Optional[str]:
        """Extract most recent session_id from JSON array."""
        if not source_sessions_json:
            return None
        try:
            sessions = json.loads(source_sessions_json)
            return sessions[-1] if sessions else None  # Most recent is last
        except (json.JSONDecodeError, TypeError):
            return None

    try:
        terms = extract_query_terms(query, max_terms=5)
        if terms:
            fts_query = ' OR '.join(f'"{t}"' for t in terms)

            rows = db.execute_read(INSIGHTS_DB, """
                SELECT e.id, e.error_type, e.error_message, e.solution_summary,
                       e.solution_details, e.occurrence_count, e.last_seen, e.source_sessions
                FROM error_patterns e
                JOIN errors_fts f ON e.id = f.rowid
                WHERE errors_fts MATCH ?
                ORDER BY e.occurrence_count DESC
                LIMIT ?
            """, (fts_query, limit))

            for row in rows:
                results.append({
                    'id': row['id'],
                    'error_type': row['error_type'],
                    'error_message': row['error_message'],
                    'solution_summary': row['solution_summary'],
                    'solution_details': row['solution_details'][:500] if row['solution_details'] else None,
                    'occurrence_count': row['occurrence_count'],
                    'last_seen': row['last_seen'],
                    'session_id': _extract_session_id(row['source_sessions']),
                })

        if not results:
            normalized_query = normalize_error_message(query)
            rows = db.execute_read(INSIGHTS_DB, """
                SELECT id, error_type, error_message, solution_summary,
                       solution_details, occurrence_count, last_seen, source_sessions
                FROM error_patterns
                WHERE normalized_message LIKE ?
                ORDER BY occurrence_count DESC
                LIMIT ?
            """, (f'%{normalized_query[:50]}%', limit))

            for row in rows:
                results.append({
                    'id': row['id'],
                    'error_type': row['error_type'],
                    'error_message': row['error_message'],
                    'solution_summary': row['solution_summary'],
                    'solution_details': row['solution_details'][:500] if row['solution_details'] else None,
                    'occurrence_count': row['occurrence_count'],
                    'last_seen': row['last_seen'],
                    'session_id': _extract_session_id(row['source_sessions']),
                })

    except Exception as e:
        log(f"Error searching error solutions: {e}")

    return results


def get_error_stats() -> Dict:
    """Get statistics about stored error patterns."""
    db = get_db_manager()

    try:
        row = db.execute_read_one(INSIGHTS_DB, "SELECT COUNT(*) as cnt FROM error_patterns")
        total = row['cnt'] if row else 0

        rows = db.execute_read(INSIGHTS_DB, """
            SELECT error_type, COUNT(*) as cnt
            FROM error_patterns
            WHERE error_type IS NOT NULL
            GROUP BY error_type
            ORDER BY cnt DESC
            LIMIT 10
        """)
        by_type = {row['error_type']: row['cnt'] for row in rows}

        return {'total': total, 'by_type': by_type}
    except Exception as e:
        log(f"Error getting error stats: {e}")
        return {'total': 0, 'by_type': {}}


# =============================================================================
# DECISION JOURNAL
# =============================================================================

# Explicit decision recording patterns - scanned in USER messages
# These are intentional recording commands with high confidence
EXPLICIT_DECISION_PATTERNS = [
    # Tier 1: Direct recording commands (confidence 0.95)
    (r"(?:record (?:this )?decision|log (?:this )?decision|document (?:this )?decision|capture (?:this )?decision)[:\s]+(.{10,300})", 0.95),
    (r"(?:ADR|architecture decision(?: record)?|design decision|technical decision)[:\s-]+(.{10,300})", 0.95),
    (r"(?:decision|final decision|our decision|team decision)[:\s]+(.{10,300})", 0.95),
    (r"for the record[,:\s]+(.{10,300})", 0.95),
    (r"(?:MIRA,? record|for MIRA)[:\s]+(.{10,300})", 0.95),

    # Tier 2: Strong declarations (confidence 0.90)
    (r"(?:policy|rule|standard|convention|guideline)[:\s]+(.{10,300})", 0.90),
    (r"(?:going forward|from now on|henceforth)[,:\s]+(.{10,300})", 0.90),
    (r"(?:we commit to|our commitment is|we are committing to)[:\s]+(.{10,300})", 0.90),
    (r"(?:requirement|mandate|mandatory)[:\s]+(.{10,300})", 0.90),

    # Tier 3: Clear statements (confidence 0.85)
    (r"we (?:will always|will never|must always|must never)\s+(.{10,200})", 0.85),
    (r"(?:always use|never use|always do|never do)\s+(.{10,200})", 0.85),
    (r"(?:resolution|resolved|we've resolved that|conclusion|we conclude that)[:\s]+(.{10,300})", 0.85),
    (r"(?:chosen approach|selected approach|final choice)[:\s]+(.{10,300})", 0.85),
]

# Technical terms that make casual triggers more likely to be real decisions
DECISION_TECH_TERMS = [
    'api', 'database', 'db', 'code', 'function', 'class', 'file', 'config',
    'deploy', 'test', 'build', 'architecture', 'framework', 'library',
    'component', 'service', 'module', 'package', 'dependency', 'schema',
    'endpoint', 'route', 'model', 'controller', 'view', 'template',
    'query', 'cache', 'storage', 'auth', 'permission', 'security',
]


def _is_decision_false_positive(trigger_match: str, content: str) -> bool:
    """Check if a decision match is a false positive."""
    content_stripped = content.strip()

    # Questions are not decisions
    if content_stripped.endswith('?'):
        return True
    if re.search(r'\b(should we|what|how|which|can we|do we|are we)\b', content_stripped, re.I):
        return True

    # Hypotheticals are not decisions
    if re.match(r'^(if|could|might|would|may|maybe)\b', content_stripped, re.I):
        return True

    # Negations indicate no decision was made
    if re.search(r"\b(haven't|hasn't|not yet|isn't final|undecided|uncertain)\b", content_stripped, re.I):
        return True

    # Too short to be meaningful - require at least 20 chars for a real decision
    if len(content_stripped) < 20:
        return True

    # Fragments: markdown list items, section headers, bullet points
    if re.match(r'^[-*+]\s+', content_stripped):  # Bullet list fragments
        return True
    if re.match(r'^\d+\.\s+', content_stripped):  # Numbered list fragments
        return True
    if re.match(r'^#+\s', content_stripped):  # Headers
        return True

    # Incomplete sentences: starts lowercase, no verb-like words
    if content_stripped[0].islower():
        # Check if it has action/decision words that make it meaningful
        if not re.search(r'\b(use|using|implement|create|build|deploy|store|avoid|prefer|must|always|never|should|require)\b', content_stripped, re.I):
            return True

    # Reject if too few words (less than 4 words = fragment)
    word_count = len(content_stripped.split())
    if word_count < 4:
        return True

    # Skip if it looks like code
    if content_stripped.startswith('{') or content_stripped.startswith('['):
        return True
    if re.match(r'^(const|let|var|function|class|def|import)\b', content_stripped):
        return True

    return False


def _has_technical_content(content: str) -> bool:
    """Check if content contains technical terms."""
    content_lower = content.lower()
    return any(term in content_lower for term in DECISION_TECH_TERMS)


def extract_decisions_from_conversation(
    conversation: dict,
    session_id: str,
    project_path: str = None,
    postgres_session_id: int = None,
    storage=None
) -> int:
    """
    Extract architectural and design decisions from a conversation.

    Scans BOTH user and assistant messages:
    - User messages: Explicit recording commands (high confidence)
    - Assistant messages: Decision patterns and recommendations

    Uses central Postgres if available, falls back to local SQLite.
    """
    messages = conversation.get('messages', [])
    if not messages:
        return 0

    # Get storage if not provided
    use_central = False
    project_id = None
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            pass

    if storage and storage.using_central and project_path:
        try:
            project_id = storage.postgres.get_or_create_project(project_path)
            use_central = True
        except Exception as e:
            log(f"Failed to get project_id for decisions: {e}")

    db = get_db_manager()
    now = datetime.now().isoformat()

    decisions_found = 0
    seen_hashes = set()  # Dedupe within this conversation

    # Collect text from user and assistant messages separately
    user_text = []
    assistant_text = []
    for msg in messages:
        content = msg.get('content', '')
        if isinstance(content, list):
            content = ' '.join(
                item.get('text', '') for item in content
                if isinstance(item, dict) and item.get('type') == 'text'
            )
        if msg.get('role') == 'user':
            user_text.append(content)
        elif msg.get('role') == 'assistant':
            assistant_text.append(content)

    combined_user_text = '\n'.join(user_text)
    combined_assistant_text = '\n'.join(assistant_text)

    def store_decision(decision_text: str, context: str, confidence: float) -> bool:
        """Store a decision and return True if successful."""
        nonlocal decisions_found

        decision_hash = hashlib.md5(decision_text[:100].lower().encode()).hexdigest()

        # Skip duplicates within this conversation
        if decision_hash in seen_hashes:
            return False
        seen_hashes.add(decision_hash)

        category = _categorize_decision(decision_text)

        # Store in central Postgres
        if use_central and project_id:
            try:
                storage.postgres.insert_decision(
                    project_id=project_id,
                    decision=decision_text[:200],
                    category=category,
                    reasoning=context[:500] if context else None,
                    session_id=postgres_session_id,
                    confidence=confidence,
                )
                decisions_found += 1
                return True
            except Exception as e:
                log(f"Central decision storage failed: {e}")

        # Fallback to local SQLite
        def insert_decision(cursor):
            cursor.execute("SELECT id FROM decisions WHERE decision_hash = ?", (decision_hash,))
            if cursor.fetchone():
                return 0

            cursor.execute("""
                INSERT INTO decisions
                (decision_hash, decision_summary, context, category, session_id, timestamp, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (decision_hash, decision_text[:200], context[:500], category, session_id, now, confidence))

            cursor.execute("""
                INSERT INTO decisions_fts(rowid, decision_summary, reasoning, context)
                VALUES (last_insert_rowid(), ?, ?, ?)
            """, (decision_text[:200], '', context[:500]))
            return 1

        result = db.execute_write_func(INSIGHTS_DB, insert_decision)
        if result:
            decisions_found += 1
        return result > 0

    # ==========================================================================
    # PHASE 1: Scan USER messages for explicit recording patterns (high confidence)
    # ==========================================================================
    try:
        for pattern, confidence in EXPLICIT_DECISION_PATTERNS:
            matches = re.findall(pattern, combined_user_text, re.IGNORECASE | re.MULTILINE)
            for match in matches[:10]:  # Limit per pattern
                decision_text = match.strip()

                if _is_decision_false_positive(pattern, decision_text):
                    continue

                # For lower-confidence patterns, require technical content
                if confidence < 0.90 and not _has_technical_content(decision_text):
                    continue

                # Extract context around the decision
                context_match = re.search(
                    rf".{{0,100}}{re.escape(decision_text[:30])}.{{0,100}}",
                    combined_user_text,
                    re.IGNORECASE | re.DOTALL
                )
                context = context_match.group(0) if context_match else ""

                store_decision(decision_text, context, confidence)

    except Exception as e:
        log(f"Error extracting explicit decisions from user messages: {e}")

    # ==========================================================================
    # PHASE 2: Scan ASSISTANT messages for decision patterns (lower confidence)
    # ==========================================================================

    # Assistant decision patterns with confidence tiers
    # These are less explicit than user recording commands
    assistant_decision_patterns = [
        # Tier 4: Explicit decisions (0.75)
        (r"(?:I (?:decided|chose|went with)|We (?:decided|chose) to use|The (?:best|chosen) approach is)\s+([^.!?\n]{15,150}[.!?])", 0.75),
        # Tier 5: Recommendations (0.65)
        (r"(?:I recommend|I suggest|My recommendation is)\s+(using [^.!?\n]{10,100}|to use [^.!?\n]{10,100})", 0.65),
        # Tier 5: Alternative explanations (0.60)
        (r"(?:instead of|rather than)\s+(using \w+[^.!?\n]{10,60}(?:because|since|for)[^.!?\n]{10,60})", 0.60),
    ]

    try:
        for pattern, confidence in assistant_decision_patterns:
            matches = re.findall(pattern, combined_assistant_text, re.IGNORECASE)
            for match in matches[:5]:  # Limit per pattern
                decision_text = match.strip()

                if len(decision_text) < 15:
                    continue

                # Skip if too much code-like content
                if decision_text.count('(') > 2 or '{' in decision_text:
                    continue

                if _is_decision_false_positive(pattern, decision_text):
                    continue

                # Extract context around the decision
                context_match = re.search(
                    rf".{{0,100}}{re.escape(decision_text[:30])}.{{0,100}}",
                    combined_assistant_text,
                    re.IGNORECASE | re.DOTALL
                )
                context = context_match.group(0) if context_match else ""

                store_decision(decision_text, context, confidence)

    except Exception as e:
        log(f"Error extracting decisions from assistant messages: {e}")

    return decisions_found


def _categorize_decision(decision_text: str) -> str:
    """Categorize a decision based on its content."""
    text_lower = decision_text.lower()

    categories = {
        'architecture': ['architecture', 'structure', 'design pattern', 'layer', 'module', 'component'],
        'technology': ['library', 'framework', 'database', 'api', 'tool', 'package'],
        'implementation': ['implement', 'approach', 'method', 'function', 'class'],
        'testing': ['test', 'testing', 'coverage', 'mock', 'stub'],
        'security': ['security', 'auth', 'permission', 'encrypt', 'token'],
        'performance': ['performance', 'optimize', 'cache', 'speed', 'memory'],
        'workflow': ['workflow', 'process', 'deploy', 'ci', 'cd', 'pipeline'],
    }

    for category, keywords in categories.items():
        if any(kw in text_lower for kw in keywords):
            return category

    return 'general'


def search_decisions(
    query: str,
    category: Optional[str] = None,
    limit: int = 10,
    project_path: str = None,
    min_confidence: float = 0.0,
    storage=None
) -> List[Dict]:
    """
    Search for past decisions matching the query.

    Uses central Postgres if available, falls back to local SQLite.

    Args:
        query: Search query
        category: Optional category filter
        limit: Maximum results
        project_path: Optional project path filter
        min_confidence: Minimum confidence threshold (0.0-1.0). Default 0.0 returns all.
                       Recommended: 0.8 for explicit decisions, 0.6 for including implicit.
        storage: Storage instance

    Returns list of matching decisions with context.
    """
    # Try central Postgres first
    if storage is None:
        try:
            from .storage import get_storage
            storage = get_storage()
        except ImportError:
            pass

    if storage and storage.using_central:
        try:
            project_id = None
            if project_path:
                project_id = storage.postgres.get_or_create_project(project_path)
            results = storage.postgres.search_decisions(
                query=query,
                project_id=project_id,
                category=category,
                limit=limit
            )
            # Filter by min_confidence (central storage may not support this natively)
            formatted = []
            for r in results:
                conf = r.get('confidence', 0.5)
                if conf >= min_confidence:
                    formatted.append({
                        'decision': r.get('decision'),
                        'reasoning': r.get('reasoning'),
                        'category': r.get('category'),
                        'alternatives': r.get('alternatives', []),
                        'confidence': conf,
                        'created_at': str(r.get('created_at', '')),
                        'project_path': r.get('project_path', ''),
                    })
            return formatted
        except Exception as e:
            log(f"Central decision search failed: {e}")

    # Fallback to local SQLite
    return _search_decisions_local(query, category, limit, min_confidence)


def _search_decisions_local(query: str, category: Optional[str] = None, limit: int = 10, min_confidence: float = 0.0) -> List[Dict]:
    """Search local SQLite for decisions."""
    db = get_db_manager()
    results = []

    try:
        terms = extract_query_terms(query, max_terms=5)
        if terms:
            fts_query = ' OR '.join(f'"{t}"' for t in terms)

            if category:
                rows = db.execute_read(INSIGHTS_DB, """
                    SELECT d.id, d.decision_summary, d.context, d.category,
                           d.session_id, d.timestamp, d.confidence
                    FROM decisions d
                    JOIN decisions_fts f ON d.id = f.rowid
                    WHERE decisions_fts MATCH ? AND d.category = ? AND d.confidence >= ?
                    ORDER BY d.confidence DESC, d.timestamp DESC
                    LIMIT ?
                """, (fts_query, category, min_confidence, limit))
            else:
                rows = db.execute_read(INSIGHTS_DB, """
                    SELECT d.id, d.decision_summary, d.context, d.category,
                           d.session_id, d.timestamp, d.confidence
                    FROM decisions d
                    JOIN decisions_fts f ON d.id = f.rowid
                    WHERE decisions_fts MATCH ? AND d.confidence >= ?
                    ORDER BY d.confidence DESC, d.timestamp DESC
                    LIMIT ?
                """, (fts_query, min_confidence, limit))

            for row in rows:
                results.append({
                    'id': row['id'],
                    'decision': row['decision_summary'],
                    'context': row['context'],
                    'category': row['category'],
                    'session_id': row['session_id'],
                    'timestamp': row['timestamp'],
                    'confidence': row['confidence'],
                })

    except Exception as e:
        log(f"Error searching decisions: {e}")

    return results


def get_decision_stats() -> Dict:
    """Get statistics about stored decisions."""
    db = get_db_manager()

    try:
        row = db.execute_read_one(INSIGHTS_DB, "SELECT COUNT(*) as cnt FROM decisions")
        total = row['cnt'] if row else 0

        rows = db.execute_read(INSIGHTS_DB, """
            SELECT category, COUNT(*) as cnt
            FROM decisions
            GROUP BY category
            ORDER BY cnt DESC
        """)
        by_category = {row['category']: row['cnt'] for row in rows}

        return {'total': total, 'by_category': by_category}
    except Exception as e:
        log(f"Error getting decision stats: {e}")
        return {'total': 0, 'by_category': {}}


# =============================================================================
# COMBINED EXTRACTION (called during ingestion)
# =============================================================================

def extract_insights_from_conversation(
    conversation: dict,
    session_id: str,
    project_path: str = None,
    postgres_session_id: int = None,
    storage=None
):
    """
    Extract all insights (errors, decisions) from a conversation.

    Called during ingestion.

    Args:
        conversation: Parsed conversation dict
        session_id: String session ID
        project_path: Filesystem path to project (for Postgres storage)
        postgres_session_id: Postgres session ID (for foreign keys)
        storage: Storage instance
    """
    import time
    from .utils import log

    short_id = session_id[:12]
    msg_count = len(conversation.get('messages', []))

    t0 = time.time()
    errors = extract_errors_from_conversation(
        conversation, session_id,
        project_path=project_path,
        storage=storage
    )
    t_errors = (time.time() - t0) * 1000

    t0 = time.time()
    decisions = extract_decisions_from_conversation(
        conversation, session_id,
        project_path=project_path,
        postgres_session_id=postgres_session_id,
        storage=storage
    )
    t_decisions = (time.time() - t0) * 1000

    log(f"[{short_id}] Insights detail: {msg_count} msgs | errors={errors or 0} ({t_errors:.0f}ms) decisions={decisions or 0} ({t_decisions:.0f}ms)")

    return {
        'errors_found': errors or 0,
        'decisions_found': decisions or 0
    }
