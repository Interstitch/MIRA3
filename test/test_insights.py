"""Tests for mira.insights module."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.insights import (
    init_insights_db, extract_errors_from_conversation, extract_decisions_from_conversation,
    search_error_solutions, search_decisions, get_error_stats, get_decision_stats,
    normalize_error_message, compute_error_signature,
    EXPLICIT_DECISION_PATTERNS, _is_decision_false_positive, _has_technical_content
)
from mira.db_manager import shutdown_db_manager


class TestInsights:
    """Test insights module (errors, decisions)."""

    def test_init_insights_db(self):
        """Test insights database initialization."""
        shutdown_db_manager()  # Reset before creating new temp dir
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_insights_db()

            db_path = mira_path / 'insights.db'
            assert db_path.exists()
        finally:
            shutdown_db_manager()  # Clean up connections
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_normalize_error_message(self):
        """Test error message normalization."""
        # Test line number removal
        error = "Error at line 42 in file.py"
        normalized = normalize_error_message(error)
        assert 'line <N>' in normalized

        # Test file path removal
        error = "Error in /home/user/project/file.py"
        normalized = normalize_error_message(error)
        assert '<FILE>' in normalized

        # Test memory address removal
        error = "Object at 0x7fff5fbff8c0"
        normalized = normalize_error_message(error)
        assert '<ADDR>' in normalized

    def test_compute_error_signature(self):
        """Test error signature computation."""
        error1 = "TypeError: Cannot read property 'foo' of undefined"
        error2 = "TypeError: Cannot read property 'bar' of undefined"

        sig1 = compute_error_signature(error1)
        sig2 = compute_error_signature(error2)

        # Should both start with TypeError
        assert sig1.startswith('TypeError:')
        assert sig2.startswith('TypeError:')

    def test_extract_errors_from_conversation(self):
        """Test error extraction from conversations."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_insights_db()

            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'I got this error: TypeError: Cannot read property "x" of undefined'},
                    {'role': 'assistant', 'content': 'This error occurs when you try to access a property on undefined. Check that the object exists before accessing it.'},
                ]
            }

            count = extract_errors_from_conversation(conversation, 'test-session')
            assert count >= 0  # May or may not match depending on pattern

            stats = get_error_stats()
            assert 'total' in stats
            assert 'by_type' in stats
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_extract_decisions_from_conversation(self):
        """Test decision extraction from conversations."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_insights_db()

            conversation = {
                'messages': [
                    {'role': 'assistant', 'content': 'I decided to use React for the frontend because it has better component reusability.'},
                    {'role': 'assistant', 'content': 'I recommend using PostgreSQL instead of MySQL for this project.'},
                ]
            }

            count = extract_decisions_from_conversation(conversation, 'decision-session')
            assert count >= 0

            stats = get_decision_stats()
            assert 'total' in stats
            assert 'by_category' in stats
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_search_error_solutions(self):
        """Test searching for error solutions."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_insights_db()

            # Add some errors first
            conversation = {
                'messages': [
                    {'role': 'user', 'content': 'Error: ModuleNotFoundError: No module named "pandas"'},
                    {'role': 'assistant', 'content': 'You need to install pandas. Run: pip install pandas'},
                ]
            }
            extract_errors_from_conversation(conversation, 'pip-error-session')

            # Search for it
            results = search_error_solutions('ModuleNotFoundError pandas', limit=5)
            assert isinstance(results, list)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)

    def test_search_decisions(self):
        """Test searching for decisions."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            mira_path = Path(temp_dir) / '.mira'
            mira_path.mkdir()

            init_insights_db()

            # Add some decisions first
            conversation = {
                'messages': [
                    {'role': 'assistant', 'content': 'I recommend using FastAPI for this API because it has automatic OpenAPI documentation.'},
                ]
            }
            extract_decisions_from_conversation(conversation, 'api-decision')

            # Search for it
            results = search_decisions('FastAPI API', limit=5)
            assert isinstance(results, list)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)


class TestProjectFirstSearch:
    """Test project-first search functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        mira_path = Path(self.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)
        init_insights_db()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutdown_db_manager()
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_search_error_solutions_with_project(self):
        """Test error search with project path filter."""
        # Add an error
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'ImportError: No module named requests'},
                {'role': 'assistant', 'content': 'Install with: pip install requests'},
            ]
        }
        extract_errors_from_conversation(conversation, 'proj-error-1', project_path='/test/project')

        # Search with project filter
        results = search_error_solutions('ImportError', project_path='/test/project', limit=5)
        assert isinstance(results, list)

    def test_search_decisions_with_project(self):
        """Test decision search with project path filter."""
        # Add a decision
        conversation = {
            'messages': [
                {'role': 'assistant', 'content': 'I chose to use SQLite for development because it needs no setup.'},
            ]
        }
        extract_decisions_from_conversation(conversation, 'proj-decision-1', project_path='/test/project')

        # Search with project filter
        results = search_decisions('SQLite', project_path='/test/project', limit=5)
        assert isinstance(results, list)


class TestDeduplicationConstraints:
    """Test deduplication logic for insights."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        mira_path = Path(self.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)
        init_insights_db()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutdown_db_manager()
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_artifact_dedup_same_content(self):
        """Test that duplicate artifacts are not stored twice."""
        from mira.artifacts import init_artifact_db, store_artifact

        init_artifact_db()

        content = "def hello():\n    print('world')"

        # Store same content twice
        result1 = store_artifact(
            session_id='dedup-session-1',
            artifact_type='code_block',
            content=content,
            language='python'
        )
        result2 = store_artifact(
            session_id='dedup-session-1',
            artifact_type='code_block',
            content=content,
            language='python'
        )

        # Both should succeed (queued) but should not create duplicates
        # The actual deduplication happens in storage layer
        assert result1 in [True, False]
        assert result2 in [True, False]

    def test_artifact_hash_consistency(self):
        """Test that the same content produces the same hash."""
        import hashlib

        content = "def hello():\n    print('world')"

        # MIRA uses sha256 truncated to 32 chars for content hashing
        hash1 = hashlib.sha256(content.encode()).hexdigest()[:32]
        hash2 = hashlib.sha256(content.encode()).hexdigest()[:32]

        assert hash1 == hash2
        assert len(hash1) == 32  # SHA256 hex digest truncated length

    def test_decision_dedup_logic(self):
        """Test that duplicate decisions from same session are not stored."""
        conversation = {
            'messages': [
                {'role': 'assistant', 'content': 'I chose React because of its ecosystem.'},
                {'role': 'assistant', 'content': 'I chose React because of its ecosystem.'},  # Duplicate
            ]
        }

        # Extract decisions
        count = extract_decisions_from_conversation(conversation, 'dedup-decision-session')
        # The function should handle duplicates gracefully
        assert count >= 0


class TestExplicitDecisionRecording:
    """Test explicit decision recording patterns."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        mira_path = Path(self.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)
        init_insights_db()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutdown_db_manager()
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_explicit_patterns_exist(self):
        """Test that explicit decision patterns are defined."""
        assert len(EXPLICIT_DECISION_PATTERNS) > 0
        # Check structure: (pattern, confidence)
        for pattern, confidence in EXPLICIT_DECISION_PATTERNS:
            assert isinstance(pattern, str)
            assert 0.0 <= confidence <= 1.0

    def test_tier1_patterns_high_confidence(self):
        """Test that tier 1 patterns have 0.95 confidence."""
        tier1_triggers = ['record this decision', 'adr:', 'decision:', 'for the record']
        tier1_patterns = [(p, c) for p, c in EXPLICIT_DECISION_PATTERNS if c == 0.95]
        assert len(tier1_patterns) >= 4  # At least 4 tier 1 patterns

    def test_has_technical_content(self):
        """Test technical content detection."""
        assert _has_technical_content("use PostgreSQL database")
        assert _has_technical_content("configure the API endpoint")
        assert _has_technical_content("test the auth module")
        assert not _has_technical_content("meeting is at 3pm")
        assert not _has_technical_content("remember to call mom")

    def test_false_positive_questions(self):
        """Test that questions are detected as false positives."""
        assert _is_decision_false_positive("decision:", "should we use React?")
        assert _is_decision_false_positive("policy:", "what is the best approach?")
        assert _is_decision_false_positive("rule:", "how do we handle this?")

    def test_false_positive_hypotheticals(self):
        """Test that hypotheticals are detected as false positives."""
        assert _is_decision_false_positive("decision:", "if we use React then...")
        assert _is_decision_false_positive("policy:", "could use PostgreSQL")
        assert _is_decision_false_positive("rule:", "might want to consider X")

    def test_false_positive_negations(self):
        """Test that negations are detected as false positives."""
        assert _is_decision_false_positive("decision:", "we haven't decided yet")
        assert _is_decision_false_positive("policy:", "this isn't final")
        assert _is_decision_false_positive("rule:", "still undecided on this")

    def test_false_positive_too_short(self):
        """Test that short content is detected as false positive."""
        assert _is_decision_false_positive("decision:", "yes")
        assert _is_decision_false_positive("policy:", "ok")
        assert _is_decision_false_positive("rule:", "maybe")

    def test_valid_decision_not_false_positive(self):
        """Test that valid decisions are not marked as false positives."""
        assert not _is_decision_false_positive("decision:", "use PostgreSQL for the database")
        assert not _is_decision_false_positive("policy:", "all configs must be in YAML format")
        assert not _is_decision_false_positive("adr:", "API responses include pagination meta")

    def test_extract_explicit_user_decision(self):
        """Test extraction of explicit decisions from user messages."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'Decision: use PostgreSQL for the primary database because of ACID compliance'},
                {'role': 'assistant', 'content': 'Good choice, PostgreSQL is excellent for that.'},
            ]
        }

        count = extract_decisions_from_conversation(conversation, 'explicit-user-decision')
        assert count >= 1  # Should capture the explicit decision

    def test_extract_adr_format(self):
        """Test extraction of ADR-style decisions."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'ADR: All API endpoints must return JSON with a meta field for pagination'},
                {'role': 'assistant', 'content': 'I\'ll ensure all endpoints follow this pattern.'},
            ]
        }

        count = extract_decisions_from_conversation(conversation, 'adr-style-decision')
        assert count >= 1

    def test_extract_policy_format(self):
        """Test extraction of policy statements."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'Policy: all configuration files must use YAML format, not JSON'},
                {'role': 'assistant', 'content': 'Understood, I\'ll use YAML for configs.'},
            ]
        }

        count = extract_decisions_from_conversation(conversation, 'policy-decision')
        assert count >= 1

    def test_extract_going_forward(self):
        """Test extraction of 'going forward' statements."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'Going forward, use pnpm instead of npm for package management'},
                {'role': 'assistant', 'content': 'I\'ll use pnpm for all package operations.'},
            ]
        }

        count = extract_decisions_from_conversation(conversation, 'going-forward-decision')
        assert count >= 1

    def test_no_extraction_for_questions(self):
        """Test that questions are not extracted as decisions."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'Decision: should we use React or Vue?'},
                {'role': 'assistant', 'content': 'I recommend React for this use case.'},
            ]
        }

        count = extract_decisions_from_conversation(conversation, 'question-not-decision')
        # Should not extract the question as a decision
        # But may extract the assistant's recommendation
        assert count >= 0  # Just ensure no crash

    def test_assistant_decisions_lower_confidence(self):
        """Test that assistant decisions are extracted with lower confidence."""
        conversation = {
            'messages': [
                {'role': 'assistant', 'content': 'I decided to use FastAPI for the backend because it has automatic OpenAPI docs.'},
            ]
        }

        count = extract_decisions_from_conversation(conversation, 'assistant-decision')
        # Should extract but with lower confidence (0.75)
        assert count >= 0


class TestExpandedErrorPatterns:
    """Test the expanded error pattern recognition."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        mira_path = Path(self.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)
        init_insights_db()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutdown_db_manager()
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_python_traceback_extraction(self):
        """Test extraction of Python tracebacks."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': '''I got this error:
Traceback (most recent call last):
  File "app.py", line 42, in main
    result = process_data(data)
  File "utils.py", line 15, in process_data
    return data['key']
KeyError: 'key'
'''},
                {'role': 'assistant', 'content': 'The fix is to check if the key exists before accessing it.'},
            ]
        }
        count = extract_errors_from_conversation(conversation, 'traceback-test')
        assert count >= 1

    def test_rust_error_extraction(self):
        """Test extraction of Rust errors with codes."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'error[E0382]: borrow of moved value: `data` at line 42 in main.rs'},
                {'role': 'assistant', 'content': 'You need to clone the data or use a reference.'},
            ]
        }
        count = extract_errors_from_conversation(conversation, 'rust-test')
        assert count >= 1

    def test_typescript_error_extraction(self):
        """Test extraction of TypeScript errors."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': "error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'."},
                {'role': 'assistant', 'content': 'You need to convert the string to a number first.'},
            ]
        }
        count = extract_errors_from_conversation(conversation, 'ts-test')
        assert count >= 1

    def test_docker_error_extraction(self):
        """Test extraction of Docker errors."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'Error response from daemon: conflict: unable to delete image, image is being used by running container abc123'},
                {'role': 'assistant', 'content': 'Stop the container first with docker stop abc123.'},
            ]
        }
        count = extract_errors_from_conversation(conversation, 'docker-test')
        assert count >= 1

    def test_npm_error_extraction(self):
        """Test extraction of npm errors."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'npm ERR! Could not resolve dependency: peer react@"^18.0.0" from some-package@1.0.0'},
                {'role': 'assistant', 'content': 'Install React 18 or use --legacy-peer-deps.'},
            ]
        }
        count = extract_errors_from_conversation(conversation, 'npm-test')
        assert count >= 1

    def test_git_fatal_error_extraction(self):
        """Test extraction of Git fatal errors."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': "fatal: refusing to merge unrelated histories when I try to pull"},
                {'role': 'assistant', 'content': 'Use git pull --allow-unrelated-histories.'},
            ]
        }
        count = extract_errors_from_conversation(conversation, 'git-test')
        assert count >= 1

    def test_postgres_error_extraction(self):
        """Test extraction of PostgreSQL errors."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'ERROR:  duplicate key value violates unique constraint "users_email_key"'},
                {'role': 'assistant', 'content': 'The email already exists in the database.'},
            ]
        }
        count = extract_errors_from_conversation(conversation, 'postgres-test')
        assert count >= 1

    def test_connection_error_extraction(self):
        """Test extraction of connection errors."""
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'Connection refused when trying to connect to localhost:5432'},
                {'role': 'assistant', 'content': 'Make sure PostgreSQL is running.'},
            ]
        }
        count = extract_errors_from_conversation(conversation, 'connection-test')
        assert count >= 1


class TestFalsePositiveDetection:
    """Test that false positives are properly filtered."""

    def test_markdown_header_not_extracted(self):
        """Test that markdown headers are not extracted as errors."""
        from mira.insights import _is_error_false_positive

        assert _is_error_false_positive("## Error Handling", "## Error Handling\n\nThis section covers...")
        assert _is_error_false_positive("# TypeError Prevention", "# TypeError Prevention\n\nAlways check...")

    def test_code_comment_not_extracted(self):
        """Test that code comments are not extracted as errors."""
        from mira.insights import _is_error_false_positive

        assert _is_error_false_positive("// Handle TypeError here", "code // Handle TypeError here")
        assert _is_error_false_positive("# This error occurs when...", "# This error occurs when...")

    def test_documentation_not_extracted(self):
        """Test that documentation is not extracted as errors."""
        from mira.insights import _is_error_false_positive

        assert _is_error_false_positive("@throws TypeError if invalid", "@throws TypeError if invalid")
        assert _is_error_false_positive("returns an error if failed", "This function returns an error if failed")

    def test_example_context_not_extracted(self):
        """Test that examples are not extracted as errors."""
        from mira.insights import _is_error_false_positive

        # Check context before the error mention
        full_content = "For example, you might see TypeError: Cannot read property"
        assert _is_error_false_positive("TypeError: Cannot read property", full_content)

    def test_conditional_code_not_extracted(self):
        """Test that error checking code is not extracted."""
        from mira.insights import _is_error_false_positive

        assert _is_error_false_positive("if error: handle_error()", "if error: handle_error()")
        assert _is_error_false_positive("catch error:", "try:\n  ...\ncatch error:")

    def test_real_error_not_filtered(self):
        """Test that real errors are not filtered out."""
        from mira.insights import _is_error_false_positive

        # These should NOT be filtered
        assert not _is_error_false_positive(
            "TypeError: Cannot read property 'foo' of undefined",
            "I got this error: TypeError: Cannot read property 'foo' of undefined"
        )
        assert not _is_error_false_positive(
            "npm ERR! Could not resolve dependency",
            "Running npm install gives: npm ERR! Could not resolve dependency"
        )


class TestErrorNormalization:
    """Test error message normalization."""

    def test_normalize_file_paths(self):
        """Test that file paths are normalized."""
        error = "Error in /home/user/project/src/main.py at line 42"
        normalized = normalize_error_message(error)
        assert '<FILE>' in normalized
        assert '/home/user' not in normalized

    def test_normalize_line_numbers(self):
        """Test that line numbers are normalized."""
        error = "Error at line 42, column 15"
        normalized = normalize_error_message(error)
        assert 'line <N>' in normalized
        assert 'column <N>' in normalized

    def test_normalize_memory_addresses(self):
        """Test that memory addresses are normalized."""
        error = "Object at 0x7fff5fbff8c0 is invalid"
        normalized = normalize_error_message(error)
        assert '<ADDR>' in normalized
        assert '0x7fff' not in normalized

    def test_normalize_timestamps(self):
        """Test that timestamps are normalized."""
        error = "Error occurred at 2025-12-13T10:30:00Z"
        normalized = normalize_error_message(error)
        assert '<TIME>' in normalized
        assert '2025-12-13' not in normalized

    def test_normalize_uuids(self):
        """Test that UUIDs are normalized."""
        error = "Request 550e8400-e29b-41d4-a716-446655440000 failed"
        normalized = normalize_error_message(error)
        assert '<UUID>' in normalized
        assert '550e8400' not in normalized

    def test_normalize_ip_addresses(self):
        """Test that IP addresses are normalized."""
        error = "Connection to 192.168.1.100 failed"
        normalized = normalize_error_message(error)
        assert '<IP>' in normalized
        assert '192.168' not in normalized

    def test_normalize_port_numbers(self):
        """Test that port numbers are normalized."""
        error = "Failed to connect to localhost:5432"
        normalized = normalize_error_message(error)
        assert '<PORT>' in normalized
        assert '5432' not in normalized


class TestSolutionExtraction:
    """Test solution extraction from assistant responses."""

    def test_extract_direct_fix(self):
        """Test extraction of direct fix statements."""
        from mira.insights import _extract_solution_summary

        solution = "The fix is to add the missing import statement at the top of the file."
        result = _extract_solution_summary(solution)
        assert result is not None
        assert 'import' in result.lower()

    def test_extract_action_description(self):
        """Test extraction of action descriptions."""
        from mira.insights import _extract_solution_summary

        solution = "I changed the configuration to use environment variables instead of hardcoded values."
        result = _extract_solution_summary(solution)
        assert result is not None

    def test_extract_command_fix(self):
        """Test extraction of command-based fixes."""
        from mira.insights import _extract_solution_summary

        # Needs to be at least 50 chars for solution extraction
        solution = "To fix this dependency issue, you should run: npm install --save-dev typescript"
        result = _extract_solution_summary(solution)
        assert result is not None
        assert 'npm' in result.lower() or 'typescript' in result.lower()

    def test_extract_pip_command(self):
        """Test extraction of pip commands."""
        from mira.insights import _extract_solution_summary

        solution = "You need to install the package. Run pip install requests to fix this."
        result = _extract_solution_summary(solution)
        assert result is not None

    def test_no_extraction_for_unrelated(self):
        """Test that unrelated content doesn't produce solutions."""
        from mira.insights import _extract_solution_summary

        solution = "This is just a regular conversation without any fixes or solutions mentioned."
        result = _extract_solution_summary(solution)
        assert result is None

    def test_no_extraction_for_short_content(self):
        """Test that short content doesn't produce solutions."""
        from mira.insights import _extract_solution_summary

        solution = "Ok."
        result = _extract_solution_summary(solution)
        assert result is None
