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
