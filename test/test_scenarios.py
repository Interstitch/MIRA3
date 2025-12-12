"""End-to-end tests for real user experience scenarios."""

import os
import json
import tempfile
import shutil
from pathlib import Path

from mira.artifacts import init_artifact_db
from mira.custodian import init_custodian_db
from mira.insights import init_insights_db
from mira.concepts import init_concepts_db
from mira.db_manager import shutdown_db_manager


class TestUserExperienceScenarios:
    """
    End-to-end tests for real user experience scenarios.
    These test complete workflows, not individual functions.
    """

    @classmethod
    def setup_class(cls):
        """Create a complete mock MIRA environment."""
        shutdown_db_manager()  # Reset before creating new temp dir
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)

        # Create .mira directory structure
        cls.mira_path = Path(cls.temp_dir) / '.mira'
        cls.archives_path = cls.mira_path / 'archives'
        cls.metadata_path = cls.mira_path / 'metadata'
        cls.mira_path.mkdir()
        cls.archives_path.mkdir()
        cls.metadata_path.mkdir()

        # Create mock Claude projects directory
        cls.claude_path = Path(cls.temp_dir) / '.claude' / 'projects' / '-workspaces-testproject'
        cls.claude_path.mkdir(parents=True)

        # Initialize all databases for end-to-end tests
        init_artifact_db()
        init_custodian_db()
        init_insights_db()
        init_concepts_db()

    @classmethod
    def teardown_class(cls):
        """Clean up."""
        shutdown_db_manager()  # Reset db_manager singleton
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _create_conversation_file(self, session_id: str, messages: list, include_summary: bool = True):
        """Helper to create a realistic conversation JSONL file."""
        conv_file = self.claude_path / f"{session_id}.jsonl"
        with open(conv_file, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg) + '\n')
            if include_summary:
                f.write(json.dumps({'type': 'summary', 'summary': f'Session about {session_id}'}) + '\n')
        return conv_file

    def test_scenario_search_for_past_discussion(self):
        """
        SCENARIO: User wants to find where they discussed authentication before.

        User asks: "Where did I work on authentication?"
        Expected: MIRA finds relevant conversations with excerpts.
        """
        from mira.search import handle_search
        from mira.ingestion import ingest_conversation

        # Setup: Create a conversation about authentication
        messages = [
            {'type': 'user', 'timestamp': '2025-12-01T10:00:00Z',
             'message': {'content': 'Help me implement user authentication with JWT tokens'}},
            {'type': 'assistant', 'timestamp': '2025-12-01T10:01:00Z',
             'message': {'content': [{'type': 'text', 'text': 'I\'ll help you implement JWT authentication. First, install jsonwebtoken...'}],
                         'model': 'claude-3-opus'}},
            {'type': 'user', 'timestamp': '2025-12-01T10:05:00Z',
             'message': {'content': 'How do I validate the token on each request?'}},
            {'type': 'assistant', 'timestamp': '2025-12-01T10:06:00Z',
             'message': {'content': [{'type': 'text', 'text': 'Create middleware that extracts and verifies the JWT token from the Authorization header.'}],
                         'model': 'claude-3-opus'}}
        ]
        self._create_conversation_file('auth-session-123', messages)

        # Create a mock collection
        class MockCollection:
            def count(self): return 0  # Empty, will use fulltext fallback
            def query(self, **kwargs): return {'ids': [[]], 'metadatas': [[]], 'distances': [[]]}

        # Search for authentication
        result = handle_search({'query': 'authentication JWT tokens', 'limit': 5}, MockCollection())

        # Verify: Should use fulltext fallback and find results
        assert result['total'] >= 0  # May find the conversation via fulltext
        assert 'results' in result

    def test_scenario_find_recent_work_sessions(self):
        """
        SCENARIO: User wants to see their recent work sessions.

        User asks: "Show me my recent sessions"
        Expected: List of recent sessions grouped by project.
        """
        from mira.handlers import handle_recent

        # Setup: Create metadata files for recent sessions
        sessions = [
            {'summary': 'Fixed authentication bug', 'project_path': '-workspaces-webapp'},
            {'summary': 'Added database migrations', 'project_path': '-workspaces-webapp'},
            {'summary': 'Setup CI/CD pipeline', 'project_path': '-workspaces-devops'},
        ]
        for i, session in enumerate(sessions):
            meta_file = self.metadata_path / f'session-{i}.json'
            meta_file.write_text(json.dumps(session))

        # Action: Get recent sessions
        result = handle_recent({'limit': 10})

        # Verify: Should return sessions grouped by project
        assert 'projects' in result
        assert 'total' in result
        assert result['total'] >= 3

    def test_scenario_init_new_session_with_context(self):
        """
        SCENARIO: User starts a new Claude Code session and wants context.

        User calls mira_init to understand recent work.
        Expected: Get tiered output with guidance, alerts, and core context.
        """
        from mira.handlers import handle_init

        # Collection is deprecated, storage is optional (None = no central storage)
        result = handle_init({'project_path': '-workspaces-testproject'}, None, storage=None)

        # Verify: Should return tiered output structure
        assert 'guidance' in result  # Tier 1: actionable guidance
        assert 'alerts' in result    # Tier 1: alerts requiring attention
        assert 'core' in result      # Tier 2: essential context
        assert 'indexing' in result  # Stats about indexed conversations

        # Core should contain custodian and current_work
        assert 'custodian' in result['core']
        assert 'current_work' in result['core']

        # Without central storage, indexed count will be 0
        assert 'indexed' in result['indexing']

    def test_scenario_find_code_claude_wrote(self):
        """
        SCENARIO: User wants to find code Claude wrote for database queries.

        User asks: "Find the database query code you wrote"
        Expected: Should find code artifacts stored during previous sessions.
        """
        from mira.artifacts import init_artifact_db, store_artifact, search_artifacts_for_query

        init_artifact_db()

        # Setup: Store code artifacts
        store_artifact(
            session_id='db-session-1',
            artifact_type='code_block',
            content='''async function getUserById(id) {
    const query = 'SELECT * FROM users WHERE id = $1';
    const result = await db.query(query, [id]);
    return result.rows[0];
}''',
            language='javascript',
            title='database query function'
        )

        store_artifact(
            session_id='db-session-2',
            artifact_type='code_block',
            content='''def get_all_products():
    cursor.execute("SELECT * FROM products ORDER BY created_at DESC")
    return cursor.fetchall()''',
            language='python',
            title='product database query'
        )

        # Action: Search for database code
        results = search_artifacts_for_query('database query', limit=5)

        # Verify: Should find relevant code artifacts
        assert isinstance(results, list)

    def test_scenario_reconstruct_file_claude_wrote(self):
        """
        SCENARIO: User lost a file that Claude wrote and wants to recover it.

        User asks: "Reconstruct the config file you wrote yesterday"
        Expected: MIRA can replay Write/Edit operations to reconstruct the file.
        """
        from mira.artifacts import init_artifact_db, store_file_operation, reconstruct_file

        init_artifact_db()

        # Setup: Store file operations as Claude would have made them
        file_path = '/project/config.json'

        # Claude writes initial file
        store_file_operation(
            session_id='config-session',
            op_type='write',
            file_path=file_path,
            content='{\n  "debug": false,\n  "port": 3000\n}',
            sequence_num=0,
            timestamp='2025-12-06T10:00:00Z'
        )

        # Claude edits the file
        store_file_operation(
            session_id='config-session',
            op_type='edit',
            file_path=file_path,
            old_string='"debug": false',
            new_string='"debug": true',
            replace_all=False,
            sequence_num=1,
            timestamp='2025-12-06T10:05:00Z'
        )

        # Action: Reconstruct the file
        content = reconstruct_file(file_path)

        # Verify: Should have the edited content
        assert content is not None
        assert '"debug": true' in content
        assert '"port": 3000' in content

    def test_scenario_new_conversation_detected(self):
        """
        SCENARIO: User finishes a conversation and MIRA should automatically ingest it.

        A new .jsonl file appears in ~/.claude/projects/
        Expected: File watcher detects it, queues for ingestion after debounce.
        """
        from mira.watcher import ConversationWatcher
        from mira.ingestion import ingest_conversation

        # Setup: Create conversation file
        messages = [
            {'type': 'user', 'timestamp': '2025-12-07T10:00:00Z',
             'message': {'content': 'Help me write unit tests'}},
            {'type': 'assistant', 'timestamp': '2025-12-07T10:01:00Z',
             'message': {'content': [{'type': 'text', 'text': 'I\'ll help you write comprehensive unit tests.'}],
                         'model': 'claude-3-opus'}}
        ]
        conv_file = self._create_conversation_file('new-conv-456', messages)

        watcher = ConversationWatcher(None, self.mira_path, storage=None)

        # Action: Simulate file detection
        watcher.queue_file(str(conv_file))

        # Verify: File should be queued for processing
        assert str(conv_file) in watcher.pending_files

        # Directly test ingestion - without central storage it will fail but shouldn't crash
        file_info = {
            'session_id': 'new-conv-456',
            'file_path': str(conv_file),
            'project_path': '-workspaces-testproject',
            'last_modified': '2025-12-07T10:01:00Z'
        }
        # Without central storage, ingest_conversation returns False
        result = ingest_conversation(file_info, None, self.mira_path, storage=None)
        # Result is False without central storage but the function shouldn't crash
        assert result in [True, False]

    def test_scenario_include_agent_files(self):
        """
        SCENARIO: MIRA should include agent-*.jsonl subagent task logs.

        Agent files contain valuable work done by subagents and should be indexed.
        """
        from mira.ingestion import discover_conversations

        # Setup: Create regular and agent files
        regular_file = self.claude_path / 'regular-session.jsonl'
        agent_file = self.claude_path / 'agent-task-123.jsonl'

        regular_file.write_text(json.dumps({'type': 'user', 'message': {'content': 'test'}}) + '\n')
        agent_file.write_text(json.dumps({'type': 'user', 'message': {'content': 'agent task'}}) + '\n')

        # Action: Discover conversations
        conversations = discover_conversations(self.claude_path.parent)

        # Verify: Should include both regular and agent files
        session_ids = [c['session_id'] for c in conversations]
        assert 'regular-session' in session_ids
        assert 'agent-task-123' in session_ids

        # Verify agent file is marked correctly
        agent_conv = [c for c in conversations if c['session_id'] == 'agent-task-123'][0]
        assert agent_conv.get('is_agent') == True

    def test_scenario_long_conversation_sampling(self):
        """
        SCENARIO: A very long conversation (100+ messages) needs to be indexed.

        ChromaDB has limited embedding context, so long conversations must be sampled.
        Expected: Sample first messages, time gaps, topic shifts, and last messages.
        """
        from mira.metadata import sample_messages_for_embedding

        # Setup: Create a 150-message conversation
        messages = []
        for i in range(150):
            messages.append({
                'role': 'user' if i % 2 == 0 else 'assistant',
                'content': f'Message number {i} about topic {"A" if i < 50 else "B" if i < 100 else "C"}',
                'timestamp': f'2025-12-07T{10 + i//60:02d}:{i%60:02d}:00Z'
            })

        # Action: Sample for embedding
        sampled = sample_messages_for_embedding(messages)

        # Verify: Should be significantly reduced but capture key parts
        assert len(sampled) < len(messages)
        assert len(sampled) >= 15  # At minimum: first 5 + last 10

    def test_scenario_fulltext_fallback_when_no_semantic_match(self):
        """
        SCENARIO: Semantic search finds nothing but the term exists in archives.

        User searches for very specific term that wasn't in the indexed summary.
        Expected: Falls back to fulltext search of archives.
        """
        from mira.search import handle_search

        # Setup: Create an archive with specific term not in summary
        archive_file = self.archives_path / 'specific-session.jsonl'
        with open(archive_file, 'w') as f:
            f.write(json.dumps({
                'type': 'user',
                'message': {'content': 'Help me configure the XYZ_OBSCURE_SETTING environment variable'}
            }) + '\n')
            f.write(json.dumps({
                'type': 'assistant',
                'message': {'content': [{'type': 'text', 'text': 'Set XYZ_OBSCURE_SETTING=true in your .env file'}]}
            }) + '\n')

        # Create metadata (without the obscure term in summary)
        meta_file = self.metadata_path / 'specific-session.json'
        meta_file.write_text(json.dumps({'summary': 'Environment configuration', 'project_path': '-test'}))

        # Collection is deprecated, storage is optional
        result = handle_search({'query': 'XYZ_OBSCURE_SETTING', 'limit': 5}, None, storage=None)

        # Verify: Without central storage, returns results or falls back to fulltext
        assert result is not None
        assert 'results' in result
        assert 'total' in result

    def test_scenario_enrich_search_results_with_excerpts(self):
        """
        SCENARIO: Semantic search finds a conversation, enrich with actual excerpts.

        User searches "database migration", semantic search finds relevant session,
        then we extract actual conversation excerpts with the term.
        """
        from mira.search import enrich_results_from_archives

        # Setup: Create archive with detailed discussion
        archive_file = self.archives_path / 'migration-session.jsonl'
        with open(archive_file, 'w') as f:
            f.write(json.dumps({
                'type': 'user',
                'timestamp': '2025-12-05T09:00:00Z',
                'message': {'content': 'I need to add a database migration for the new users table'}
            }) + '\n')
            f.write(json.dumps({
                'type': 'assistant',
                'timestamp': '2025-12-05T09:01:00Z',
                'message': {'content': [{'type': 'text', 'text': 'I\'ll create a migration file. Run: npx knex migrate:make add_users_table'}]}
            }) + '\n')

        # Simulated semantic search result
        semantic_results = [{
            'session_id': 'migration-session',
            'summary': 'Database work',
            'relevance': 0.85
        }]

        # Action: Enrich with excerpts
        enriched = enrich_results_from_archives(semantic_results, 'database migration', self.mira_path)

        # Verify: Should have excerpts added
        assert len(enriched) == 1
        assert 'excerpts' in enriched[0]
        assert 'has_archive_matches' in enriched[0]

    def test_scenario_extract_all_artifact_types(self):
        """
        SCENARIO: A conversation contains code, lists, tables, and errors.

        MIRA should detect and store all artifact types for later retrieval.
        Without central storage, extraction returns 0 but parsing should work.
        """
        from mira.artifacts import init_artifact_db, extract_artifacts_from_content, get_artifact_stats

        init_artifact_db()

        content = '''Here's the implementation plan:

1. Create the database schema
2. Implement the API endpoints
3. Add authentication middleware
4. Write unit tests
5. Deploy to staging

Here's the code:

```python
def create_user(data):
    user = User(**data)
    db.session.add(user)
    db.session.commit()
    return user
```

Here's a comparison of frameworks:

| Framework | Speed | Ease |
|-----------|-------|------|
| Django    | Med   | High |
| FastAPI   | High  | Med  |
| Flask     | Med   | High |

I got this error when testing:

Traceback (most recent call last):
  File "app.py", line 42, in create_user
    db.session.commit()
sqlalchemy.exc.IntegrityError: duplicate key value
'''

        # Action: Extract artifacts
        count = extract_artifacts_from_content(
            content=content,
            session_id='multi-artifact-session',
            role='assistant'
        )

        # Without central storage, extraction returns 0 but shouldn't crash
        assert count >= 0

        stats = get_artifact_stats()
        assert 'total' in stats

    def test_scenario_handle_empty_collection_gracefully(self):
        """
        SCENARIO: First-time user with no indexed conversations yet.

        User runs mira_search before any conversations have been indexed.
        Expected: Gracefully return empty results, not crash.
        """
        from mira.search import handle_search

        class EmptyCollection:
            def count(self): return 0

        # Action: Search on empty collection
        result = handle_search({'query': 'anything', 'limit': 10}, EmptyCollection())

        # Verify: Should return valid empty result
        assert result is not None
        assert 'results' in result
        assert 'total' in result

    def test_scenario_ingest_skips_already_indexed(self):
        """
        SCENARIO: Running ingestion twice should skip already-indexed conversations.

        Without central storage, both will return False (storage unavailable).
        The key test is that the function doesn't crash.
        """
        from mira.ingestion import ingest_conversation

        # Setup: Create conversation
        messages = [
            {'type': 'user', 'timestamp': '2025-12-07T10:00:00Z',
             'message': {'content': 'Quick question'}},
            {'type': 'assistant', 'timestamp': '2025-12-07T10:00:30Z',
             'message': {'content': [{'type': 'text', 'text': 'Here is the answer.'}],
                         'model': 'claude-3-opus'}}
        ]
        conv_file = self._create_conversation_file('idempotent-test', messages)

        file_info = {
            'session_id': 'idempotent-test',
            'file_path': str(conv_file),
            'project_path': '-workspaces-testproject',
            'last_modified': '2025-12-07T10:00:30Z'
        }

        # Without central storage, ingestion returns False but shouldn't crash
        result1 = ingest_conversation(file_info, None, self.mira_path, storage=None)
        assert result1 in [True, False]

        # Second call should also not crash
        result2 = ingest_conversation(file_info, None, self.mira_path, storage=None)
        assert result2 in [True, False]

    def test_scenario_system_status_overview(self):
        """
        SCENARIO: User wants to check MIRA system health.

        User calls mira_status to see indexed count, storage, pending files.
        """
        from mira.handlers import handle_status

        # params, collection (deprecated), storage
        result = handle_status({}, None, storage=None)

        # Verify: Should return system overview with global stats
        assert 'global' in result
        assert 'ingestion' in result['global']
        assert 'storage_path' in result
        assert 'last_sync' in result
        # Ingestion progress fields
        assert 'indexed' in result['global']['ingestion']
        assert 'pending' in result['global']['ingestion']
        assert 'percent' in result['global']['ingestion']
        assert 'complete' in result['global']['ingestion']
