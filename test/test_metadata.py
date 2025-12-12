"""Tests for mira.metadata module."""

from mira.metadata import (
    build_summary, extract_keywords, extract_key_facts,
    clean_task_description, extract_todo_topics, sample_messages_for_embedding,
    extract_metadata
)


class TestMetadata:
    """Test metadata extraction functions."""

    def test_clean_task_description_greeting(self):
        result = clean_task_description("Hi Claude, please help me fix this bug")
        assert result.startswith("Fix this bug") or "fix" in result.lower()
        assert "hi claude" not in result.lower()

    def test_clean_task_description_polite(self):
        result = clean_task_description("Can you please add a new feature?")
        assert "can you" not in result.lower()
        assert "please" not in result.lower()

    def test_extract_keywords_basic(self):
        messages = [
            {'content': 'import chromadb\nfrom sentence_transformers import SentenceTransformer'},
            {'content': 'def get_embedding_model():\n    pass'}
        ]
        keywords = extract_keywords(messages)
        assert len(keywords) > 0
        # Should extract package names and function names
        assert any('chromadb' in k.lower() for k in keywords) or \
               any('embedding' in k.lower() for k in keywords)

    def test_extract_key_facts_rules(self):
        messages = [
            {'role': 'assistant', 'content': 'You must always use HTTPS for API calls.'},
            {'role': 'assistant', 'content': 'Never store passwords in plain text.'}
        ]
        facts = extract_key_facts(messages)
        assert len(facts) > 0

    def test_extract_todo_topics(self):
        snapshots = [
            ('2025-12-07T10:00:00Z', [
                {'task': 'Implement authentication', 'status': 'pending'},
                {'task': 'Write tests', 'status': 'completed'}
            ])
        ]
        topics = extract_todo_topics(snapshots)
        assert 'Implement authentication' in topics
        assert 'Write tests' in topics

    def test_build_summary_with_existing(self):
        summary = build_summary([], '', 'Existing summary from Claude')
        assert summary == 'Existing summary from Claude'

    def test_build_summary_from_first_message(self):
        messages = [{'role': 'user', 'content': 'Fix the login bug'}]
        summary = build_summary(messages, 'Fix the login bug', '')
        assert 'login' in summary.lower() or 'fix' in summary.lower()

    def test_sample_messages_short_conversation(self):
        messages = [{'role': 'user', 'content': f'Message {i}'} for i in range(10)]
        sampled = sample_messages_for_embedding(messages)
        # Short conversations should keep all messages
        assert len(sampled) == 10

    def test_sample_messages_long_conversation(self):
        messages = [{'role': 'user', 'content': f'Message {i}'} for i in range(100)]
        sampled = sample_messages_for_embedding(messages)
        # Long conversations should be sampled
        assert len(sampled) < 100
        assert len(sampled) >= 15  # Should keep first 5 + last 10 at minimum

    def test_extract_metadata(self):
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'Fix the authentication bug'},
                {'role': 'assistant', 'content': 'I\'ll help fix that authentication issue.'}
            ],
            'summary': 'Auth bug fix',
            'first_user_message': 'Fix the authentication bug',
            'todo_snapshots': [],
            'session_meta': {
                'slug': 'test-session',
                'git_branch': 'main',
                'models_used': ['claude-3-opus'],
                'tools_used': {'Read': 2, 'Edit': 1},
                'files_touched': ['/src/auth.py']
            }
        }
        file_info = {
            'session_id': 'test-123',
            'project_path': '-workspaces-test',
            'last_modified': '2025-12-07T10:00:00Z'
        }
        metadata = extract_metadata(conversation, file_info)
        assert 'summary' in metadata
        assert 'keywords' in metadata
        assert 'task_description' in metadata
        assert metadata['session_id'] == 'test-123'
