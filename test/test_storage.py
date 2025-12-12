"""Tests for mira.storage and mira.local_store modules."""

import os
import json
import tempfile
import shutil
from pathlib import Path

from mira.db_manager import shutdown_db_manager


class TestLocalStore:
    """Test local SQLite storage module."""

    @classmethod
    def setup_class(cls):
        # Reset db_manager to ensure fresh state
        shutdown_db_manager()
        cls.mira_path = Path(tempfile.mkdtemp())
        os.environ['MIRA_PATH'] = str(cls.mira_path)
        # Initialize local store
        from mira import local_store
        local_store._initialized = False  # Reset initialization flag
        local_store.init_local_db()

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        if cls.mira_path.exists():
            shutil.rmtree(cls.mira_path, ignore_errors=True)
        if 'MIRA_PATH' in os.environ:
            del os.environ['MIRA_PATH']

    def test_local_store_init(self):
        """Test local store database initialization."""
        from mira.local_store import LOCAL_DB
        from mira.db_manager import get_db_manager

        db = get_db_manager()

        # Verify tables exist
        tables = db.execute_read(LOCAL_DB,
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", ())
        table_names = [t['name'] for t in tables]
        assert 'projects' in table_names
        assert 'sessions' in table_names
        assert 'archives' in table_names
        assert 'custodian' in table_names

    def test_local_project_operations(self):
        """Test get_or_create_project in local store."""
        from mira.local_store import get_or_create_project

        # Create project
        project_id = get_or_create_project('/workspaces/test', slug='test')
        assert project_id > 0

        # Get same project
        project_id2 = get_or_create_project('/workspaces/test')
        assert project_id2 == project_id

        # Create project with git remote
        project_id3 = get_or_create_project(
            '/other/path', git_remote='git@github.com:user/repo.git'
        )
        assert project_id3 > 0
        assert project_id3 != project_id

    def test_local_session_operations(self):
        """Test session upsert and retrieval in local store."""
        from mira.local_store import get_or_create_project, upsert_session, get_recent_sessions

        project_id = get_or_create_project('/workspaces/test2')

        # Create session
        session_id = upsert_session(
            project_id=project_id,
            session_id='test-session-123',
            summary='Test session summary',
            keywords=['test', 'local'],
            task_description='Testing local storage',
            message_count=5,
        )
        assert session_id > 0

        # Get recent sessions
        sessions = get_recent_sessions(project_id=project_id, limit=10)
        assert len(sessions) >= 1
        assert any(s['session_id'] == 'test-session-123' for s in sessions)

    def test_local_archive_operations(self):
        """Test archive storage in local store."""
        from mira.local_store import (
            get_or_create_project, upsert_session, upsert_archive, get_archive
        )

        project_id = get_or_create_project('/workspaces/test3')
        session_db_id = upsert_session(
            project_id=project_id,
            session_id='archive-test-session',
            summary='Archive test',
        )

        # Store archive
        content = '{"type":"user","message":{"content":"test"}}\n'
        archive_id = upsert_archive(
            session_db_id=session_db_id,
            content=content,
            content_hash='abc123',
        )
        assert archive_id > 0

        # Retrieve archive
        retrieved = get_archive('archive-test-session')
        assert retrieved == content

    def test_local_fts_search(self):
        """Test full-text search in local store."""
        from mira.local_store import (
            get_or_create_project, upsert_session, search_sessions_fts
        )

        project_id = get_or_create_project('/workspaces/fts-test')
        upsert_session(
            project_id=project_id,
            session_id='fts-session-1',
            summary='Discussion about Python and Flask web development',
            keywords=['python', 'flask', 'web'],
        )
        upsert_session(
            project_id=project_id,
            session_id='fts-session-2',
            summary='Debugging JavaScript React components',
            keywords=['javascript', 'react'],
        )

        # Search for Python
        results = search_sessions_fts('python', project_id=project_id)
        assert len(results) >= 1
        assert any(s['session_id'] == 'fts-session-1' for s in results)

        # Search for React
        results = search_sessions_fts('react', project_id=project_id)
        assert len(results) >= 1
        assert any(s['session_id'] == 'fts-session-2' for s in results)

    def test_local_custodian_operations(self):
        """Test custodian preferences in local store."""
        from mira.local_store import upsert_custodian, get_custodian_all

        upsert_custodian(
            key='preference:editor',
            value='vscode',
            category='preference',
            confidence=0.8,
            source_session='test-session',
        )

        prefs = get_custodian_all()
        assert len(prefs) >= 1
        editor_pref = next((p for p in prefs if p['key'] == 'preference:editor'), None)
        assert editor_pref is not None
        assert editor_pref['value'] == 'vscode'


class TestStorageFallback:
    """Test storage abstraction layer fallback behavior."""

    @classmethod
    def setup_class(cls):
        # Reset db_manager and storage to ensure fresh state
        shutdown_db_manager()
        from mira.storage import reset_storage
        from mira import local_store
        reset_storage()
        local_store._initialized = False

        cls.mira_path = Path(tempfile.mkdtemp())
        os.environ['MIRA_PATH'] = str(cls.mira_path)
        # Ensure no server.json exists (forces local mode)
        server_json = cls.mira_path / 'server.json'
        if server_json.exists():
            server_json.unlink()

    @classmethod
    def teardown_class(cls):
        from mira.storage import reset_storage
        reset_storage()
        shutdown_db_manager()
        if cls.mira_path.exists():
            shutil.rmtree(cls.mira_path, ignore_errors=True)
        if 'MIRA_PATH' in os.environ:
            del os.environ['MIRA_PATH']

    def test_storage_mode_without_config(self):
        """Test that storage falls back to local when no config exists."""
        from mira.storage import Storage, reset_storage

        reset_storage()
        storage = Storage()

        # Should not be using central (no config)
        assert storage.using_central == False

        mode = storage.get_storage_mode()
        assert mode['mode'] == 'local'
        assert 'limitations' in mode
        assert len(mode['limitations']) > 0

    def test_storage_local_fallback_operations(self):
        """Test that storage operations work in local mode."""
        from mira.storage import Storage, reset_storage

        reset_storage()
        storage = Storage()

        # Project operations should work locally
        project_id = storage.get_or_create_project('/test/project')
        assert project_id is not None

        # Session operations should work locally
        session_id = storage.upsert_session(
            project_path='/test/project',
            session_id='local-test-session',
            summary='Testing local fallback',
            keywords=['test'],
        )
        assert session_id is not None

        # Recent sessions should work
        sessions = storage.get_recent_sessions(limit=5)
        assert isinstance(sessions, list)

        # FTS search should work
        results = storage.search_sessions_fts('test')
        assert isinstance(results, list)

    def test_local_first_writes(self):
        """Test that writes always go to local SQLite first."""
        from mira.storage import Storage, reset_storage

        reset_storage()
        storage = Storage()

        # Upsert session should succeed even without central
        result = storage.upsert_session(
            project_path='/test/project',
            session_id='test-session-local-first',
            summary='Test summary',
            keywords=['test', 'local'],
        )
        # Should return local ID (not None)
        assert result is not None

        # Should be able to get it back from local
        from mira import local_store
        session = local_store.get_session_by_uuid('test-session-local-first')
        assert session is not None
        assert session['summary'] == 'Test summary'

    def test_storage_health_check_local_mode(self):
        """Test health check in local mode."""
        from mira.storage import Storage, reset_storage

        reset_storage()
        storage = Storage()

        health = storage.health_check()
        assert health['mode'] == 'local'
        assert health['using_central'] == False

    def test_session_exists_in_central_local_mode(self):
        """Test session_exists_in_central returns False in local mode."""
        from mira.storage import Storage, reset_storage

        reset_storage()
        storage = Storage()

        # In local mode, should always return False (no central storage)
        assert storage.session_exists_in_central('any-session-id') == False


class TestLocalToCentralSync:
    """Test sync detection from local to central storage."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        from mira.storage import reset_storage
        from mira import local_store
        reset_storage()
        local_store._initialized = False

        cls.mira_path = Path(tempfile.mkdtemp())
        os.environ['MIRA_PATH'] = str(cls.mira_path)

        # Create metadata directory
        (cls.mira_path / 'metadata').mkdir(parents=True, exist_ok=True)

    @classmethod
    def teardown_class(cls):
        from mira.storage import reset_storage
        reset_storage()
        shutdown_db_manager()
        if cls.mira_path.exists():
            shutil.rmtree(cls.mira_path, ignore_errors=True)
        if 'MIRA_PATH' in os.environ:
            del os.environ['MIRA_PATH']

    def test_sync_detection_skips_unchanged_in_local_mode(self):
        """Test that unchanged local sessions are skipped in local mode."""
        from mira.storage import Storage, reset_storage
        from mira.ingestion import ingest_conversation

        reset_storage()
        storage = Storage()

        # Create a test conversation file
        test_session_id = 'sync-test-session-001'
        test_file = self.mira_path / 'test_conversation.jsonl'
        test_file.write_text(json.dumps({
            'type': 'user',
            'message': {'role': 'user', 'content': 'Test message'}
        }) + '\n')

        # Create metadata file to simulate already-ingested session
        meta_file = self.mira_path / 'metadata' / f'{test_session_id}.json'
        meta_file.write_text(json.dumps({
            'last_modified': '2025-01-01T00:00:00',
            'summary': 'Test session'
        }))

        # Ingest should skip (already processed, local mode)
        file_info = {
            'session_id': test_session_id,
            'file_path': str(test_file),
            'project_path': '/test/project',
            'last_modified': '2025-01-01T00:00:00'  # Same as metadata
        }

        result = ingest_conversation(file_info, None, self.mira_path, storage)
        assert result == False  # Should be skipped

    def test_sync_detection_processes_modified_file(self):
        """Test that modified files are re-ingested."""
        from mira.storage import Storage, reset_storage
        from mira.ingestion import ingest_conversation

        reset_storage()
        storage = Storage()

        test_session_id = 'sync-test-session-002'
        test_file = self.mira_path / 'test_conversation2.jsonl'
        test_file.write_text(json.dumps({
            'type': 'user',
            'message': {'role': 'user', 'content': 'Test message for sync'}
        }) + '\n' + json.dumps({
            'type': 'assistant',
            'message': {'role': 'assistant', 'content': 'Response message'}
        }) + '\n')

        # Create OLD metadata file
        meta_file = self.mira_path / 'metadata' / f'{test_session_id}.json'
        meta_file.write_text(json.dumps({
            'last_modified': '2025-01-01T00:00:00',
            'summary': 'Old session'
        }))

        # Ingest with NEWER modification time
        file_info = {
            'session_id': test_session_id,
            'file_path': str(test_file),
            'project_path': '/test/project',
            'last_modified': '2025-01-02T00:00:00'  # Newer than metadata
        }

        result = ingest_conversation(file_info, None, self.mira_path, storage)
        assert result == True  # Should be ingested (file modified)


class TestCentralStorageIntegration:
    """Integration tests that verify central storage flow without actual connections."""

    def test_storage_init_without_config(self):
        """Test Storage initialization without central config."""
        from mira.storage import Storage
        from mira.config import ServerConfig

        config = ServerConfig(version=1, central=None)
        storage = Storage(config=config)

        assert storage._using_central is False
        assert storage._qdrant is None
        assert storage._postgres is None

    def test_storage_central_enabled_property(self):
        """Test storage correctly reports central availability."""
        from mira.storage import Storage
        from mira.config import ServerConfig

        config = ServerConfig(version=1, central=None)
        storage = Storage(config=config)

        # Without central config, should report local mode
        assert storage.config.central_enabled is False
