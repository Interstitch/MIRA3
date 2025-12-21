"""Test that all major imports work correctly."""

import pytest


class TestCoreImports:
    """Test core module imports."""

    def test_core_imports(self):
        from mira.core import (
            log,
            VERSION,
            get_mira_path,
            get_config,
            get_db_manager,
            shutdown_db_manager,
        )
        assert VERSION is not None
        assert callable(log)
        assert callable(get_mira_path)

    def test_core_constants(self):
        from mira.core import (
            DB_LOCAL_STORE,
            DB_ARTIFACTS,
            DB_CUSTODIAN,
            DB_INSIGHTS,
            DB_CONCEPTS,
        )
        assert DB_LOCAL_STORE.endswith('.db')

    def test_core_config(self):
        from mira.core import ServerConfig, CentralConfig, VERSION
        config = ServerConfig(version=VERSION)
        assert hasattr(config, 'central')


class TestStorageImports:
    """Test storage module imports."""

    def test_storage_imports(self):
        from mira.storage import get_storage, Storage
        assert callable(get_storage)


class TestExtractionImports:
    """Test extraction module imports."""

    def test_extraction_imports(self):
        from mira.extraction import (
            extract_metadata,
            extract_artifacts_from_messages,
            extract_insights_from_conversation,
            extract_concepts_from_conversation,
            extract_errors_from_conversation,
            extract_decisions_from_conversation,
        )
        assert callable(extract_metadata)
        assert callable(extract_artifacts_from_messages)

    def test_code_history_imports(self):
        from mira.extraction import (
            init_code_history_db,
            get_file_timeline,
            get_code_history_stats,
        )
        assert callable(init_code_history_db)


class TestSearchImports:
    """Test search module imports."""

    def test_search_imports(self):
        from mira.search import (
            handle_search,
            fulltext_search_archives,
        )
        assert callable(handle_search)

    def test_fuzzy_imports(self):
        from mira.search import (
            damerau_levenshtein_distance,
            find_closest_match,
        )
        assert callable(damerau_levenshtein_distance)

    def test_local_semantic_imports(self):
        from mira.search import (
            start_local_indexer,
            stop_local_indexer,
        )
        assert callable(start_local_indexer)


class TestCustodianImports:
    """Test custodian module imports."""

    def test_custodian_imports(self):
        from mira.custodian import (
            init_custodian_db,
            extract_custodian_learnings,
            get_full_custodian_profile,
        )
        assert callable(init_custodian_db)


class TestIngestionImports:
    """Test ingestion module imports."""

    def test_ingestion_imports(self):
        from mira.ingestion import (
            ingest_conversation,
            run_full_ingestion,
            discover_conversations,
        )
        assert callable(ingest_conversation)

    def test_watcher_imports(self):
        from mira.ingestion.watcher import (
            ConversationWatcher,
            run_file_watcher,
        )
        assert callable(run_file_watcher)


class TestToolsImports:
    """Test tools module imports."""

    def test_tools_imports(self):
        from mira.tools import (
            handle_init,
            handle_search,
            handle_recent,
            handle_error_lookup,
            handle_decisions,
            handle_code_history,
            handle_status,
        )
        assert callable(handle_init)
        assert callable(handle_search)
        assert callable(handle_status)
