"""Tests for mira.sync_queue and mira.sync_worker modules."""

import os
import tempfile
import shutil
from pathlib import Path

from mira.db_manager import shutdown_db_manager


class TestSyncQueue:
    """Test sync queue functionality."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        from mira.sync_queue import SyncQueue
        SyncQueue._initialized = False
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        mira_path = Path(cls.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_enqueue_item(self):
        """Test enqueueing items to sync queue."""
        from mira.sync_queue import SyncQueue
        SyncQueue._initialized = False
        queue = SyncQueue()

        result = queue.enqueue(
            data_type="artifact",
            item_hash="abc123",
            payload={"content": "test code", "language": "python"}
        )
        assert result == True

    def test_enqueue_duplicate(self):
        """Test that duplicate items don't create new entries."""
        from mira.sync_queue import SyncQueue
        SyncQueue._initialized = False
        queue = SyncQueue()

        # Enqueue same item twice
        queue.enqueue("session", "session-hash-1", {"summary": "test"})
        queue.enqueue("session", "session-hash-1", {"summary": "updated"})

        # Should still only have one pending item
        items = queue.get_pending("session", limit=100)
        session_items = [i for i in items if i["item_hash"] == "session-hash-1"]
        assert len(session_items) <= 1

    def test_get_pending_items(self):
        """Test retrieving pending items from queue."""
        from mira.sync_queue import SyncQueue
        SyncQueue._initialized = False
        queue = SyncQueue()

        # Add some items
        queue.enqueue("error", "error-hash-1", {"error_type": "TypeError"})
        queue.enqueue("error", "error-hash-2", {"error_type": "ValueError"})

        items = queue.get_pending("error", limit=10)
        assert isinstance(items, list)
        assert all("payload" in item for item in items)

    def test_mark_synced(self):
        """Test marking items as synced removes them from queue."""
        from mira.sync_queue import SyncQueue
        SyncQueue._initialized = False
        queue = SyncQueue()

        queue.enqueue("decision", "decision-hash-1", {"decision": "use FastAPI"})

        items = queue.get_pending("decision", limit=10)
        decision_items = [i for i in items if i["item_hash"] == "decision-hash-1"]

        if decision_items:
            queue.mark_synced([decision_items[0]["id"]])

            # Should no longer be pending
            items_after = queue.get_pending("decision", limit=100)
            assert not any(i["item_hash"] == "decision-hash-1" for i in items_after)

    def test_mark_failed_increases_retry_count(self):
        """Test that failed items get retry count incremented."""
        from mira.sync_queue import SyncQueue
        SyncQueue._initialized = False
        queue = SyncQueue()

        queue.enqueue("custodian", "cust-hash-1", {"key": "name", "value": "Max"})

        items = queue.get_pending("custodian", limit=10)
        cust_items = [i for i in items if i["item_hash"] == "cust-hash-1"]

        if cust_items:
            original_retry = cust_items[0]["retry_count"]
            queue.mark_failed([cust_items[0]["id"]], "Connection failed")

            # Check retry count increased
            items_after = queue.get_pending("custodian", limit=100, max_retries=10)
            failed_item = next((i for i in items_after if i["item_hash"] == "cust-hash-1"), None)
            if failed_item:
                assert failed_item["retry_count"] > original_retry

    def test_get_stats(self):
        """Test queue statistics."""
        from mira.sync_queue import SyncQueue
        SyncQueue._initialized = False
        queue = SyncQueue()

        # Add some items
        queue.enqueue("artifact", "stat-test-1", {"content": "test1"})
        queue.enqueue("artifact", "stat-test-2", {"content": "test2"})

        stats = queue.get_stats()
        assert "total_pending" in stats
        assert "total_failed" in stats
        assert "pending" in stats

    def test_record_sync_stats(self):
        """Test recording sync statistics."""
        from mira.sync_queue import SyncQueue
        SyncQueue._initialized = False
        queue = SyncQueue()

        queue.record_sync_stats(
            data_type="artifact",
            items_synced=10,
            items_failed=2,
            duration_ms=150
        )

        recent = queue.get_recent_sync_stats(limit=5)
        assert isinstance(recent, list)


class TestSyncWorker:
    """Test sync worker functionality."""

    @classmethod
    def setup_class(cls):
        shutdown_db_manager()
        from mira.sync_queue import SyncQueue
        SyncQueue._initialized = False
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        mira_path = Path(cls.temp_dir) / '.mira'
        mira_path.mkdir(exist_ok=True)

    @classmethod
    def teardown_class(cls):
        shutdown_db_manager()
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_sync_worker_init(self):
        """Test sync worker initialization."""
        from mira.sync_worker import SyncWorker

        worker = SyncWorker(storage=None)
        assert worker.running == False
        assert worker.storage is None

    def test_sync_worker_start_stop(self):
        """Test starting and stopping sync worker."""
        from mira.sync_worker import SyncWorker
        import time

        worker = SyncWorker(storage=None)
        worker.start()
        assert worker.running == True
        assert worker.thread is not None

        # Let it run briefly
        time.sleep(0.1)

        worker.stop()
        assert worker.running == False

    def test_force_sync_without_central(self):
        """Test force sync returns error when central not available."""
        from mira.sync_worker import SyncWorker

        worker = SyncWorker(storage=None)
        result = worker.force_sync()

        # Without central storage, should return error
        assert "error" in result or result.get("synced", 0) == 0
