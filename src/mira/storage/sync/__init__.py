"""
MIRA Sync Module

Background sync worker for pushing local changes to central storage.
"""

import logging
import threading
from typing import Optional

log = logging.getLogger(__name__)

# Global sync worker state
_sync_worker_thread: Optional[threading.Thread] = None
_sync_worker_stop_event: Optional[threading.Event] = None


def start_sync_worker(storage) -> bool:
    """
    Start the background sync worker.

    The sync worker periodically pushes queued items to central storage.

    Args:
        storage: Storage instance with central backend

    Returns:
        True if worker started, False if already running or not needed
    """
    global _sync_worker_thread, _sync_worker_stop_event

    if _sync_worker_thread is not None and _sync_worker_thread.is_alive():
        log.debug("Sync worker already running")
        return False

    if not storage or not storage.using_central:
        log.debug("Sync worker not needed (no central storage)")
        return False

    _sync_worker_stop_event = threading.Event()

    def sync_loop():
        """Background sync loop."""
        from .queue import get_sync_queue

        log.info("Sync worker started")
        queue = get_sync_queue()

        while not _sync_worker_stop_event.is_set():
            try:
                # Process pending items
                pending = queue.get_pending(limit=10)
                for item in pending:
                    if _sync_worker_stop_event.is_set():
                        break
                    try:
                        _sync_item(storage, item)
                        queue.mark_completed(item['id'])
                    except Exception as e:
                        log.error(f"Sync failed for {item['data_type']}: {e}")
                        queue.mark_failed(item['id'], str(e))

                # Wait before next poll
                _sync_worker_stop_event.wait(timeout=30)

            except Exception as e:
                log.error(f"Sync loop error: {e}")
                _sync_worker_stop_event.wait(timeout=60)

        log.info("Sync worker stopped")

    _sync_worker_thread = threading.Thread(
        target=sync_loop,
        daemon=True,
        name="SyncWorker"
    )
    _sync_worker_thread.start()
    return True


def stop_sync_worker():
    """Stop the background sync worker."""
    global _sync_worker_thread, _sync_worker_stop_event

    if _sync_worker_stop_event:
        _sync_worker_stop_event.set()

    if _sync_worker_thread and _sync_worker_thread.is_alive():
        _sync_worker_thread.join(timeout=5)

    _sync_worker_thread = None
    _sync_worker_stop_event = None


def _sync_item(storage, item: dict):
    """Sync a single item to central storage."""
    data_type = item.get('data_type')
    payload = item.get('payload', {})

    if data_type == 'session':
        # Sync session to central
        if storage.postgres:
            storage.postgres.upsert_session(**payload)
    elif data_type == 'project':
        # Sync project to central
        if storage.postgres:
            storage.postgres.get_or_create_project(**payload)
    else:
        log.warning(f"Unknown sync data type: {data_type}")


__all__ = [
    "start_sync_worker",
    "stop_sync_worker",
]
