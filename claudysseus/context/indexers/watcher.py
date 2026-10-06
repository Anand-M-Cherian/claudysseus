from __future__ import annotations


import platform
import threading
from pathlib import Path


from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler, FileSystemEvent


from claudysseus.context.indexers.code_parser import ALL_EXTENSIONS, SKIP_DIRS
from claudysseus.context.indexers.factory import get_file_indexer
from claudysseus.observability.logger import get_logger


logger = get_logger(__name__)


# ------------------------------------------------------------------
# How the watcher works
# ------------------------------------------------------------------
# A watchdog Observer runs in a background daemon thread watching the
# project directory recursively. When a file changes, the OS fires an
# event → _CodebaseEventHandler receives it → debounces it (editors
# emit multiple events per save) → calls index_single_file() or
# remove_file_from_index() on the active indexer (chosen by
# factory.get_file_indexer(), same config rules as get_indexer()).
#
# This keeps the vector store up-to-date in real time so /ask queries reflect
# changes made by /plan tasks or manual edits without needing a restart.
# ------------------------------------------------------------------


# Editors often fire several events for a single save (write + chmod +
# rename swap). Wait this long after the last event before acting —
# any new event for the same file resets the timer.
_DEBOUNCE_SECONDS = 1.5


def _get_observer() -> Observer:
    """
    Return the best available watchdog observer for the current OS.


    macOS  → FSEvents   (native, event-driven, instant)
    Linux  → inotify    (native, event-driven, instant)
    Windows → PollingObserver — ReadDirectoryChangesW has edge cases
              under heavy load and on network drives, so we poll every
              2 seconds instead. Slightly delayed but reliable.
    """
    if platform.system() == "Windows":
        return PollingObserver(timeout=2)
    return Observer()


class _CodebaseEventHandler(FileSystemEventHandler):
    """
    Translates raw watchdog filesystem events into debounced indexer calls.


    Only reacts to files with extensions the indexer understands (ALL_EXTENSIONS
    from code_parser). Directory events and unrecognised extensions are ignored.


    Debounce pattern:
      Each file gets its own threading.Timer. If a new event arrives for the
      same file before the timer fires, the old timer is cancelled and a new
      one starts. The indexer is only called once the file has been stable for
      _DEBOUNCE_SECONDS.
    """

    def __init__(self) -> None:
        # {filepath: threading.Timer} — one pending debounced call per file.
        # Timers are daemon threads so they don't block process exit.
        self._timers: dict[str, threading.Timer] = {}
        # Resolved once from config so the watcher writes to the same
        # vector store that get_indexer() built and /ask searches.
        self._index_file, self._remove_file = get_file_indexer()

    def _is_indexable(self, path: str) -> bool:
        """
        True if the indexer would pick this file up on a full scan:
        a known extension, and not inside a skipped dir (.venv, .git, ...).
        """
        p = Path(path)
        return p.suffix.lower() in ALL_EXTENSIONS and not any(part in SKIP_DIRS for part in p.parts)

    def _schedule(self, action: str, filepath: str) -> None:
        """
        Schedule a debounced indexer call for filepath.
        Cancels any already-pending call for the same file first.
        action is either "upsert" (create/modify) or "delete".
        """
        existing = self._timers.pop(filepath, None)
        if existing:
            existing.cancel()

        timer = threading.Timer(
            _DEBOUNCE_SECONDS, self._run, args=(action, filepath))
        timer.daemon = True
        timer.start()
        self._timers[filepath] = timer

    def _run(self, action: str, filepath: str) -> None:
        """
        Actual indexer call, executed after the debounce window expires.


        Runs on a Timer thread, so errors are logged here rather than
        surfacing as an unhandled thread exception.
        """
        self._timers.pop(filepath, None)
        try:
            if action == "delete":
                self._remove_file(filepath)
            else:
                self._index_file(filepath)
        except Exception as e:
            logger.error(f"Watchdog failed to {action} {filepath}: {e}")

    # ── watchdog event hooks ──────────────────────────────────────────

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_indexable(event.src_path):
            self._schedule("upsert", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_indexable(event.src_path):
            self._schedule("upsert", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_indexable(event.src_path):
            self._schedule("delete", event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        # A rename/move = delete the old path + upsert the new path.
        # Both paths are checked independently since either could be non-indexable.
        if not event.is_directory:
            if self._is_indexable(event.src_path):
                self._schedule("delete", event.src_path)
            if self._is_indexable(event.dest_path):
                self._schedule("upsert", event.dest_path)


def start_watcher(repo_path: str) -> Observer:
    """
    Start a filesystem observer on repo_path in a background daemon thread.


    The observer watches recursively — all subdirectories are covered.
    Daemon=True means the thread won't prevent the process from exiting
    if the user hits Ctrl+C without going through /exit.


    Returns the Observer so the caller can call stop_watcher() on shutdown.
    """
    handler = _CodebaseEventHandler()
    observer = _get_observer()
    observer.schedule(handler, repo_path, recursive=True)
    observer.daemon = True
    observer.start()
    logger.info(
        f"Watchdog started on {repo_path} (backend: {type(observer).__name__})")
    return observer


def stop_watcher(observer: Observer) -> None:
    """
    Cleanly stop the observer and wait for its thread to finish.
    Called in the finally block of _run_async() in main.py so it
    always runs regardless of how the REPL exits.
    """
    observer.stop()
    observer.join()
    logger.info("Watchdog stopped")
