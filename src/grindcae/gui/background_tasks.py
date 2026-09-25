"""Shared mutual-exclusion gate for workbench background calculations."""

from __future__ import annotations

from threading import Lock


class BackgroundTaskGate:
    def __init__(self) -> None:
        self._lock = Lock()
        self._owner: str | None = None

    @property
    def owner(self) -> str | None:
        with self._lock:
            return self._owner

    def try_acquire(self, owner: str) -> bool:
        with self._lock:
            if self._owner is not None:
                return False
            self._owner = owner
            return True

    def release(self, owner: str) -> None:
        with self._lock:
            if self._owner == owner:
                self._owner = None
