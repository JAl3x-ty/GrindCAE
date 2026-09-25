"""Shared Gmsh initialization policy for main-thread and worker execution."""

from __future__ import annotations

from threading import current_thread, main_thread
from typing import Protocol, Sequence


class GmshApi(Protocol):
    def initialize(
        self,
        argv: Sequence[str],
        *,
        interruptible: bool,
    ) -> None: ...


def initialize_gmsh(gmsh_api: GmshApi, argv: Sequence[str]) -> None:
    """Initialize Gmsh without registering Python signals in worker threads."""

    gmsh_api.initialize(
        list(argv),
        interruptible=current_thread() is main_thread(),
    )


__all__ = ["initialize_gmsh"]
