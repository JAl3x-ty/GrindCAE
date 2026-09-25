"""Phase 5B2 Tkinter desktop GUI and testable application services.

The package stays lightweight so the diagnostics entry can report a missing
runtime dependency instead of failing during package initialization. Public
phase 5A names remain available through lazy imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "GuiController",
    "GuiEvent",
    "GuiForm",
    "GuiInputError",
    "GuiRunResult",
]


_PUBLIC_MODULES = {
    "GuiController": ".controller",
    "GuiEvent": ".controller",
    "GuiRunResult": ".controller",
    "GuiForm": ".form",
    "GuiInputError": ".form",
}


def __getattr__(name: str) -> Any:
    module_name = _PUBLIC_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
